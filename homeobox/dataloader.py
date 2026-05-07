"""Fast batch dataloader for ML training from homeobox atlases.

:class:`UnimodalHoxDataset` is a pure data-access object: it resolves zarr
remaps and exposes ``__getitems__`` for batched async I/O.

Designed for the ``query -> UnimodalHoxDataset -> SparseBatch -> collate_fn ->
GPU`` pipeline.  Reader initialisation is deferred to the worker
process, making the dataset safely picklable for spawn-based
multiprocessing.

Pointer structs and metadata columns are loaded lazily per-batch via
lance's ``take_row_ids`` API rather than materializing the full obs
table at init time.

Usage::

    dataset = atlas.query().to_unimodal_dataset("gene_expression", "counts", metadata_columns=["cell_type"])
    loader = make_loader(dataset, batch_size=256, shuffle=True, num_workers=4)
    for batch in loader:
        X = sparse_to_dense_collate(batch)["X"]
"""

import asyncio
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import lancedb
import numpy as np
import polars as pl
from polars.dataframe.group_by import GroupBy

if TYPE_CHECKING:
    from homeobox.atlas import RaggedAtlas
from homeobox.batch_types import (
    DenseFeatureBatch,
    MultimodalBatch,
    SparseBatch,
    SpatialTileBatch,
)
from homeobox.group_reader import GroupReader
from homeobox.group_specs import FeatureSpaceSpec, get_spec
from homeobox.pointer_types import DenseZarrPointer, DiscreteSpatialPointer, SparseZarrPointer
from homeobox.read import _prepare_obs_and_groups
from homeobox.reconstruction_functional import (
    RowOrderMapping,
    _reorder_dense_feature_batch_rows,  # transitional: until dense take is refactored
    _reorder_spatial_tile_batch_rows,  # transitional: until spatial take is refactored
    collect_group_readers_from_atlas,
    collect_remapped_layout_readers_from_atlas,
    concat_remapped_batches,
    get_array_paths_to_read,
    read_arrays_by_group,
    reorder_batch_rows,
)

# ---------------------------------------------------------------------------
# Shared helpers / mixin
# ---------------------------------------------------------------------------


def _build_present_arrays(
    present_indices: np.ndarray,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build presence mask and per-row position index for one modality.

    Returns ``(present_mask, row_positions)`` where:

    - ``present_mask[i]`` is True if row *i* has this modality
    - ``row_positions[i]`` is the index into the modality's present-row arrays, or -1 if absent
    """
    present_mask = np.zeros(n_rows, dtype=bool)
    row_positions = np.full(n_rows, -1, dtype=np.int64)
    if len(present_indices) > 0:
        present_mask[present_indices] = True
        row_positions[present_indices] = np.arange(len(present_indices), dtype=np.int64)
    return present_mask, row_positions


def _build_modality_data(
    atlas: "RaggedAtlas",
    rows_indexed: pl.DataFrame,
    # TODO: type these
    pf,
    spec: FeatureSpaceSpec,
    layer_overrides: list[str] | None,
    wanted_globals: np.ndarray | None,
    n_rows: int,
) -> "tuple[pl.DataFrame, _ModalityData]":
    """Build ``_ModalityData`` for any pointer-field modality.

    Spec-driven: structural array paths and the layer paths come from
    :func:`get_array_paths_to_read`; group-reader assembly branches on
    ``spec.has_var_df`` (feature-registry-backed modalities get remapped
    layouts, others don't). When ``layer_overrides`` is None, all required
    layers from the spec are read.

    Returns ``(filtered_rows, modality_data)`` where *filtered_rows*
    is the DataFrame after empty-row removal (with internal columns
    added).
    """
    if wanted_globals is not None and not spec.has_var_df:
        raise ValueError(
            f"wanted_globals is only valid for feature spaces with has_var_df=True; "
            f"feature space '{spec.feature_space}' has has_var_df=False"
        )

    fs = spec.feature_space
    required_array_paths, layer_array_paths = get_array_paths_to_read(spec, layer_overrides)

    filtered, groups = _prepare_obs_and_groups(rows_indexed, spec.pointer_type, pf.field_name)

    present_indices = filtered["_orig_idx"].to_numpy().astype(np.int64)
    present_mask, row_positions = _build_present_arrays(present_indices, n_rows)

    if spec.has_var_df:
        layouts_per_group = collect_remapped_layout_readers_from_atlas(
            atlas, groups, spec, wanted_globals=wanted_globals
        )
    else:
        layouts_per_group = None
    group_readers = collect_group_readers_from_atlas(
        atlas, groups, spec, layouts_per_group=layouts_per_group, for_worker=True
    )

    if spec.has_var_df:
        n_features = (
            len(wanted_globals)
            if wanted_globals is not None
            else atlas.registry_tables[fs].count_rows()
        )
    else:
        n_features = 0

    if group_readers:
        first_reader = next(iter(group_readers.values()))
        layer_dtypes = {
            name: first_reader.get_array_reader(path)._native_dtype
            for name, path in layer_array_paths.items()
        }
    else:
        layer_dtypes = {name: np.dtype(np.float32) for name in layer_array_paths}

    mod_data = _ModalityData(
        spec=spec,
        group_readers=group_readers,
        n_features=n_features,
        required_array_paths=list(required_array_paths),
        layer_array_paths=dict(layer_array_paths),
        layer_dtypes=layer_dtypes,
        present_mask=present_mask,
        row_positions=row_positions,
    )
    return filtered, mod_data


def _single_layer(layers: dict):
    """Return the sole layer's data, or raise if there isn't exactly one."""
    if len(layers) != 1:
        raise ValueError(
            f"Built-in collate expected exactly one layer, got {list(layers)}. "
            "Use a custom collate to handle multi-layer batches."
        )
    return next(iter(layers.values()))


def _select_obs_metadata(obs_pl: pl.DataFrame) -> pl.DataFrame | None:
    """Select non-internal, non-Struct columns to carry as batch metadata."""
    cols = [
        col
        for col, dtype in obs_pl.schema.items()
        if not col.startswith("_") and dtype.base_type() != pl.Struct
    ]
    return obs_pl.select(cols) if cols else None


def _metadata_to_tensor_dict(metadata: pl.DataFrame | None) -> dict:
    """Convert a metadata DataFrame to a dict of tensors (numeric) / numpy arrays (other)."""
    if metadata is None:
        return {}
    import torch

    out: dict = {}
    for col in metadata.columns:
        arr = metadata[col].to_numpy()
        if arr.dtype.kind in ("i", "u", "f"):
            out[col] = torch.from_numpy(arr)
        else:
            out[col] = arr
    return out


def _sparse_batch_to_dense_tensor(batch: "SparseBatch"):
    """Scatter a SparseBatch into a dense float32 torch tensor (n_rows, n_features)."""
    import torch

    values = _single_layer(batch.layers)
    n_rows = len(batch.offsets) - 1
    X = torch.zeros(n_rows, batch.n_features, dtype=torch.float32)
    if n_rows > 0 and len(batch.indices) > 0:
        lengths = np.diff(batch.offsets)
        row_indices = np.repeat(np.arange(n_rows), lengths)
        X[row_indices, batch.indices] = torch.from_numpy(values.astype(np.float32))
    return X


def _reorder_take_result(result: pl.DataFrame, batch_row_ids: np.ndarray) -> pl.DataFrame:
    """Reorder ``take_row_ids`` result to match the input order of *batch_row_ids*.

    ``take_row_ids`` returns rows sorted by ``_rowid``.  This function
    computes the inverse permutation so the output aligns with the
    caller's requested order.
    """
    returned_ids = result["_rowid"].to_numpy().astype(np.uint64)
    # Build mapping: for each returned_id, find its position in batch_row_ids.
    # Both are uint64; we sort batch_row_ids and use searchsorted.
    sort_perm = np.argsort(batch_row_ids)
    inv_perm = np.empty_like(sort_perm)
    inv_perm[sort_perm] = np.arange(len(sort_perm))
    # returned_ids are already sorted by _rowid; sort batch_row_ids to match
    sorted_batch_ids = batch_row_ids[sort_perm]
    # Map each returned_id to its position in the original batch_row_ids order
    positions = np.searchsorted(sorted_batch_ids, returned_ids)
    reorder = inv_perm[positions]
    return result[reorder.tolist()]


class _AsyncDataset:
    """Mixin providing shared async event loop lifecycle for dataset classes."""

    def _start_event_loop(self) -> None:
        """Start the background async event loop thread."""
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def __del__(self) -> None:
        if hasattr(self, "_loop") and self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=5)
            self._loop.close()


# Used insted of a lambda function because pickle doesn't like lambdas
def _identity_collate(x):
    return x


# TODO: This should be a frozen dataclass
@dataclass
class _ModalityData:
    """Pre-computed per-modality metadata for UnimodalHoxDataset and MultimodalHoxDataset.

    Built at ``__init__`` time; all fields are picklable.  Does NOT store
    per-row pointer arrays (starts/ends/groups_np) — those are loaded
    lazily per batch via lance ``take_row_ids``.
    """

    spec: FeatureSpaceSpec
    group_readers: dict[str, GroupReader]
    # Full size of the feature registry, unless wanted_globals filtered it; 0
    # when ``spec.has_var_df`` is False (e.g. spatial tiles have no registry).
    n_features: int
    # Structural array paths from spec.reconstructor.required_arrays; consumers
    # that depend on a single index array (e.g. sparse readers) assert len == 1.
    required_array_paths: list[str]
    # ``{layer_name: full zarr path}`` (e.g. ``{"counts": "csr/layers/counts"}``).
    layer_array_paths: dict[str, str]
    layer_dtypes: dict[str, np.dtype]  # layer_name -> native dtype
    # TODO: Multimodal fields; maybe move to a separate class that inherits
    present_mask: np.ndarray | None = None  # bool, (n_total_rows,); None for UnimodalHoxDataset
    row_positions: np.ndarray | None = None  # int64, (n_total_rows,); None for UnimodalHoxDataset


# ---------------------------------------------------------------------------
# Async primitives
# ---------------------------------------------------------------------------


# TODO: This is assuming that just because a pointer is sparse, that it also
# has layers. This is true for gene_expression but not necessarily a fundamental
# feature of sparse data.
async def _take_sparse_from_pointers(
    groups: GroupBy,
    spec: FeatureSpaceSpec,
    batch_row_ids: np.ndarray,
    mod_data: _ModalityData,
) -> SparseBatch:
    """Fetch a sparse batch from per-row pointer arrays."""
    layer_names = list(mod_data.layer_array_paths.keys())
    group_batches = read_arrays_by_group(
        mod_data.group_readers,
        groups,
        spec=spec,
        required_array_paths=mod_data.required_array_paths,
        layer_array_paths=mod_data.layer_array_paths,
    )
    if not group_batches:
        return SparseBatch(
            indices=np.array([], dtype=np.int32),
            offsets=np.zeros(len(batch_row_ids) + 1, dtype=np.int64),
            layers={name: np.array([], dtype=mod_data.layer_dtypes[name]) for name in layer_names},
            n_features=mod_data.n_features,
        )

    layouts_per_group = {
        zg: gr.layout_reader
        for zg, gr in mod_data.group_readers.items()
        if gr.layout_reader is not None
    }
    batch = concat_remapped_batches(
        group_batches,
        layouts_per_group=layouts_per_group or None,
        n_features=mod_data.n_features,
    )

    mapping = RowOrderMapping(
        source_row_ids=batch.metadata["_rowid"].to_numpy().astype(batch_row_ids.dtype, copy=False),
        target_row_ids=batch_row_ids,
    )
    batch = reorder_batch_rows(batch, mapping)
    batch.metadata = _select_obs_metadata(batch.metadata)
    return batch


async def _take_dense_feature_from_pointers(
    groups: GroupBy,
    spec: FeatureSpaceSpec,
    batch_row_ids: np.ndarray,
    mod_data: _ModalityData,
) -> DenseFeatureBatch:
    """Fetch a dense feature batch from per-row pointer arrays."""
    out_dtype = np.dtype(np.float32)
    layer_names = list(mod_data.layer_array_paths.keys())
    layer_paths = list(mod_data.layer_array_paths.values())
    group_obs_data, results = read_arrays_by_group(
        mod_data.group_readers,
        groups,
        spec=spec,
        array_names=layer_paths,
        read_method="boxes",
        stack_uniform=True,
    )
    if not group_obs_data:
        return DenseFeatureBatch(
            layers={
                name: np.zeros((0, mod_data.n_features), dtype=out_dtype) for name in layer_names
            },
            n_features=mod_data.n_features,
        )

    obs_parts = []
    data_parts_per_layer: dict[str, list[np.ndarray]] = {name: [] for name in layer_names}
    for (_zg, group_rows), group_results in zip(group_obs_data, results, strict=True):
        for name, group_data in zip(layer_names, group_results, strict=True):
            if group_data.dtype != out_dtype:
                group_data = group_data.astype(out_dtype)
            data_parts_per_layer[name].append(group_data)
        obs_parts.append(group_rows)

    layers_data = {
        name: np.concatenate(parts, axis=0) for name, parts in data_parts_per_layer.items()
    }

    obs_pl = pl.concat(obs_parts, how="diagonal_relaxed")
    batch = DenseFeatureBatch(
        layers=layers_data,
        n_features=mod_data.n_features,
        metadata=_select_obs_metadata(obs_pl),
    )

    grouped_row_ids = obs_pl["_rowid"].to_numpy().astype(batch_row_ids.dtype, copy=False)
    row_id_order = np.argsort(grouped_row_ids, kind="stable")
    row_perm = row_id_order[np.searchsorted(grouped_row_ids[row_id_order], batch_row_ids)]
    return _reorder_dense_feature_batch_rows(batch, row_perm)


async def _take_spatial_tile_from_pointers(
    groups: GroupBy,
    spec: FeatureSpaceSpec,
    batch_row_ids: np.ndarray,
    mod_data: _ModalityData,
) -> SpatialTileBatch:
    """Fetch a spatial tile/crop batch from per-row pointer arrays."""
    layer_names = list(mod_data.layer_array_paths.keys())
    layer_paths = list(mod_data.layer_array_paths.values())
    group_obs_data, results = read_arrays_by_group(
        mod_data.group_readers,
        groups,
        spec=spec,
        array_names=layer_paths,
        read_method="boxes",
        stack_uniform=False,
    )
    if not group_obs_data:
        return SpatialTileBatch(layers={name: [] for name in layer_names})

    obs_parts = []
    layers_data: dict[str, list[np.ndarray]] = {name: [] for name in layer_names}
    for (_zg, group_rows), group_results in zip(group_obs_data, results, strict=True):
        for name, group_data in zip(layer_names, group_results, strict=True):
            if isinstance(group_data, list):
                rows = group_data
            else:
                rows = [group_data[i] for i in range(group_data.shape[0])]
            target_dtype = mod_data.layer_dtypes[name]
            layers_data[name].extend(
                row if row.dtype == target_dtype else row.astype(target_dtype) for row in rows
            )
        obs_parts.append(group_rows)

    obs_pl = pl.concat(obs_parts, how="diagonal_relaxed")
    batch = SpatialTileBatch(
        layers=layers_data,
        metadata=_select_obs_metadata(obs_pl),
    )

    grouped_row_ids = obs_pl["_rowid"].to_numpy().astype(batch_row_ids.dtype, copy=False)
    row_id_order = np.argsort(grouped_row_ids, kind="stable")
    row_perm = row_id_order[np.searchsorted(grouped_row_ids[row_id_order], batch_row_ids)]
    return _reorder_spatial_tile_batch_rows(batch, row_perm)


async def _take_multimodal(
    take_result: pl.DataFrame,
    modality_data: dict[str, _ModalityData],
    specs: dict[str, FeatureSpaceSpec],
    pointer_fields: dict[str, str],
    metadata_columns: list[str] | None,
) -> MultimodalBatch:
    """Fetch a multimodal batch from a lance take result."""
    n_rows = take_result.height

    tasks: list = []
    task_fs: list[str] = []
    present_masks: dict[str, np.ndarray] = {}

    for fs, mod_data in modality_data.items():
        pf_name = pointer_fields[fs]

        # Extract pointers for ALL batch rows for this modality
        pointer_df = take_result[pf_name].struct.unnest()
        zg_series = pointer_df["zarr_group"]
        pointer_type = mod_data.spec.pointer_type
        if pointer_type is DiscreteSpatialPointer:
            batch_present = (zg_series.is_not_null() & (zg_series != "")).to_numpy()
        else:
            batch_present = zg_series.is_not_null().to_numpy()

        present_masks[fs] = batch_present
        present_indices = np.where(batch_present)[0]

        # Extract pointers only for present rows
        present_take = take_result[present_indices.tolist()]
        present_row_ids = present_take["_rowid"].to_numpy().astype(np.uint64, copy=False)
        obs_pl, groups = _prepare_obs_and_groups(present_take, pointer_type, pf_name)
        assert len(obs_pl) == len(present_take)

        if pointer_type is SparseZarrPointer:
            tasks.append(_take_sparse_from_pointers(groups, specs[fs], present_row_ids, mod_data))
        elif pointer_type is DenseZarrPointer and specs[fs].has_var_df:
            tasks.append(
                _take_dense_feature_from_pointers(groups, specs[fs], present_row_ids, mod_data)
            )
        else:
            tasks.append(
                _take_spatial_tile_from_pointers(groups, specs[fs], present_row_ids, mod_data)
            )
        task_fs.append(fs)

    results = list(await asyncio.gather(*tasks)) if tasks else []

    modalities: dict[str, SparseBatch | DenseFeatureBatch | SpatialTileBatch] = {}
    for fs, result in zip(task_fs, results, strict=True):
        modalities[fs] = result

    metadata = None
    if metadata_columns:
        present_cols = [col for col in metadata_columns if col in take_result.columns]
        if present_cols:
            metadata = take_result.select(present_cols)

    return MultimodalBatch(
        n_rows=n_rows,
        metadata=metadata,
        modalities=modalities,
        present=present_masks,
    )


# ---------------------------------------------------------------------------
# UnimodalHoxDataset
# ---------------------------------------------------------------------------


class UnimodalHoxDataset(_AsyncDataset):
    """Map-style dataset for fast batch access over an atlas query.

    Pure data-access object: resolves zarr remaps and exposes
    :meth:`__getitems__` for batched async I/O.  Use :func:`make_loader`
    to wrap it in a ``torch.utils.data.DataLoader``.

    Pointer structs and metadata are loaded lazily per-batch via lance's
    ``take_row_ids`` API rather than being stored in memory at init time.

    Parameters
    ----------
    atlas:
        The atlas to read from.
    obs_pl:
        Polars DataFrame of row records (from a query). Must include
        ``_rowid`` column (via ``with_row_id(True)``).
    field_name:
        Pointer-field attribute name on the row schema.
    layer_overrides:
        Which layers to read within the pointer field's feature space. When
        ``None``, all required layers from the spec are read.
    metadata_columns:
        Obs column names to include as metadata on each batch.
    wanted_globals:
        Optional sorted int64 array of global feature indices to keep.
        When set, :attr:`n_features` reflects the filtered count and
        batch ``indices`` are bounded by that value.  Only valid for
        sparse feature spaces with a feature registry.
    stack_dense:
        Deprecated for spatial modalities; spatial batches are always returned
        as one ndarray per row.
    """

    def __init__(
        self,
        atlas: "RaggedAtlas",
        obs_pl: pl.DataFrame,
        # TODO: Shouldn't default this
        field_name: str = "gene_expression",
        layer_overrides: list[str] | None = None,
        metadata_columns: list[str] | None = None,
        wanted_globals: np.ndarray | None = None,
        stack_dense: bool = True,
    ) -> None:
        pf = atlas.pointer_fields[field_name]
        self.spec = get_spec(pf.feature_space)

        # Store the obstore ObjectStore (picklable via __getnewargs_ex__)
        # Workers reconstruct the zarr root lazily from this store.
        self._store = atlas.store
        self._pointer_type = self.spec.pointer_type

        # Build modality data (filters empty rows, builds remaps & readers)
        rows_indexed = obs_pl.with_row_index("_orig_idx")

        filtered, self._mod_data = _build_modality_data(
            atlas,
            rows_indexed,
            pf,
            self.spec,
            layer_overrides,
            wanted_globals,
            obs_pl.height,
        )

        self._row_ids = filtered["_rowid"].to_numpy().astype(np.uint64)
        self._n_rows = len(self._row_ids)
        self._pointer_field = field_name
        self._metadata_columns = metadata_columns
        self._lance_info = (
            atlas.db_uri,
            atlas.obs_table.name,
            atlas.obs_table.version,
            getattr(atlas.db, "storage_options", None),
        )

        # Worker-local state — initialized lazily in _ensure_initialized()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._obs_table: lancedb.table.Table | None = None

    @property
    def n_rows(self) -> int:
        return self._n_rows

    @property
    def n_features(self) -> int:
        return self._mod_data.n_features

    def __len__(self) -> int:
        return self._n_rows

    def __getitems__(
        self, row_indices: list[int]
    ) -> "SparseBatch | DenseFeatureBatch | SpatialTileBatch":
        """Fetch a batch of rows by index.

        Called by PyTorch's DataLoader when ``batch_sampler`` yields a list of
        indices (PyTorch >= 2.0 ``__getitems__`` protocol).

        Returns :class:`SparseBatch` for sparse feature spaces,
        :class:`DenseFeatureBatch` for dense feature spaces, or
        :class:`SpatialTileBatch` for spatial arrays.

        Parameters
        ----------
        row_indices:
            List of 0-based row indices into this dataset's row arrays.
        """
        self._ensure_initialized()
        indices_arr = np.array(row_indices, dtype=np.int64)

        # 1. Lance take: pointer + metadata in one call
        batch_row_ids = self._row_ids[indices_arr]
        select_cols = [self._pointer_field]
        if self._metadata_columns:
            select_cols.extend(self._metadata_columns)

        take_result = (
            self._obs_table.take_row_ids(batch_row_ids.tolist())
            .with_row_id()
            .select(select_cols)
            .to_polars()
        )

        # 2. Reorder to match input order (take_row_ids sorts by _rowid)
        take_result = _reorder_take_result(take_result, batch_row_ids)

        # 3. Extract pointer data and dispatch async read
        if self._pointer_type is SparseZarrPointer:
            take_fn = _take_sparse_from_pointers
        elif self._pointer_type is DenseZarrPointer and self.spec.has_var_df:
            take_fn = _take_dense_feature_from_pointers
        else:
            take_fn = _take_spatial_tile_from_pointers

        # This is safe because we already filtered at __init__, that
        # guarantees that this op will not drop any rows from take_result
        obs_pl, groups = _prepare_obs_and_groups(
            take_result, self._pointer_type, self._pointer_field
        )
        # Sanity check
        assert len(obs_pl) == len(take_result)
        future = asyncio.run_coroutine_threadsafe(
            take_fn(groups, self.spec, batch_row_ids, self._mod_data),
            self._loop,
        )
        return future.result()

    def __getitem__(self, idx: int) -> "SparseBatch | DenseFeatureBatch | SpatialTileBatch":
        """Fetch a single row as a batch."""
        return self.__getitems__([idx])

    def _ensure_initialized(self) -> None:
        """Start the background event loop and open the lance table if not yet done.

        Safe to call multiple times; subsequent calls are no-ops.
        Called automatically on the first ``__getitem__`` in each process,
        including spawned worker processes.
        """
        if self._loop is not None:
            return
        self._start_event_loop()
        db_uri, table_name, table_version, storage_options = self._lance_info
        db = lancedb.connect(db_uri, storage_options=storage_options)
        self._obs_table = db.open_table(table_name)
        self._obs_table.checkout(table_version)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # Drop worker-local state so the dataset is safely picklable for spawn.
        # Workers call _ensure_initialized() on their first __getitems__.
        # GroupReader.__getstate__ zeroes its own transient zarr state.
        state["_loop"] = None
        state["_loop_thread"] = None
        state["_obs_table"] = None
        return state


# ---------------------------------------------------------------------------
# MultimodalHoxDataset
# ---------------------------------------------------------------------------


class MultimodalHoxDataset(_AsyncDataset):
    """Map-style multimodal dataset for fast batch access over an atlas query.

    Supports within-row multimodal batches where each row may have data
    from multiple modalities (e.g. CITE-seq RNA + protein, multiome RNA +
    ATAC).  Yields :class:`MultimodalBatch` via :meth:`__getitems__`.

    Each modality's sub-batch contains only the rows that have it; a
    ``present`` mask tracks membership.  No synthetic fill values.

    Parameters
    ----------
    atlas:
        The atlas to read from.
    obs_pl:
        Polars DataFrame of row records (from a query). Must include
        ``_rowid`` column.
    field_names:
        Ordered list of pointer-field attribute names.
    layer_overrides:
        Optional ``{field_name: list_of_layer_names | None}`` mapping. Per
        field, ``None`` (or a missing entry) reads all required layers from
        the spec.
    metadata_columns:
        Obs column names to include as metadata on each batch.
    wanted_globals:
        Optional ``{field_name: sorted int64 array}`` of global feature
        indices to keep per modality.
    stack_dense:
        Deprecated for spatial modalities; spatial batches are always returned
        as one ndarray per row.
    """

    def __init__(
        self,
        atlas: "RaggedAtlas",
        obs_pl: pl.DataFrame,
        field_names: list[str],
        layer_overrides: dict[str, list[str] | None] | None = None,
        metadata_columns: list[str] | None = None,
        wanted_globals: dict[str, np.ndarray] | None = None,
        stack_dense: bool | dict[str, bool] = True,
    ) -> None:
        self._field_names = field_names
        self._n_rows = obs_pl.height
        self._metadata_columns = metadata_columns

        # Store lance info for lazy table reconstruction in workers
        self._lance_info = (
            atlas.db_uri,
            atlas.obs_table.name,
            atlas.obs_table.version,
            getattr(atlas.db, "storage_options", None),
        )

        # Store row IDs for lazy loading
        self._row_ids = obs_pl["_rowid"].to_numpy().astype(np.uint64)

        # Map field_name -> pointer field name (identical here, kept for parity
        # with lance take() call shape).
        self._pointer_fields: dict[str, str] = {}
        self._specs: dict[str, FeatureSpaceSpec] = {}

        # Attach row indices so we can track original positions after per-modality filters
        rows_indexed = obs_pl.with_row_index("_orig_idx")

        modality_data: dict[str, _ModalityData] = {}

        for fn in field_names:
            pf = atlas.pointer_fields[fn]
            self._pointer_fields[fn] = pf.field_name
            spec = get_spec(pf.feature_space)
            self._specs[fn] = spec
            field_layer_overrides = layer_overrides.get(fn) if layer_overrides is not None else None

            wg = wanted_globals.get(fn) if wanted_globals is not None else None
            _, modality_data[fn] = _build_modality_data(
                atlas, rows_indexed, pf, spec, field_layer_overrides, wg, self._n_rows
            )

        self._modality_data = modality_data
        self._n_features = {fn: modality_data[fn].n_features for fn in field_names}

        # Worker-local state — initialized lazily in _ensure_initialized()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._obs_table: lancedb.table.Table | None = None

    @property
    def n_rows(self) -> int:
        return self._n_rows

    @property
    def n_features(self) -> dict[str, int]:
        """Per-modality feature counts."""
        return self._n_features

    def __len__(self) -> int:
        return self._n_rows

    def __getitems__(self, row_indices: list[int]) -> MultimodalBatch:
        """Fetch a multimodal batch of rows by index."""
        self._ensure_initialized()
        indices_arr = np.array(row_indices, dtype=np.int64)

        # Single lance take: all pointer columns + metadata
        batch_row_ids = self._row_ids[indices_arr]
        select_cols = list(self._pointer_fields.values())
        if self._metadata_columns:
            select_cols.extend(self._metadata_columns)
        select_cols = list(dict.fromkeys(select_cols))  # dedupe

        take_result = (
            self._obs_table.take_row_ids(batch_row_ids.tolist())
            .with_row_id()
            .select(select_cols)
            .to_polars()
        )
        take_result = _reorder_take_result(take_result, batch_row_ids)

        future = asyncio.run_coroutine_threadsafe(
            _take_multimodal(
                take_result,
                self._modality_data,
                self._specs,
                self._pointer_fields,
                self._metadata_columns,
            ),
            self._loop,
        )
        return future.result()

    def __getitem__(self, idx: int) -> MultimodalBatch:
        return self.__getitems__([idx])

    def _ensure_initialized(self) -> None:
        if self._loop is not None:
            return
        self._start_event_loop()
        db_uri, table_name, table_version, storage_options = self._lance_info
        db = lancedb.connect(db_uri, storage_options=storage_options)
        self._obs_table = db.open_table(table_name)
        self._obs_table.checkout(table_version)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_loop"] = None
        state["_loop_thread"] = None
        state["_obs_table"] = None
        return state


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------


def sparse_to_dense_collate(batch: SparseBatch) -> dict:
    """Convert a SparseBatch to a dense float32 tensor via scatter.

    Returns ``{"X": dense_tensor, **metadata_tensors}``.
    """
    return {"X": _sparse_batch_to_dense_tensor(batch), **_metadata_to_tensor_dict(batch.metadata)}


def sparse_to_csr_collate(batch: SparseBatch) -> dict:
    """Convert a SparseBatch to a sparse CSR tensor.

    Returns ``{"X": sparse_csr_tensor, **metadata_tensors}``.
    """
    import torch

    values = _single_layer(batch.layers)
    n_rows = len(batch.offsets) - 1
    X = torch.sparse_csr_tensor(
        crow_indices=torch.from_numpy(batch.offsets),
        col_indices=torch.from_numpy(batch.indices.astype(np.int64)),
        values=torch.from_numpy(values.astype(np.float32)),
        size=(n_rows, batch.n_features),
    )
    return {"X": X, **_metadata_to_tensor_dict(batch.metadata)}


def multimodal_to_dense_collate(batch: MultimodalBatch) -> dict:
    """Convert a MultimodalBatch to dense tensors for model consumption.

    Returns::

        {
            "present": {"gene_expression": bool_tensor, ...},
            "gene_expression": {"X": float32_tensor},  # (n_present, n_features)
            "protein_abundance": {"X": float32_tensor},
            "metadata": {"cell_type": tensor, ...},    # omitted if no metadata
        }

    For sparse modalities the scatter fill is applied (same as
    :func:`sparse_to_dense_collate`). Dense feature data is wrapped directly
    in a tensor. Spatial data is returned as one tensor per tile/crop.
    """
    import torch

    result: dict = {}

    result["present"] = {fs: torch.from_numpy(mask) for fs, mask in batch.present.items()}

    for fs, mod_batch in batch.modalities.items():
        if isinstance(mod_batch, SparseBatch):
            result[fs] = {"X": _sparse_batch_to_dense_tensor(mod_batch)}
        elif isinstance(mod_batch, DenseFeatureBatch):
            result[fs] = {"X": torch.from_numpy(_single_layer(mod_batch.layers))}
        elif isinstance(mod_batch, SpatialTileBatch):
            result[fs] = {
                "X": [
                    torch.from_numpy(np.ascontiguousarray(x))
                    for x in _single_layer(mod_batch.layers)
                ]
            }
        else:
            raise TypeError(f"Unsupported modality batch type: {type(mod_batch).__name__}")

    if batch.metadata is not None:
        result["metadata"] = _metadata_to_tensor_dict(batch.metadata)

    return result


def dense_to_tensor_collate(batch: DenseFeatureBatch | SpatialTileBatch) -> dict:
    """Convert a dense feature or spatial tile batch to a tensor dict.

    Returns ``{"X": tensor, **metadata_tensors}``.  The tensor preserves the
    batch's dtype for spatial tiles.
    """
    import torch

    if isinstance(batch, SpatialTileBatch):
        X = [torch.from_numpy(np.ascontiguousarray(x)) for x in _single_layer(batch.layers)]
    else:
        X = torch.from_numpy(np.ascontiguousarray(_single_layer(batch.layers)))
    return {"X": X, **_metadata_to_tensor_dict(batch.metadata)}


# ---------------------------------------------------------------------------
# Torch integration
# ---------------------------------------------------------------------------


def make_loader(
    dataset: "UnimodalHoxDataset | MultimodalHoxDataset",
    *,
    batch_size: int = 1024,
    shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
    batch_sampler=None,
    **kwargs,
):
    """Create a DataLoader with the right defaults for UnimodalHoxDataset.

    By default uses PyTorch's automatic ``BatchSampler`` driven by
    ``shuffle`` + ``batch_size``, so ``dataset.__getitems__(indices)``
    is called per batch.  Pass a custom ``batch_sampler`` to override;
    ``batch_size``, ``shuffle``, and ``drop_last`` are then ignored
    (PyTorch's requirement).

    Defaults: ``collate_fn=_identity_collate``,
    ``multiprocessing_context="spawn"`` when ``num_workers > 0``,
    ``persistent_workers=False``.

    Parameters
    ----------
    dataset:
        A :class:`UnimodalHoxDataset` or :class:`MultimodalHoxDataset`.
    batch_size:
        Cells per batch.
    shuffle:
        Whether to shuffle rows each epoch (requires ``__len__`` on
        the dataset).
    drop_last:
        Drop the trailing incomplete batch.
    num_workers:
        DataLoader worker count.
    batch_sampler:
        Optional custom batch sampler. Mutually exclusive with
        ``batch_size``/``shuffle``/``drop_last``.
    **kwargs:
        Forwarded to ``torch.utils.data.DataLoader``, overriding defaults.

    Returns
    -------
    torch.utils.data.DataLoader
    """
    from torch.utils.data import DataLoader

    defaults: dict = dict(
        num_workers=num_workers,
        collate_fn=_identity_collate,
        persistent_workers=False,
    )
    if batch_sampler is not None:
        defaults["batch_sampler"] = batch_sampler
    else:
        defaults["batch_size"] = batch_size
        defaults["shuffle"] = shuffle
        defaults["drop_last"] = drop_last
    if num_workers > 0:
        defaults["multiprocessing_context"] = "spawn"
    defaults.update(kwargs)
    return DataLoader(dataset, **defaults)
