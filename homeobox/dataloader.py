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
from homeobox.read import (
    _group_key_to_zg,
    _prepare_obs_and_groups,
)
from homeobox.reconstruction_functional import (
    collect_group_readers_from_atlas,
    collect_remapped_layout_readers_from_atlas,
    get_array_paths_to_read,
    read_arrays_by_group,
    remap_sparse_indices_and_values,
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


def _build_sparse_group_readers(
    atlas: "RaggedAtlas",
    groups: GroupBy,
    spec,
    wanted_globals_for_fs: np.ndarray | None,
) -> dict[str, GroupReader]:
    """Build per-group GroupReader instances for a sparse feature space.

    Resolves each group's remap and applies the optional feature filter.
    """
    layouts_per_group = collect_remapped_layout_readers_from_atlas(
        atlas, groups, spec, wanted_globals=wanted_globals_for_fs
    )
    group_readers = collect_group_readers_from_atlas(
        atlas, groups, spec, layouts_per_group=layouts_per_group, for_worker=True
    )
    return group_readers


def _build_group_readers(
    atlas: "RaggedAtlas",
    groups: GroupBy,
    feature_space: str,
) -> dict[str, GroupReader]:
    """Build per-group readers and matching unique group order."""
    group_readers: dict[str, GroupReader] = {}
    for key, _group_rows in groups:
        zg = _group_key_to_zg(key)
        group_readers[zg] = GroupReader.for_worker(
            zarr_group=zg,
            feature_space=feature_space,
            store=atlas.store,
        )
    return group_readers


def _build_sparse_modality_data(
    atlas: "RaggedAtlas",
    rows_indexed: pl.DataFrame,
    # TODO: type these
    pf,
    spec,
    layer: str,
    wanted_globals_for_fs: np.ndarray | None,
    n_rows: int,
) -> "tuple[pl.DataFrame, _ModalityData]":
    """Build ``_ModalityData`` for a sparse pointer-field modality.

    Returns ``(filtered_rows, modality_data)`` where *filtered_rows*
    is the DataFrame after empty-row removal (with internal columns
    added).
    """
    fs = pf.feature_space
    required_array_paths, layer_array_paths_dict = get_array_paths_to_read(spec, [layer])
    [index_array_name] = required_array_paths
    layer_path = layer_array_paths_dict[layer]

    filtered, groups = _prepare_obs_and_groups(rows_indexed, spec.pointer_type, pf.field_name)

    present_indices = filtered["_orig_idx"].to_numpy().astype(np.int64)
    present_mask, row_positions = _build_present_arrays(present_indices, n_rows)

    group_readers = _build_sparse_group_readers(atlas, groups, spec, wanted_globals_for_fs)

    n_features = (
        len(wanted_globals_for_fs)
        if wanted_globals_for_fs is not None
        else atlas.registry_tables[fs].count_rows()
    )
    if group_readers:
        first_key = list(group_readers.keys())[0]
        layer_dtype = group_readers[first_key].get_array_reader(layer_path)._native_dtype
    else:
        layer_dtype = np.dtype(np.float32)

    mod_data = _ModalityData(
        pointer_type=SparseZarrPointer,
        group_readers=group_readers,
        n_features=n_features,
        index_array_name=index_array_name,
        layer_path=layer_path,
        layer_dtype=layer_dtype,
        present_mask=present_mask,
        row_positions=row_positions,
    )
    return filtered, mod_data


def _build_dense_feature_modality_data(
    atlas: "RaggedAtlas",
    rows_indexed: pl.DataFrame,
    # TODO: Should type these
    pf,
    spec,
    layer: str,
    n_rows: int,
) -> "tuple[pl.DataFrame, _ModalityData]":
    """Build ``_ModalityData`` for a dense feature pointer-field modality.

    Returns ``(filtered_rows, modality_data)`` where *filtered_rows*
    is the DataFrame after empty-row removal.
    """
    fs = pf.feature_space
    filtered, groups = _prepare_obs_and_groups(rows_indexed, spec.pointer_type, pf.field_name)

    present_indices = filtered["_orig_idx"].to_numpy().astype(np.int64)
    present_mask, row_positions = _build_present_arrays(present_indices, n_rows)

    group_readers = _build_group_readers(atlas, groups, fs)

    _, layer_array_paths_dict = get_array_paths_to_read(spec, [layer])
    layer_path = layer_array_paths_dict[layer]

    if group_readers:
        first_key = list(group_readers.keys())[0]
        reader = group_readers[first_key].get_array_reader(layer_path)
        layer_dtype = reader._native_dtype
    else:
        layer_dtype = np.dtype(np.float32)
    n_features = atlas.registry_tables[fs].count_rows()

    mod_data = _ModalityData(
        pointer_type=DenseZarrPointer,
        group_readers=group_readers,
        n_features=n_features,
        index_array_name="",
        layer_path=layer_path,
        layer_dtype=layer_dtype,
        present_mask=present_mask,
        row_positions=row_positions,
    )
    return filtered, mod_data


def _build_spatial_modality_data(
    atlas: "RaggedAtlas",
    rows_indexed: pl.DataFrame,
    # TODO: Should type these
    pf,
    spec,
    layer: str,
    n_rows: int,
) -> "tuple[pl.DataFrame, _ModalityData]":
    """Build ``_ModalityData`` for a spatial tile/crop modality."""
    fs = pf.feature_space
    filtered, groups = _prepare_obs_and_groups(rows_indexed, spec.pointer_type, pf.field_name)

    present_indices = filtered["_orig_idx"].to_numpy().astype(np.int64)
    present_mask, row_positions = _build_present_arrays(present_indices, n_rows)

    group_readers = _build_group_readers(atlas, groups, fs)

    _, layer_array_paths_dict = get_array_paths_to_read(spec, [layer])
    layer_path = layer_array_paths_dict[layer]

    if group_readers:
        first_key = list(group_readers.keys())[0]
        reader = group_readers[first_key].get_array_reader(layer_path)
        layer_dtype = reader._native_dtype
    else:
        layer_dtype = np.dtype(np.float32)

    mod_data = _ModalityData(
        pointer_type=spec.pointer_type,
        group_readers=group_readers,
        n_features=0,
        index_array_name="",
        layer_path=layer_path,
        layer_dtype=layer_dtype,
        present_mask=present_mask,
        row_positions=row_positions,
    )
    return filtered, mod_data


def _sparse_batch_to_dense_tensor(batch: "SparseBatch"):
    """Scatter a SparseBatch into a dense float32 torch tensor (n_rows, n_features)."""
    import torch

    n_rows = len(batch.offsets) - 1
    X = torch.zeros(n_rows, batch.n_features, dtype=torch.float32)
    if n_rows > 0 and len(batch.indices) > 0:
        lengths = np.diff(batch.offsets)
        row_indices = np.repeat(np.arange(n_rows), lengths)
        X[row_indices, batch.indices] = torch.from_numpy(batch.values.astype(np.float32))
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

    pointer_type: type
    group_readers: dict[str, GroupReader]
    # This is the full size of the feature registry for the data
    # modality, unless specific features are provided by wanted_globals
    n_features: int
    # Can we keep `spec` or just feature_space instead
    index_array_name: str  # sparse only; "" for dense
    layer_path: str  # full zarr path, e.g. "csr/layers/counts" or "layers/raw"
    layer_dtype: np.dtype
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
    """Fetch a sparse batch from per-row pointer arrays.

    Unlike the old ``_take_sparse`` which indexed into stored arrays,
    this receives the pointer data directly (loaded from lance per-batch).
    """
    group_obs_data, results = read_arrays_by_group(
        mod_data.group_readers,
        groups,
        spec=spec,
        array_names=[mod_data.index_array_name, mod_data.layer_path],
        read_method="ranges",
    )
    if not group_obs_data:
        return SparseBatch(
            indices=np.array([], dtype=np.int32),
            values=np.array([], dtype=mod_data.layer_dtype),
            offsets=np.zeros(len(batch_row_ids) + 1, dtype=np.int64),
            n_features=mod_data.n_features,
        )

    obs_parts = []
    all_indices = []
    all_values = []
    all_lengths = []
    for (zg, group_rows), group_results in zip(group_obs_data, results, strict=True):
        (flat_indices, lengths), (flat_values, _) = group_results
        flat_indices, flat_values_per_layer, lengths = remap_sparse_indices_and_values(
            remapping_array=mod_data.group_readers[zg].get_remap(),
            flat_indices=flat_indices,
            flat_values_per_layer={"layer": flat_values},
            lengths=lengths,
        )
        all_indices.append(flat_indices)
        all_values.append(flat_values_per_layer.pop("layer"))
        all_lengths.append(lengths)
        obs_parts.append(group_rows)

    flat_indices = np.concatenate(all_indices) if all_indices else np.array([], dtype=np.int32)
    flat_values = (
        np.concatenate(all_values) if all_values else np.array([], dtype=mod_data.layer_dtype)
    )
    lengths = np.concatenate(all_lengths) if all_lengths else np.array([], dtype=np.int64)

    obs_pl = pl.concat(obs_parts, how="diagonal_relaxed")
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])

    metadata = {
        col: obs_pl[col].to_numpy()
        for col, dtype in obs_pl.schema.items()
        if not col.startswith("_") and dtype.base_type() != pl.Struct
    }
    batch = SparseBatch(
        indices=flat_indices,
        values=flat_values,
        offsets=offsets,
        n_features=mod_data.n_features,
        metadata=metadata or None,
    )

    grouped_row_ids = obs_pl["_rowid"].to_numpy().astype(batch_row_ids.dtype, copy=False)
    row_id_order = np.argsort(grouped_row_ids, kind="stable")
    row_perm = row_id_order[np.searchsorted(grouped_row_ids[row_id_order], batch_row_ids)]
    return _reorder_sparse_batch_rows(batch, row_perm)


def _reorder_sparse_batch_rows(batch: SparseBatch, perm: np.ndarray) -> SparseBatch:
    """Reorder rows of a SparseBatch; ``perm[i]`` is the source row for output row ``i``."""
    n_rows = len(perm)
    sorted_lengths = np.diff(batch.offsets)
    new_lengths = sorted_lengths[perm]
    new_offsets = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(new_lengths, out=new_offsets[1:])

    reordered_metadata = (
        {col: arr[perm] for col, arr in batch.metadata.items()}
        if batch.metadata is not None
        else None
    )

    total = int(new_lengths.sum())
    if total == 0:
        return SparseBatch(
            batch.indices, batch.values, new_offsets, batch.n_features, reordered_metadata
        )

    # Segment-arange gather: for each output row i, collect elements from source row perm[i]
    src_starts = batch.offsets[:-1][perm]
    cumlen = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(new_lengths, out=cumlen[1:])
    within = np.arange(total, dtype=np.int64) - np.repeat(cumlen[:-1], new_lengths)
    gather = np.repeat(src_starts, new_lengths) + within
    return SparseBatch(
        indices=batch.indices[gather],
        values=batch.values[gather],
        offsets=new_offsets,
        n_features=batch.n_features,
        metadata=reordered_metadata,
    )


def _reorder_dense_feature_batch_rows(
    batch: DenseFeatureBatch, perm: np.ndarray
) -> DenseFeatureBatch:
    """Reorder rows of a DenseFeatureBatch; ``perm[i]`` is the source row for output row ``i``."""
    reordered_metadata = (
        {col: arr[perm] for col, arr in batch.metadata.items()}
        if batch.metadata is not None
        else None
    )
    return DenseFeatureBatch(
        data=batch.data[perm],
        n_features=batch.n_features,
        metadata=reordered_metadata,
    )


def _reorder_spatial_tile_batch_rows(batch: SpatialTileBatch, perm: np.ndarray) -> SpatialTileBatch:
    """Reorder rows of a SpatialTileBatch; ``perm[i]`` is the source row for output row ``i``."""
    reordered_metadata = (
        {col: arr[perm] for col, arr in batch.metadata.items()}
        if batch.metadata is not None
        else None
    )
    return SpatialTileBatch(
        data=[batch.data[int(i)] for i in perm],
        metadata=reordered_metadata,
    )


async def _take_dense_feature_from_pointers(
    groups: GroupBy,
    spec: FeatureSpaceSpec,
    batch_row_ids: np.ndarray,
    mod_data: _ModalityData,
) -> DenseFeatureBatch:
    """Fetch a dense feature batch from per-row pointer arrays."""
    out_dtype = np.dtype(np.float32)
    group_obs_data, results = read_arrays_by_group(
        mod_data.group_readers,
        groups,
        spec=spec,
        array_names=[mod_data.layer_path],
        read_method="boxes",
        stack_uniform=True,
    )
    if not group_obs_data:
        return DenseFeatureBatch(
            data=np.zeros((0, mod_data.n_features), dtype=out_dtype),
            n_features=mod_data.n_features,
        )

    obs_parts = []
    data_parts = []
    for (_zg, group_rows), group_results in zip(group_obs_data, results, strict=True):
        group_data = group_results[0]
        if group_data.dtype != out_dtype:
            group_data = group_data.astype(out_dtype)

        obs_parts.append(group_rows)
        data_parts.append(group_data)

    data = np.concatenate(data_parts, axis=0)

    obs_pl = pl.concat(obs_parts, how="diagonal_relaxed")
    metadata = {
        col: obs_pl[col].to_numpy()
        for col, dtype in obs_pl.schema.items()
        if not col.startswith("_") and dtype.base_type() != pl.Struct
    }
    batch = DenseFeatureBatch(
        data=data,
        n_features=mod_data.n_features,
        metadata=metadata or None,
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
    group_obs_data, results = read_arrays_by_group(
        mod_data.group_readers,
        groups,
        spec=spec,
        array_names=[mod_data.layer_path],
        read_method="boxes",
        stack_uniform=False,
    )
    if not group_obs_data:
        return SpatialTileBatch(data=[])

    obs_parts = []
    data: list[np.ndarray] = []
    for (_zg, group_rows), group_results in zip(group_obs_data, results, strict=True):
        group_data = group_results[0]
        if isinstance(group_data, list):
            rows = group_data
        else:
            rows = [group_data[i] for i in range(group_data.shape[0])]
        data.extend(
            row if row.dtype == mod_data.layer_dtype else row.astype(mod_data.layer_dtype)
            for row in rows
        )
        obs_parts.append(group_rows)

    obs_pl = pl.concat(obs_parts, how="diagonal_relaxed")
    metadata = {
        col: obs_pl[col].to_numpy()
        for col, dtype in obs_pl.schema.items()
        if not col.startswith("_") and dtype.base_type() != pl.Struct
    }
    batch = SpatialTileBatch(
        data=data,
        metadata=metadata or None,
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
        if mod_data.pointer_type is DiscreteSpatialPointer:
            batch_present = (zg_series.is_not_null() & (zg_series != "")).to_numpy()
        else:
            batch_present = zg_series.is_not_null().to_numpy()

        present_masks[fs] = batch_present
        present_indices = np.where(batch_present)[0]

        # Extract pointers only for present rows
        present_take = take_result[present_indices.tolist()]
        present_row_ids = present_take["_rowid"].to_numpy().astype(np.uint64, copy=False)
        obs_pl, groups = _prepare_obs_and_groups(present_take, mod_data.pointer_type, pf_name)
        assert len(obs_pl) == len(present_take)

        if mod_data.pointer_type is SparseZarrPointer:
            tasks.append(_take_sparse_from_pointers(groups, specs[fs], present_row_ids, mod_data))
        elif mod_data.pointer_type is DenseZarrPointer and specs[fs].has_var_df:
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
        metadata = {
            col: take_result[col].to_numpy()
            for col in metadata_columns
            if col in take_result.columns
        }

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
    layer:
        Which layer to read within the pointer field's feature space.
        May be ``""`` for layer-less specs such as ``image_tiles``.
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
        # TODO: Should be `layer: str | None = None`
        layer: str = "counts",
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

        if self.spec.pointer_type is SparseZarrPointer:
            filtered, self._mod_data = _build_sparse_modality_data(
                atlas,
                rows_indexed,
                pf,
                self.spec,
                layer,
                wanted_globals,
                obs_pl.height,
            )
        elif self.spec.pointer_type is DiscreteSpatialPointer or (
            self.spec.pointer_type is DenseZarrPointer and not self.spec.has_var_df
        ):
            filtered, self._mod_data = _build_spatial_modality_data(
                atlas,
                rows_indexed,
                pf,
                self.spec,
                layer,
                obs_pl.height,
            )
        else:
            filtered, self._mod_data = _build_dense_feature_modality_data(
                atlas,
                rows_indexed,
                pf,
                self.spec,
                layer,
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
    layers:
        ``{field_name: layer_name}`` mapping.
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
        # TODO: layers should be dict[str, str | None]
        layers: dict[str, str],
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
            layer = layers.get(fn, "counts")

            if spec.pointer_type is SparseZarrPointer:
                wg = wanted_globals.get(fn) if wanted_globals is not None else None
                _, modality_data[fn] = _build_sparse_modality_data(
                    atlas, rows_indexed, pf, spec, layer, wg, self._n_rows
                )
            elif spec.pointer_type is DiscreteSpatialPointer or (
                spec.pointer_type is DenseZarrPointer and not spec.has_var_df
            ):
                _, modality_data[fn] = _build_spatial_modality_data(
                    atlas,
                    rows_indexed,
                    pf,
                    spec,
                    layer,
                    self._n_rows,
                )
            else:
                _, modality_data[fn] = _build_dense_feature_modality_data(
                    atlas,
                    rows_indexed,
                    pf,
                    spec,
                    layer,
                    self._n_rows,
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
    import torch

    result: dict = {"X": _sparse_batch_to_dense_tensor(batch)}
    if batch.metadata:
        for col, arr in batch.metadata.items():
            if arr.dtype.kind in ("i", "u", "f"):
                result[col] = torch.from_numpy(arr)
            else:
                result[col] = arr
    return result


def sparse_to_csr_collate(batch: SparseBatch) -> dict:
    """Convert a SparseBatch to a sparse CSR tensor.

    Returns ``{"X": sparse_csr_tensor, **metadata_tensors}``.
    """
    import torch

    n_rows = len(batch.offsets) - 1
    X = torch.sparse_csr_tensor(
        crow_indices=torch.from_numpy(batch.offsets),
        col_indices=torch.from_numpy(batch.indices.astype(np.int64)),
        values=torch.from_numpy(batch.values.astype(np.float32)),
        size=(n_rows, batch.n_features),
    )

    result: dict = {"X": X}
    if batch.metadata:
        for col, arr in batch.metadata.items():
            if arr.dtype.kind in ("i", "u", "f"):
                result[col] = torch.from_numpy(arr)
            else:
                result[col] = arr
    return result


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
            result[fs] = {"X": torch.from_numpy(mod_batch.data)}
        elif isinstance(mod_batch, SpatialTileBatch):
            result[fs] = {"X": [torch.from_numpy(np.ascontiguousarray(x)) for x in mod_batch.data]}
        else:
            raise TypeError(f"Unsupported modality batch type: {type(mod_batch).__name__}")

    if batch.metadata:
        result["metadata"] = {}
        for col, arr in batch.metadata.items():
            if arr.dtype.kind in ("i", "u", "f"):
                result["metadata"][col] = torch.from_numpy(arr)
            else:
                result["metadata"][col] = arr

    return result


def dense_to_tensor_collate(batch: DenseFeatureBatch | SpatialTileBatch) -> dict:
    """Convert a dense feature or spatial tile batch to a tensor dict.

    Returns ``{"X": tensor, **metadata_tensors}``.  The tensor preserves the
    batch's dtype for spatial tiles.
    """
    import torch

    if isinstance(batch, SpatialTileBatch):
        result: dict = {"X": [torch.from_numpy(np.ascontiguousarray(x)) for x in batch.data]}
    else:
        result = {"X": torch.from_numpy(np.ascontiguousarray(batch.data))}
    if batch.metadata:
        for col, arr in batch.metadata.items():
            if arr.dtype.kind in ("i", "u", "f"):
                result[col] = torch.from_numpy(arr)
            else:
                result[col] = arr
    return result


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
