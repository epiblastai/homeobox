"""Fast batch dataloader for ML training from homeobox atlases.

:class:`UnimodalHoxDataset` is a pure data-access object: it resolves zarr
remaps and exposes ``__getitems__`` for batched async I/O.

Designed for the ``query -> UnimodalHoxDataset -> SparseBatch`` pipeline.
Reader initialisation is deferred to the worker process, making the dataset
safely picklable for spawn-based multiprocessing.

Pointer structs and metadata columns are loaded lazily per-batch via
lance's ``take_row_ids`` API rather than materializing the full obs
table at init time.

Usage::

    dataset = atlas.query().to_unimodal_dataset("gene_expression", "counts", metadata_columns=["cell_type"])
    loader = make_loader(dataset, batch_size=256, shuffle=True, num_workers=4)
    for batch in loader:
        ...
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
from homeobox.group_specs import get_spec
from homeobox.pointer_types import DiscreteSpatialPointer
from homeobox.read import _prepare_obs_and_groups
from homeobox.reconstruction_functional import (
    FeatureReadPlan,
    build_feature_read_plan,
    finalize_grouped_read,
    read_arrays_by_group,
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
    pf,
    layer_overrides: list[str] | None,
    wanted_globals: np.ndarray | None,
    n_rows: int,
) -> "tuple[pl.DataFrame, _ModalityData]":
    """Build ``_ModalityData`` for any pointer-field modality.

    Filters empty rows once, then resolves the per-feature-space I/O plan
    (worker-local readers, remapped layouts, joined feature width) via
    :func:`build_feature_read_plan`. Returns
    ``(filtered_rows, modality_data)`` where *filtered_rows* is the
    DataFrame after empty-row removal (with internal columns added) and
    *modality_data* wraps the plan plus multimodal presence arrays.
    """
    spec = get_spec(pf.feature_space)
    filtered, groups = _prepare_obs_and_groups(rows_indexed, spec.pointer_type, pf.field_name)

    present_indices = filtered["_orig_idx"].to_numpy().astype(np.int64)
    present_mask, row_positions = _build_present_arrays(present_indices, n_rows)

    plan = build_feature_read_plan(
        atlas,
        groups,
        pf,
        layer_overrides=layer_overrides,
        wanted_globals=wanted_globals,
        for_worker=True,
    )
    return filtered, _ModalityData(
        plan=plan, present_mask=present_mask, row_positions=row_positions
    )


def _select_obs_metadata(obs_pl: pl.DataFrame) -> pl.DataFrame | None:
    """Select non-internal, non-Struct columns to carry as batch metadata."""
    cols = [
        col
        for col, dtype in obs_pl.schema.items()
        if not col.startswith("_") and dtype.base_type() != pl.Struct
    ]
    return obs_pl.select(cols) if cols else None


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


@dataclass(frozen=True)
class _ModalityData:
    """Per-modality state for UnimodalHoxDataset and MultimodalHoxDataset.

    Wraps the picklable :class:`FeatureReadPlan` and adds the
    multimodal-only presence arrays. Does NOT store per-row pointer arrays
    (starts/ends/groups_np) — those are loaded lazily per batch via lance
    ``take_row_ids``.
    """

    plan: FeatureReadPlan
    # Multimodal-only: ``present_mask[i]`` is True if row *i* has this
    # modality; ``row_positions[i]`` is the index into the modality's
    # present-row arrays (or -1 if absent). Both ``None`` for unimodal.
    present_mask: np.ndarray | None = None
    row_positions: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Async primitives
# ---------------------------------------------------------------------------


async def _take_from_pointers(
    groups: GroupBy,
    batch_row_ids: np.ndarray,
    plan: FeatureReadPlan,
) -> "SparseBatch | DenseFeatureBatch | SpatialTileBatch":
    """Fetch a batch from per-row pointer arrays.

    Pointer-type-agnostic: the spec's reconstructor handles batch construction
    (per-group via :meth:`Reconstructor.build_group_batch`, and the empty case
    via :meth:`Reconstructor.build_empty_batch`), so the same flow works for
    sparse CSR, dense feature, and spatial tile pointers.
    """
    group_batches = read_arrays_by_group(plan, groups)
    if not group_batches:
        return plan.spec.reconstructor.build_empty_batch(
            n_rows=len(batch_row_ids),
            n_features=plan.n_features,
            layer_dtypes=plan.layer_dtypes,
            layer_names=plan.layer_names,
        )

    batch = finalize_grouped_read(plan, group_batches, target_row_ids=batch_row_ids)
    batch.metadata = _select_obs_metadata(batch.metadata)
    return batch


async def _take_multimodal(
    take_result: pl.DataFrame,
    modality_data: dict[str, _ModalityData],
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
        pointer_type = mod_data.plan.spec.pointer_type
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

        tasks.append(_take_from_pointers(groups, present_row_ids, mod_data.plan))
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
        return self._mod_data.plan.n_features

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

        # 2. Extract pointer data and dispatch async read
        # This is safe because we already filtered at __init__, that
        # guarantees that this op will not drop any rows from take_result
        obs_pl, groups = _prepare_obs_and_groups(
            take_result, self._pointer_type, self._pointer_field
        )
        # Sanity check
        assert len(obs_pl) == len(take_result)
        future = asyncio.run_coroutine_threadsafe(
            _take_from_pointers(groups, batch_row_ids, self._mod_data.plan),
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
    """

    def __init__(
        self,
        atlas: "RaggedAtlas",
        obs_pl: pl.DataFrame,
        field_names: list[str],
        layer_overrides: dict[str, list[str] | None] | None = None,
        metadata_columns: list[str] | None = None,
        wanted_globals: dict[str, np.ndarray] | None = None,
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

        # Attach row indices so we can track original positions after per-modality filters
        rows_indexed = obs_pl.with_row_index("_orig_idx")

        modality_data: dict[str, _ModalityData] = {}

        for fn in field_names:
            pf = atlas.pointer_fields[fn]
            self._pointer_fields[fn] = pf.field_name
            field_layer_overrides = layer_overrides.get(fn) if layer_overrides is not None else None

            wg = wanted_globals.get(fn) if wanted_globals is not None else None
            _, modality_data[fn] = _build_modality_data(
                atlas, rows_indexed, pf, field_layer_overrides, wg, self._n_rows
            )

        self._modality_data = modality_data
        self._n_features = {fn: modality_data[fn].plan.n_features for fn in field_names}

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
