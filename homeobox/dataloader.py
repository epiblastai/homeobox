"""Fast batch dataloader for ML training from homeobox atlases.

:class:`UnimodalHoxDataset` is a pure data-access object: it owns the
worker-local event loop and lance table, and delegates per-modality
reads to the feature space's :class:`~homeobox.reconstructor_base.Reconstructor`
via :meth:`Reconstructor.build_modality_data` (init-time) and
:meth:`Reconstructor.take_batch_async` (per-batch).

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
from typing import TYPE_CHECKING

import lancedb
import numpy as np
import polars as pl

from homeobox.batches import DenseBatch, ModalityData, MultimodalBatch, SparseBatch
from homeobox.group_specs import get_spec

if TYPE_CHECKING:
    from homeobox.atlas import RaggedAtlas
    from homeobox.reconstructor_base import Reconstructor


__all__ = [
    "DenseBatch",
    "MultimodalBatch",
    "SparseBatch",
    "UnimodalHoxDataset",
    "MultimodalHoxDataset",
    "make_loader",
    "sparse_to_dense_collate",
    "sparse_to_csr_collate",
    "multimodal_to_dense_collate",
    "dense_to_tensor_collate",
]


# ---------------------------------------------------------------------------
# Shared helpers / mixin
# ---------------------------------------------------------------------------


def _sparse_batch_to_dense_tensor(batch: SparseBatch):
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
    sort_perm = np.argsort(batch_row_ids)
    inv_perm = np.empty_like(sort_perm)
    inv_perm[sort_perm] = np.arange(len(sort_perm))
    sorted_batch_ids = batch_row_ids[sort_perm]
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


# ---------------------------------------------------------------------------
# UnimodalHoxDataset
# ---------------------------------------------------------------------------


class UnimodalHoxDataset(_AsyncDataset):
    """Map-style dataset for fast batch access over an atlas query.

    Pure data-access object: resolves zarr remaps via the feature space's
    :class:`~homeobox.reconstructor_base.Reconstructor` and exposes
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
        Whether dense batches should be stacked into a single ndarray. Set
        to ``False`` to return one array per row, which allows variable-size
        image tiles.
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
        spec = get_spec(pf.feature_space)

        self._reconstructor: Reconstructor = spec.reconstructor
        rows_indexed = obs_pl.with_row_index("_orig_idx")

        filtered, self._mod_data = self._reconstructor.build_modality_data(
            atlas,
            rows_indexed,
            pf,
            spec,
            layer,
            n_rows=obs_pl.height,
            wanted_globals=wanted_globals,
            stack_dense=stack_dense,
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

    @property
    def per_row_shape(self) -> tuple[int, ...] | None:
        """Per-row array shape, or ``None`` for flat/sparse feature spaces."""
        return self._mod_data.per_row_shape

    def __len__(self) -> int:
        return self._n_rows

    def __getitems__(self, row_indices: list[int]) -> "SparseBatch | DenseBatch":
        """Fetch a batch of rows by index.

        Called by PyTorch's DataLoader when ``batch_sampler`` yields a list of
        indices (PyTorch >= 2.0 ``__getitems__`` protocol).
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

        # 3. Dispatch async read on the worker-local event loop
        future = asyncio.run_coroutine_threadsafe(
            self._reconstructor.take_batch_async(self._mod_data, take_result, self._pointer_field),
            self._loop,
        )
        batch = future.result()

        # 4. Extract metadata from same take result
        if self._metadata_columns:
            batch.metadata = {
                col: take_result[col].to_numpy()
                for col in self._metadata_columns
                if col in take_result.columns
            }
        return batch

    def __getitem__(self, idx: int) -> "SparseBatch | DenseBatch":
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


async def _take_multimodal(
    take_result: pl.DataFrame,
    modality_data: dict[str, ModalityData],
    reconstructors: "dict[str, Reconstructor]",
    pointer_fields: dict[str, str],
    metadata_columns: list[str] | None,
) -> MultimodalBatch:
    """Fetch a multimodal batch from a lance take result.

    Each modality's reconstructor handles its own per-batch read; this
    function coordinates presence masks, dispatches per-modality
    coroutines concurrently, and assembles the :class:`MultimodalBatch`.
    """
    from homeobox.group_specs import PointerKind  # local import to keep top-level slim

    n_rows = take_result.height

    tasks: list = []
    task_fs: list[str] = []
    present_masks: dict[str, np.ndarray] = {}
    empty_modalities: dict[str, SparseBatch | DenseBatch] = {}

    for fs, mod_data in modality_data.items():
        pf_name = pointer_fields[fs]
        reconstructor = reconstructors[fs]

        pointer_df = take_result[pf_name].struct.unnest()
        zg_series = pointer_df["zarr_group"]
        if mod_data.kind is PointerKind.DISCRETE_SPATIAL:
            batch_present = (zg_series.is_not_null() & (zg_series != "")).to_numpy()
        else:
            batch_present = zg_series.is_not_null().to_numpy()

        present_masks[fs] = batch_present
        present_indices = np.where(batch_present)[0]

        if len(present_indices) == 0:
            if mod_data.kind is PointerKind.SPARSE:
                empty_modalities[fs] = SparseBatch(
                    indices=np.array([], dtype=np.int32),
                    values=np.array([], dtype=mod_data.layer_dtype),
                    offsets=np.zeros(1, dtype=np.int64),
                    n_features=mod_data.n_features,
                )
            elif mod_data.kind is PointerKind.DISCRETE_SPATIAL:
                # Box dims are per-row; with zero rows there is no stacked shape to fall back on.
                empty_modalities[fs] = DenseBatch(
                    data=[],
                    n_features=mod_data.n_features,
                    per_row_shape=mod_data.per_row_shape,
                )
            else:
                if mod_data.per_row_shape is not None:
                    empty_shape = (0, *mod_data.per_row_shape)
                else:
                    empty_shape = (0, mod_data.n_features)
                empty_modalities[fs] = DenseBatch(
                    data=np.zeros(empty_shape, dtype=mod_data.layer_dtype),
                    n_features=mod_data.n_features,
                    per_row_shape=mod_data.per_row_shape,
                )
            continue

        # Restrict the take_result to rows present for this modality before dispatch.
        present_take = take_result[present_indices.tolist()]
        tasks.append(reconstructor.take_batch_async(mod_data, present_take, pf_name))
        task_fs.append(fs)

    results = list(await asyncio.gather(*tasks)) if tasks else []

    modalities: dict[str, SparseBatch | DenseBatch] = dict(empty_modalities)
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
        Whether dense batches should be stacked into a single ndarray. May be
        a single bool for all dense modalities or a mapping by field name.
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

        # Attach row indices so we can track original positions after per-modality filters
        rows_indexed = obs_pl.with_row_index("_orig_idx")

        modality_data: dict[str, ModalityData] = {}
        reconstructors: dict[str, Reconstructor] = {}

        for fn in field_names:
            pf = atlas.pointer_fields[fn]
            self._pointer_fields[fn] = pf.field_name
            spec = get_spec(pf.feature_space)
            layer = layers.get(fn, "counts")
            stack_dense_for_field = (
                stack_dense.get(fn, True) if isinstance(stack_dense, dict) else stack_dense
            )
            wg = wanted_globals.get(fn) if wanted_globals is not None else None

            reconstructors[fn] = spec.reconstructor
            _, modality_data[fn] = spec.reconstructor.build_modality_data(
                atlas,
                rows_indexed,
                pf,
                spec,
                layer,
                n_rows=self._n_rows,
                wanted_globals=wg,
                stack_dense=stack_dense_for_field,
            )

        self._modality_data = modality_data
        self._reconstructors = reconstructors
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
                self._reconstructors,
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
    :func:`sparse_to_dense_collate`).  For dense modalities the data array
    is wrapped directly in a tensor.
    """
    import torch

    result: dict = {}

    result["present"] = {fs: torch.from_numpy(mask) for fs, mask in batch.present.items()}

    for fs, mod_batch in batch.modalities.items():
        if isinstance(mod_batch, SparseBatch):
            result[fs] = {"X": _sparse_batch_to_dense_tensor(mod_batch)}
        elif isinstance(mod_batch.data, list):
            result[fs] = {"X": [torch.from_numpy(np.ascontiguousarray(x)) for x in mod_batch.data]}
        else:
            result[fs] = {"X": torch.from_numpy(mod_batch.data)}

    if batch.metadata:
        result["metadata"] = {}
        for col, arr in batch.metadata.items():
            if arr.dtype.kind in ("i", "u", "f"):
                result["metadata"][col] = torch.from_numpy(arr)
            else:
                result["metadata"][col] = arr

    return result


def dense_to_tensor_collate(batch: DenseBatch) -> dict:
    """Convert a DenseBatch to a tensor dict.

    Returns ``{"X": tensor, **metadata_tensors}``.  The tensor preserves the
    batch's native dtype (e.g. ``uint16`` for image tiles).
    """
    import torch

    if isinstance(batch.data, list):
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
