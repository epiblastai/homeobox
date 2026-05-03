"""Fast batch dataloader for ML training from homeobox atlases.

:class:`CellDataset` is a pure data-access object: it resolves zarr
remaps and exposes ``__getitems__`` for batched async I/O.

Designed for the ``query -> CellDataset -> SparseBatch -> collate_fn ->
GPU`` pipeline.  Reader initialisation is deferred to the worker
process, making the dataset safely picklable for spawn-based
multiprocessing.

Pointer structs and metadata columns are loaded lazily per-batch via
lance's ``take_row_ids`` API rather than materializing the full cell
table at init time.

Usage::

    dataset = atlas.query().to_cell_dataset("gene_expression", "counts", metadata_columns=["cell_type"])
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

if TYPE_CHECKING:
    from homeobox.atlas import RaggedAtlas
from homeobox.batch_array import BatchAsyncArray
from homeobox.group_reader import GroupReader
from homeobox.group_specs import PointerKind, get_spec
from homeobox.read import (
    _apply_wanted_globals_remap,
    _prepare_dense_cells,
    _prepare_sparse_cells,
)

# ---------------------------------------------------------------------------
# Shared helpers / mixin
# ---------------------------------------------------------------------------


def _build_groups_np(zg_series: pl.Series, groups: list[str]) -> np.ndarray:
    """Map group-name strings to contiguous integer IDs (groups must be sorted)."""
    mapping = pl.DataFrame({"_zg": groups, "_gid": np.arange(len(groups), dtype=np.int32)})
    return zg_series.to_frame("_zg").join(mapping, on="_zg", how="left")["_gid"].to_numpy()


def _build_present_arrays(
    present_indices: np.ndarray,
    n_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build presence mask and per-cell position index for one modality.

    Returns ``(present_mask, cell_positions)`` where:

    - ``present_mask[i]`` is True if cell *i* has this modality
    - ``cell_positions[i]`` is the index into the modality's present-cell arrays, or -1 if absent
    """
    present_mask = np.zeros(n_cells, dtype=bool)
    cell_positions = np.full(n_cells, -1, dtype=np.int64)
    if len(present_indices) > 0:
        present_mask[present_indices] = True
        cell_positions[present_indices] = np.arange(len(present_indices), dtype=np.int64)
    return present_mask, cell_positions


def _build_sparse_group_readers(
    atlas: "RaggedAtlas",
    groups: list[str],
    feature_space: str,
    wanted_globals_for_fs: np.ndarray | None,
) -> "dict[str, GroupReader]":
    """Build per-group GroupReader instances for a sparse feature space.

    Resolves each group's remap and applies the optional feature filter.
    """
    group_readers: dict[str, GroupReader] = {}
    for zg in groups:
        raw_remap = atlas.get_group_reader(zg, feature_space).get_remap()
        effective_remap = (
            _apply_wanted_globals_remap(raw_remap, wanted_globals_for_fs)
            if wanted_globals_for_fs is not None
            else raw_remap
        )
        group_readers[zg] = GroupReader.for_worker(
            zarr_group=zg,
            feature_space=feature_space,
            store=atlas.store,
            remap=effective_remap,
        )
    return group_readers


def _build_sparse_modality_data(
    atlas: "RaggedAtlas",
    cells_indexed: pl.DataFrame,
    # TODO: type these
    pf,
    spec,
    layer: str,
    wanted_globals_for_fs: np.ndarray | None,
    n_cells: int,
) -> "tuple[pl.DataFrame, _ModalityData]":
    """Build ``_ModalityData`` for a sparse pointer-field modality.

    Returns ``(filtered_cells, modality_data)`` where *filtered_cells*
    is the DataFrame after empty-cell removal (with internal columns
    added).
    """
    fs = pf.feature_space
    if len(spec.zarr_group_spec.required_arrays) != 1:
        raise NotImplementedError(
            f"Sparse modality requires exactly 1 index array, "
            f"got {len(spec.zarr_group_spec.required_arrays)} for '{fs}'"
        )
    index_array_name = spec.zarr_group_spec.required_arrays[0].array_name

    filtered, groups = _prepare_sparse_cells(cells_indexed, pf)
    groups = sorted(groups)

    present_indices = filtered["_orig_idx"].to_numpy().astype(np.int64)
    present_mask, cell_positions = _build_present_arrays(present_indices, n_cells)

    group_readers = _build_sparse_group_readers(atlas, groups, fs, wanted_globals_for_fs)

    layers_path = spec.zarr_group_spec.find_layers_path()
    n_features = (
        len(wanted_globals_for_fs)
        if wanted_globals_for_fs is not None
        else atlas.registry_tables[fs].count_rows()
    )
    layer_dtype = (
        group_readers[groups[0]].get_array_reader(f"{layers_path}/{layer}")._native_dtype
        if groups
        else np.dtype(np.float32)
    )

    mod_data = _ModalityData(
        kind=PointerKind.SPARSE,
        unique_groups=groups,
        group_readers=group_readers,
        n_features=n_features,
        index_array_name=index_array_name,
        layer=layer,
        layer_dtype=layer_dtype,
        layers_path=layers_path,
        present_mask=present_mask,
        cell_positions=cell_positions,
    )
    return filtered, mod_data


def _build_dense_modality_data(
    atlas: "RaggedAtlas",
    cells_indexed: pl.DataFrame,
    # TODO: Should type these
    pf,
    spec,
    layer: str,
    n_cells: int,
    stack_dense: bool = True,
) -> "tuple[pl.DataFrame, _ModalityData]":
    """Build ``_ModalityData`` for a dense pointer-field modality.

    Returns ``(filtered_cells, modality_data)`` where *filtered_cells*
    is the DataFrame after empty-cell removal.
    """
    fs = pf.feature_space
    filtered, groups = _prepare_dense_cells(cells_indexed, pf)
    groups = sorted(groups)

    present_indices = filtered["_orig_idx"].to_numpy().astype(np.int64)
    present_mask, cell_positions = _build_present_arrays(present_indices, n_cells)

    group_readers: dict[str, GroupReader] = {
        zg: GroupReader.for_worker(
            zarr_group=zg,
            feature_space=fs,
            store=atlas.store,
            remap=np.array([], dtype=np.int32),
        )
        for zg in groups
    }

    # Determine read path and shape based on spec capabilities
    has_layers = bool(spec.zarr_group_spec.layers.required) or bool(
        spec.zarr_group_spec.layers.allowed
    )
    per_cell_shape: tuple[int, ...] | None = None
    array_name = ""

    if has_layers:
        layers_path = spec.zarr_group_spec.find_layers_path()
        array_path = f"{layers_path}/{layer}"
    else:
        layers_path = ""
        array_name = (
            spec.zarr_group_spec.required_arrays[0].array_name
            if spec.zarr_group_spec.required_arrays
            else "data"
        )
        array_path = array_name

    if groups:
        reader = group_readers[groups[0]].get_array_reader(array_path)
        layer_dtype = reader._native_dtype
        if spec.has_var_df:
            n_features = atlas.registry_tables[fs].count_rows()
        else:
            per_cell_shape = tuple(reader.shape[1:])
            n_features = int(np.prod(per_cell_shape)) if per_cell_shape else 0
    else:
        layer_dtype = np.dtype(np.float32)
        n_features = atlas.registry_tables[fs].count_rows() if spec.has_var_df else 0

    mod_data = _ModalityData(
        kind=PointerKind.DENSE,
        unique_groups=groups,
        group_readers=group_readers,
        n_features=n_features,
        index_array_name="",
        layer=layer,
        layer_dtype=layer_dtype,
        layers_path=layers_path,
        present_mask=present_mask,
        cell_positions=cell_positions,
        per_cell_shape=per_cell_shape,
        array_name=array_name,
        stack_dense=stack_dense,
    )
    return filtered, mod_data


def _sparse_batch_to_dense_tensor(batch: "SparseBatch"):
    """Scatter a SparseBatch into a dense float32 torch tensor (n_cells, n_features)."""
    import torch

    n_cells = len(batch.offsets) - 1
    X = torch.zeros(n_cells, batch.n_features, dtype=torch.float32)
    if n_cells > 0 and len(batch.indices) > 0:
        lengths = np.diff(batch.offsets)
        row_indices = np.repeat(np.arange(n_cells), lengths)
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


@dataclass
class SparseBatch:
    """Minimal sparse batch for ML training.

    Represents a batch of cells as flat CSR-style arrays, avoiding
    the overhead of full AnnData/scipy/var DataFrame construction.

    Attributes
    ----------
    indices:
        int32, flat global feature indices (remapped from local).
    values:
        Native dtype, flat expression values.
    offsets:
        int64, CSR-style indptr (length = n_cells + 1).
    n_features:
        Global feature space width (registry size).
    metadata:
        Optional dict of obs columns as numpy arrays, aligned to cells.
    """

    indices: np.ndarray
    values: np.ndarray
    offsets: np.ndarray
    n_features: int
    metadata: dict[str, np.ndarray] | None = None


@dataclass
class DenseBatch:
    """Dense batch for ML training.

    Represents a batch of cells as dense arrays. Only cells that have this
    modality are included (no fill values).

    Attributes
    ----------
    data:
        Stacked ndarray with leading cell axis, or one ndarray per cell when
        dense stacking is disabled. Rows/items are in query order.
    n_features:
        Feature space width.
    """

    data: np.ndarray | list[np.ndarray]
    n_features: int
    metadata: dict[str, np.ndarray] | None = None
    per_cell_shape: tuple[int, ...] | None = None


@dataclass
class MultimodalBatch:
    """Container for a within-cell multimodal training batch.

    Analogous to MuData at training time: each modality contains only the
    cells that have it, and ``present`` tracks membership.  No synthetic
    fill values are added for absent cells.

    Attributes
    ----------
    n_cells:
        Total cells in the batch (query order).
    metadata:
        Optional dict of obs columns aligned to ``n_cells`` (query order).
    modalities:
        ``{feature_space: SparseBatch | DenseBatch}``. Each sub-batch has
        ``present[fs].sum()`` rows in query order.
    present:
        ``{feature_space: bool ndarray}``, shape ``(n_cells,)`` per modality.
    """

    n_cells: int
    metadata: dict[str, np.ndarray] | None
    modalities: dict[str, "SparseBatch | DenseBatch"]
    present: dict[str, np.ndarray]


@dataclass
class _ModalityData:
    """Pre-computed per-modality metadata for CellDataset and MultimodalCellDataset.

    Built at ``__init__`` time; all fields are picklable.  Does NOT store
    per-cell pointer arrays (starts/ends/groups_np) — those are loaded
    lazily per batch via lance ``take_row_ids``.
    """

    kind: PointerKind
    unique_groups: list[str]
    group_readers: dict[str, GroupReader]
    n_features: int
    index_array_name: str  # sparse only; "" for dense
    layer: str
    layer_dtype: np.dtype
    layers_path: str = ""  # e.g. "csr/layers" or "layers"
    present_mask: np.ndarray | None = None  # bool, (n_total_cells,); None for CellDataset
    cell_positions: np.ndarray | None = None  # int64, (n_total_cells,); None for CellDataset
    per_cell_shape: tuple[int, ...] | None = None  # (C, H, W) for tiles; None for sparse/2D dense
    array_name: str = ""  # direct zarr array for layer-less specs (e.g., "data")
    stack_dense: bool = True


# ---------------------------------------------------------------------------
# Async primitives
# ---------------------------------------------------------------------------


async def _take_group_sparse(
    index_reader: BatchAsyncArray,
    layer_reader: BatchAsyncArray,
    remap: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read indices and values for one zarr group concurrently.

    Dispatches two concurrent ``run_in_executor`` calls (indices + values)
    for maximum I/O overlap with GIL released.
    """
    (flat_indices, lengths), (flat_values, _) = await asyncio.gather(
        index_reader.read_ranges(starts, ends),
        layer_reader.read_ranges(starts, ends),
    )
    remapped = remap[flat_indices.astype(np.intp)]
    mask = remapped >= 0
    if not mask.all():
        cell_ids = np.repeat(np.arange(len(lengths)), lengths)
        remapped = remapped[mask]
        flat_values = flat_values[mask]
        lengths = np.bincount(cell_ids[mask], minlength=len(lengths)).astype(np.int64)
    return remapped, flat_values, lengths


# TODO: This is assuming that just because a pointer is sparse, that it also
# has layers. This is true for gene_expression but not necessarily a fundamental
# feature of sparse data.
async def _take_sparse_from_pointers(
    groups_np: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    mod_data: _ModalityData,
) -> SparseBatch:
    """Fetch a sparse batch from per-cell pointer arrays.

    Unlike the old ``_take_sparse`` which indexed into stored arrays,
    this receives the pointer data directly (loaded from lance per-batch).
    """
    n_cells = len(groups_np)

    # Sort by group for ordered concatenation
    sort_order = np.argsort(groups_np, kind="stable")
    sorted_groups = groups_np[sort_order]
    sorted_starts = starts[sort_order]
    sorted_ends = ends[sort_order]

    # Dispatch one task per unique group
    tasks = []
    for gid in np.unique(sorted_groups):
        mask = sorted_groups == gid
        zg = mod_data.unique_groups[gid]
        gr = mod_data.group_readers[zg]
        tasks.append(
            _take_group_sparse(
                gr.get_array_reader(mod_data.index_array_name),
                gr.get_array_reader(f"{mod_data.layers_path}/{mod_data.layer}"),
                gr.get_remap(),
                sorted_starts[mask],
                sorted_ends[mask],
            )
        )

    results = await asyncio.gather(*tasks)

    # Assemble: concatenate in group order
    all_indices = []
    all_values = []
    all_lengths = []
    for remapped_indices, values, lengths in results:
        all_indices.append(remapped_indices)
        all_values.append(values)
        all_lengths.append(lengths)

    flat_indices = np.concatenate(all_indices) if all_indices else np.array([], dtype=np.int32)
    flat_values = (
        np.concatenate(all_values) if all_values else np.array([], dtype=mod_data.layer_dtype)
    )
    lengths = np.concatenate(all_lengths) if all_lengths else np.array([], dtype=np.int64)

    # Build CSR-style offsets
    offsets = np.zeros(n_cells + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])

    batch = SparseBatch(
        indices=flat_indices,
        values=flat_values,
        offsets=offsets,
        n_features=mod_data.n_features,
    )

    # Reorder to input order
    inv_sort = np.argsort(sort_order, kind="stable")
    return _reorder_sparse_batch_rows(batch, inv_sort)


def _reorder_sparse_batch_rows(batch: SparseBatch, perm: np.ndarray) -> SparseBatch:
    """Reorder rows of a SparseBatch; ``perm[i]`` is the source row for output row ``i``."""
    n_cells = len(perm)
    sorted_lengths = np.diff(batch.offsets)
    new_lengths = sorted_lengths[perm]
    new_offsets = np.zeros(n_cells + 1, dtype=np.int64)
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
    cumlen = np.zeros(n_cells + 1, dtype=np.int64)
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


async def _take_group_dense(
    reader: BatchAsyncArray,
    starts: np.ndarray,
    ends: np.ndarray,
    cell_shape: tuple[int, ...],
    dtype: np.dtype | None = None,
) -> np.ndarray:
    """Read dense data for one zarr group.

    Parameters
    ----------
    cell_shape:
        Per-cell shape, e.g. ``(n_features,)`` for 2D or ``(C, H, W)`` for tiles.
    dtype:
        Output dtype.  ``None`` means cast to float32 (legacy 2D behaviour).
    """
    flat_data, _ = await reader.read_ranges(starts, ends)
    out = flat_data.reshape(len(starts), *cell_shape)
    if dtype is None:
        return out.astype(np.float32)
    return out if out.dtype == dtype else out.astype(dtype)


async def _take_dense_from_pointers(
    groups_np: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    mod_data: _ModalityData,
) -> DenseBatch:
    """Fetch a dense batch from per-cell pointer arrays."""
    n_present = len(groups_np)

    # Determine per-cell shape and output dtype
    if mod_data.per_cell_shape is not None:
        cell_shape = mod_data.per_cell_shape
        out_dtype = mod_data.layer_dtype
    else:
        cell_shape = (mod_data.n_features,)
        out_dtype = np.dtype(np.float32)

    # Determine which zarr array to read
    array_path = (
        mod_data.array_name if mod_data.array_name else f"{mod_data.layers_path}/{mod_data.layer}"
    )

    sort_order = np.argsort(groups_np, kind="stable")
    sorted_groups = groups_np[sort_order]
    sorted_starts = starts[sort_order]
    sorted_ends = ends[sort_order]

    tasks = []
    group_slices: list[tuple[int, int]] = []
    pos = 0
    for gid in np.unique(sorted_groups):
        mask = sorted_groups == gid
        count = int(mask.sum())
        zg = mod_data.unique_groups[gid]
        gr = mod_data.group_readers[zg]
        group_reader = gr.get_array_reader(array_path)
        group_cell_shape = tuple(group_reader.shape[1:]) if not mod_data.stack_dense else cell_shape
        tasks.append(
            _take_group_dense(
                group_reader,
                sorted_starts[mask],
                sorted_ends[mask],
                group_cell_shape,
                mod_data.layer_dtype if mod_data.per_cell_shape is not None else None,
            )
        )
        group_slices.append((pos, pos + count))
        pos += count

    results = await asyncio.gather(*tasks)

    if not mod_data.stack_dense:
        sorted_items: list[np.ndarray] = []
        for group_data in results:
            sorted_items.extend(group_data[i] for i in range(group_data.shape[0]))
        inv_sort = np.argsort(sort_order, kind="stable")
        return DenseBatch(
            data=[sorted_items[i] for i in inv_sort],
            n_features=mod_data.n_features,
            per_cell_shape=mod_data.per_cell_shape,
        )

    sorted_data = np.empty((n_present, *cell_shape), dtype=out_dtype)
    for (s, e), group_data in zip(group_slices, results, strict=True):
        sorted_data[s:e] = group_data

    inv_sort = np.argsort(sort_order, kind="stable")
    return DenseBatch(
        data=sorted_data[inv_sort],
        n_features=mod_data.n_features,
        per_cell_shape=mod_data.per_cell_shape,
    )


async def _fetch_modality_from_pointers(
    groups_np: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    mod_data: _ModalityData,
) -> "SparseBatch | DenseBatch":
    """Dispatch to sparse or dense fetch using pointer arrays."""
    if mod_data.kind is PointerKind.SPARSE:
        return await _take_sparse_from_pointers(groups_np, starts, ends, mod_data)
    return await _take_dense_from_pointers(groups_np, starts, ends, mod_data)


def _extract_pointers_sparse(
    take_result: pl.DataFrame,
    pointer_field: str,
    unique_groups: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (groups_np, starts, ends) from a take result for a sparse pointer field."""
    pointer_df = take_result[pointer_field].struct.unnest()
    zg_series = pointer_df["zarr_group"]
    groups_np = _build_groups_np(zg_series, unique_groups)
    starts = pointer_df["start"].to_numpy().astype(np.int64)
    ends = pointer_df["end"].to_numpy().astype(np.int64)
    return groups_np, starts, ends


def _extract_pointers_dense(
    take_result: pl.DataFrame,
    pointer_field: str,
    unique_groups: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (groups_np, starts, ends) from a take result for a dense pointer field."""
    pointer_df = take_result[pointer_field].struct.unnest()
    zg_series = pointer_df["zarr_group"]
    groups_np = _build_groups_np(zg_series, unique_groups)
    pos_arr = pointer_df["position"].to_numpy().astype(np.int64)
    return groups_np, pos_arr, pos_arr + 1


async def _take_multimodal(
    take_result: pl.DataFrame,
    modality_data: dict[str, _ModalityData],
    pointer_fields: dict[str, str],
    metadata_columns: list[str] | None,
) -> MultimodalBatch:
    """Fetch a multimodal batch from a lance take result."""
    n_cells = take_result.height

    tasks: list = []
    task_fs: list[str] = []
    present_masks: dict[str, np.ndarray] = {}
    empty_modalities: dict[str, SparseBatch | DenseBatch] = {}

    for fs, mod_data in modality_data.items():
        pf_name = pointer_fields[fs]

        # Extract pointers for ALL batch cells for this modality
        pointer_df = take_result[pf_name].struct.unnest()
        zg_series = pointer_df["zarr_group"]
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
            else:
                if mod_data.per_cell_shape is not None:
                    empty_shape = (0, *mod_data.per_cell_shape)
                else:
                    empty_shape = (0, mod_data.n_features)
                empty_modalities[fs] = DenseBatch(
                    data=np.zeros(empty_shape, dtype=mod_data.layer_dtype),
                    n_features=mod_data.n_features,
                    per_cell_shape=mod_data.per_cell_shape,
                )
            continue

        # Extract pointers only for present cells
        present_take = take_result[present_indices.tolist()]
        if mod_data.kind is PointerKind.SPARSE:
            groups_np, starts, ends = _extract_pointers_sparse(
                present_take, pf_name, mod_data.unique_groups
            )
        else:
            groups_np, starts, ends = _extract_pointers_dense(
                present_take, pf_name, mod_data.unique_groups
            )

        tasks.append(_fetch_modality_from_pointers(groups_np, starts, ends, mod_data))
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
        n_cells=n_cells,
        metadata=metadata,
        modalities=modalities,
        present=present_masks,
    )


# ---------------------------------------------------------------------------
# CellDataset
# ---------------------------------------------------------------------------


class CellDataset(_AsyncDataset):
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
    cells_pl:
        Polars DataFrame of cell records (from a query). Must include
        ``_rowid`` column (via ``with_row_id(True)``).
    field_name:
        Pointer-field attribute name on the cell schema.
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
        to ``False`` to return one array per cell, which allows variable-size
        image tiles.
    """

    def __init__(
        self,
        atlas: "RaggedAtlas",
        cells_pl: pl.DataFrame,
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

        # Store the obstore ObjectStore (picklable via __getnewargs_ex__)
        # Workers reconstruct the zarr root lazily from this store.
        self._store = atlas.store
        self._pointer_kind = spec.pointer_kind

        # Build modality data (filters empty cells, builds remaps & readers)
        cells_indexed = cells_pl.with_row_index("_orig_idx")

        if spec.pointer_kind is PointerKind.SPARSE:
            filtered, self._mod_data = _build_sparse_modality_data(
                atlas,
                cells_indexed,
                pf,
                spec,
                layer,
                wanted_globals,
                cells_pl.height,
            )
        else:
            filtered, self._mod_data = _build_dense_modality_data(
                atlas,
                cells_indexed,
                pf,
                spec,
                layer,
                cells_pl.height,
                stack_dense=stack_dense,
            )

        self._row_ids = filtered["_rowid"].to_numpy().astype(np.uint64)
        self._n_cells = len(self._row_ids)
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
    def n_cells(self) -> int:
        return self._n_cells

    @property
    def n_features(self) -> int:
        return self._mod_data.n_features

    @property
    def per_cell_shape(self) -> tuple[int, ...] | None:
        """Per-cell array shape, or ``None`` for flat/sparse feature spaces."""
        return self._mod_data.per_cell_shape

    def __len__(self) -> int:
        return self._n_cells

    def __getitems__(self, cell_indices: list[int]) -> "SparseBatch | DenseBatch":
        """Fetch a batch of cells by index.

        Called by PyTorch's DataLoader when ``batch_sampler`` yields a list of
        indices (PyTorch >= 2.0 ``__getitems__`` protocol).

        Returns :class:`SparseBatch` for sparse feature spaces or
        :class:`DenseBatch` for dense feature spaces (including image tiles).

        Parameters
        ----------
        cell_indices:
            List of 0-based cell indices into this dataset's cell arrays.
        """
        self._ensure_initialized()
        indices_arr = np.array(cell_indices, dtype=np.int64)

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
        if self._pointer_kind is PointerKind.SPARSE:
            groups_np, starts, ends = _extract_pointers_sparse(
                take_result, self._pointer_field, self._mod_data.unique_groups
            )
            future = asyncio.run_coroutine_threadsafe(
                _take_sparse_from_pointers(groups_np, starts, ends, self._mod_data),
                self._loop,
            )
        else:
            groups_np, starts, ends = _extract_pointers_dense(
                take_result, self._pointer_field, self._mod_data.unique_groups
            )
            future = asyncio.run_coroutine_threadsafe(
                _take_dense_from_pointers(groups_np, starts, ends, self._mod_data),
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
        """Fetch a single cell as a batch."""
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
# MultimodalCellDataset
# ---------------------------------------------------------------------------


class MultimodalCellDataset(_AsyncDataset):
    """Map-style multimodal dataset for fast batch access over an atlas query.

    Supports within-cell multimodal batches where each cell may have data
    from multiple modalities (e.g. CITE-seq RNA + protein, multiome RNA +
    ATAC).  Yields :class:`MultimodalBatch` via :meth:`__getitems__`.

    Each modality's sub-batch contains only the cells that have it; a
    ``present`` mask tracks membership.  No synthetic fill values.

    Parameters
    ----------
    atlas:
        The atlas to read from.
    cells_pl:
        Polars DataFrame of cell records (from a query). Must include
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
        cells_pl: pl.DataFrame,
        field_names: list[str],
        # TODO: layers should be dict[str, str | None]
        layers: dict[str, str],
        metadata_columns: list[str] | None = None,
        wanted_globals: dict[str, np.ndarray] | None = None,
        stack_dense: bool | dict[str, bool] = True,
    ) -> None:
        self._field_names = field_names
        self._n_cells = cells_pl.height
        self._metadata_columns = metadata_columns

        # Store lance info for lazy table reconstruction in workers
        self._lance_info = (
            atlas.db_uri,
            atlas.obs_table.name,
            atlas.obs_table.version,
            getattr(atlas.db, "storage_options", None),
        )

        # Store row IDs for lazy loading
        self._row_ids = cells_pl["_rowid"].to_numpy().astype(np.uint64)

        # Map field_name -> pointer field name (identical here, kept for parity
        # with lance take() call shape).
        self._pointer_fields: dict[str, str] = {}

        # Attach row indices so we can track original positions after per-modality filters
        cells_indexed = cells_pl.with_row_index("_orig_idx")

        modality_data: dict[str, _ModalityData] = {}

        for fn in field_names:
            pf = atlas.pointer_fields[fn]
            self._pointer_fields[fn] = pf.field_name
            spec = get_spec(pf.feature_space)
            layer = layers.get(fn, "counts")

            if spec.pointer_kind is PointerKind.SPARSE:
                wg = wanted_globals.get(fn) if wanted_globals is not None else None
                _, modality_data[fn] = _build_sparse_modality_data(
                    atlas, cells_indexed, pf, spec, layer, wg, self._n_cells
                )
            else:
                stack_dense_for_field = (
                    stack_dense.get(fn, True) if isinstance(stack_dense, dict) else stack_dense
                )
                _, modality_data[fn] = _build_dense_modality_data(
                    atlas,
                    cells_indexed,
                    pf,
                    spec,
                    layer,
                    self._n_cells,
                    stack_dense=stack_dense_for_field,
                )

        self._modality_data = modality_data
        self._n_features = {fn: modality_data[fn].n_features for fn in field_names}

        # Worker-local state — initialized lazily in _ensure_initialized()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._obs_table: lancedb.table.Table | None = None

    @property
    def n_cells(self) -> int:
        return self._n_cells

    @property
    def n_features(self) -> dict[str, int]:
        """Per-modality feature counts."""
        return self._n_features

    def __len__(self) -> int:
        return self._n_cells

    def __getitems__(self, cell_indices: list[int]) -> MultimodalBatch:
        """Fetch a multimodal batch of cells by index."""
        self._ensure_initialized()
        indices_arr = np.array(cell_indices, dtype=np.int64)

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

    n_cells = len(batch.offsets) - 1
    X = torch.sparse_csr_tensor(
        crow_indices=torch.from_numpy(batch.offsets),
        col_indices=torch.from_numpy(batch.indices.astype(np.int64)),
        values=torch.from_numpy(batch.values.astype(np.float32)),
        size=(n_cells, batch.n_features),
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
    dataset: "CellDataset | MultimodalCellDataset",
    *,
    batch_size: int = 1024,
    shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
    batch_sampler=None,
    **kwargs,
):
    """Create a DataLoader with the right defaults for CellDataset.

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
        A :class:`CellDataset` or :class:`MultimodalCellDataset`.
    batch_size:
        Cells per batch.
    shuffle:
        Whether to shuffle cells each epoch (requires ``__len__`` on
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
