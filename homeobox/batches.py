"""Lightweight batch types and per-modality state shared by the dataloader and reconstructors.

This is a leaf module so :mod:`homeobox.dataloader` and
:mod:`homeobox.reconstruction` can both import these types without a
circular dependency.
"""

from dataclasses import dataclass

import numpy as np

from homeobox.group_reader import GroupReader
from homeobox.group_specs import PointerKind


@dataclass
class SparseBatch:
    """Minimal sparse batch for ML training.

    Represents a batch of rows as flat CSR-style arrays, avoiding
    the overhead of full AnnData/scipy/var DataFrame construction.

    Attributes
    ----------
    indices:
        int32, flat global feature indices (remapped from local).
    values:
        Native dtype, flat expression values.
    offsets:
        int64, CSR-style indptr (length = n_rows + 1).
    n_features:
        Global feature space width (registry size).
    metadata:
        Optional dict of obs columns as numpy arrays, aligned to rows.
    """

    indices: np.ndarray
    values: np.ndarray
    offsets: np.ndarray
    n_features: int
    metadata: dict[str, np.ndarray] | None = None


@dataclass
class DenseBatch:
    """Dense batch for ML training.

    Represents a batch of rows as dense arrays. Only rows that have this
    modality are included (no fill values).

    Attributes
    ----------
    data:
        Stacked ndarray with leading row axis, or one ndarray per row when
        dense stacking is disabled. Rows/items are in query order.
    n_features:
        Feature space width.
    """

    data: np.ndarray | list[np.ndarray]
    n_features: int
    metadata: dict[str, np.ndarray] | None = None
    per_row_shape: tuple[int, ...] | None = None


@dataclass
class MultimodalBatch:
    """Container for a within-row multimodal training batch.

    Analogous to MuData at training time: each modality contains only the
    rows that have it, and ``present`` tracks membership.  No synthetic
    fill values are added for absent rows.

    Attributes
    ----------
    n_rows:
        Total rows in the batch (query order).
    metadata:
        Optional dict of obs columns aligned to ``n_rows`` (query order).
    modalities:
        ``{feature_space: SparseBatch | DenseBatch}``. Each sub-batch has
        ``present[fs].sum()`` rows in query order.
    present:
        ``{feature_space: bool ndarray}``, shape ``(n_rows,)`` per modality.
    """

    n_rows: int
    metadata: dict[str, np.ndarray] | None
    modalities: dict[str, SparseBatch | DenseBatch]
    present: dict[str, np.ndarray]


@dataclass
class ModalityData:
    """Pre-computed per-modality metadata for UnimodalHoxDataset and MultimodalHoxDataset.

    Built at ``__init__`` time; all fields are picklable.  Does NOT store
    per-row pointer arrays (starts/ends/groups_np) — those are loaded
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
    present_mask: np.ndarray | None = None  # bool, (n_total_rows,); None for UnimodalHoxDataset
    row_positions: np.ndarray | None = None  # int64, (n_total_rows,); None for UnimodalHoxDataset
    per_row_shape: tuple[int, ...] | None = None  # (C, H, W) for tiles; None for sparse/2D dense
    array_name: str = ""  # direct zarr array for layer-less specs (e.g., "data")
    stack_dense: bool = True
