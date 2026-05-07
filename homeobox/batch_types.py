"""Batch type data structures"""

from dataclasses import dataclass

import numpy as np


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
    modalities: dict[str, "SparseBatch | DenseBatch"]
    present: dict[str, np.ndarray]
