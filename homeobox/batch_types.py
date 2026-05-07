"""Batch type data structures"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SparseBatch:
    """Minimal sparse batch for ML training.

    Represents a batch of rows as flat CSR-style arrays, avoiding
    the overhead of full AnnData/scipy/var DataFrame construction.

    The CSR skeleton (``indices`` and ``offsets``) is shared across all
    layers; ``layers`` holds one flat values array per requested layer.

    Attributes
    ----------
    indices:
        int32, flat global feature indices (remapped from local).
    offsets:
        int64, CSR-style indptr (length = n_rows + 1).
    layers:
        ``{layer_name: values_array}``. Each values array uses the layer's
        native dtype and is aligned to ``indices``.
    n_features:
        Global feature space width (registry size).
    metadata:
        Optional dict of obs columns as numpy arrays, aligned to rows.
    """

    indices: np.ndarray
    offsets: np.ndarray
    layers: dict[str, np.ndarray]
    n_features: int
    metadata: dict[str, np.ndarray] | None = None


@dataclass
class DenseFeatureBatch:
    """Dense feature batch for ML training.

    Represents a batch of rows as one stacked dense matrix per layer.
    Only rows that have this modality are included (no fill values).

    Attributes
    ----------
    layers:
        ``{layer_name: ndarray}``. Each ndarray has shape ``(n_rows, n_features)``
        and rows are in query order.
    n_features:
        Feature space width.
    metadata:
        Optional dict of obs columns as numpy arrays, aligned to rows.
    """

    layers: dict[str, np.ndarray]
    n_features: int
    metadata: dict[str, np.ndarray] | None = None


@dataclass
class SpatialTileBatch:
    """Spatial tile/crop batch for ML training.

    Represents a batch of spatial arrays as one ndarray per row, per layer.
    Spatial batches are always list-backed so uniform and ragged reads expose
    the same shape contract.

    Attributes
    ----------
    layers:
        ``{layer_name: list_of_ndarrays}``. Each list has one ndarray per row
        in query order.
    metadata:
        Optional dict of obs columns as numpy arrays, aligned to rows.
    """

    layers: dict[str, list[np.ndarray]]
    metadata: dict[str, np.ndarray] | None = None


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
        ``{feature_space: SparseBatch | DenseFeatureBatch | SpatialTileBatch}``.
        Each sub-batch has
        ``present[fs].sum()`` rows in query order.
    present:
        ``{feature_space: bool ndarray}``, shape ``(n_rows,)`` per modality.
    """

    n_rows: int
    metadata: dict[str, np.ndarray] | None
    modalities: dict[str, "SparseBatch | DenseFeatureBatch | SpatialTileBatch"]
    present: dict[str, np.ndarray]
