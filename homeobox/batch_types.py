"""Batch type data structures"""

from dataclasses import dataclass

import numpy as np
import polars as pl


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
        resolved dtype and is aligned to ``indices``.
    n_features:
        Global feature space width (registry size).
    metadata:
        Optional polars DataFrame of obs columns, aligned to rows.
    """

    indices: np.ndarray
    offsets: np.ndarray
    layers: dict[str, np.ndarray]
    n_features: int
    metadata: pl.DataFrame | None = None

    @classmethod
    def empty(
        cls,
        n_rows: int,
        n_features: int,
        layer_dtypes: dict[str, np.dtype],
        metadata: pl.DataFrame | None = None,
    ) -> "SparseBatch":
        """Construct an empty batch with ``n_rows`` rows, each having zero values."""
        return cls(
            indices=np.array([], dtype=np.int32),
            offsets=np.zeros(n_rows + 1, dtype=np.int64),
            layers={name: np.array([], dtype=dtype) for name, dtype in layer_dtypes.items()},
            n_features=n_features,
            metadata=metadata,
        )

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, sl: slice) -> "SparseBatch":
        """Return a sub-batch holding the rows in *sl* (contiguous slices only)."""
        if not isinstance(sl, slice):
            raise TypeError(f"SparseBatch only supports slice indexing, got {type(sl).__name__}")
        start, stop, step = sl.indices(len(self))
        if step != 1:
            raise ValueError("SparseBatch slicing requires step == 1")
        offsets = self.offsets[start : stop + 1]
        nnz_start = int(offsets[0])
        nnz_end = int(offsets[-1])
        return SparseBatch(
            indices=self.indices[nnz_start:nnz_end],
            offsets=offsets - offsets[0],
            layers={name: arr[nnz_start:nnz_end] for name, arr in self.layers.items()},
            n_features=self.n_features,
            metadata=self.metadata[start:stop] if self.metadata is not None else None,
        )


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
        Optional polars DataFrame of obs columns, aligned to rows.
    """

    layers: dict[str, np.ndarray]
    n_features: int
    metadata: pl.DataFrame | None = None

    @classmethod
    def empty(
        cls,
        n_features: int,
        layer_dtypes: dict[str, np.dtype],
        metadata: pl.DataFrame | None = None,
    ) -> "DenseFeatureBatch":
        """Construct an empty batch with zero rows."""
        return cls(
            layers={
                name: np.zeros((0, n_features), dtype=dtype) for name, dtype in layer_dtypes.items()
            },
            n_features=n_features,
            metadata=metadata,
        )

    def __len__(self) -> int:
        return next(iter(self.layers.values())).shape[0]

    def __getitem__(self, sl: slice) -> "DenseFeatureBatch":
        """Return a sub-batch holding the rows in *sl*."""
        if not isinstance(sl, slice):
            raise TypeError(
                f"DenseFeatureBatch only supports slice indexing, got {type(sl).__name__}"
            )
        return DenseFeatureBatch(
            layers={name: arr[sl] for name, arr in self.layers.items()},
            n_features=self.n_features,
            metadata=self.metadata[sl] if self.metadata is not None else None,
        )


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
        Optional polars DataFrame of obs columns, aligned to rows.
    """

    layers: dict[str, list[np.ndarray]]
    metadata: pl.DataFrame | None = None

    @classmethod
    def empty(
        cls,
        layer_names: list[str],
        metadata: pl.DataFrame | None = None,
    ) -> "SpatialTileBatch":
        """Construct an empty batch with zero rows."""
        return cls(
            layers={name: [] for name in layer_names},
            metadata=metadata,
        )

    def __len__(self) -> int:
        return len(next(iter(self.layers.values())))

    def __getitem__(self, sl: slice) -> "SpatialTileBatch":
        """Return a sub-batch holding the rows in *sl*."""
        if not isinstance(sl, slice):
            raise TypeError(
                f"SpatialTileBatch only supports slice indexing, got {type(sl).__name__}"
            )
        return SpatialTileBatch(
            layers={name: arrs[sl] for name, arrs in self.layers.items()},
            metadata=self.metadata[sl] if self.metadata is not None else None,
        )


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
        Optional polars DataFrame aligned to ``n_rows`` (query order).
    modalities:
        ``{feature_space: SparseBatch | DenseFeatureBatch | SpatialTileBatch}``.
        Each sub-batch has
        ``present[fs].sum()`` rows in query order.
    present:
        ``{feature_space: bool ndarray}``, shape ``(n_rows,)`` per modality.
    """

    n_rows: int
    metadata: pl.DataFrame | None
    modalities: dict[str, "SparseBatch | DenseFeatureBatch | SpatialTileBatch"]
    present: dict[str, np.ndarray]
