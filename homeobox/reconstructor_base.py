"""Base class and decorator for reconstructors.

Reconstructors translate atlas query results into modality-native objects
(AnnData, raw arrays, fragment intervals, ...). Each user-facing method
is marked with :func:`endpoint` so the query layer can enumerate valid
endpoints and produce helpful errors when a feature space is queried
through the wrong API.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, TypeVar

if TYPE_CHECKING:
    import numpy as np
    import polars as pl

    from homeobox.batch_types import DenseFeatureBatch, SparseBatch, SpatialTileBatch
    from homeobox.group_reader import GroupReader

F = TypeVar("F", bound=Callable)


def endpoint(method: F) -> F:
    """Mark a reconstructor method as a user-facing endpoint."""
    method.__is_endpoint__ = True  # type: ignore[attr-defined]
    return method


class Reconstructor:
    """Base class for reconstructors.

    Subclasses implement one or more endpoint methods (decorated with
    :func:`endpoint`) such as ``as_anndata``, ``as_spatial_batch``, or
    ``as_fragments``. Class attributes declare spec-level requirements
    validated by :class:`homeobox.group_specs.FeatureSpaceSpec`.

    Reconstructors that drive
    :func:`homeobox.reconstruction_functional.read_arrays_by_group` also
    declare:

    - :attr:`read_method`: ``"ranges"`` (sparse / CSR-style) or ``"boxes"``
      (dense / spatial).
    - :attr:`stack_uniform`: only meaningful when ``read_method == "boxes"``;
      controls whether the box reader stacks per-group rows into a single
      ndarray (dense feature matrices) or returns a list (spatial tiles).
    - :meth:`build_group_batch`: turns the raw per-group read results into a
      local-space batch (:class:`SparseBatch`, :class:`DenseFeatureBatch`, or
      :class:`SpatialTileBatch`).

    Reconstructors with their own bespoke read flow (e.g.
    :class:`IntervalReconstructor`) leave these unset.
    """

    required_arrays: list[str] = []
    require_var_df: bool = False

    # Set by subclasses that drive ``read_arrays_by_group``.
    read_method: Literal["ranges", "boxes"]
    stack_uniform: bool = True

    def build_group_batch(
        self,
        group_reader: "GroupReader",
        group_rows: "pl.DataFrame",
        layer_names: list[str],
        results: list,
    ) -> "SparseBatch | DenseFeatureBatch | SpatialTileBatch":
        """Package one group's raw read results as a local-space batch.

        Implemented by reconstructors that drive ``read_arrays_by_group``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement build_group_batch")

    def build_empty_batch(
        self,
        *,
        n_rows: int,
        n_features: int,
        layer_dtypes: "dict[str, np.dtype]",
        layer_names: list[str],
    ) -> "SparseBatch | DenseFeatureBatch | SpatialTileBatch":
        """Construct an empty batch matching this reconstructor's batch type.

        Called when ``read_arrays_by_group`` returns no groups (empty query).
        Each subclass uses the subset of arguments relevant to its batch type;
        callers pass all of them so the call site does not branch on type.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement build_empty_batch")

    @classmethod
    def endpoints(cls) -> list[str]:
        """Return the names of user-facing endpoint methods."""
        names: list[str] = []
        for name in dir(cls):
            attr = getattr(cls, name, None)
            if callable(attr) and getattr(attr, "__is_endpoint__", False):
                names.append(name)
        return names
