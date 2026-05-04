"""Base class and decorator for reconstructors.

Reconstructors translate atlas query results into modality-native objects
(AnnData, raw arrays, fragment intervals, ...). Each user-facing method
is marked with :func:`endpoint` so the query layer can enumerate valid
endpoints and produce helpful errors when a feature space is queried
through the wrong API.

Reconstructors also own the per-batch read path used by the dataloader.
The two relevant hooks are :meth:`Reconstructor.build_modality_data`
(init-time work — filter empty rows, build group readers, resolve dtypes)
and :meth:`Reconstructor.take_batch_async` (per-batch coroutine that
extracts pointer arrays from a lance ``take_result`` and returns a
:class:`~homeobox.batches.SparseBatch` or
:class:`~homeobox.batches.DenseBatch`). Reconstructors that don't
participate in the batch path (e.g. ``IntervalReconstructor`` for
fragments) inherit the default ``NotImplementedError`` behaviour.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    import polars as pl

    from homeobox.atlas import RaggedAtlas
    from homeobox.batches import DenseBatch, ModalityData, SparseBatch
    from homeobox.group_specs import FeatureSpaceSpec
    from homeobox.schema import PointerField

F = TypeVar("F", bound=Callable)


def endpoint(method: F) -> F:
    """Mark a reconstructor method as a user-facing endpoint."""
    method.__is_endpoint__ = True  # type: ignore[attr-defined]
    return method


class Reconstructor:
    """Base class for reconstructors.

    Subclasses implement one or more endpoint methods (decorated with
    :func:`endpoint`) such as ``as_anndata``, ``as_array``, or
    ``as_fragments``, and may also implement the dataloader-facing
    :meth:`build_modality_data` / :meth:`take_batch_async` pair.
    """

    @classmethod
    def endpoints(cls) -> list[str]:
        """Return the names of user-facing endpoint methods."""
        names: list[str] = []
        for name in dir(cls):
            attr = getattr(cls, name, None)
            if callable(attr) and getattr(attr, "__is_endpoint__", False):
                names.append(name)
        return names

    def build_modality_data(
        self,
        atlas: "RaggedAtlas",
        rows_indexed: "pl.DataFrame",
        pf: "PointerField",
        spec: "FeatureSpaceSpec",
        layer: str,
        *,
        n_rows: int,
        **opts: Any,
    ) -> "tuple[pl.DataFrame, ModalityData]":
        """Build per-modality state for the dataloader hot path.

        Filters the obs rows that have this modality, resolves per-group
        readers, and packages everything into a picklable
        :class:`~homeobox.batches.ModalityData`.

        Subclasses that don't participate in the dataloader path leave
        this raising ``NotImplementedError``.

        Returns
        -------
        (filtered_rows, modality_data)
            ``filtered_rows`` is the obs DataFrame after empty-row
            removal (with internal helper columns added by the
            corresponding ``read.py`` ``_prepare_*_obs`` helper).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement build_modality_data; "
            "this feature space is not usable from the dataloader."
        )

    async def take_batch_async(
        self,
        mod_data: "ModalityData",
        take_result: "pl.DataFrame",
        pointer_field: str,
    ) -> "SparseBatch | DenseBatch":
        """Per-batch coroutine: lance ``take_result`` -> SparseBatch / DenseBatch.

        Subclasses that don't participate in the dataloader path leave
        this raising ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement take_batch_async; "
            "this feature space is not usable from the dataloader."
        )
