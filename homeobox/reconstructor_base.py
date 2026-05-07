"""Base class and decorator for reconstructors.

Reconstructors translate atlas query results into modality-native objects
(AnnData, raw arrays, fragment intervals, ...). Each user-facing method
is marked with :func:`endpoint` so the query layer can enumerate valid
endpoints and produce helpful errors when a feature space is queried
through the wrong API.
"""

from collections.abc import Callable
from typing import TypeVar

F = TypeVar("F", bound=Callable)


def endpoint(method: F) -> F:
    """Mark a reconstructor method as a user-facing endpoint."""
    method.__is_endpoint__ = True  # type: ignore[attr-defined]
    return method


class Reconstructor:
    """Base class for reconstructors.

    Subclasses implement one or more endpoint methods (decorated with
    :func:`endpoint`) such as ``as_anndata``, ``as_array``, or
    ``as_fragments``. Class attributes declare spec-level requirements
    validated by :class:`homeobox.group_specs.FeatureSpaceSpec`.
    """

    required_arrays: list[str] = []
    require_var_df: bool = False

    @classmethod
    def endpoints(cls) -> list[str]:
        """Return the names of user-facing endpoint methods."""
        names: list[str] = []
        for name in dir(cls):
            attr = getattr(cls, name, None)
            if callable(attr) and getattr(attr, "__is_endpoint__", False):
                names.append(name)
        return names
