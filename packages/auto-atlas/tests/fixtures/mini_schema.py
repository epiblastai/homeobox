"""Minimal homeobox schema used by the ingestion tests.

One obs class (``CellRow``) with a single ``image_features`` pointer and a
``has_features`` presence flag; one dataset class (``MiniDataset``) with an
``n_cells`` SummaryField; a per-feature-space registry (``FeatureSchema``); and a
collection-level registry-key target table (``StudySchema``).

``image_features`` is a dense feature space (``DenseZarrPointer``,
``has_var_df=True``), so a loader feeds it a dense matrix and a var table.
"""

from typing import Self

# Importing homeobox registers the builtin feature-space specs (incl.
# ``image_features``); the pointer field below is validated against it at
# class-definition time, so the import must precede the class bodies.
import homeobox  # noqa: F401
from homeobox.pointer_types import DenseZarrPointer
from homeobox.schema import (
    DatasetSchema,
    FeatureBaseSchema,
    HoxBaseSchema,
    PointerField,
    StableUIDBaseSchema,
    StableUIDField,
    SummaryField,
)
from pydantic import model_validator

FEATURE_SPACE = "image_features"


class FeatureSchema(FeatureBaseSchema):
    """Feature registry for the ``image_features`` space."""

    feature_id: str | None = StableUIDField.declare(default=None)
    name: str | None = None


class StudySchema(StableUIDBaseSchema):
    """A collection-level registry-key target table."""

    study_accession: str | None = StableUIDField.declare(default=None)
    title: str | None = None


class MiniDataset(DatasetSchema):
    """One row per feature-space write; ``n_cells`` filled at ingestion."""

    study_uid: str | None = None
    n_cells: int | None = SummaryField.declare(
        target_schema="CellRow", target_field="uid", op="count", default=None
    )


class CellRow(HoxBaseSchema):
    """One cell, pointing into the ``image_features`` space."""

    study_uid: str | None = None
    features: DenseZarrPointer | None = PointerField.declare(
        feature_space=FEATURE_SPACE, feature_registry_schema="FeatureSchema"
    )
    has_features: bool = False

    @model_validator(mode="after")
    def _generate_has_features(self) -> Self:
        self.has_features = self.features is not None
        return self
