import dataclasses
import datetime
import uuid
from types import UnionType
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin

if TYPE_CHECKING:
    import pandas as pd

import pyarrow as pa
from lancedb.pydantic import LanceModel
from pydantic import Field, model_validator

from homeobox.group_specs import PointerKind, get_spec


class SparseZarrPointer(LanceModel):
    zarr_group: str | None = None
    start: int | None = None
    end: int | None = None
    zarr_row: int | None = None  # 0-indexed position within this zarr group (for CSC lookup)


class DenseZarrPointer(LanceModel):
    zarr_group: str | None = None
    position: int | None = None


class DiscreteSpatialPointer(LanceModel):
    """N-D discrete bounding box into a zarr array.

    ``min_corner`` / ``max_corner`` apply to the leading axes of the referenced
    zarr array; trailing axes without a corner slice everything. For a ``(H, W)``
    array with ``min_corner=[0]``, ``max_corner=[10]`` the referenced region is
    ``array[0:10, :]``.
    """

    zarr_group: str | None = None
    min_corner: list[int] | None = None
    max_corner: list[int] | None = None

    @model_validator(mode="after")
    def _validate_corners(self):
        # Pointer doesn't have a group, so there's nothing else to validate
        if self.zarr_group is None:
            return self

        # TODO: Improve error message for null corners. Currently will raise NoneType has no length
        # if corners are None but the Zarr group is not empty
        if len(self.min_corner) != len(self.max_corner):
            raise ValueError(
                f"min_corner and max_corner must have the same length, "
                f"got {len(self.min_corner)} and {len(self.max_corner)}"
            )
        for i, (lo, hi) in enumerate(zip(self.min_corner, self.max_corner, strict=True)):
            if lo > hi:
                raise ValueError(f"min_corner[{i}]={lo} exceeds max_corner[{i}]={hi}")
        return self


ZarrPointer = SparseZarrPointer | DenseZarrPointer | DiscreteSpatialPointer


# Arrow field metadata key used to persist the feature_space for each pointer column.
# Written by HoxBaseSchema.to_arrow_schema() and read back by
# _infer_pointer_fields_from_arrow() when a schema class is not available.
POINTER_FEATURE_SPACE_METADATA_KEY: bytes = b"homeobox.feature_space"

# Arrow field metadata key used to persist which column drives stable UID generation.
STABLE_UID_METADATA_KEY: bytes = b"homeobox.stable_uid"


def make_uid() -> str:
    """Generate a random 16-character hex uid."""
    return uuid.uuid4().hex[:16]


_HOX_NS = uuid.UUID("b3e7a9f1-6c2d-4a8b-9f01-3d5e7a2b8c4f")


def make_stable_uid(*identity_values: str) -> str:
    """Deterministic 16-char hex UID from identity values.

    Same inputs always produce the same UID. Used for entity deduplication
    across datasets (genes, proteins, molecules, perturbations, publications).
    """
    return uuid.uuid5(_HOX_NS, "|".join(identity_values)).hex[:16]


def _stable_uid_identity_str(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


@dataclasses.dataclass(frozen=True)
class PointerField:
    """Runtime metadata for a single pointer field on a HoxBaseSchema subclass.

    Decouples the HoxBaseSchema table column name from the feature_space it references,
    so a schema can declare multiple columns in the same feature space (e.g.
    ``cycle1_image_tiles`` and ``cycle2_image_tiles``, both with
    ``feature_space="image_tiles"``).

    The concrete pointer type is derivable from ``pointer_kind`` and
    intentionally not stored here.
    """

    field_name: str
    feature_space: str
    # TODO: Is pointer kind even necessary? As long as a spec is registered with
    # a reconstructor we know how to parse it's pointer type no? I think maybe the issue
    # was that pointer type; which would be a better indicator than a string just couldn't
    # be serialized. Perhaps we can use the same arrow metadata trick to get around it.
    # Just double checked, we can 100% remove it
    pointer_kind: PointerKind

    @staticmethod
    def declare(*, feature_space: str, default: Any = None, **kwargs: Any) -> Any:
        """Factory used in schema class bodies to mark a pointer field.

        Returns a pydantic ``Field`` whose ``json_schema_extra`` carries
        ``{"is_pointer": True, "feature_space": feature_space}``. The feature
        space is resolved against the registered spec at schema definition
        time by :meth:`HoxBaseSchema.__init_subclass__`.
        """
        return Field(
            default=default,
            json_schema_extra={"is_pointer": True, "feature_space": feature_space},
            **kwargs,
        )


@dataclasses.dataclass(frozen=True)
class StableUIDField:
    """Marker for a schema field that defines a deterministic ``uid`` value."""

    @staticmethod
    def declare(*, default: Any = None, **kwargs: Any) -> Any:
        """Factory used in schema class bodies to mark a stable UID field."""
        extra = dict(kwargs.pop("json_schema_extra", {}) or {})
        extra["stable_uid"] = True
        return Field(default=default, json_schema_extra=extra, **kwargs)


def _read_field_json_schema_extra(cls: type, name: str) -> dict | None:
    """Read the ``json_schema_extra`` dict for a field on *cls*.

    Works both at ``__init_subclass__`` time (when pydantic has not yet
    populated ``model_fields``) by inspecting the ``FieldInfo`` object
    directly on the class, and after class definition via ``model_fields``.
    Returns ``None`` when the field has no ``json_schema_extra``.
    """
    model_fields = getattr(cls, "model_fields", None)
    if model_fields and name in model_fields:
        extra = model_fields[name].json_schema_extra
    else:
        raw = cls.__dict__.get(name)
        extra = getattr(raw, "json_schema_extra", None)
    if not isinstance(extra, dict):
        return None
    return extra


def _iter_pointer_annotations(cls: type) -> list[tuple[str, type]]:
    """Yield ``(field_name, pointer_type)`` for each pointer-typed annotation on *cls*.

    Unwraps ``X | None`` and ``Optional[X]`` unions to find the underlying
    pointer type.
    """
    result: list[tuple[str, type]] = []
    for name, annotation in cls.__annotations__.items():
        if name in ("uid", "dataset_uid"):
            continue
        origin = get_origin(annotation)
        if origin is Union or isinstance(annotation, UnionType):
            inner_types = get_args(annotation)
        else:
            inner_types = (annotation,)
        for t in inner_types:
            if t is SparseZarrPointer or t is DenseZarrPointer or t is DiscreteSpatialPointer:
                result.append((name, t))
                break
    return result


class StableUIDBaseSchema(LanceModel):
    """Base schema for tables whose ``uid`` may be derived from one stable field."""

    uid: str = Field(default_factory=make_uid)

    @classmethod
    def stable_uid_field_names(cls) -> list[str]:
        """Return fields marked with :meth:`StableUIDField.declare`."""
        return [
            name
            for name in getattr(cls, "model_fields", {})
            if (_read_field_json_schema_extra(cls, name) or {}).get("stable_uid")
        ]

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        stable_uid_fields = cls.stable_uid_field_names()
        if len(stable_uid_fields) > 1:
            fields = ", ".join(stable_uid_fields)
            raise TypeError(
                f"{cls.__name__} may declare at most one StableUIDField; found {fields}"
            )

    @classmethod
    def compute_stable_uids(cls, df: "pd.DataFrame") -> "pd.DataFrame":
        """Populate deterministic ``uid`` values for rows with a stable UID value."""
        stable_uid_fields = cls.stable_uid_field_names()
        if not stable_uid_fields:
            return df

        field_name = stable_uid_fields[0]
        if field_name not in df.columns:
            raise KeyError(f"{cls.__name__}.compute_stable_uids requires column '{field_name}'")
        if "uid" not in df.columns:
            df["uid"] = [make_uid() for _ in range(len(df))]

        mask = df[field_name].notna()
        df.loc[mask, "uid"] = df.loc[mask, field_name].map(
            lambda value: make_stable_uid(_stable_uid_identity_str(value))
        )
        return df

    @model_validator(mode="after")
    def _validate_stable_uid(self):
        stable_uid_fields = type(self).stable_uid_field_names()
        if not stable_uid_fields:
            return self

        field_name = stable_uid_fields[0]
        stable_value = getattr(self, field_name)
        if stable_value is None:
            return self

        expected_uid = make_stable_uid(_stable_uid_identity_str(stable_value))
        if self.uid != expected_uid:
            raise ValueError(
                f"{type(self).__name__}.uid must equal make_stable_uid(str({field_name})) "
                f"when {field_name} is not None"
            )
        return self

    @classmethod
    def to_arrow_schema(cls) -> pa.Schema:
        """Return the Arrow schema with stable UID field metadata stamped."""
        schema: pa.Schema = super().to_arrow_schema()
        for name in cls.stable_uid_field_names():
            idx = schema.get_field_index(name)
            field = schema.field(idx)
            new_metadata = dict(field.metadata or {})
            new_metadata[STABLE_UID_METADATA_KEY] = b"true"
            schema = schema.set(idx, field.with_metadata(new_metadata))
        return schema


class HoxBaseSchema(LanceModel):
    """
    Base schema for all homeobox datasets. The only requirements are a uid string
    that allows for safe parallel-write scenarios, and at least one ZarrPointer
    into a feature space declared via :meth:`PointerField.declare`.
    """

    uid: str = Field(default_factory=make_uid)
    dataset_uid: str = ""

    @classmethod
    def compute_auto_fields(cls, obs_df: "pd.DataFrame") -> "pd.DataFrame":
        """Compute auto-generated fields on an obs DataFrame.

        Override in subclasses to populate fields that are derived from other columns.
        Must return the DataFrame (may modify in-place).
        """
        return obs_df

    def __init_subclass__(cls, **kwargs):
        """Class-definition-time: subclass must declare at least one pointer field.

        Each pointer-typed field must be declared with
        :meth:`PointerField.declare`, which attaches ``is_pointer=True`` and the
        feature_space to the pydantic ``Field``. The declared feature_space must
        be registered, and its ``pointer_kind`` must match the annotation
        (``SparseZarrPointer`` ↔ sparse spec, ``DenseZarrPointer`` ↔ dense spec,
        ``DiscreteSpatialPointer`` ↔ discrete_spatial spec).
        """
        super().__init_subclass__(**kwargs)
        pointer_annotations = _iter_pointer_annotations(cls)
        if not pointer_annotations:
            raise TypeError(
                f"{cls.__name__} must declare at least one SparseZarrPointer, "
                f"DenseZarrPointer, or DiscreteSpatialPointer field via "
                f"PointerField.declare(...)"
            )
        for name, pointer_type in pointer_annotations:
            extra = _read_field_json_schema_extra(cls, name)
            if extra is None or not extra.get("is_pointer"):
                raise TypeError(
                    f"{cls.__name__}.{name}: pointer-typed fields must be declared via "
                    f"PointerField.declare(feature_space=...)"
                )
            feature_space = extra.get("feature_space")
            if not isinstance(feature_space, str) or not feature_space:
                raise TypeError(
                    f"{cls.__name__}.{name}: PointerField.declare must be called with "
                    f"a non-empty feature_space string"
                )
            spec = get_spec(feature_space)
            if pointer_type is SparseZarrPointer:
                annotation_kind = PointerKind.SPARSE
            elif pointer_type is DenseZarrPointer:
                annotation_kind = PointerKind.DENSE
            else:
                annotation_kind = PointerKind.DISCRETE_SPATIAL
            if annotation_kind is not spec.pointer_kind:
                raise TypeError(
                    f"{cls.__name__}.{name}: {annotation_kind.value} pointer annotation "
                    f"does not match feature_space '{feature_space}' which expects "
                    f"{spec.pointer_kind.value}"
                )

    @model_validator(mode="after")
    def _require_at_least_one_pointer(self):
        """Instance-time: at least one pointer must be non-None."""
        for name in self.model_fields:
            if isinstance(
                getattr(self, name),
                SparseZarrPointer | DenseZarrPointer | DiscreteSpatialPointer,
            ):
                return self
        raise ValueError(
            f"{type(self).__name__} requires at least one populated zarr pointer field"
        )

    @classmethod
    def to_arrow_schema(cls) -> pa.Schema:
        """Return the Arrow schema with ``homeobox.feature_space`` metadata stamped.

        Overrides :meth:`LanceModel.to_arrow_schema` to persist each pointer
        field's declared feature_space as Arrow field-level metadata. The
        schema-less read path (:func:`_infer_pointer_fields_from_arrow`)
        uses this metadata to reconstruct pointer-field info without the
        Python schema class.
        """
        schema: pa.Schema = super().to_arrow_schema()
        for name, _ in _iter_pointer_annotations(cls):
            feature_space = _read_field_json_schema_extra(cls, name)["feature_space"]
            idx = schema.get_field_index(name)
            field = schema.field(idx)
            new_metadata = dict(field.metadata or {})
            new_metadata[POINTER_FEATURE_SPACE_METADATA_KEY] = feature_space.encode("utf-8")
            schema = schema.set(idx, field.with_metadata(new_metadata))
        return schema


# Fields set automatically by the atlas — never expected in user-provided obs.
AUTO_FIELDS: frozenset[str] = frozenset(HoxBaseSchema.model_fields)


class FeatureBaseSchema(StableUIDBaseSchema):
    """
    Minimal schema for a global feature registry entry.

    Each feature space (e.g. genes, proteins) maintains its own registry where
    every row is a unique feature. Subclass this to add modality-specific fields.

    Fields:
        uid: Canonical stable identifier. Safe to preserve across registry rebuilds.
        global_index: Unique stable integer, assigned incrementally (new features get
            max(existing) + 1). Used as a scatter/gather key in compute paths. Never
            reassigned once set — use uid for durable references.
    """

    global_index: int | None = None


class DatasetSchema(LanceModel):
    """Metadata for a single ingested dataset.

    ``zarr_group`` is the per-row primary key (unique per modality write).
    ``dataset_uid`` is the logical dataset identifier referenced by ``HoxBaseSchema.dataset_uid``;
    it is shared across rows that belong to the same multimodal batch (one row per
    feature space).
    """

    dataset_uid: str = Field(default_factory=make_uid)
    zarr_group: str
    feature_space: str  # FeatureSpace value
    n_rows: int
    # TODO: Layout UID is updated automatically by add_or_reuse_layout. If a user forgets
    # to call that method during ingestion, this will break. add_or_reuse_layout should
    # probably be called automatically somewhere to avoid mistakes.
    layout_uid: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


class FeatureLayout(LanceModel):
    """Per-feature-per-layout index row for the ``_feature_layouts`` table.

    Each row maps a single feature within a unique feature ordering (layout)
    to its local position and its global position in the feature registry.
    Multiple datasets sharing the same feature ordering reference the same
    layout_uid, dramatically reducing row count.

    Parameters
    ----------
    layout_uid:
        Content-hash of the ordered feature list (shared across datasets
        with identical feature orderings).
    feature_uid:
        Global feature UID (FTS indexed for feature-to-layout lookups).
    local_index:
        0-based position of the feature within this layout.
    global_index:
        Position in the global feature registry. Denormalized from the registry and
        kept in sync by ``sync_layouts_global_index``.
    """

    layout_uid: str
    feature_uid: str
    local_index: int
    global_index: int | None = None


class AtlasVersionRecord(LanceModel):
    """One row per atlas snapshot created by ``RaggedAtlas.snapshot()``.

    Captures the Lance table versions for every table in the atlas at the
    time of the snapshot, enabling reproducible point-in-time queries via
    ``RaggedAtlas.checkout(version)``.

    Parameters
    ----------
    version:
        Monotonically increasing snapshot version number.
    obs_table_name:
        Name of the HoxBaseSchema Lance table.
    obs_table_version:
        Lance version of the HoxBaseSchema table at snapshot time.
    dataset_table_name:
        Name of the datasets Lance table.
    dataset_table_version:
        Lance version of the datasets table at snapshot time.
    registry_table_names:
        JSON-encoded mapping of ``{feature_space: table_name}`` for feature registries.
    registry_table_versions:
        JSON-encoded mapping of ``{feature_space: version_int}`` for feature registries.
    feature_layouts_table_version:
        Lance version of the ``_feature_layouts`` table at snapshot time.
    total_rows:
        Total number of rows across all datasets at snapshot time.
    created_at:
        ISO-8601 UTC timestamp of when the snapshot was created.
    """

    version: int
    obs_table_name: str
    obs_table_version: int
    dataset_table_name: str
    dataset_table_version: int
    registry_table_names: str
    registry_table_versions: str
    feature_layouts_table_version: int
    total_rows: int
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
