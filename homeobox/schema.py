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

from homeobox.group_specs import get_spec, registered_feature_spaces
from homeobox.pointer_types import (
    ZARR_POINTER_TYPES,
    ZarrPointer,
)

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

    The concrete pointer type is carried by the matching FeatureSpaceSpec.
    ``feature_registry_schema`` is optional, informational metadata for code
    parsers and visualizers; it is not stored in Arrow metadata.
    """

    field_name: str
    feature_space: str
    feature_registry_schema: str | None = None

    @staticmethod
    def declare(
        *,
        feature_space: str,
        feature_registry_schema: type | str | None = None,
        default: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Factory used in schema class bodies to mark a pointer field.

        Returns a pydantic ``Field`` whose ``json_schema_extra`` carries
        ``{"is_pointer": True, "feature_space": feature_space}``. When
        provided, ``feature_registry_schema`` is recorded as lightweight
        Python-level metadata only. The feature space is resolved against the
        registered spec at schema definition time by
        :meth:`HoxBaseSchema.__init_subclass__`.
        """
        extra = dict(kwargs.pop("json_schema_extra", {}) or {})
        extra["is_pointer"] = True
        extra["feature_space"] = feature_space
        if feature_registry_schema is not None:
            if isinstance(feature_registry_schema, str):
                feature_registry_schema_name = feature_registry_schema
            else:
                feature_registry_schema_name = feature_registry_schema.__name__
            if not feature_registry_schema_name:
                raise TypeError("PointerField.declare requires a non-empty feature_registry_schema")
            extra["feature_registry_schema"] = feature_registry_schema_name
        return Field(
            default=default,
            json_schema_extra=extra,
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


@dataclasses.dataclass(frozen=True)
class RegistryKeyField:
    """Runtime metadata for a schema field that references another schema field.

    This marker is informational only. It is not written to Arrow metadata and
    does not add validation or database constraints.
    """

    field_name: str
    target_schema: str
    target_field: str

    @staticmethod
    def declare(
        *,
        target_schema: type | str,
        target_field: str = "uid",
        default: Any = ...,
        **kwargs: Any,
    ) -> Any:
        """Factory used in schema class bodies to mark a foreign-key-like field."""
        if isinstance(target_schema, str):
            target_schema_name = target_schema
        else:
            target_schema_name = target_schema.__name__
        if not target_schema_name:
            raise TypeError("RegistryKeyField.declare requires a non-empty target_schema")
        if not isinstance(target_field, str) or not target_field:
            raise TypeError("RegistryKeyField.declare requires a non-empty target_field string")

        extra = dict(kwargs.pop("json_schema_extra", {}) or {})
        extra["registry_key"] = {
            "target_schema": target_schema_name,
            "target_field": target_field,
        }
        return Field(default=default, json_schema_extra=extra, **kwargs)


@dataclasses.dataclass(frozen=True)
class PolymorphicRegistryKeyField:
    """Runtime metadata for a value field whose target schema depends on another field.

    The value field (marked with :meth:`declare`) holds foreign-key values. The
    parallel ``type_field`` holds discriminator values that select which target
    schema each value refers to. This marker is informational only.
    """

    field_name: str
    type_field: str
    target_field: str
    variants: dict[str, str]

    @staticmethod
    def declare(
        *,
        type_field: str,
        variants: dict[str, type | str],
        target_field: str = "uid",
        default: Any = ...,
        **kwargs: Any,
    ) -> Any:
        """Factory used in schema class bodies to mark a polymorphic foreign-key field."""
        if not isinstance(type_field, str) or not type_field:
            raise TypeError(
                "PolymorphicRegistryKeyField.declare requires a non-empty type_field string"
            )
        if not isinstance(target_field, str) or not target_field:
            raise TypeError(
                "PolymorphicRegistryKeyField.declare requires a non-empty target_field string"
            )
        if not variants:
            raise TypeError(
                "PolymorphicRegistryKeyField.declare requires a non-empty variants dict"
            )

        variant_names: dict[str, str] = {}
        for key, target_schema in variants.items():
            if not isinstance(key, str) or not key:
                raise TypeError(
                    "PolymorphicRegistryKeyField.declare variants keys must be non-empty strings"
                )
            if isinstance(target_schema, str):
                target_schema_name = target_schema
            else:
                target_schema_name = target_schema.__name__
            if not target_schema_name:
                raise TypeError(
                    "PolymorphicRegistryKeyField.declare requires a non-empty target schema "
                    f"for variant {key!r}"
                )
            variant_names[key] = target_schema_name

        extra = dict(kwargs.pop("json_schema_extra", {}) or {})
        extra["polymorphic_registry_key"] = {
            "type_field": type_field,
            "target_field": target_field,
            "variants": variant_names,
        }
        return Field(default=default, json_schema_extra=extra, **kwargs)


@dataclasses.dataclass(frozen=True)
class OntologyAlignedField:
    """Runtime metadata marking a schema field as aligned to an ontology.

    This marker is informational only. It is not written to Arrow metadata and
    does not add validation or database constraints.
    """

    field_name: str
    ontology_name: str

    @staticmethod
    def declare(
        *,
        ontology_name: str,
        default: Any = ...,
        **kwargs: Any,
    ) -> Any:
        """Factory used in schema class bodies to mark an ontology-aligned field."""
        if not isinstance(ontology_name, str) or not ontology_name:
            raise TypeError(
                "OntologyAlignedField.declare requires a non-empty ontology_name string"
            )

        extra = dict(kwargs.pop("json_schema_extra", {}) or {})
        extra["ontology_aligned"] = {"ontology_name": ontology_name}
        return Field(default=default, json_schema_extra=extra, **kwargs)


@dataclasses.dataclass(frozen=True)
class CrossReferenceField:
    """Runtime metadata marking a schema field as a cross-reference to a database.

    Like :class:`OntologyAlignedField`, but the field references an external
    database (e.g. ``doi``, ``pubmed``, ``pubchem``, ``uniprot``) rather than an
    ontology. This marker is informational only. It is not written to Arrow
    metadata and does not add validation or database constraints.
    """

    field_name: str
    database_name: str

    @staticmethod
    def declare(
        *,
        database_name: str,
        default: Any = ...,
        **kwargs: Any,
    ) -> Any:
        """Factory used in schema class bodies to mark a cross-reference field."""
        if not isinstance(database_name, str) or not database_name:
            raise TypeError("CrossReferenceField.declare requires a non-empty database_name string")

        extra = dict(kwargs.pop("json_schema_extra", {}) or {})
        extra["cross_reference"] = {"database_name": database_name}
        return Field(default=default, json_schema_extra=extra, **kwargs)


def combine_markers(*markers: Any, default: Any = ...) -> Any:
    """Attach several informational field markers to a single schema field.

    Each marker factory (``StableUIDField.declare``, ``CrossReferenceField.declare``,
    ``RegistryKeyField.declare``, …) writes its metadata under a distinct top-level
    key in the field's ``json_schema_extra``. Because those keys are orthogonal, the
    markers can be merged into one field without conflict:

    ```python
    pubchem_cid: int | None = combine_markers(
        StableUIDField.declare(),
        CrossReferenceField.declare(database_name="pubchem"),
        default=None,
    )
    ```

    The resulting field's ``json_schema_extra`` is the union of every marker's
    metadata, so readers that look up a single key (e.g. ``extra.get("stable_uid")``)
    keep working unchanged whether the field carries one marker or several.

    ``combine_markers`` owns the field's ``default``; the ``default`` passed to the
    inner ``declare()`` calls is ignored (each marker factory supplies its own factory
    default, which is not meaningful once combined). Two markers writing the same
    metadata key (e.g. two ``CrossReferenceField`` markers) is a class-definition-time
    ``TypeError``.
    """
    if len(markers) < 2:
        raise TypeError("combine_markers requires at least two markers")

    combined_extra: dict[str, Any] = {}
    for marker in markers:
        extra = getattr(marker, "json_schema_extra", None)
        if not isinstance(extra, dict):
            raise TypeError(
                "combine_markers expects marker fields produced by a "
                "<Marker>.declare(...) factory; got an object without "
                "json_schema_extra metadata"
            )
        for key, value in extra.items():
            if key in combined_extra:
                raise TypeError(
                    f"combine_markers: conflicting marker metadata for key {key!r}; "
                    "two markers cannot both write the same key onto one field"
                )
            combined_extra[key] = value

    return Field(default=default, json_schema_extra=combined_extra)


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
            if t in ZARR_POINTER_TYPES:
                result.append((name, t))
                break
    return result


def _validate_pointer_field(
    *,
    field_name: str,
    pointer_type: type,
    feature_space: str,
    feature_registry_schema: str | None = None,
    context: str,
) -> PointerField:
    """Validate feature-space metadata and return runtime pointer metadata."""
    spec = get_spec(feature_space)
    if pointer_type is not spec.pointer_type:
        raise TypeError(
            f"{context}: {pointer_type.pointer_type_name} pointer does not match "
            f"feature_space '{feature_space}' which expects "
            f"{spec.pointer_type.pointer_type_name}"
        )
    return PointerField(
        field_name=field_name,
        feature_space=feature_space,
        feature_registry_schema=feature_registry_schema,
    )


def _extract_pointer_fields(
    schema_cls: type["HoxBaseSchema"],
) -> dict[str, PointerField]:
    """Introspect a schema class and return :class:`PointerField` for each pointer field.

    Reads the ``json_schema_extra`` that :meth:`PointerField.declare` attaches
    to each pydantic field. Keys are the Python attribute names — which may
    differ from the feature_space value carried by each pointer — so schemas
    can declare multiple columns in the same feature space.
    """
    result: dict[str, PointerField] = {}
    for name, pointer_type in _iter_pointer_annotations(schema_cls):
        extra = _read_field_json_schema_extra(schema_cls, name) or {}
        feature_space = extra.get("feature_space")
        if not feature_space:
            raise TypeError(
                f"{schema_cls.__name__}.{name}: pointer field missing feature_space "
                f"metadata; declare with PointerField.declare(feature_space=...)"
            )
        feature_registry_schema = extra.get("feature_registry_schema")
        if feature_registry_schema is not None and (
            not isinstance(feature_registry_schema, str) or not feature_registry_schema
        ):
            raise TypeError(
                f"{schema_cls.__name__}.{name}: feature_registry_schema metadata must be "
                f"a non-empty string when provided"
            )
        result[name] = _validate_pointer_field(
            field_name=name,
            pointer_type=pointer_type,
            feature_space=feature_space,
            feature_registry_schema=feature_registry_schema,
            context=f"{schema_cls.__name__}.{name}",
        )
    return result


def _infer_pointer_type_from_struct_fields(sub_names: set[str]) -> type | None:
    for pointer_type in sorted(
        ZARR_POINTER_TYPES,
        key=lambda cls: len(frozenset(cls.model_fields)),
        reverse=True,
    ):
        if frozenset(pointer_type.model_fields) == sub_names:
            return pointer_type
    return None


def _infer_pointer_fields_from_arrow(
    arrow_schema: pa.Schema,
) -> dict[str, PointerField]:
    """Infer pointer fields from a obs table's Arrow schema.

    Detects struct columns whose sub-field names match the signatures of
    ``SparseZarrPointer``, ``DenseZarrPointer``, or ``DiscreteSpatialPointer``,
    then reads the declared feature_space from Arrow field metadata (key
    :data:`POINTER_FEATURE_SPACE_METADATA_KEY`) stamped by
    :meth:`HoxBaseSchema.to_arrow_schema`.
    """
    result: dict[str, PointerField] = {}
    for i in range(len(arrow_schema)):
        field = arrow_schema.field(i)
        if not pa.types.is_struct(field.type):
            continue
        sub_names = {field.type.field(j).name for j in range(field.type.num_fields)}
        pointer_type = _infer_pointer_type_from_struct_fields(sub_names)
        if pointer_type is None:
            continue

        metadata = field.metadata or {}
        fs_bytes = metadata.get(POINTER_FEATURE_SPACE_METADATA_KEY)
        if fs_bytes is not None:
            feature_space = fs_bytes.decode("utf-8")
        elif field.name in registered_feature_spaces():
            # Legacy-atlas fallback: tables written before PointerField.declare
            # existed carry no per-field metadata, but the old convention required
            # field_name == feature_space. Fall back to that only if it resolves
            # to a registered spec.
            feature_space = field.name
        else:
            raise TypeError(
                f"Arrow field '{field.name}' looks like a {pointer_type.pointer_type_name} pointer "
                f"but is missing the '{POINTER_FEATURE_SPACE_METADATA_KEY.decode()}' "
                f"metadata key, and its name does not match any registered feature "
                f"space. Open with an explicit obs_schema or re-create the atlas with "
                f"a schema that uses PointerField.declare(feature_space=...)."
            )
        result[field.name] = _validate_pointer_field(
            field_name=field.name,
            pointer_type=pointer_type,
            feature_space=feature_space,
            context=f"Arrow field '{field.name}'",
        )
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
        """Populate ``uid`` values on *df*.

        Adds a ``uid`` column when missing. When the schema declares a
        :class:`StableUIDField`, rows whose stable-uid source value is non-null
        get a deterministic ``uid = make_stable_uid(source)``; rows with a null
        source value fall back to a random uid. When the schema has no
        :class:`StableUIDField`, every row gets a random uid.
        """
        if "uid" not in df.columns:
            df["uid"] = [make_uid() for _ in range(len(df))]

        stable_uid_fields = cls.stable_uid_field_names()
        if not stable_uid_fields:
            return df

        field_name = stable_uid_fields[0]
        if field_name not in df.columns:
            raise KeyError(f"{cls.__name__}.compute_stable_uids requires column '{field_name}'")

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


class RegistryBaseSchema(StableUIDBaseSchema):
    """Base schema for registry tables referenced by foreign keys.

    A registry table holds a set of unique, dedup-able entities (genes,
    proteins, molecules, perturbations, publications, donors, …) keyed by a
    stable ``uid``. By convention — though not enforced — the
    ``target_schema`` of every :class:`RegistryKeyField` in a database should be
    a :class:`RegistryBaseSchema` subclass.

    This adds no behavior over :class:`StableUIDBaseSchema`; it exists to give
    foreign-key targets an explicit, greppable type name that documents intent
    and aids schema parsers and visualizers.
    """


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
        be registered, and its ``pointer_type`` must match the annotation
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
            feature_registry_schema = extra.get("feature_registry_schema")
            if feature_registry_schema is not None and (
                not isinstance(feature_registry_schema, str) or not feature_registry_schema
            ):
                raise TypeError(
                    f"{cls.__name__}.{name}: PointerField.declare must be called with "
                    f"a non-empty feature_registry_schema when provided"
                )
            _validate_pointer_field(
                field_name=name,
                pointer_type=pointer_type,
                feature_space=feature_space,
                feature_registry_schema=feature_registry_schema,
                context=f"{cls.__name__}.{name}",
            )

    @model_validator(mode="after")
    def _require_at_least_one_pointer(self):
        """Instance-time: at least one pointer must be non-None."""
        for name in self.model_fields:
            if isinstance(getattr(self, name), ZarrPointer):
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


class FeatureBaseSchema(RegistryBaseSchema):
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

    @model_validator(mode="after")
    def _validate_global_index_unset(self):
        if self.global_index is not None:
            raise ValueError("global_index must be None when creating feature records")
        return self


class DatasetSchema(LanceModel):
    """Metadata for a single ingested dataset.

    ``zarr_group`` is the per-row primary key (unique per modality write) and
    auto-generates a random uid when not supplied. ``dataset_uid`` is the logical
    dataset identifier referenced by ``HoxBaseSchema.dataset_uid``; it also
    auto-generates, but is shared across rows that belong to the same multimodal
    batch (one row per feature space) by passing the same value to each row.
    """

    dataset_uid: str = Field(default_factory=make_uid)
    zarr_group: str = Field(default_factory=make_uid)
    feature_space: str  # FeatureSpace value
    n_rows: int
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
    obs_table_versions:
        JSON-encoded mapping of ``{obs_table_name: version_int}`` covering every
        obs table in the atlas at snapshot time.
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
    obs_table_versions: str
    dataset_table_name: str
    dataset_table_version: int
    registry_table_names: str
    registry_table_versions: str
    feature_layouts_table_version: int
    total_rows: int
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
