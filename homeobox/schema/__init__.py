import importlib

from homeobox.schema.definitions import (
    AUTO_FIELDS,
    POINTER_FEATURE_SPACE_METADATA_KEY,
    STABLE_UID_METADATA_KEY,
    AtlasVersionRecord,
    CrossReferenceField,
    DatasetSchema,
    FeatureBaseSchema,
    FeatureLayout,
    HoxBaseSchema,
    OntologyAlignedField,
    PointerField,
    PolymorphicRegistryKeyField,
    RegistryBaseSchema,
    RegistryKeyField,
    StableUIDBaseSchema,
    StableUIDField,
    SummaryField,
    _extract_pointer_fields,
    _infer_pointer_fields_from_arrow,
    _infer_pointer_type_from_struct_fields,
    _iter_pointer_annotations,
    combine_markers,
    make_stable_uid,
    make_uid,
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "emit": ("homeobox.schema.codegen", "emit"),
    "emit_to_file": ("homeobox.schema.codegen", "emit_to_file"),
    "model_from_file": ("homeobox.schema.ingest", "model_from_file"),
    "model_from_module": ("homeobox.schema.ingest", "model_from_module"),
    "model_from_source": ("homeobox.schema.ingest", "model_from_source"),
}

__all__ = [
    "AUTO_FIELDS",
    "POINTER_FEATURE_SPACE_METADATA_KEY",
    "STABLE_UID_METADATA_KEY",
    "AtlasVersionRecord",
    "CrossReferenceField",
    "DatasetSchema",
    "FeatureBaseSchema",
    "FeatureLayout",
    "HoxBaseSchema",
    "OntologyAlignedField",
    "PointerField",
    "PolymorphicRegistryKeyField",
    "RegistryBaseSchema",
    "RegistryKeyField",
    "StableUIDBaseSchema",
    "StableUIDField",
    "SummaryField",
    "_extract_pointer_fields",
    "_infer_pointer_fields_from_arrow",
    "_infer_pointer_type_from_struct_fields",
    "_iter_pointer_annotations",
    "combine_markers",
    "emit",
    "emit_to_file",
    "make_stable_uid",
    "make_uid",
    "model_from_file",
    "model_from_module",
    "model_from_source",
]


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_name, attr = _LAZY_EXPORTS[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
