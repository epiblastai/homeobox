"""Project a :class:`~homeobox.schema_ir.SchemaModel` into the review-UI result.

The review UI needs the tables, fields, and relationships a schema declares --
not the raw Python. That structured view is now derived from a single source of
truth: the YAML-backed intermediate representation (:mod:`homeobox.schema_ir`).
There is exactly one path here, ``IR -> parsed_result``; the schema source is
parsed elsewhere (``schema.py -> IR`` lives in :mod:`homeobox.schema_ingest`).

``parsed_result`` is a plain dict::

    {
        "obs": table | None,        # the single HoxBaseSchema table
        "dataset": table | None,    # the single DatasetSchema table
        "tables": [table, ...],     # feature registries, entities, plain tables
        "relationships": [...],     # pointer / registry-key / summary edges
        "warnings": [...],          # e.g. missing obs or dataset table
    }

where each ``table`` is ``{"class_name", "kind", "fields": [...]}``. Inherited
built-in fields (``uid``, ``dataset_uid``, ...) are read from the trusted
homeobox base classes -- the IR omits them, but the review UI shows them -- so
parser output tracks the installed homeobox version as its base fields evolve.
"""

import types
from functools import lru_cache
from typing import Any, Union, get_args, get_origin

from homeobox.schema_ir import FieldDef, SchemaModel, TableDef

# IR base kind -> the "kind" string the review UI renders. The IR's "registry"
# bucket (RegistryBaseSchema / StableUIDBaseSchema subclasses) is surfaced as
# "entity"; every other kind keeps its name.
_IR_KIND_TO_PARSER_KIND: dict[str, str] = {
    "obs": "obs",
    "dataset": "dataset",
    "feature_registry": "feature_registry",
    "registry": "entity",
    "table": "table",
}

_FALLBACK_HOMEBOX_BASE_FIELDS: dict[str, list[dict]] = {
    "HoxBaseSchema": [
        {"name": "uid", "type": "str"},
        {"name": "dataset_uid", "type": "str"},
    ],
    "DatasetSchema": [
        {"name": "dataset_uid", "type": "str"},
        {"name": "zarr_group", "type": "str"},
        {"name": "feature_space", "type": "str"},
        {"name": "n_rows", "type": "int"},
        {"name": "layout_uid", "type": "str"},
        {"name": "created_at", "type": "str"},
    ],
    "FeatureBaseSchema": [
        {"name": "uid", "type": "str"},
        {"name": "global_index", "type": "int | None"},
    ],
    "RegistryBaseSchema": [{"name": "uid", "type": "str"}],
    "LanceModel": [],
}


# ---------------------------------------------------------------------------
# Inherited base-class fields
# ---------------------------------------------------------------------------


def _annotation_to_string(annotation: Any) -> str:
    """Render a runtime annotation the way the IR writes field types.

    Unions come out as ``A | B`` and custom classes as their bare name, so a
    base field typed ``int | None`` reads identically to an IR-declared field.
    """
    if annotation is None or annotation is type(None):
        return "None"
    if annotation is Any:
        return "Any"

    origin = get_origin(annotation)
    if origin is Union or isinstance(annotation, types.UnionType):
        return " | ".join(_annotation_to_string(arg) for arg in get_args(annotation))
    if origin is not None:
        args = get_args(annotation)
        name = getattr(origin, "__name__", None) or str(origin).replace("typing.", "")
        if not args:
            return name
        return f"{name}[{', '.join(_annotation_to_string(arg) for arg in args)}]"
    if isinstance(annotation, type):
        return annotation.__name__

    return str(annotation).replace("typing.", "").replace("<class '", "").replace("'>", "")


def _field_copy(field: dict) -> dict:
    return {
        key: (_field_copy(value) if isinstance(value, dict) else value)
        for key, value in field.items()
    }


@lru_cache
def _homeobox_base_fields_by_class() -> dict[str, tuple[tuple[tuple[str, object], ...], ...]]:
    """Read trusted homeobox base fields straight off the installed classes.

    The IR drops inherited built-ins, but ``parsed_result`` re-attaches them so
    the review UI can show ``uid`` / ``dataset_uid`` and so the obs ``dataset_uid``
    foreign key resolves. Only homeobox's own base classes are imported here; no
    user schema is executed.
    """
    result: dict[str, list[dict]] = {}
    try:
        from homeobox import schema as homeobox_schema
    except Exception:
        result = {
            base_name: [{**_field_copy(field), "inherited": True} for field in fields]
            for base_name, fields in _FALLBACK_HOMEBOX_BASE_FIELDS.items()
        }
    else:
        for base_name, fallback_fields in _FALLBACK_HOMEBOX_BASE_FIELDS.items():
            base_cls = getattr(homeobox_schema, base_name, None)
            model_fields = getattr(base_cls, "model_fields", None)
            if not model_fields:
                result[base_name] = [
                    {**_field_copy(field), "inherited": True} for field in fallback_fields
                ]
                continue
            result[base_name] = [
                {
                    "name": name,
                    "type": _annotation_to_string(field_info.annotation),
                    "inherited": True,
                }
                for name, field_info in model_fields.items()
            ]

    return {
        base_name: tuple(tuple(field.items()) for field in fields)
        for base_name, fields in result.items()
    }


def _homeobox_base_fields(base_name: str) -> list[dict]:
    fields = _homeobox_base_fields_by_class().get(base_name, ())
    return [dict(field) for field in fields]


# ---------------------------------------------------------------------------
# IR -> table dicts
# ---------------------------------------------------------------------------


def _field_dict_from_ir(field: FieldDef) -> dict:
    """Render one IR field as a ``parsed_result`` field dict.

    Marker payloads are copied verbatim, except ``registry_key`` /
    ``polymorphic_registry_key`` get their implicit ``target_field="uid"``
    re-filled (the IR omits it when it is the default).
    """
    out: dict[str, Any] = {"name": field.name, "type": field.type}
    markers = field.markers

    if "pointer" in markers:
        out["pointer"] = dict(markers["pointer"])

    registry_key = markers.get("registry_key")
    if registry_key is not None:
        out["registry_key"] = {
            "target_schema": registry_key["target_schema"],
            "target_field": registry_key.get("target_field", "uid"),
        }

    polymorphic = markers.get("polymorphic_registry_key")
    if polymorphic is not None:
        out["polymorphic_registry_key"] = {
            "type_field": polymorphic["type_field"],
            "target_field": polymorphic.get("target_field", "uid"),
            "variants": dict(polymorphic["variants"]),
        }

    if "ontology_aligned" in markers:
        out["ontology_aligned"] = dict(markers["ontology_aligned"])
    if "cross_reference" in markers:
        out["cross_reference"] = dict(markers["cross_reference"])
    if markers.get("stable_uid"):
        out["stable_uid"] = True
    if "summary" in markers:
        out["summary"] = dict(markers["summary"])

    return out


def _table_dict_from_ir(table: TableDef) -> dict:
    """Render one IR table as a ``parsed_result`` table dict.

    Inherited base-class fields lead (marked ``inherited``), followed by the
    fields the schema declares in IR order.
    """
    fields = _homeobox_base_fields(table.base_class)
    fields.extend(_field_dict_from_ir(field) for field in table.fields)
    return {
        "class_name": table.name,
        "kind": _IR_KIND_TO_PARSER_KIND[table.base],
        "fields": fields,
    }


# ---------------------------------------------------------------------------
# Relationships and assembly
# ---------------------------------------------------------------------------


def _field_by_name(table: dict, field_name: str) -> dict | None:
    for field in table["fields"]:
        if field["name"] == field_name:
            return field
    return None


def _add_default_relationship_metadata(obs: dict | None, dataset: dict | None) -> None:
    if obs is None or dataset is None:
        return
    obs_dataset_uid = _field_by_name(obs, "dataset_uid")
    dataset_dataset_uid = _field_by_name(dataset, "dataset_uid")
    if obs_dataset_uid is None or dataset_dataset_uid is None:
        return
    obs_dataset_uid.setdefault(
        "registry_key",
        {
            "target_schema": dataset["class_name"],
            "target_field": "dataset_uid",
        },
    )


def _relationships_for_table(table: dict) -> list[dict]:
    relationships: list[dict] = []
    for field in table["fields"]:
        source = {
            "source_table": table["class_name"],
            "source_field": field["name"],
        }
        pointer = field.get("pointer")
        if isinstance(pointer, dict) and pointer.get("feature_registry_schema"):
            relationships.append(
                {
                    "kind": "pointer_feature_registry",
                    **source,
                    "target_schema": pointer["feature_registry_schema"],
                    "target_field": "uid",
                    "feature_space": pointer.get("feature_space"),
                }
            )

        registry_key = field.get("registry_key")
        if isinstance(registry_key, dict):
            relationships.append(
                {
                    "kind": "registry_key",
                    **source,
                    "target_schema": registry_key["target_schema"],
                    "target_field": registry_key.get("target_field", "uid"),
                }
            )

        polymorphic = field.get("polymorphic_registry_key")
        if isinstance(polymorphic, dict):
            for variant, target_schema in polymorphic.get("variants", {}).items():
                relationships.append(
                    {
                        "kind": "polymorphic_registry_key",
                        **source,
                        "target_schema": target_schema,
                        "target_field": polymorphic.get("target_field", "uid"),
                        "type_field": polymorphic["type_field"],
                        "variant": variant,
                    }
                )

        summary = field.get("summary")
        if isinstance(summary, dict):
            relationships.append(
                {
                    "kind": "summary",
                    **source,
                    "target_schema": summary["target_schema"],
                    "target_field": summary["target_field"],
                    "op": summary["op"],
                }
            )
    return relationships


def _assemble_parse_result(parsed_tables: list[dict]) -> dict:
    """Bucket ``{class_name, kind, fields}`` tables into the parser result shape.

    Selects the single obs and datasets tables (warning on duplicates and on
    absence), attaches the default ``dataset_uid`` relationship, and derives the
    relationship list.
    """
    obs: dict | None = None
    dataset: dict | None = None
    tables: list[dict] = []
    warnings: list[str] = []

    for table in parsed_tables:
        kind = table["kind"]
        if kind == "obs":
            if obs is None:
                obs = table
            else:
                warnings.append(
                    f"Multiple obs tables found ({obs['class_name']}, {table['class_name']}); "
                    f"using {obs['class_name']}."
                )
                tables.append(table)
        elif kind == "dataset":
            if dataset is None:
                dataset = table
            else:
                warnings.append(
                    f"Multiple dataset tables found ({dataset['class_name']}, "
                    f"{table['class_name']}); using {dataset['class_name']}."
                )
                tables.append(table)
        else:
            tables.append(table)

    if obs is None:
        warnings.append("No obs table (HoxBaseSchema subclass) found.")
    if dataset is None:
        warnings.append("No datasets table (DatasetSchema subclass) found.")

    _add_default_relationship_metadata(obs, dataset)

    all_tables = [table for table in (obs, dataset, *tables) if table is not None]
    relationships = [
        relationship for table in all_tables for relationship in _relationships_for_table(table)
    ]

    return {
        "obs": obs,
        "dataset": dataset,
        "tables": tables,
        "relationships": relationships,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parsed_result_from_model(model: SchemaModel) -> dict:
    """Project a :class:`SchemaModel` into the review-UI ``parsed_result`` dict.

    The single obs table and dataset table are selected by base kind; feature
    registries, entities, and plain tables land in ``tables``. Relationships
    (pointers, registry keys, polymorphic keys, summaries) are derived from the
    declared markers, plus the implicit obs ``dataset_uid`` foreign key.
    """
    parsed_tables = [_table_dict_from_ir(table) for table in model.obs_tables]
    if model.dataset_table is not None:
        parsed_tables.append(_table_dict_from_ir(model.dataset_table))
    parsed_tables.extend(_table_dict_from_ir(t) for t in model.feature_registry_tables)
    parsed_tables.extend(_table_dict_from_ir(t) for t in model.fk_registry_tables)
    parsed_tables.extend(_table_dict_from_ir(t) for t in model.other_tables)
    return _assemble_parse_result(parsed_tables)
