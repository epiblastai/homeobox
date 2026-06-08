"""Statically parse a generated homeobox ``schema.py`` into structured tables.

The review UI needs the tables and fields the agent wrote, not the raw Python.
This module reads the schema source with :mod:`ast` -- it never imports or
executes the user file -- and classifies each declared class by its homeobox
base:

- exactly one ``HoxBaseSchema`` subclass -> the obs table,
- exactly one ``DatasetSchema`` subclass -> the datasets table,
- ``FeatureBaseSchema`` and any other ``LanceModel`` tables -> collapsed
  "tables" (feature registries plus additional entity/relationship tables).

Static parsing avoids arbitrary-code execution, and user-authored field types
are reported exactly as written (``str | None``, ``SparseZarrPointer | None``,
``list[str] | None``). Inherited built-in fields are read from the trusted
homeobox base classes when homeobox is importable, with a small fallback for
the known base fields.
"""

import ast
import inspect
import types
from functools import lru_cache
from pathlib import Path
from typing import Any, Union, get_args, get_origin

# homeobox base classes mapped to the table "kind" the review UI renders, in
# priority order (most specific first) so a class is bucketed by its tightest
# recognised base -- FeatureBaseSchema is a RegistryBaseSchema is a
# StableUIDBaseSchema, etc. RegistryBaseSchema and StableUIDBaseSchema share the
# "entity" kind; the separate entry exists so registry tables declared directly
# against RegistryBaseSchema are still recognised by the name-matching AST path.
_BASE_KINDS: list[tuple[str, str]] = [
    ("HoxBaseSchema", "obs"),
    ("DatasetSchema", "dataset"),
    ("FeatureBaseSchema", "feature_registry"),
    ("RegistryBaseSchema", "entity"),
    ("StableUIDBaseSchema", "entity"),
    ("LanceModel", "table"),
]
_KIND_PRIORITY = {kind: index for index, (_, kind) in enumerate(_BASE_KINDS)}

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
    "StableUIDBaseSchema": [{"name": "uid", "type": "str"}],
    "LanceModel": [],
}


def _attribute_parts(node: ast.AST, aliases: dict[str, str]) -> list[str] | None:
    """Return dotted-name parts for a name/attribute expression."""
    if isinstance(node, ast.Name):
        return [aliases.get(node.id, node.id)]
    if isinstance(node, ast.Attribute):
        parent = _attribute_parts(node.value, aliases)
        if parent is None:
            return None
        return [*parent, node.attr]
    return None


def _call_parts(node: ast.Call, aliases: dict[str, str]) -> list[str] | None:
    return _attribute_parts(node.func, aliases)


def _declare_marker(node: ast.Call, aliases: dict[str, str]) -> str | None:
    """Return the marker class for ``Foo.declare(...)`` calls."""
    parts = _call_parts(node, aliases)
    if not parts or parts[-1] != "declare" or len(parts) < 2:
        return None
    return parts[-2]


def _keyword(node: ast.Call, name: str) -> ast.AST | None:
    for keyword in node.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _string_value(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _schema_name(node: ast.AST | None) -> str | None:
    """Return the schema name represented by a class object or string literal."""
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _annotation_to_string(annotation: Any) -> str:
    """Render a runtime annotation similarly to ``ast.unparse`` output.

    Unions are rendered as ``A | B`` and custom classes as their bare name, so
    ``SparseZarrPointer | None`` and ``list[str] | None`` come out the same way
    the static parser reports them from source.
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
    """Read trusted homeobox base fields without importing the user schema.

    The uploaded/generated schema is still parsed statically. We only import
    homeobox's known base classes here so parser output follows the installed
    homeobox version as its built-in fields evolve.
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
                result[base_name] = [_field_copy(field) for field in fallback_fields]
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


def _string_schema_dict(node: ast.AST | None) -> dict[str, str]:
    if not isinstance(node, ast.Dict):
        return {}
    result: dict[str, str] = {}
    for key_node, value_node in zip(node.keys, node.values, strict=False):
        key = _string_value(key_node) if key_node is not None else None
        value = _schema_name(value_node)
        if key and value:
            result[key] = value
    return result


def _literal_jsonish(node: ast.AST | None) -> object:
    """Evaluate only simple literal JSON-like syntax used in Field metadata."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Dict):
        result: dict[object, object] = {}
        for key_node, value_node in zip(node.keys, node.values, strict=False):
            key = _literal_jsonish(key_node) if key_node is not None else None
            if isinstance(key, str):
                result[key] = _literal_jsonish(value_node)
        return result
    if isinstance(node, ast.List | ast.Tuple):
        return [_literal_jsonish(element) for element in node.elts]
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _marker_metadata_from_extra(extra: dict) -> dict:
    """Translate a field's ``json_schema_extra`` into parser field metadata.

    Shared by the static path (where ``extra`` is reconstructed from a literal
    ``Field(json_schema_extra=...)`` call) and the runtime path (where ``extra``
    is read directly off ``model_fields[...].json_schema_extra``). The two markers
    that combine onto one field via ``combine_markers`` land in the same dict, so
    every key is read independently and no key is mutually exclusive.
    """
    metadata: dict = {}
    if extra.get("is_pointer") and isinstance(extra.get("feature_space"), str):
        pointer = {"feature_space": extra["feature_space"]}
        if isinstance(extra.get("feature_registry_schema"), str):
            pointer["feature_registry_schema"] = extra["feature_registry_schema"]
        metadata["pointer"] = pointer

    registry_key = extra.get("registry_key")
    if isinstance(registry_key, dict) and isinstance(registry_key.get("target_schema"), str):
        metadata["registry_key"] = {
            "target_schema": registry_key["target_schema"],
            "target_field": registry_key.get("target_field", "uid"),
        }

    polymorphic = extra.get("polymorphic_registry_key")
    if isinstance(polymorphic, dict) and isinstance(polymorphic.get("type_field"), str):
        variants = polymorphic.get("variants")
        if isinstance(variants, dict):
            metadata["polymorphic_registry_key"] = {
                "type_field": polymorphic["type_field"],
                "target_field": polymorphic.get("target_field", "uid"),
                "variants": {
                    key: value
                    for key, value in variants.items()
                    if isinstance(key, str) and isinstance(value, str)
                },
            }

    ontology = extra.get("ontology_aligned")
    if isinstance(ontology, dict) and isinstance(ontology.get("ontology_name"), str):
        metadata["ontology_aligned"] = {"ontology_name": ontology["ontology_name"]}

    cross_reference = extra.get("cross_reference")
    if isinstance(cross_reference, dict) and isinstance(cross_reference.get("database_name"), str):
        metadata["cross_reference"] = {"database_name": cross_reference["database_name"]}

    if extra.get("stable_uid"):
        metadata["stable_uid"] = True

    summary = extra.get("summary")
    if (
        isinstance(summary, dict)
        and isinstance(summary.get("target_schema"), str)
        and isinstance(summary.get("target_field"), str)
        and isinstance(summary.get("op"), str)
    ):
        metadata["summary"] = {
            "target_schema": summary["target_schema"],
            "target_field": summary["target_field"],
            "op": summary["op"],
        }

    return metadata


def _field_call_metadata(value: ast.AST | None, aliases: dict[str, str]) -> dict:
    """Extract homeobox Field.declare metadata without importing the schema."""
    if not isinstance(value, ast.Call):
        return {}

    metadata: dict = {}
    marker = _declare_marker(value, aliases)

    # ``combine_markers(MarkerA.declare(...), MarkerB.declare(...), default=...)``
    # attaches several markers to one field. Each positional argument is itself a
    # ``<Marker>.declare(...)`` call writing a distinct metadata key, so recurse
    # into each and merge -- the union mirrors the field's runtime json_schema_extra.
    if _call_parts(value, aliases) == ["combine_markers"]:
        for argument in value.args:
            metadata.update(_field_call_metadata(argument, aliases))
        return metadata

    if marker == "PointerField":
        feature_space = _string_value(_keyword(value, "feature_space"))
        feature_registry_schema = _schema_name(_keyword(value, "feature_registry_schema"))
        pointer = {}
        if feature_space:
            pointer["feature_space"] = feature_space
        if feature_registry_schema:
            pointer["feature_registry_schema"] = feature_registry_schema
        if pointer:
            metadata["pointer"] = pointer

    elif marker == "RegistryKeyField":
        target_schema = _schema_name(_keyword(value, "target_schema"))
        target_field = _string_value(_keyword(value, "target_field")) or "uid"
        if target_schema:
            metadata["registry_key"] = {
                "target_schema": target_schema,
                "target_field": target_field,
            }

    elif marker == "PolymorphicRegistryKeyField":
        type_field = _string_value(_keyword(value, "type_field"))
        variants = _string_schema_dict(_keyword(value, "variants"))
        target_field = _string_value(_keyword(value, "target_field")) or "uid"
        if type_field and variants:
            metadata["polymorphic_registry_key"] = {
                "type_field": type_field,
                "target_field": target_field,
                "variants": variants,
            }

    elif marker == "OntologyAlignedField":
        ontology_name = _string_value(_keyword(value, "ontology_name"))
        if ontology_name:
            metadata["ontology_aligned"] = {"ontology_name": ontology_name}

    elif marker == "CrossReferenceField":
        database_name = _string_value(_keyword(value, "database_name"))
        if database_name:
            metadata["cross_reference"] = {"database_name": database_name}

    elif marker == "StableUIDField":
        metadata["stable_uid"] = True

    elif marker == "SummaryField":
        target_schema = _schema_name(_keyword(value, "target_schema"))
        target_field = _string_value(_keyword(value, "target_field"))
        op = _string_value(_keyword(value, "op"))
        if target_schema and target_field and op:
            metadata["summary"] = {
                "target_schema": target_schema,
                "target_field": target_field,
                "op": op,
            }

    parts = _call_parts(value, aliases)
    if parts and parts[-1] == "Field":
        extra = _literal_jsonish(_keyword(value, "json_schema_extra"))
        if isinstance(extra, dict):
            metadata.update(_marker_metadata_from_extra(extra))

    return metadata


def _import_aliases(tree: ast.Module) -> dict[str, str]:
    """Map locally-bound names back to their original imported names.

    Handles ``from homeobox.schema import DatasetSchema as HoxDatasetSchema``
    so a base written as ``HoxDatasetSchema`` resolves to ``DatasetSchema``.
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for name in node.names:
                if name.asname:
                    aliases[name.asname] = name.name
        elif isinstance(node, ast.Import):
            for name in node.names:
                if name.asname:
                    aliases[name.asname] = name.name.split(".")[-1]
    return aliases


def _base_names(node: ast.ClassDef, aliases: dict[str, str]) -> list[str]:
    names: list[str] = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            raw = base.id
        elif isinstance(base, ast.Attribute):
            raw = base.attr
        else:
            continue
        names.append(aliases.get(raw, raw))
    return names


def _resolve_kind(
    class_name: str,
    bases_by_class: dict[str, list[str]],
    _seen: frozenset[str] = frozenset(),
) -> str | None:
    """Resolve a class to its table kind, following local subclass chains.

    A class may subclass a homeobox base directly or via another class defined
    in the same file; the most specific recognised base wins.
    """
    if class_name in _seen:
        return None
    candidates: list[str] = []
    for base in bases_by_class.get(class_name, []):
        for known_base, kind in _BASE_KINDS:
            if base == known_base:
                candidates.append(kind)
        if base in bases_by_class:
            inherited = _resolve_kind(base, bases_by_class, _seen | {class_name})
            if inherited is not None:
                candidates.append(inherited)
    if not candidates:
        return None
    return min(candidates, key=lambda kind: _KIND_PRIORITY[kind])


def _own_fields(node: ast.ClassDef, aliases: dict[str, str]) -> list[dict]:
    """Annotated fields declared directly in a class body."""
    fields: list[dict] = []
    for statement in node.body:
        if not isinstance(statement, ast.AnnAssign):
            continue
        if not isinstance(statement.target, ast.Name):
            continue
        name = statement.target.id
        field = {"name": name, "type": ast.unparse(statement.annotation)}
        field.update(_field_call_metadata(statement.value, aliases))
        fields.append(field)
    return fields


def _merge_fields(fields: list[dict]) -> list[dict]:
    merged: list[dict] = []
    indexes: dict[str, int] = {}
    for field in fields:
        name = field["name"]
        if name in indexes:
            merged[indexes[name]] = field
        else:
            indexes[name] = len(merged)
            merged.append(field)
    return merged


def _fields_for_class(
    class_name: str,
    class_defs_by_name: dict[str, ast.ClassDef],
    bases_by_class: dict[str, list[str]],
    aliases: dict[str, str],
    _seen: frozenset[str] = frozenset(),
) -> list[dict]:
    if class_name in _seen:
        return []
    node = class_defs_by_name[class_name]
    fields: list[dict] = []
    for base_name in bases_by_class.get(class_name, []):
        if base_name in class_defs_by_name:
            inherited_fields = _fields_for_class(
                base_name,
                class_defs_by_name,
                bases_by_class,
                aliases,
                _seen | {class_name},
            )
            fields.extend({**field, "inherited": True} for field in inherited_fields)
        else:
            fields.extend(_homeobox_base_fields(base_name))
    fields.extend(_own_fields(node, aliases))
    return _merge_fields(fields)


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
    relationship list. Shared by the static and runtime entry points so they
    produce identical output structure.
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


def parse_schema_source(source: str) -> dict:
    """Statically parse schema *source* into the structured table/relationship result.

    Never imports or executes the source. Field types and marker metadata are
    recovered from the AST, which means constructs the parser cannot evaluate
    statically (e.g. ``variants=SOME_MODULE_CONSTANT``) are reported as absent.
    Use :func:`parse_schema_classes` / :func:`parse_schema_module` when the schema
    is importable and you want metadata read straight from the live classes.
    """
    tree = ast.parse(source)
    aliases = _import_aliases(tree)

    class_defs = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    class_defs_by_name = {node.name: node for node in class_defs}
    bases_by_class = {node.name: _base_names(node, aliases) for node in class_defs}

    parsed_tables: list[dict] = []
    for node in class_defs:
        kind = _resolve_kind(node.name, bases_by_class)
        if kind is None:
            continue
        parsed_tables.append(
            {
                "class_name": node.name,
                "kind": kind,
                "fields": _fields_for_class(
                    node.name,
                    class_defs_by_name,
                    bases_by_class,
                    aliases,
                ),
            }
        )

    return _assemble_parse_result(parsed_tables)


def parse_schema_file(schema_path: Path) -> dict:
    return parse_schema_source(schema_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Runtime parsing
# ---------------------------------------------------------------------------
#
# When the schema module is importable, introspecting the live pydantic classes
# is exact where static parsing is brittle: marker metadata is read straight off
# ``model_fields[...].json_schema_extra`` (already fully resolved -- polymorphic
# variants as schema names, combine_markers merged into one dict, etc.), and the
# table kind comes from real ``issubclass`` checks rather than base-name matching.
# The trade-off is that importing executes the module, so only use this on schemas
# you trust. The output structure is identical to ``parse_schema_source``.


@lru_cache
def _runtime_base_kinds() -> tuple[tuple[type, str], ...]:
    """Homeobox base classes paired with their table kind, most specific first.

    Mirrors ``_BASE_KINDS`` but with the real classes so the runtime path can use
    ``issubclass``. Lazily imported to avoid an import cycle with ``homeobox``.
    """
    from lancedb.pydantic import LanceModel

    from homeobox.schema import (
        DatasetSchema,
        FeatureBaseSchema,
        HoxBaseSchema,
        RegistryBaseSchema,
        StableUIDBaseSchema,
    )

    return (
        (HoxBaseSchema, "obs"),
        (DatasetSchema, "dataset"),
        (FeatureBaseSchema, "feature_registry"),
        (RegistryBaseSchema, "entity"),
        (StableUIDBaseSchema, "entity"),
        (LanceModel, "table"),
    )


def _runtime_kind(cls: type) -> str | None:
    """Resolve a live schema class to its table kind via ``issubclass``."""
    for base, kind in _runtime_base_kinds():
        if issubclass(cls, base):
            return kind
    return None


def _runtime_fields(cls: type) -> list[dict]:
    """Build field dicts from a live pydantic model's ``model_fields``.

    A field is marked ``inherited`` when it is not declared in *cls*'s own
    annotations (i.e. it comes from a base class), matching the static parser.
    ``inspect.get_annotations`` returns own-only annotations and is robust to the
    PEP 649 change (Python 3.14 no longer stores ``__annotations__`` on the class
    ``__dict__``).
    """
    own_annotations = inspect.get_annotations(cls)
    fields: list[dict] = []
    for name, field_info in cls.model_fields.items():
        field = {"name": name, "type": _annotation_to_string(field_info.annotation)}
        if name not in own_annotations:
            field["inherited"] = True
        extra = field_info.json_schema_extra
        if isinstance(extra, dict):
            field.update(_marker_metadata_from_extra(extra))
        fields.append(field)
    return fields


def parse_schema_classes(classes: list[type]) -> dict:
    """Parse an explicit list of live schema classes into the parser result shape.

    Each class is classified by ``issubclass`` and its fields and marker metadata
    are read from the live pydantic model. Classes that are not homeobox schema
    tables (no recognised base) are skipped. Output matches ``parse_schema_source``.
    """
    parsed_tables: list[dict] = []
    for cls in classes:
        kind = _runtime_kind(cls)
        if kind is None:
            continue
        parsed_tables.append(
            {
                "class_name": cls.__name__,
                "kind": kind,
                "fields": _runtime_fields(cls),
            }
        )

    return _assemble_parse_result(parsed_tables)


def parse_schema_module(module: Any) -> dict:
    """Parse every schema table defined in an imported *module*.

    Collects classes defined in the module itself (in definition order, skipping
    classes merely imported into it) and delegates to :func:`parse_schema_classes`.
    """
    classes = [
        obj
        for obj in vars(module).values()
        if isinstance(obj, type) and getattr(obj, "__module__", None) == module.__name__
    ]
    return parse_schema_classes(classes)
