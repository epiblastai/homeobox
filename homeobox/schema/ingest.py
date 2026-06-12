"""Parse a homeobox ``schema.py`` back into a :class:`SchemaModel` (round-trip).

This is the inverse of :mod:`homeobox.schema.codegen`. It **imports** the schema
module and introspects the live pydantic classes: table kind comes from real
``issubclass`` checks, field types / defaults from each ``FieldInfo``, marker
metadata straight off ``model_fields[...].json_schema_extra`` (already fully
resolved -- polymorphic variants as schema names, ``combine_markers`` merged into
one dict), and enums from the live ``StrEnum`` members. The trade-off is that
importing executes the module, so only ingest schemas you trust.

Constraints, computed fields, and presence flags have no declarative runtime
form -- they are emitted as ``@model_validator`` methods -- so those bodies are
recovered with :func:`inspect.getsource` and classified. Every model validator a
schema declares must reduce to one of:

- a **dead enum re-check** (``value not in SomeEnum.__members__.values()``) -- the
  field is already typed as the enum, so this is dropped;
- a **require_any** / **equal_length** constraint;
- a **join_list** computed field; or
- the **presence-flag** generator.

Anything else is a hard error. The IR is a schema description, not a place to
smuggle hand-written validation logic, so a bespoke validator must be simplified
or removed before its schema can be represented.
"""

import ast
import dataclasses
import importlib.util
import inspect
import os
import sys
import tempfile
import textwrap
import types
import uuid
from enum import StrEnum
from typing import Any, Union, get_args, get_origin

from homeobox.schema.ir import (
    REQUIRED,
    ComputedDef,
    ConstraintDef,
    EnumDef,
    FieldDef,
    SchemaModel,
    TableDef,
    dump_yaml,
)

# Auto / base-class fields that a generated schema never declares; if a source
# file redeclares one (e.g. ``uid: str = Field(default_factory=make_uid)``) it is
# dropped on ingest because the base class already provides it.
_SKIP_FIELDS: frozenset[str] = frozenset({"uid", "dataset_uid", "global_index"})

# Non-validator methods the IR regenerates; allowed in a schema class body.
_REGENERATED_METHODS: frozenset[str] = frozenset({"compute_auto_fields", "has_pointer_field_map"})


def _runtime_base_kinds() -> tuple[tuple[type, str], ...]:
    """Homeobox base classes paired with their IR base kind, most specific first.

    Lazily imported to avoid any import-time coupling with ``homeobox.schema``.
    """
    from lancedb.pydantic import LanceModel

    from homeobox.schema.definitions import (
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
        (RegistryBaseSchema, "registry"),
        (StableUIDBaseSchema, "registry"),
        (LanceModel, "table"),
    )


# ---------------------------------------------------------------------------
# Annotation rendering
# ---------------------------------------------------------------------------


def _annotation_to_string(annotation: Any) -> str:
    """Render a runtime annotation the way the IR writes field types.

    Unions come out as ``A | B`` and custom classes as their bare name, so a
    live ``SparseZarrPointer | None`` / ``list[str] | None`` field reads exactly
    like its IR-declared ``type``.
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


# ---------------------------------------------------------------------------
# Marker metadata
# ---------------------------------------------------------------------------


def _marker_metadata_from_extra(extra: dict) -> dict:
    """Translate a field's ``json_schema_extra`` into intermediate marker metadata.

    The two markers that combine onto one field via ``combine_markers`` land in
    the same dict, so every key is read independently and none are mutually
    exclusive.
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


def _ir_markers(metadata: dict) -> dict:
    """Convert marker metadata into IR marker payloads."""
    markers: dict[str, object] = {}
    for key in ("pointer", "ontology_aligned", "cross_reference", "summary"):
        if key in metadata:
            markers[key] = dict(metadata[key])
    if metadata.get("stable_uid"):
        markers["stable_uid"] = True
    if "registry_key" in metadata:
        payload = dict(metadata["registry_key"])
        if payload.get("target_field") == "uid":
            payload.pop("target_field", None)
        markers["registry_key"] = payload
    if "polymorphic_registry_key" in metadata:
        payload = dict(metadata["polymorphic_registry_key"])
        if payload.get("target_field") == "uid":
            payload.pop("target_field", None)
        markers["polymorphic_registry_key"] = payload
    return markers


# ---------------------------------------------------------------------------
# Validator classification
# ---------------------------------------------------------------------------


def _self_attr(node: ast.AST) -> str | None:
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return node.attr
    return None


def _classify_validator(func: ast.FunctionDef, class_name: str) -> tuple[str, object]:
    """Return ``(kind, payload)`` for a recognised validator, else hard-error.

    ``kind`` is one of ``enum_recheck`` (drop), ``require_any``, ``equal_length``,
    ``join_list``, ``presence``.
    """
    nodes = list(ast.walk(func))

    # Dead enum re-check: `value not in SomeEnum.__members__.values()`.
    if any(isinstance(n, ast.Attribute) and n.attr == "__members__" for n in nodes):
        return ("enum_recheck", None)

    # Presence flags: references `has_pointer_field_map`.
    if any(
        (isinstance(n, ast.Attribute) and n.attr == "has_pointer_field_map")
        or (isinstance(n, ast.Name) and n.id == "has_pointer_field_map")
        for n in nodes
    ):
        return ("presence", None)

    # require_any: `if not any([self.a, self.b, ...]): raise ...`.
    for n in nodes:
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "any":
            if n.args and isinstance(n.args[0], ast.List | ast.Tuple):
                attrs = [a for a in (_self_attr(e) for e in n.args[0].elts) if a]
                if len(attrs) >= 2:
                    return ("require_any", tuple(attrs))

    # join_list computed: `self.X = "<sep>".join(self.Y or [])`.
    for n in nodes:
        if not isinstance(n, ast.Assign) or len(n.targets) != 1:
            continue
        target = _self_attr(n.targets[0])
        value = n.value
        if (
            target
            and isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "join"
            and isinstance(value.func.value, ast.Constant)
            and isinstance(value.func.value.value, str)
            and value.args
        ):
            source = next((_self_attr(a) for a in ast.walk(value.args[0]) if _self_attr(a)), None)
            if source:
                return (
                    "join_list",
                    {"field": target, "source": source, "separator": value.func.value.value},
                )

    # equal_length: builds a list of >=2 self attributes and compares lengths.
    has_len = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "len"
        for n in nodes
    )
    if has_len:
        for n in nodes:
            if isinstance(n, ast.List):
                attrs = [a for a in (_self_attr(e) for e in n.elts) if a]
                if len(attrs) >= 2:
                    return ("equal_length", tuple(attrs))

    raise ValueError(
        f"{class_name}.{func.name} is not a recognised validator shape "
        "(enum re-check, require_any, equal_length, join_list computed, or presence flags). "
        "Hand-written validators are not supported by the schema IR; simplify or remove it."
    )


def _own_model_validators(cls: type) -> dict[str, ast.FunctionDef]:
    """Return ``{name: FunctionDef}`` for model validators declared on *cls* itself.

    Inherited base-class validators (e.g. ``_require_at_least_one_pointer``) are
    skipped via a qualname check. Each validator's body is recovered from the
    live function with :func:`inspect.getsource` and parsed.
    """
    own: dict[str, ast.FunctionDef] = {}
    for name, decorator in cls.__pydantic_decorators__.model_validators.items():
        func = decorator.func
        if func.__qualname__.rsplit(".", 1)[0] != cls.__qualname__:
            continue
        own[name] = ast.parse(textwrap.dedent(inspect.getsource(func))).body[0]
    return own


def _check_no_unexpected_methods(cls: type, validator_names: set[str]) -> None:
    """Hard-error on any method the IR cannot regenerate."""
    for name, member in cls.__dict__.items():
        if name.startswith("__"):
            continue
        if not (isinstance(member, classmethod | staticmethod) or inspect.isfunction(member)):
            continue
        if name in validator_names or name in _REGENERATED_METHODS:
            continue
        raise ValueError(
            f"{cls.__name__}.{name}: unexpected method; the IR only models fields, "
            "markers, constraints, computed fields and presence flags."
        )


# ---------------------------------------------------------------------------
# Docstrings / defaults
# ---------------------------------------------------------------------------


def _own_doc(obj: type) -> str | None:
    """Return *obj*'s own (non-inherited) docstring, cleaned like the source."""
    doc = obj.__dict__.get("__doc__")
    return inspect.cleandoc(doc) if doc else None


def _field_default(cls: type, field_name: str, field_info: Any) -> object:
    if field_info.is_required():
        return REQUIRED
    if field_info.default_factory is not None:
        raise ValueError(
            f"{cls.__name__}.{field_name}: default_factory is not a literal; "
            "IR field defaults must be literals"
        )
    return field_info.default


# ---------------------------------------------------------------------------
# Class extraction
# ---------------------------------------------------------------------------


def _extract_enum(cls: type) -> EnumDef:
    values = {member.name: member.value for member in cls}
    if not values:
        raise ValueError(f"enum {cls.__name__} has no members")
    return EnumDef(name=cls.__name__, values=values, doc=_own_doc(cls))


def _extract_table(cls: type, ir_base: str) -> TableDef:
    own_field_names = [
        name
        for name in inspect.get_annotations(cls)
        if name in cls.model_fields and name not in _SKIP_FIELDS
    ]

    raw_fields: dict[str, FieldDef] = {}
    for name in own_field_names:
        field_info = cls.model_fields[name]
        extra = field_info.json_schema_extra
        markers = _ir_markers(_marker_metadata_from_extra(extra if isinstance(extra, dict) else {}))
        raw_fields[name] = FieldDef(
            name=name,
            type=_annotation_to_string(field_info.annotation),
            default=_field_default(cls, name, field_info),
            markers=markers,
        )

    constraints: list[ConstraintDef] = []
    presence = False
    computed_specs: dict[str, ComputedDef] = {}

    validators = _own_model_validators(cls)
    for node in validators.values():
        verdict, payload = _classify_validator(node, cls.__name__)
        if verdict == "enum_recheck":
            continue
        if verdict == "presence":
            presence = True
        elif verdict == "require_any":
            constraints.append(ConstraintDef("require_any", payload))
        elif verdict == "equal_length":
            constraints.append(ConstraintDef("equal_length", payload))
        elif verdict == "join_list":
            computed_specs[payload["field"]] = ComputedDef(
                "join_list", {"source": payload["source"], "separator": payload["separator"]}
            )

    _check_no_unexpected_methods(cls, set(validators))

    # Drop generated presence-flag fields and attach computed specs.
    pointer_names = {n for n, f in raw_fields.items() if "pointer" in f.markers}
    drop = {f"has_{n}" for n in pointer_names} if presence else set()

    fields: list[FieldDef] = []
    for name, field in raw_fields.items():
        if name in drop:
            continue
        if name in computed_specs:
            field = dataclasses.replace(field, computed=computed_specs[name])
        fields.append(field)

    return TableDef(
        name=cls.__name__,
        base=ir_base,
        fields=tuple(fields),
        doc=_own_doc(cls),
        constraints=tuple(constraints),
        presence_flags=presence,
    )


def _table_ir_base(cls: type) -> str | None:
    for base, ir_base in _runtime_base_kinds():
        if issubclass(cls, base):
            return ir_base
    return None


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def model_from_module(module: types.ModuleType, name: str | None = None) -> SchemaModel:
    """Introspect every schema class defined in an imported *module*."""
    enums: list[EnumDef] = []
    sections: dict[str, list[TableDef]] = {
        "obs": [],
        "feature_registry": [],
        "registry": [],
        "table": [],
    }
    dataset_table: TableDef | None = None

    for obj in vars(module).values():
        if not isinstance(obj, type) or getattr(obj, "__module__", None) != module.__name__:
            continue
        if issubclass(obj, StrEnum) and obj is not StrEnum:
            enums.append(_extract_enum(obj))
            continue
        ir_base = _table_ir_base(obj)
        if ir_base is None:
            continue
        table = _extract_table(obj, ir_base)
        if ir_base == "dataset":
            dataset_table = table
        else:
            sections[ir_base].append(table)

    module_doc = inspect.cleandoc(module.__doc__) if module.__doc__ else None
    return SchemaModel(
        name=name or "schema",
        doc=module_doc,
        enums=tuple(enums),
        obs_tables=tuple(sections["obs"]),
        dataset_table=dataset_table,
        feature_registry_tables=tuple(sections["feature_registry"]),
        fk_registry_tables=tuple(sections["registry"]),
        other_tables=tuple(sections["table"]),
    )


def _ingest_path(path: str, name: str | None) -> SchemaModel:
    """Import the module at *path* under a throwaway name and introspect it."""
    module_name = "_hox_ingest_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        return model_from_module(module, name=name)
    finally:
        sys.modules.pop(module_name, None)


def model_from_file(path: str, name: str | None = None) -> SchemaModel:
    """Import the schema file at *path* and introspect it into a :class:`SchemaModel`."""
    return _ingest_path(path, name)


def model_from_source(source: str, name: str | None = None) -> SchemaModel:
    """Import schema *source* (written to a temp module) and introspect it."""
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "schema_module.py")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(source)
        return _ingest_path(path, name)


def yaml_from_source(source: str, name: str | None = None) -> str:
    return dump_yaml(model_from_source(source, name=name))
