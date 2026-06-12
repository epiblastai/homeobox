"""Parse a homeobox ``schema.py`` back into a :class:`SchemaModel` (round-trip).

This is the inverse of :mod:`homeobox.schema_codegen`. It reuses the static AST
machinery in :mod:`homeobox.parser` for table classification and marker
extraction, and adds what the parser does not capture: enum definitions, field
defaults, docstrings, declarative constraints, computed fields, and presence
flags.

Recognition is deliberately strict. Every ``@model_validator`` in a schema
subclass must reduce to one of:

- a **dead enum re-check** (``value not in SomeEnum.__members__.values()``) -- the
  field is already typed as the enum, so this is dropped;
- a **require_any** / **equal_length** constraint;
- a **join_list** computed field; or
- the **presence-flag** generator.

Anything else is a hard error. The IR is a schema description, not a place to
smuggle hand-written validation logic, so a bespoke validator must be simplified
or removed before its schema can be represented. Inline ``# comments`` on fields
are not recovered (they are not part of the AST).
"""

import ast

from homeobox import parser
from homeobox.schema_ir import (
    REQUIRED,
    ComputedDef,
    ConstraintDef,
    EnumDef,
    FieldDef,
    SchemaModel,
    TableDef,
    dump_yaml,
)

# Parser table "kind" -> IR base kind (the parser collapses RegistryBaseSchema
# and StableUIDBaseSchema to "entity"; the IR calls that bucket "registry").
KIND_TO_IR: dict[str, str] = {
    "obs": "obs",
    "dataset": "dataset",
    "feature_registry": "feature_registry",
    "entity": "registry",
    "table": "table",
}

# Auto / base-class fields that a generated schema never declares; if a source
# file redeclares one (e.g. ``uid: str = Field(default_factory=make_uid)``) it is
# dropped on ingest because the base class already provides it.
_SKIP_FIELDS: frozenset[str] = frozenset({"uid", "dataset_uid", "global_index"})

# Marker name -> factory default when ``declare()`` is called without ``default``.
_MARKER_FACTORY_DEFAULT: dict[str, object] = {
    "pointer": None,
    "stable_uid": None,
    "registry_key": REQUIRED,
    "polymorphic_registry_key": REQUIRED,
    "ontology_aligned": REQUIRED,
    "cross_reference": REQUIRED,
    "summary": REQUIRED,
}

# Non-validator methods the IR regenerates; allowed in a source class body.
_REGENERATED_METHODS: frozenset[str] = frozenset({"compute_auto_fields", "has_pointer_field_map"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _literal_default(node: ast.AST | None) -> object:
    if node is None:
        return REQUIRED
    if isinstance(node, ast.Constant) and node.value is Ellipsis:
        return REQUIRED
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        raise ValueError(
            f"non-literal default {ast.unparse(node)!r}; IR field defaults must be literals"
        ) from None


def _self_attr(node: ast.AST) -> str | None:
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return node.attr
    return None


def _ir_markers(metadata: dict) -> dict:
    """Convert parser marker metadata into IR marker payloads."""
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


def _extract_default(value: ast.AST | None, markers: dict) -> object:
    """Effective default for a field, given its assignment expression and markers."""
    if not isinstance(value, ast.Call):
        return _literal_default(value)

    keyword = parser._keyword(value, "default")
    parts = parser._call_parts(value, {}) or []

    if parts == ["combine_markers"]:
        return _literal_default(keyword) if keyword is not None else REQUIRED

    if keyword is not None:
        return _literal_default(keyword)

    # A single ``Marker.declare(...)`` with no explicit default: use its factory
    # default (None for pointer/stable_uid, required otherwise).
    if len(markers) == 1:
        (name,) = markers
        return _MARKER_FACTORY_DEFAULT.get(name, REQUIRED)
    return REQUIRED


# ---------------------------------------------------------------------------
# Validator classification
# ---------------------------------------------------------------------------


def _is_model_validator(func: ast.FunctionDef) -> bool:
    return any("model_validator" in ast.unparse(dec) for dec in func.decorator_list)


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


# ---------------------------------------------------------------------------
# Class extraction
# ---------------------------------------------------------------------------


def _extract_enum(node: ast.ClassDef) -> EnumDef:
    values: dict[str, str] = {}
    for stmt in node.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            values[stmt.targets[0].id] = stmt.value.value
    if not values:
        raise ValueError(f"enum {node.name} has no string members")
    return EnumDef(name=node.name, values=values, doc=ast.get_docstring(node))


def _extract_table(node: ast.ClassDef, kind: str, aliases: dict) -> TableDef:
    ir_kind = KIND_TO_IR[kind]

    raw_fields: dict[str, FieldDef] = {}
    for stmt in node.body:
        if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
            continue
        name = stmt.target.id
        if name in _SKIP_FIELDS:
            continue
        markers = _ir_markers(parser._field_call_metadata(stmt.value, aliases))
        default = _extract_default(stmt.value, markers)
        raw_fields[name] = FieldDef(
            name=name,
            type=ast.unparse(stmt.annotation),
            default=default,
            markers=markers,
        )

    constraints: list[ConstraintDef] = []
    presence = False
    computed_specs: dict[str, ComputedDef] = {}

    for stmt in node.body:
        if not isinstance(stmt, ast.FunctionDef):
            continue
        if not _is_model_validator(stmt):
            if stmt.name not in _REGENERATED_METHODS:
                raise ValueError(
                    f"{node.name}.{stmt.name}: unexpected method; the IR only models fields, "
                    "markers, constraints, computed fields and presence flags."
                )
            continue
        verdict, payload = _classify_validator(stmt, node.name)
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

    # Drop generated presence-flag fields and attach computed specs.
    pointer_names = {n for n, f in raw_fields.items() if "pointer" in f.markers}
    drop = {f"has_{n}" for n in pointer_names} if presence else set()

    fields: list[FieldDef] = []
    for name, field in raw_fields.items():
        if name in drop:
            continue
        if name in computed_specs:
            field = FieldDef(
                name=field.name,
                type=field.type,
                default=field.default,
                doc=field.doc,
                markers=field.markers,
                computed=computed_specs[name],
            )
        fields.append(field)

    return TableDef(
        name=node.name,
        base=ir_kind,
        fields=tuple(fields),
        doc=ast.get_docstring(node),
        constraints=tuple(constraints),
        presence_flags=presence,
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def model_from_source(source: str, name: str | None = None) -> SchemaModel:
    """Parse schema *source* into a :class:`SchemaModel`."""
    tree = ast.parse(source)
    aliases = parser._import_aliases(tree)

    class_defs = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    bases_by_class = {n.name: parser._base_names(n, aliases) for n in class_defs}

    enums: list[EnumDef] = []
    sections: dict[str, list[TableDef]] = {
        "obs": [],
        "feature_registry": [],
        "registry": [],
        "table": [],
    }
    dataset_table: TableDef | None = None

    for node in class_defs:
        if "StrEnum" in bases_by_class.get(node.name, []):
            enums.append(_extract_enum(node))
            continue
        kind = parser._resolve_kind(node.name, bases_by_class)
        if kind is None:
            continue
        table = _extract_table(node, kind, aliases)
        if table.base == "dataset":
            dataset_table = table
        else:
            sections[table.base].append(table)

    module_doc = ast.get_docstring(tree)
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


def model_from_file(path: str, name: str | None = None) -> SchemaModel:
    with open(path, encoding="utf-8") as handle:
        return model_from_source(handle.read(), name=name)


def yaml_from_source(source: str, name: str | None = None) -> str:
    return dump_yaml(model_from_source(source, name=name))
