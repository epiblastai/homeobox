"""Parse a homeobox ``schema.py`` back into a :class:`SchemaModel` (round-trip).

This is the inverse of :mod:`homeobox.schema_codegen`. It reads schema source
with :mod:`ast` -- it never imports or executes the user file -- classifying each
declared class by its homeobox base and extracting fields, marker metadata, enum
definitions, defaults, docstrings, declarative constraints, computed fields, and
presence flags.

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

# homeobox base classes mapped to the table "kind" used during classification, in
# priority order (most specific first) so a class is bucketed by its tightest
# recognised base -- FeatureBaseSchema is a RegistryBaseSchema is a
# StableUIDBaseSchema, etc. RegistryBaseSchema and StableUIDBaseSchema share the
# "entity" kind.
_BASE_KINDS: list[tuple[str, str]] = [
    ("HoxBaseSchema", "obs"),
    ("DatasetSchema", "dataset"),
    ("FeatureBaseSchema", "feature_registry"),
    ("RegistryBaseSchema", "entity"),
    ("StableUIDBaseSchema", "entity"),
    ("LanceModel", "table"),
]
_KIND_PRIORITY = {kind: index for index, (_, kind) in enumerate(_BASE_KINDS)}

# Parser table "kind" -> IR base kind (classification collapses RegistryBaseSchema
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
# AST extraction helpers
# ---------------------------------------------------------------------------


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

    Used when ``extra`` is reconstructed from a literal ``Field(json_schema_extra=...)``
    call. The two markers that combine onto one field via ``combine_markers`` land
    in the same dict, so every key is read independently and none are mutually
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


# ---------------------------------------------------------------------------
# Default / validator helpers
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

    keyword = _keyword(value, "default")
    parts = _call_parts(value, {}) or []

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
        markers = _ir_markers(_field_call_metadata(stmt.value, aliases))
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
    aliases = _import_aliases(tree)

    class_defs = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    bases_by_class = {n.name: _base_names(n, aliases) for n in class_defs}

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
        kind = _resolve_kind(node.name, bases_by_class)
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
