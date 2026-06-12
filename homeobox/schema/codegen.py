"""Generate a homeobox ``schema.py`` source string from a :class:`SchemaModel`.

Emission is deterministic and ordered (imports -> enums -> tables in dependency
order -> derived ``REGISTRY_SCHEMAS``) so the same IR always produces the same
source. The output is run through :func:`ast.parse` before it is returned, so a
successful return guarantees syntactically valid Python.

Schema cross-references (``target_schema``, ``feature_registry_schema``,
polymorphic ``variants``) are emitted as bare class names when the target is
already defined earlier in the file, and as string literals otherwise -- which
reproduces the forward-reference idiom of a hand-written schema (e.g. a dataset
table summarising the obs table that is defined below it).
"""

import ast

from homeobox.schema.ir import (
    MARKER_FACTORIES,
    POINTER_TYPE_NAMES,
    REQUIRED,
    ConstraintDef,
    EnumDef,
    FieldDef,
    SchemaModel,
    TableDef,
)

INDENT = "    "


# ---------------------------------------------------------------------------
# Literals and references
# ---------------------------------------------------------------------------


def _py_literal(value: object) -> str:
    if value is REQUIRED:
        return "..."
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, int | float):
        return repr(value)
    raise TypeError(f"cannot render default literal of type {type(value).__name__}: {value!r}")


def _schema_ref(name: str, defined: set[str]) -> str:
    """Bare class name if already defined above, else a forward-reference string."""
    return name if name in defined else repr(name)


def _annotation_names(type_str: str) -> set[str]:
    tree = ast.parse(type_str, mode="eval")
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


def _declare_kwargs(name: str, payload: object, defined: set[str]) -> list[str]:
    """Render a marker's ``declare(...)`` keyword arguments (no ``default``)."""
    if name == "stable_uid":
        return []
    if name == "pointer":
        kwargs = [f"feature_space={_py_literal(payload['feature_space'])}"]
        registry = payload.get("feature_registry_schema")
        if registry:
            kwargs.append(f"feature_registry_schema={_schema_ref(registry, defined)}")
        return kwargs
    if name == "registry_key":
        kwargs = [f"target_schema={_schema_ref(payload['target_schema'], defined)}"]
        if payload.get("target_field", "uid") != "uid":
            kwargs.append(f"target_field={_py_literal(payload['target_field'])}")
        return kwargs
    if name == "polymorphic_registry_key":
        variants = ", ".join(
            f"{_py_literal(key)}: {_schema_ref(value, defined)}"
            for key, value in payload["variants"].items()
        )
        kwargs = [f"type_field={_py_literal(payload['type_field'])}", f"variants={{{variants}}}"]
        if payload.get("target_field", "uid") != "uid":
            kwargs.append(f"target_field={_py_literal(payload['target_field'])}")
        return kwargs
    if name == "ontology_aligned":
        return [f"ontology_name={_py_literal(payload['ontology_name'])}"]
    if name == "cross_reference":
        return [f"database_name={_py_literal(payload['database_name'])}"]
    if name == "summary":
        return [
            f"target_schema={_schema_ref(payload['target_schema'], defined)}",
            f"target_field={_py_literal(payload['target_field'])}",
            f"op={_py_literal(payload['op'])}",
        ]
    raise ValueError(f"unknown marker {name!r}")  # pragma: no cover


def _ordered_markers(markers: dict) -> list[tuple[str, object]]:
    """Markers in a canonical order so emission is stable regardless of authoring."""
    return [(name, markers[name]) for name in MARKER_FACTORIES if name in markers]


def _render_field_value(field: FieldDef, defined: set[str]) -> str:
    """Render the right-hand side of a marked field (``= ...`` part)."""
    default_literal = _py_literal(field.default)
    ordered = _ordered_markers(field.markers)
    if len(ordered) == 1:
        name, payload = ordered[0]
        kwargs = _declare_kwargs(name, payload, defined)
        kwargs.append(f"default={default_literal}")
        return f"{MARKER_FACTORIES[name]}.declare({', '.join(kwargs)})"

    inner = []
    for name, payload in ordered:
        kwargs = _declare_kwargs(name, payload, defined)
        inner.append(f"{MARKER_FACTORIES[name]}.declare({', '.join(kwargs)})")
    inner.append(f"default={default_literal}")
    return f"combine_markers({', '.join(inner)})"


# ---------------------------------------------------------------------------
# Field and validator emission
# ---------------------------------------------------------------------------


def _emit_field(field: FieldDef, defined: set[str]) -> list[str]:
    lines: list[str] = []
    if field.doc is not None:
        lines.append(f"{INDENT}# {field.doc}")
    if field.markers:
        rhs = _render_field_value(field, defined)
        lines.append(f"{INDENT}{field.name}: {field.type} = {rhs}")
    elif field.required:
        lines.append(f"{INDENT}{field.name}: {field.type}")
    else:
        lines.append(f"{INDENT}{field.name}: {field.type} = {_py_literal(field.default)}")
    return lines


def _emit_require_any(constraint: ConstraintDef, index: int, table: TableDef) -> list[str]:
    values = ", ".join(f"self.{name}" for name in constraint.fields)
    names = ", ".join(constraint.fields)
    return [
        f'{INDENT}@model_validator(mode="after")',
        f"{INDENT}def _require_any_{index}(self) -> Self:",
        f"{INDENT}{INDENT}if not any([{values}]):",
        f"{INDENT}{INDENT}{INDENT}raise ValueError(",
        f'{INDENT}{INDENT}{INDENT}{INDENT}"{table.name} requires at least one of: {names}"',
        f"{INDENT}{INDENT}{INDENT})",
        f"{INDENT}{INDENT}return self",
    ]


def _emit_equal_length(constraint: ConstraintDef, index: int, table: TableDef) -> list[str]:
    values = ", ".join(f"self.{name}" for name in constraint.fields)
    names = ", ".join(constraint.fields)
    return [
        f'{INDENT}@model_validator(mode="after")',
        f"{INDENT}def _equal_length_{index}(self) -> Self:",
        f"{INDENT}{INDENT}_values = [{values}]",
        f"{INDENT}{INDENT}_present = [v for v in _values if v is not None]",
        f"{INDENT}{INDENT}if _present and len({{len(v) for v in _present}}) > 1:",
        f'{INDENT}{INDENT}{INDENT}raise ValueError("Fields {names} must all have the same length")',
        f"{INDENT}{INDENT}return self",
    ]


def _emit_computed_validator(field: FieldDef) -> list[str]:
    computed = field.computed
    assert computed is not None and computed.op == "join_list"
    sep = _py_literal(computed.args["separator"])
    source = computed.args["source"]
    return [
        f'{INDENT}@model_validator(mode="after")',
        f"{INDENT}def _compute_{field.name}(self) -> Self:",
        f"{INDENT}{INDENT}self.{field.name} = {sep}.join(self.{source} or [])",
        f"{INDENT}{INDENT}return self",
    ]


def _emit_presence(table: TableDef) -> list[str]:
    return [
        f"{INDENT}@classmethod",
        f"{INDENT}def has_pointer_field_map(cls) -> dict[str, str]:",
        f'{INDENT}{INDENT}return {{f"has_{{name}}": name for name, _ in _iter_pointer_annotations(cls)}}',
        "",
        f'{INDENT}@model_validator(mode="after")',
        f"{INDENT}def _generate_has_pointer_flags(self) -> Self:",
        f"{INDENT}{INDENT}for flag, source in type(self).has_pointer_field_map().items():",
        f"{INDENT}{INDENT}{INDENT}setattr(self, flag, getattr(self, source) is not None)",
        f"{INDENT}{INDENT}return self",
    ]


def _emit_compute_auto_fields(computed_fields: list[FieldDef]) -> list[str]:
    lines = [
        f"{INDENT}@classmethod",
        f'{INDENT}def compute_auto_fields(cls, obs_df: "pd.DataFrame") -> "pd.DataFrame":',
        f"{INDENT}{INDENT}import json",
        "",
        f"{INDENT}{INDENT}import pandas as pd",
        "",
    ]
    for field in computed_fields:
        computed = field.computed
        assert computed is not None and computed.op == "join_list"
        sep = _py_literal(computed.args["separator"])
        source = computed.args["source"]
        lines.extend(
            [
                f"{INDENT}{INDENT}def _join_{field.name}(value):",
                f"{INDENT}{INDENT}{INDENT}if value is None or (isinstance(value, float) and pd.isna(value)):",
                f'{INDENT}{INDENT}{INDENT}{INDENT}return ""',
                f"{INDENT}{INDENT}{INDENT}items = json.loads(value) if isinstance(value, str) else list(value)",
                f"{INDENT}{INDENT}{INDENT}return {sep}.join(items)",
                "",
                f'{INDENT}{INDENT}if "{source}" in obs_df.columns:',
                f'{INDENT}{INDENT}{INDENT}obs_df["{field.name}"] = obs_df["{source}"].map(_join_{field.name})',
                f"{INDENT}{INDENT}else:",
                f'{INDENT}{INDENT}{INDENT}obs_df["{field.name}"] = ""',
                "",
            ]
        )
    lines.append(f"{INDENT}{INDENT}return obs_df")
    return lines


def _emit_table(table: TableDef, defined: set[str]) -> list[str]:
    lines = [f"class {table.name}({table.base_class}):"]
    if table.doc is not None:
        lines.extend(_emit_docstring(table.doc, INDENT))

    body: list[str] = []
    for field in table.fields:
        body.extend(_emit_field(field, defined))

    if table.presence_flags:
        pointer_fields = [f for f in table.fields if "pointer" in f.markers]
        if len(pointer_fields) > 1:
            body.append("")
            for field in pointer_fields:
                body.append(f"{INDENT}has_{field.name}: bool = False")

    for index, constraint in enumerate(table.constraints):
        body.append("")
        if constraint.kind == "require_any":
            body.extend(_emit_require_any(constraint, index, table))
        else:
            body.extend(_emit_equal_length(constraint, index, table))

    for field in table.fields:
        if field.computed is not None:
            body.append("")
            body.extend(_emit_computed_validator(field))

    if table.presence_flags and len([f for f in table.fields if "pointer" in f.markers]) > 1:
        body.append("")
        body.extend(_emit_presence(table))

    computed_fields = [f for f in table.fields if f.computed is not None]
    if computed_fields:
        body.append("")
        body.extend(_emit_compute_auto_fields(computed_fields))

    lines.extend(body)
    return lines


def _emit_docstring(doc: str, indent: str) -> list[str]:
    text = doc.rstrip("\n")
    if "\n" in text:
        lines = [f'{indent}"""']
        lines.extend(f"{indent}{line}".rstrip() for line in text.split("\n"))
        lines.append(f'{indent}"""')
        return lines
    return [f'{indent}"""{text}"""']


def _emit_enum(enum: EnumDef) -> list[str]:
    lines = [f"class {enum.name}(StrEnum):"]
    if enum.doc is not None:
        lines.extend(_emit_docstring(enum.doc, INDENT))
        lines.append("")
    for member, value in enum.values.items():
        lines.append(f"{INDENT}{member} = {_py_literal(value)}")
    return lines


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


def _collect_imports(model: SchemaModel) -> list[str]:
    tables = model.emit_order()
    all_fields = [f for t in tables for f in t.fields]

    annotation_names: set[str] = set()
    for field in all_fields:
        annotation_names |= _annotation_names(field.type)

    pointer_types = sorted(n for n in annotation_names if n in POINTER_TYPE_NAMES)
    needs_datetime = "datetime" in annotation_names

    markers_used: set[str] = set()
    has_combined = False
    for field in all_fields:
        markers_used |= set(field.markers)
        if len(field.markers) > 1:
            has_combined = True

    base_classes = {t.base_class for t in tables}
    needs_iter_pointer = any(t.presence_flags for t in model.obs_tables)
    has_validators = any(
        t.constraints or any(f.computed for f in t.fields) or t.presence_flags for t in tables
    )
    has_compute_auto = any(any(f.computed for f in t.fields) for t in tables)

    stdlib: list[str] = []
    if needs_datetime:
        stdlib.append("from datetime import datetime")
    if model.enums:
        stdlib.append("from enum import StrEnum")
    typing_names = ["Self"] if has_validators else []
    if has_compute_auto:
        typing_names = ["TYPE_CHECKING", *typing_names]
    if typing_names:
        stdlib.append(f"from typing import {', '.join(typing_names)}")

    type_checking: list[str] = []
    if has_compute_auto:
        type_checking = ["", "if TYPE_CHECKING:", f"{INDENT}import pandas as pd"]

    third_party: list[str] = []
    if "LanceModel" in base_classes:
        third_party.append("from lancedb.pydantic import LanceModel")
    if has_validators:
        third_party.append("from pydantic import model_validator")

    homeobox_lines: list[str] = []
    if pointer_types:
        homeobox_lines.append(f"from homeobox.pointer_types import {', '.join(pointer_types)}")

    schema_names: set[str] = {cls for cls in base_classes if cls != "LanceModel"}
    schema_names |= {MARKER_FACTORIES[m] for m in markers_used}
    if has_combined:
        schema_names.add("combine_markers")
    if needs_iter_pointer:
        schema_names.add("_iter_pointer_annotations")
    if schema_names:
        names = ", ".join(sorted(schema_names))
        homeobox_lines.append(f"from homeobox.schema import {names}")

    blocks = [b for b in (stdlib, third_party, homeobox_lines) if b]
    out: list[str] = []
    for i, block in enumerate(blocks):
        if i > 0:
            out.append("")
        out.extend(block)
        if block is stdlib and type_checking:
            out.extend(type_checking)
    return out


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


def emit(model: SchemaModel) -> str:
    """Render *model* to a homeobox ``schema.py`` source string."""
    parts: list[str] = []
    if model.doc is not None:
        parts.extend(_emit_docstring(model.doc, ""))
        parts.append("")

    parts.extend(_collect_imports(model))

    for enum in model.enums:
        parts.append("")
        parts.append("")
        parts.extend(_emit_enum(enum))

    defined: set[str] = set()
    for table in model.emit_order():
        parts.append("")
        parts.append("")
        parts.extend(_emit_table(table, defined))
        defined.add(table.name)

    registry_schemas = model.registry_schemas()
    if registry_schemas:
        parts.append("")
        parts.append("")
        parts.append("REGISTRY_SCHEMAS: dict[str, type[FeatureBaseSchema]] = {")
        for feature_space, schema_name in registry_schemas.items():
            parts.append(f"{INDENT}{_py_literal(feature_space)}: {schema_name},")
        parts.append("}")

    source = "\n".join(parts).rstrip("\n") + "\n"
    ast.parse(source)  # guarantee syntactic validity
    return source


def emit_to_file(model: SchemaModel, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(emit(model))
