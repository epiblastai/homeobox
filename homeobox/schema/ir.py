"""In-memory intermediate representation (IR) for homeobox database schemas.

This module defines :class:`SchemaModel` -- a small tree of frozen dataclasses
that describes a homeobox schema declaratively -- plus a strict YAML loader and
dumper. The IR is the single source of truth shared by:

- :mod:`homeobox.schema.codegen` (IR -> ``schema.py`` source), and
- :mod:`homeobox.schema.ingest` (``schema.py`` -> IR).

The YAML surface is intentionally small (see ``specs/schema_yaml_ir.md``). The
loader hard-errors on anything it does not recognise: unknown keys, unknown
markers, malformed payloads. There is no escape hatch for arbitrary Python --
the IR is a schema description, not a programming language.
"""

import dataclasses
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Table section name -> (IR base kind, generated base class name).
SECTION_BASES: dict[str, tuple[str, str]] = {
    "obs_tables": ("obs", "HoxBaseSchema"),
    "dataset_table": ("dataset", "DatasetSchema"),
    "feature_registry_tables": ("feature_registry", "FeatureBaseSchema"),
    "fk_registry_tables": ("registry", "RegistryBaseSchema"),
    "other_tables": ("table", "LanceModel"),
}

# IR base kind -> generated base class name (inverse of SECTION_BASES values).
BASE_CLASS_NAMES: dict[str, str] = {kind: cls for kind, cls in SECTION_BASES.values()}

# IR base kind -> table section name.
KIND_SECTIONS: dict[str, str] = {kind: section for section, (kind, _) in SECTION_BASES.items()}

# Marker name -> factory class in homeobox.schema. ``stable_uid`` is a bare flag.
MARKER_FACTORIES: dict[str, str] = {
    "pointer": "PointerField",
    "stable_uid": "StableUIDField",
    "registry_key": "RegistryKeyField",
    "polymorphic_registry_key": "PolymorphicRegistryKeyField",
    "ontology_aligned": "OntologyAlignedField",
    "cross_reference": "CrossReferenceField",
    "summary": "SummaryField",
}

# Markers whose YAML payload may be written as a single bare string.
MARKER_STRING_SHORTHAND: dict[str, str] = {
    "cross_reference": "database_name",
    "ontology_aligned": "ontology_name",
}

# Pointer class name -> the spec ``pointer_type_name`` it must match.
POINTER_TYPE_NAMES: dict[str, str] = {
    "SparseZarrPointer": "sparse",
    "DenseZarrPointer": "dense",
    "DiscreteSpatialPointer": "discrete_spatial",
}

ALLOWED_SUMMARY_OPS: frozenset[str] = frozenset({"count", "nunique", "unique"})
ALLOWED_COMPUTED_OPS: frozenset[str] = frozenset({"join_list"})
ALLOWED_CONSTRAINT_KINDS: frozenset[str] = frozenset({"require_any", "equal_length"})


class _Required:
    """Sentinel for a field with no default (pydantic-required)."""

    _instance: "_Required | None" = None

    def __new__(cls) -> "_Required":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "REQUIRED"


REQUIRED = _Required()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EnumDef:
    name: str
    values: dict[str, str]
    doc: str | None = None


@dataclasses.dataclass(frozen=True)
class ComputedDef:
    op: str
    args: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class FieldDef:
    name: str
    type: str
    default: Any = REQUIRED
    doc: str | None = None
    # marker name -> normalised payload (dict, or True for ``stable_uid``).
    markers: dict[str, Any] = dataclasses.field(default_factory=dict)
    computed: ComputedDef | None = None

    @property
    def required(self) -> bool:
        return self.default is REQUIRED


@dataclasses.dataclass(frozen=True)
class ConstraintDef:
    kind: str
    fields: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class TableDef:
    name: str
    base: str  # IR base kind, e.g. "obs"
    fields: tuple[FieldDef, ...]
    doc: str | None = None
    constraints: tuple[ConstraintDef, ...] = ()
    presence_flags: bool = False

    @property
    def base_class(self) -> str:
        return BASE_CLASS_NAMES[self.base]


@dataclasses.dataclass(frozen=True)
class SchemaModel:
    name: str
    doc: str | None = None
    enums: tuple[EnumDef, ...] = ()
    obs_tables: tuple[TableDef, ...] = ()
    dataset_table: TableDef | None = None
    feature_registry_tables: tuple[TableDef, ...] = ()
    fk_registry_tables: tuple[TableDef, ...] = ()
    other_tables: tuple[TableDef, ...] = ()

    def emit_order(self) -> list[TableDef]:
        """Tables in code-emission order: every reference defined before use.

        Registries and other LanceModel tables (FK / pointer targets) come
        first, then the dataset table, then the obs tables that reference them.
        """
        tables: list[TableDef] = [
            *self.fk_registry_tables,
            *self.feature_registry_tables,
            *self.other_tables,
        ]
        if self.dataset_table is not None:
            tables.append(self.dataset_table)
        tables.extend(self.obs_tables)
        return tables

    def registry_schemas(self) -> dict[str, str]:
        """Derive ``{feature_space: feature_registry_schema}`` from obs pointers."""
        result: dict[str, str] = {}
        for table in self.obs_tables:
            for field in table.fields:
                pointer = field.markers.get("pointer")
                if isinstance(pointer, dict) and pointer.get("feature_registry_schema"):
                    result[pointer["feature_space"]] = pointer["feature_registry_schema"]
        return result


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _require_keys(mapping: dict, allowed: set[str], context: str) -> None:
    if not isinstance(mapping, dict):
        raise ValueError(f"{context}: expected a mapping, got {type(mapping).__name__}")
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(
            f"{context}: unknown key(s) {sorted(unknown)!r}; allowed keys are {sorted(allowed)!r}"
        )


def _normalize_marker(name: str, raw: Any, context: str) -> Any:
    if name == "stable_uid":
        if raw is not True:
            raise ValueError(f"{context}: stable_uid must be the literal `true`")
        return True

    if name in MARKER_STRING_SHORTHAND:
        key = MARKER_STRING_SHORTHAND[name]
        if isinstance(raw, str):
            payload = {key: raw}
        elif isinstance(raw, dict):
            _require_keys(raw, {key}, f"{context}.{name}")
            payload = dict(raw)
        else:
            raise ValueError(f"{context}.{name}: expected a string or mapping")
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"{context}.{name}: requires a non-empty {key!r} string")
        return payload

    if name == "registry_key":
        _require_keys(raw, {"target_schema", "target_field"}, f"{context}.{name}")
        if not isinstance(raw.get("target_schema"), str) or not raw["target_schema"]:
            raise ValueError(f"{context}.{name}: requires a non-empty target_schema")
        payload = {"target_schema": raw["target_schema"]}
        if "target_field" in raw:
            payload["target_field"] = raw["target_field"]
        return payload

    if name == "polymorphic_registry_key":
        _require_keys(raw, {"type_field", "target_field", "variants"}, f"{context}.{name}")
        if not isinstance(raw.get("type_field"), str) or not raw["type_field"]:
            raise ValueError(f"{context}.{name}: requires a non-empty type_field")
        variants = raw.get("variants")
        if not isinstance(variants, dict) or not variants:
            raise ValueError(f"{context}.{name}: requires a non-empty variants mapping")
        for key, value in variants.items():
            if not isinstance(key, str) or not isinstance(value, str) or not key or not value:
                raise ValueError(f"{context}.{name}: variants must map non-empty strings")
        payload = {"type_field": raw["type_field"], "variants": dict(variants)}
        if "target_field" in raw:
            payload["target_field"] = raw["target_field"]
        return payload

    if name == "summary":
        _require_keys(raw, {"target_schema", "target_field", "op"}, f"{context}.{name}")
        for key in ("target_schema", "target_field", "op"):
            if not isinstance(raw.get(key), str) or not raw[key]:
                raise ValueError(f"{context}.{name}: requires a non-empty {key}")
        if raw["op"] not in ALLOWED_SUMMARY_OPS:
            raise ValueError(f"{context}.{name}: op must be one of {sorted(ALLOWED_SUMMARY_OPS)!r}")
        return {
            "target_schema": raw["target_schema"],
            "target_field": raw["target_field"],
            "op": raw["op"],
        }

    if name == "pointer":
        _require_keys(raw, {"feature_space", "feature_registry_schema"}, f"{context}.{name}")
        if not isinstance(raw.get("feature_space"), str) or not raw["feature_space"]:
            raise ValueError(f"{context}.{name}: requires a non-empty feature_space")
        payload = {"feature_space": raw["feature_space"]}
        registry = raw.get("feature_registry_schema")
        if registry is not None:
            if not isinstance(registry, str) or not registry:
                raise ValueError(
                    f"{context}.{name}: feature_registry_schema must be a non-empty string"
                )
            payload["feature_registry_schema"] = registry
        return payload

    raise ValueError(f"{context}: unknown marker {name!r}")


def _load_computed(raw: Any, context: str) -> ComputedDef:
    if not isinstance(raw, dict):
        raise ValueError(f"{context}: computed must be a mapping")
    op = raw.get("op")
    if op not in ALLOWED_COMPUTED_OPS:
        raise ValueError(f"{context}: computed op must be one of {sorted(ALLOWED_COMPUTED_OPS)!r}")
    if op == "join_list":
        _require_keys(raw, {"op", "source", "separator"}, context)
        if not isinstance(raw.get("source"), str) or not raw["source"]:
            raise ValueError(f"{context}: join_list requires a non-empty source")
        if not isinstance(raw.get("separator"), str):
            raise ValueError(f"{context}: join_list requires a separator string")
        return ComputedDef(
            op="join_list", args={"source": raw["source"], "separator": raw["separator"]}
        )
    raise ValueError(f"{context}: unsupported computed op {op!r}")  # pragma: no cover


def _load_field(raw: Any, context: str) -> FieldDef:
    marker_keys = set(MARKER_FACTORIES)
    allowed = {"name", "type", "default", "doc", "computed", "markers"} | marker_keys
    _require_keys(raw, allowed, context)

    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{context}: field requires a non-empty name")
    context = f"{context}.{name}"
    type_ = raw.get("type")
    if not isinstance(type_, str) or not type_:
        raise ValueError(f"{context}: field requires a non-empty type")

    markers: dict[str, Any] = {}
    for key in marker_keys:
        if key in raw:
            markers[key] = _normalize_marker(key, raw[key], context)
    block = raw.get("markers")
    if block is not None:
        if not isinstance(block, dict):
            raise ValueError(f"{context}.markers: expected a mapping")
        for key, value in block.items():
            if key not in marker_keys:
                raise ValueError(f"{context}.markers: unknown marker {key!r}")
            if key in markers:
                raise ValueError(f"{context}: marker {key!r} given both inline and under `markers`")
            markers[key] = _normalize_marker(key, value, context)

    computed = _load_computed(raw["computed"], f"{context}.computed") if "computed" in raw else None
    default = raw["default"] if "default" in raw else REQUIRED
    doc = raw.get("doc")

    return FieldDef(
        name=name,
        type=type_,
        default=default,
        doc=doc,
        markers=markers,
        computed=computed,
    )


def _load_constraint(raw: Any, context: str) -> ConstraintDef:
    if not isinstance(raw, dict) or len(raw) != 1:
        raise ValueError(f"{context}: each constraint must be a single-key mapping")
    ((kind, fields),) = raw.items()
    if kind not in ALLOWED_CONSTRAINT_KINDS:
        raise ValueError(
            f"{context}: unknown constraint {kind!r}; allowed: {sorted(ALLOWED_CONSTRAINT_KINDS)!r}"
        )
    if not isinstance(fields, list) or not all(isinstance(f, str) and f for f in fields):
        raise ValueError(f"{context}.{kind}: expected a list of field names")
    if len(fields) < 2:
        raise ValueError(f"{context}.{kind}: needs at least two fields")
    return ConstraintDef(kind=kind, fields=tuple(fields))


def _load_table(raw: Any, kind: str, context: str) -> TableDef:
    _require_keys(raw, {"name", "doc", "fields", "constraints", "presence_flags"}, context)
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{context}: table requires a non-empty name")
    context = f"{context}[{name}]"
    fields_raw = raw.get("fields")
    if not isinstance(fields_raw, list) or not fields_raw:
        raise ValueError(f"{context}: table requires a non-empty fields list")
    fields = tuple(_load_field(f, f"{context}.fields") for f in fields_raw)

    constraints_raw = raw.get("constraints", [])
    if not isinstance(constraints_raw, list):
        raise ValueError(f"{context}.constraints: expected a list")
    constraints = tuple(_load_constraint(c, f"{context}.constraints") for c in constraints_raw)

    presence = raw.get("presence_flags", False)
    if not isinstance(presence, bool):
        raise ValueError(f"{context}.presence_flags: expected a boolean")
    if presence and kind != "obs":
        raise ValueError(f"{context}: presence_flags is only valid on obs tables")

    return TableDef(
        name=name,
        base=kind,
        fields=fields,
        doc=raw.get("doc"),
        constraints=constraints,
        presence_flags=presence,
    )


def _load_enum(name: str, raw: Any, context: str) -> EnumDef:
    _require_keys(raw, {"doc", "values"}, context)
    values = raw.get("values")
    if not isinstance(values, dict) or not values:
        raise ValueError(f"{context}: enum requires a non-empty values mapping")
    normalized: dict[str, str] = {}
    for member, value in values.items():
        if not isinstance(member, str) or not member:
            raise ValueError(f"{context}: enum member names must be non-empty strings")
        normalized[member] = str(value)
    return EnumDef(name=name, values=normalized, doc=raw.get("doc"))


def model_from_dict(document: dict) -> SchemaModel:
    """Build a :class:`SchemaModel` from an already-parsed YAML document."""
    allowed = {"schema", "enums", *SECTION_BASES}
    _require_keys(document, allowed, "<root>")

    schema_meta = document.get("schema")
    if not isinstance(schema_meta, dict):
        raise ValueError("<root>.schema: required mapping with at least a name")
    _require_keys(schema_meta, {"name", "doc"}, "schema")
    schema_name = schema_meta.get("name")
    if not isinstance(schema_name, str) or not schema_name:
        raise ValueError("schema.name: required non-empty string")

    enums_raw = document.get("enums", {})
    if not isinstance(enums_raw, dict):
        raise ValueError("enums: expected a mapping of enum name -> definition")
    enums = tuple(_load_enum(name, raw, f"enums.{name}") for name, raw in enums_raw.items())

    def load_section(section: str) -> tuple[TableDef, ...]:
        raw = document.get(section)
        if raw is None:
            return ()
        kind = SECTION_BASES[section][0]
        if not isinstance(raw, list):
            raise ValueError(f"{section}: expected a list of tables")
        return tuple(_load_table(t, kind, section) for t in raw)

    dataset_raw = document.get("dataset_table")
    dataset_table: TableDef | None = None
    if dataset_raw is not None:
        dataset_table = _load_table(dataset_raw, "dataset", "dataset_table")

    return SchemaModel(
        name=schema_name,
        doc=schema_meta.get("doc"),
        enums=enums,
        obs_tables=load_section("obs_tables"),
        dataset_table=dataset_table,
        feature_registry_tables=load_section("feature_registry_tables"),
        fk_registry_tables=load_section("fk_registry_tables"),
        other_tables=load_section("other_tables"),
    )


def load_yaml(text: str) -> SchemaModel:
    """Parse YAML *text* into a :class:`SchemaModel`, hard-erroring on bad input."""
    document = yaml.safe_load(text)
    if not isinstance(document, dict):
        raise ValueError("schema YAML must be a top-level mapping")
    return model_from_dict(document)


def load_yaml_file(path: str) -> SchemaModel:
    with open(path, encoding="utf-8") as handle:
        return load_yaml(handle.read())


# ---------------------------------------------------------------------------
# Dumping
# ---------------------------------------------------------------------------


def _marker_to_yaml(name: str, payload: Any) -> Any:
    """Render a normalised marker payload back to its compact YAML form."""
    if name == "stable_uid":
        return True
    if name in MARKER_STRING_SHORTHAND:
        return payload[MARKER_STRING_SHORTHAND[name]]
    return dict(payload)


def _field_to_dict(field: FieldDef) -> dict:
    out: dict[str, Any] = {"name": field.name, "type": field.type}
    if not field.required:
        out["default"] = field.default
    if field.doc is not None:
        out["doc"] = field.doc
    if field.computed is not None:
        out["computed"] = {"op": field.computed.op, **field.computed.args}
    if len(field.markers) == 1:
        ((name, payload),) = field.markers.items()
        out[name] = _marker_to_yaml(name, payload)
    elif field.markers:
        out["markers"] = {
            name: _marker_to_yaml(name, payload) for name, payload in field.markers.items()
        }
    return out


def _table_to_dict(table: TableDef) -> dict:
    out: dict[str, Any] = {"name": table.name}
    if table.doc is not None:
        out["doc"] = table.doc
    if table.presence_flags:
        out["presence_flags"] = True
    if table.constraints:
        out["constraints"] = [{c.kind: list(c.fields)} for c in table.constraints]
    out["fields"] = [_field_to_dict(f) for f in table.fields]
    return out


def model_to_dict(model: SchemaModel) -> dict:
    """Project a :class:`SchemaModel` back to the plain YAML document structure."""
    schema_meta: dict[str, Any] = {"name": model.name}
    if model.doc is not None:
        schema_meta["doc"] = model.doc

    document: dict[str, Any] = {"schema": schema_meta}
    if model.enums:
        enums: dict[str, Any] = {}
        for enum in model.enums:
            entry: dict[str, Any] = {}
            if enum.doc is not None:
                entry["doc"] = enum.doc
            entry["values"] = dict(enum.values)
            enums[enum.name] = entry
        document["enums"] = enums

    if model.obs_tables:
        document["obs_tables"] = [_table_to_dict(t) for t in model.obs_tables]
    if model.dataset_table is not None:
        document["dataset_table"] = _table_to_dict(model.dataset_table)
    if model.feature_registry_tables:
        document["feature_registry_tables"] = [
            _table_to_dict(t) for t in model.feature_registry_tables
        ]
    if model.fk_registry_tables:
        document["fk_registry_tables"] = [_table_to_dict(t) for t in model.fk_registry_tables]
    if model.other_tables:
        document["other_tables"] = [_table_to_dict(t) for t in model.other_tables]
    return document


def dump_yaml(model: SchemaModel) -> str:
    """Serialise a :class:`SchemaModel` to YAML text."""
    return yaml.safe_dump(
        model_to_dict(model),
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=100,
    )
