"""Shared internal utilities.

No ``pathlib``: paths are plain strings joined with ``os.path`` so s3 urls work.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any

import lancedb
import pyarrow as pa
from homeobox.parser import parse_schema_module
from homeobox.schema import PolymorphicRegistryKeyField, RegistryKeyField, SummaryField

from auto_atlas.types import SchemaInfo, TableRef


def extract_h5ad_obs_var(h5ad_path: str) -> tuple[str, str]:
    """Write the obs and var dataframes of an h5ad file to separate CSV files.

    The CSVs are written alongside the input, reusing its name: ``foo.h5ad``
    yields ``foo_obs.csv`` and ``foo_var.csv``. The dataframes keep their index
    (cell barcodes for obs, feature ids for var). The file is read in backed
    mode so X is never loaded into memory. Returns ``(obs_csv_path, var_csv_path)``.
    """
    # Imported lazily so the rest of this module does not depend on anndata.
    import anndata as ad

    base = os.path.splitext(h5ad_path)[0]
    obs_csv_path = f"{base}_obs.csv"
    var_csv_path = f"{base}_var.csv"

    adata = ad.read_h5ad(h5ad_path, backed="r")
    adata.obs.to_csv(obs_csv_path)
    adata.var.to_csv(var_csv_path)
    return obs_csv_path, var_csv_path


# ===========================================================================
# Finalization helpers
# ===========================================================================
#
# Finalization turns a set of independently-harmonized Lance tables into a
# linked, schema-conformant collection. The helpers below load a target homeobox
# schema as live pydantic classes *and* its parsed relationship graph, discover
# the concrete Lance tables across a collection, order them so every registry-key
# target precedes its referrers, and read / mutate / overwrite a table at the
# Arrow level so harmonized column types (lists, pointer structs) survive
# untouched.


# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------


def load_schema_module(schema_path: str) -> Any:
    """Import a homeobox ``schema.py`` from a file path and return the module.

    Importing executes the file (needed to get live pydantic classes for
    ``compute_stable_uids`` / validation); only run on schemas you trust.
    """
    schema_path = os.fspath(schema_path)
    base = os.path.splitext(os.path.basename(schema_path))[0]
    mod_name = f"_finalize_schema_{base}"
    spec = importlib.util.spec_from_file_location(mod_name, schema_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load schema module from {schema_path!r}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def load_schema_info(schema_path: str) -> SchemaInfo:
    """Load the schema module and derive the kinds map and registry-key markers.

    Registry keys are returned as homeobox's own ``RegistryKeyField`` /
    ``PolymorphicRegistryKeyField`` markers. The parser flattens a polymorphic FK
    into one relationship per variant; those are recombined here into a single
    marker per field. ``SummaryField`` markers are read off each field dict (the
    parser hangs a ``summary`` key on derived fields) and grouped by class.
    """
    module = load_schema_module(schema_path)
    parsed = parse_schema_module(module)

    kinds: dict[str, str] = {}
    summary_fields: dict[str, list[SummaryField]] = {}
    for table in [parsed.get("obs"), parsed.get("dataset"), *parsed.get("tables", [])]:
        if table is None:
            continue
        class_name = table["class_name"]
        kinds[class_name] = table["kind"]
        for f in table.get("fields", []):
            summary = f.get("summary")
            if summary is None:
                continue
            summary_fields.setdefault(class_name, []).append(
                SummaryField(
                    field_name=f["name"],
                    target_schema=summary["target_schema"],
                    target_field=summary["target_field"],
                    op=summary["op"],
                )
            )

    scalar_fks: dict[str, list[RegistryKeyField]] = {}
    # (source, field, type_field, target_field) -> {variant: target_schema}
    poly_acc: dict[tuple[str, str, str, str], dict[str, str]] = {}

    for rel in parsed.get("relationships", []):
        kind = rel["kind"]
        source = rel["source_table"]
        if kind == "registry_key":
            target_field = rel.get("target_field", "uid")
            # ``dataset_uid`` is stamped separately from collection.json, not joined.
            if target_field != "uid":
                continue
            scalar_fks.setdefault(source, []).append(
                RegistryKeyField(
                    field_name=rel["source_field"],
                    target_schema=rel["target_schema"],
                    target_field=target_field,
                )
            )
        elif kind == "polymorphic_registry_key":
            key = (source, rel["source_field"], rel["type_field"], rel.get("target_field", "uid"))
            poly_acc.setdefault(key, {})[rel["variant"]] = rel["target_schema"]
        # pointer_feature_registry relationships are zarr pointers, not uid FKs.

    poly_fks: dict[str, list[PolymorphicRegistryKeyField]] = {}
    for (source, field_name, type_field, target_field), variants in poly_acc.items():
        poly_fks.setdefault(source, []).append(
            PolymorphicRegistryKeyField(
                field_name=field_name,
                type_field=type_field,
                target_field=target_field,
                variants=variants,
            )
        )

    return SchemaInfo(
        module=module,
        kinds=kinds,
        scalar_fks=scalar_fks,
        poly_fks=poly_fks,
        summary_fields=summary_fields,
    )


# ---------------------------------------------------------------------------
# Table discovery
# ---------------------------------------------------------------------------


def _lance_db_dirs(collection_root: str) -> list[tuple[str | None, str]]:
    """Return ``(dataset_name, lance_db_path)`` for every lance_db in the collection.

    The collection-level ``lance_db`` (registries / FK targets) is reported with
    ``dataset_name=None``; each ``<dataset>/lance_db`` with its directory name.
    """
    collection_root = os.fspath(collection_root)
    dirs: list[tuple[str | None, str]] = []

    root_db = os.path.join(collection_root, "lance_db")
    if os.path.isdir(root_db):
        dirs.append((None, root_db))

    for entry in sorted(os.listdir(collection_root)):
        sub = os.path.join(collection_root, entry, "lance_db")
        if os.path.isdir(sub):
            dirs.append((entry, sub))

    return dirs


def _class_for_table(table_name: str, class_names: list[str]) -> str | None:
    """Map a concrete Lance table name to its schema class by exact match."""
    if table_name in class_names:
        return table_name
    return None


def discover_tables(collection_root: str, info: SchemaInfo) -> list[TableRef]:
    """Find every Lance table in the collection that maps to a schema class."""
    class_names = list(info.kinds.keys())
    refs: list[TableRef] = []
    for dataset, lance_db_path in _lance_db_dirs(collection_root):
        db = lancedb.connect(lance_db_path)
        for table_name in db.list_tables().tables:
            cls = _class_for_table(table_name, class_names)
            if cls is None:
                continue
            refs.append(
                TableRef(
                    lance_db_path=lance_db_path,
                    table_name=table_name,
                    class_name=cls,
                    dataset=dataset,
                )
            )
    return refs


def tables_for_class(refs: list[TableRef], class_name: str) -> list[TableRef]:
    return [r for r in refs if r.class_name == class_name]


# ---------------------------------------------------------------------------
# Dependency ordering
# ---------------------------------------------------------------------------


def finalization_order(info: SchemaInfo) -> list[str]:
    """Topologically sort schema classes so every FK target precedes its referrers.

    Edges come from the schema's own registry-key declarations (target -> source),
    so the order is derived, never hard-coded. Cycles raise rather than silently
    producing a wrong order.
    """
    nodes = set(info.kinds.keys())
    # dependencies[source] = set of target classes that must finalize first
    deps: dict[str, set[str]] = {n: set() for n in nodes}
    for source, fks in info.scalar_fks.items():
        for fk in fks:
            if fk.target_schema in nodes:
                deps.setdefault(source, set()).add(fk.target_schema)
    for source, pfks in info.poly_fks.items():
        for pfk in pfks:
            for target in pfk.variants.values():
                if target in nodes:
                    deps.setdefault(source, set()).add(target)

    ordered: list[str] = []
    done: set[str] = set()
    while len(done) < len(nodes):
        ready = sorted(n for n in nodes if n not in done and deps.get(n, set()) <= done)
        if not ready:
            remaining = sorted(nodes - done)
            raise ValueError(f"Foreign-key dependency cycle among: {remaining}")
        for n in ready:
            ordered.append(n)
            done.add(n)
    return ordered


# ---------------------------------------------------------------------------
# Arrow read / write helpers
# ---------------------------------------------------------------------------


def open_table(ref: TableRef):
    return lancedb.connect(ref.lance_db_path).open_table(ref.table_name)


def read_arrow(ref: TableRef) -> pa.Table:
    return open_table(ref).to_arrow()


def overwrite_table(ref: TableRef, table: pa.Table) -> None:
    """Overwrite a Lance table from a pyarrow ``Table``, preserving its types."""
    db = lancedb.connect(ref.lance_db_path)
    db.create_table(ref.table_name, data=table, mode="overwrite")


def set_arrow_column(table: pa.Table, name: str, array: pa.Array | pa.ChunkedArray) -> pa.Table:
    """Replace ``name`` if present, else append it."""
    field_ = pa.field(name, array.type)
    if name in table.column_names:
        return table.set_column(table.column_names.index(name), field_, array)
    return table.append_column(field_, array)


def drop_arrow_columns(table: pa.Table, names: list[str]) -> pa.Table:
    keep = [c for c in table.column_names if c not in set(names)]
    return table.select(keep)


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------


def is_null(value: Any) -> bool:
    if isinstance(value, list):
        return False  # a list (even empty) is a present value, not null
    if value is None:
        return True
    if isinstance(value, float):
        return value != value  # NaN
    return False


def join_key(value: Any) -> str | None:
    """Canonicalize a natural-key cell to a comparable string, or None if absent.

    Stringify-and-strip is a deterministic, exact transform applied identically to
    both sides of a join so Arrow type variance (int vs str ids) does not cause a
    spurious miss. Any further normalization is the harmonization step's job.
    """
    if is_null(value):
        return None
    text = str(value).strip()
    return text or None
