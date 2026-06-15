"""Stage publication metadata from ``publication.json`` into collection ``lance_db/``.

Reads the collection's ``publication.json`` sidecar (typically under
``other_files/`` after coalesce) and writes one or two Lance tables at
``<collection_root>/lance_db/``. Only run this when the target schema defines
collection-level publication registry tables.

Field mapping follows ``publication.json`` exactly — only keys present in the
file are written. Top-level keys other than ``text_data`` go to the
publication table; ``text_data.section_title`` / ``text_data.section_text``
become section rows.

Each collection has a single publication, so staging also seeds the ``*_join``
scaffolding that ``finalize-tables`` expects: ``{pub_schema}_join = 0`` on the
publication table, and ``{field}_{pub_schema}_join = 0`` on the section table
when it references the publication registry (discovered via ``--schema`` or
``--pub-fk-field``). Staged columns usually will not yet conform to the homeobox
schema; downstream skills align and finalize them.

Three modes (provide at least one schema argument):

1. Publication registry only (``--pub-schema``):
   One row with all top-level fields except ``text_data``.
2. Publication + sections (``--pub-schema`` and ``--pub-section-schema``):
   One publication row plus one section row per ``text_data`` entry.
3. Denormalized sections only (``--pub-section-schema``):
   Section rows with top-level publication fields repeated on each row.

Usage:
    python scripts/stage_publication_tables.py <collection_root> \\
        [--pub-schema PublicationSchema] \\
        [--pub-section-schema PublicationSectionSchema] \\
        [--schema <path/to/schema.py>] \\
        [--pub-fk-field FIELD] \\
        [--publication-json PATH]

Arguments:
    collection_root       Root directory of a coalesced collection
    --pub-schema          CamelCase Lance table for the publication registry
    --pub-section-schema  CamelCase Lance table for publication text sections
    --schema              Homeobox schema file (discover section-table RegistryKeyFields)
    --pub-fk-field        Registry-key field on the section table (repeatable; overrides --schema)
    --publication-json    Path to publication.json (default: discover from manifest)
"""

from __future__ import annotations

import argparse
import json
import os

import lancedb
import pandas as pd

from auto_atlas.collection import FileTypeTag
from auto_atlas.types import SchemaInfo
from auto_atlas.util import load_schema_info

COLLECTION_MANIFEST = "collection.json"
LANCE_DB_DIR = "lance_db"
PUBLICATION_FILENAME = "publication.json"
PUBLICATION_JOIN_VALUE = 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage publication.json into collection-level lance_db."
    )
    parser.add_argument("collection_root", help="Root directory of a coalesced collection")
    parser.add_argument(
        "--pub-schema",
        help="CamelCase Lance table name for the publication registry (e.g. PublicationSchema)",
    )
    parser.add_argument(
        "--pub-section-schema",
        help=(
            "CamelCase Lance table name for publication sections (e.g. PublicationSectionSchema)"
        ),
    )
    parser.add_argument(
        "--publication-json",
        help="Path to publication.json (absolute or relative to collection root)",
    )
    parser.add_argument(
        "--schema",
        help="Path to the homeobox schema Python file (for RegistryKeyField discovery)",
    )
    parser.add_argument(
        "--pub-fk-field",
        action="append",
        default=[],
        dest="pub_fk_fields",
        metavar="FIELD",
        help=(
            "Registry-key field on the section table that references --pub-schema "
            "(repeatable; overrides --schema discovery)"
        ),
    )
    args = parser.parse_args(argv)
    if not args.pub_schema and not args.pub_section_schema:
        parser.error("Provide --pub-schema and/or --pub-section-schema")
    return args


def resolve_path(collection_root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(collection_root, path)


def find_publication_json(collection_root: str, override: str | None) -> str:
    if override is not None:
        path = resolve_path(collection_root, override)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"publication.json not found: {path}")
        return path

    manifest_path = os.path.join(collection_root, COLLECTION_MANIFEST)
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            payload = json.load(f)
        for entry in payload.get("shared_files", []):
            path = entry.get("path", "")
            if os.path.basename(path) != PUBLICATION_FILENAME:
                continue
            resolved = resolve_path(collection_root, path)
            if os.path.isfile(resolved):
                return resolved

    for candidate in (
        os.path.join(collection_root, "other_files", PUBLICATION_FILENAME),
        os.path.join(collection_root, PUBLICATION_FILENAME),
    ):
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not find {PUBLICATION_FILENAME} under {collection_root}. "
        "Pass --publication-json or add the file to the collection first."
    )


def warn_if_not_tagged_publication(collection_root: str, publication_path: str) -> None:
    manifest_path = os.path.join(collection_root, COLLECTION_MANIFEST)
    if not os.path.isfile(manifest_path):
        return
    with open(manifest_path) as f:
        payload = json.load(f)
    abs_publication = os.path.abspath(publication_path)
    for entry in payload.get("shared_files", []):
        if entry.get("tag") != str(FileTypeTag.OTHER):
            continue
        tagged = os.path.abspath(resolve_path(collection_root, entry["path"]))
        if tagged == abs_publication:
            return
    print(f"warning: {publication_path} is not listed as an OTHER file in {COLLECTION_MANIFEST}")


def load_publication_json(path: str) -> dict:
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def extract_pub_fields(publication: dict) -> dict:
    """Top-level publication.json fields, excluding ``text_data``."""
    return {key: value for key, value in publication.items() if key != "text_data"}


def target_join_column(pub_schema: str) -> str:
    return f"{pub_schema}_join"


def referencing_join_column(fk_field: str, pub_schema: str) -> str:
    return f"{fk_field}_{pub_schema}_join"


def resolve_pub_fk_fields(
    pub_schema: str,
    section_schema: str,
    schema_info: SchemaInfo | None,
    explicit_fields: list[str],
) -> list[str]:
    if explicit_fields:
        return explicit_fields
    if schema_info is None:
        return []
    return [
        fk.field_name
        for fk in schema_info.scalar_fks.get(section_schema, [])
        if fk.target_schema == pub_schema
    ]


def add_target_join(row: dict, pub_schema: str) -> dict:
    row[target_join_column(pub_schema)] = PUBLICATION_JOIN_VALUE
    return row


def add_referencing_joins(row: dict, pub_schema: str, fk_fields: list[str]) -> dict:
    for field in fk_fields:
        row[referencing_join_column(field, pub_schema)] = PUBLICATION_JOIN_VALUE
    return row


def build_section_rows(
    publication: dict,
    *,
    denormalize: bool,
    pub_schema: str | None = None,
    pub_fk_fields: list[str] | None = None,
) -> list[dict]:
    text_data = publication.get("text_data") or {}
    titles = text_data.get("section_title") or []
    texts = text_data.get("section_text") or []

    if len(titles) != len(texts):
        raise ValueError(
            "text_data.section_title and text_data.section_text must have the same length "
            f"({len(titles)} vs {len(texts)})"
        )

    pub_fields = extract_pub_fields(publication) if denormalize else {}

    rows: list[dict] = []
    for title, text in zip(titles, texts, strict=True):
        row = {"section_title": title, "section_text": text}
        if denormalize:
            row.update(pub_fields)
        elif pub_schema is not None and pub_fk_fields:
            add_referencing_joins(row, pub_schema, pub_fk_fields)
        rows.append(row)
    return rows


def stage_table(db: lancedb.DBConnection, table_name: str, df: pd.DataFrame) -> None:
    db.create_table(table_name, data=df, mode="overwrite")
    print(f"{table_name}: {len(df)} row(s), {len(df.columns)} column(s)")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    collection_root = os.path.abspath(args.collection_root)
    publication_path = find_publication_json(collection_root, args.publication_json)
    warn_if_not_tagged_publication(collection_root, publication_path)
    publication = load_publication_json(publication_path)
    schema_info = load_schema_info(args.schema) if args.schema else None

    lance_path = os.path.join(collection_root, LANCE_DB_DIR)
    os.makedirs(lance_path, exist_ok=True)
    db = lancedb.connect(lance_path)

    print(f"Loaded {publication_path}")

    pub_fk_fields: list[str] = []
    if args.pub_schema and args.pub_section_schema:
        pub_fk_fields = resolve_pub_fk_fields(
            args.pub_schema,
            args.pub_section_schema,
            schema_info,
            args.pub_fk_fields,
        )
        if not pub_fk_fields:
            raise ValueError(
                "Section table references the publication registry but no join field "
                "was found. Pass --schema to discover RegistryKeyFields on "
                f"{args.pub_section_schema!r}, or pass --pub-fk-field explicitly."
            )

    if args.pub_schema:
        pub_row = add_target_join(extract_pub_fields(publication), args.pub_schema)
        pub_df = pd.DataFrame([pub_row])
        stage_table(db, args.pub_schema, pub_df)

    if args.pub_section_schema:
        denormalize = args.pub_schema is None
        section_rows = build_section_rows(
            publication,
            denormalize=denormalize,
            pub_schema=args.pub_schema,
            pub_fk_fields=pub_fk_fields,
        )
        if not section_rows:
            print("warning: no text sections found in publication.json text_data")
        section_df = pd.DataFrame(section_rows)
        stage_table(db, args.pub_section_schema, section_df)

    print(f"-> {lance_path}")


if __name__ == "__main__":
    main()
