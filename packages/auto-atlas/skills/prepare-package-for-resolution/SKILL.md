---
name: prepare-package-for-resolution
description: Use after create-data-package when a coalesced data package and homeobox schema file are ready. Stages per-dataset OBS and VAR, the per-dataset DatasetSchema scaffold, and collection-level LIBRARY tables into Lance.
---

# Prepare package for resolution

## Scope: raw staging only

This skill loads raw tables into Lance and names them after homeobox schema classes. It does not download files, standardize columns, or ingest arrays. Run `create-data-package` first.

Tables are named after schema classes (e.g. `GeneticPerturbationSchema`) but their **columns are kept exactly as found in the source file**. This skill does not rename, reshape, or otherwise align columns to the schema's fields, so a staged table will usually not conform to its schema yet. That is expected. Evolving these raw tables into the final schema-aligned form is the job of other downstream skills.

### Preparing source files before staging

The staging scripts load tables into Lance without changing cell values. Delimited text files (`.csv`, `.tsv`, `.tsv.gz`) skip lines that start with `#`; other formats are read as-is.

Before running the scripts, inspect the head of each source file. When the file is not ready to load cleanly, you may make small **file-level** fixes first — for example converting to the expected delimiter, removing a non-`#` header or preamble block, or stripping a duplicate header row embedded in the data. These edits should only adjust file structure (comments, headers, delimiters); do not modify actual data cell values. Apply fixes on the coalesced package paths, then run the staging scripts.

Stage the **`DatasetSchema` scaffold** here (see step 3): the identity rows and their `dataset_uid` come straight from `collection.json`, so they belong with the other raw staging. The scaffold creates only those identity columns — `zarr_group`, descriptive metadata, and the `SummaryField` aggregates are each added later by the step that fills them.

**Publication tables** (step 4) are also staged here, but only when the target schema defines collection-level publication registry tables and the collection includes a `publication.json` sidecar (typically under `other_files/` after coalesce). Skip step 4 if the schema has no publication registry.

## Expected input

- A coalesced collection with `collection.json` at the root
- A schema file path from the user (ask if missing). Never assume the schema file, this is important to avoid wasted work. The schema file path must be explicitly provided by the user somewhere in your conversation before proceeding.

## Workflow

### 1. Stage OBS and VAR

Unless the user already said whether they trust the schema file, ask. Use `--parse-mode runtime` only if they do; otherwise omit the flag (defaults to safe AST parsing).

```
python scripts/stage_lance_tables.py <collection_root> \
  --schema <path/to/schema.py> \
  [--parse-mode runtime] \
  [--obs-class CellIndex]
```

If the script fails because the schema defines multiple obs tables, ask the user which class to use and re-run with `--obs-class`.

### 2. Stage LIBRARY tables

Collection-level `LIBRARY` files (in `collection.json` → `shared_files`) may be staged into `<collection_root>/lance_db/`. Read the schema and decide which CamelCase table each library file belongs to (e.g. `GeneticPerturbationSchema`). If more than one table is plausible, ask the user.

Run once per library file:

```
python scripts/stage_library_table.py <collection_root> \
  --library <path/to/library.csv> \
  --table <SchemaClassName>
```

Supported formats: `.parquet`, `.csv`, `.tsv`, `.tsv.gz`, `.xlsx`.

#### Multi-sheet `.xlsx` files

A single `.xlsx` library file may contain many sheets, and often only one (or a few) are actual reference tables. The script loads one sheet per run (`--sheet-name`, default is the first sheet), so never assume the default sheet is the right one.

Before staging, inspect the workbook: list the sheet names and preview each sheet's columns. Then stage only the sheet(s) that are genuine reference/library tables, choosing the matching schema class for each:

```
python scripts/stage_library_table.py <collection_root> \
  --library <path/to/library.xlsx> \
  --table <SchemaClassName> \
  --sheet-name <SheetName>
```

If it is unclear which sheet is the library, or which schema table a sheet maps to, ask the user rather than guessing.

### 3. Stage the DatasetSchema scaffold

Create the per-dataset `DatasetSchema` table — one row per feature space — in each `<dataset>/lance_db/`. The script reads `collection.json` for every dataset's `dataset_uid` and feature spaces and builds the rows through the schema's dataset class, so the columns and types come straight from the schema:

```
python scripts/stage_dataset_table.py <collection_root> \
  --schema <path/to/schema.py> \
  [--dataset NAME]
```

The scaffold creates only the columns known at staging; every other column is added by the step that fills it:

- `dataset_uid` ← `collection.json` (the same value `set_dataset_uid` later broadcasts onto obs rows) and `feature_space` ← each space present in the dataset. These are the only columns the scaffold creates.
- `zarr_group` and other automatic columns ← finalization (like `uid`).
- accession codes and dataset description ← **schema-harmonization**.
- publication registry keys ← **publication harmonization** (future; not schema-harmonization).
- `SummaryField` aggregates (`n_rows`, `organism`, …) ← ingestion.

### 4. Stage publication tables

Only run this step when the schema defines collection-level publication registry tables **and** the collection has a `publication.json` sidecar. Read the schema to decide which CamelCase table names apply.

The script reads `publication.json` and copies its fields into Lance as-is — only keys present in the JSON are written; staged columns usually will not yet match the schema. Top-level keys other than `text_data` go to a publication registry table. If there's also a table for storing publication sections as documents for search, then `text_data` will be used to fill its rows.

Because each collection has one publication, the script also seeds the join scaffolding `finalize-tables` expects:

- On the publication table: `{PubSchema}_join = 0` (target-side key).
- On the section table (publication + sections mode): `{field}_{PubSchema}_join = 0` for each `RegistryKeyField` on the section class that points at the publication table (discovered via `--schema`, or pass `--pub-fk-field` explicitly).

There are three modes:

| Mode | Arguments | What gets staged |
|------|-----------|------------------|
| Publication only | `--pub-schema` | One row: publication fields + `{PubSchema}_join = 0` |
| Publication + sections | `--pub-schema` + `--pub-section-schema` | One publication row (with target join) plus section rows (`section_title`, `section_text`, referencing join columns) |
| Denormalized sections | `--pub-section-schema` only | One row per section with top-level publication fields repeated on each row (no join columns — no separate publication table) |

```
python scripts/stage_publication_tables.py <collection_root> \
  --schema <path/to/schema.py> \
  --pub-schema PublicationSchema \
  [--pub-section-schema PublicationSectionSchema]
```

Use `--publication-json` only when the file is not listed in `collection.json` or not at the usual `other_files/publication.json` path.

Note that we do not currently support storing figures and captions in tables, as these are not included in the `publication.json` payload. If a schema includes fields or tables for these, then raise this limitation to the user.

## Scripts

| Script | Usage | Purpose |
|--------|-------|---------|
| `scripts/stage_lance_tables.py` | `python scripts/stage_lance_tables.py <collection_root> --schema <schema.py> [--parse-mode runtime] [--obs-class NAME]` | Stage OBS/VAR into per-dataset `lance_db/` |
| `scripts/stage_library_table.py` | `python scripts/stage_library_table.py <collection_root> --library <file> --table <SchemaClassName> [--sheet-name SHEET]` | Stage one LIBRARY file into collection `lance_db/` |
| `scripts/stage_dataset_table.py` | `python scripts/stage_dataset_table.py <collection_root> --schema <schema.py> [--dataset NAME]` | Stage the per-dataset `DatasetSchema` scaffold (one row per feature space) into `<dataset>/lance_db/` |
| `scripts/stage_publication_tables.py` | `python scripts/stage_publication_tables.py <collection_root> [--pub-schema NAME] [--pub-section-schema NAME] [--schema schema.py] [--pub-fk-field FIELD] [--publication-json PATH]` | Stage `publication.json` into collection `lance_db/` with join scaffolding (requires at least one schema argument) |
