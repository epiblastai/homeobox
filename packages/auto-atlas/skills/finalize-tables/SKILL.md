---
name: finalize-tables
description: Finalize harmonized Lance tables in a data package — assign automatic columns (uid, dataset_uid, derived fields), connect tables by populating RegistryKeyFields, and validate every table against its target homeobox schema.
---

# Finalize tables

This skill runs *after* `schema-harmonization` has finished on **every** table in a collection (and *after* `multimodal-alignment` on multimodal datasets). Harmonization aligns columns and resolves values (each change applied through the audited `CurationApplicator`). For multimodal datasets, finalization first **joins** per-feature-space obs tables into one table named after the obs schema class, then fills the deterministic, automatic columns, **connects the tables** by resolving registry keys, and validates that each table is consistent with its schema for the columns present.

Finalization is the step that turns a set of independently-harmonized tables into a linked collection, so it operates on the **whole collection at once**, not a single table in isolation.

## Inputs

- A **collection root** containing harmonized tables: collection-level registries/registry-key targets in `<collection_root>/lance_db/`, and per-dataset obs/var in `<dataset_dir>/lance_db/`. Every table should have been harmonized to its schema aside from the fields handled by this skill.
- The **target homeobox schema file**. Each table name corresponds to one of its schema classes.
- The `collection.json` manifest (feature spaces, dataset uids).
- **Standardized join columns** written by harmonization (see below) that record how each registry key is linked.

## Output

For every table: all automatic columns assigned, all registry keys populated, transient join/leftover columns removed, and the table validated against its schema class. The collection will be internally linked and ready for ingestion.

Finalization materializes **every** schema field as a column except those deferred to ingestion. A field whose value is unknown is fine to leave empty — a nullable field is null-initialized rather than dropped, because the schema default does not excuse the column from existing: downstream writers expect it. The only fields allowed to be absent are those filled at **ingestion** time: `PointerField` modality pointers (and their `has_<pointer>` flags), `SummaryField` aggregates (e.g. `n_rows`, dataset-level rollups of obs metadata), and `global_index`. Validation checks that present values conform to the schema; it does not require those deferred columns to exist yet.

## Responsibilities (per table, in order)

1. **Assign the automatic row key.** For `StableUIDBaseSchema` tables (var / feature / registries) compute `uid` via `cls.compute_stable_uids(df)`. For other tables that declare a `uid` field (incl. `HoxBaseSchema` obs tables) assign random `make_uid()`. `DatasetSchema` tables key on `zarr_group` instead of `uid` — assign it a random `make_uid()` (one modality write per feature-space row), creating the column since the staging scaffold omits it. Skip tables with neither key.
1b. **Produce the per-feature-space `uid` artifact** — every feature space of every dataset ends with a `{obs_class}_{feature_space}` table whose `uid` column is in DATA-file row order, so ingestion aligns DATA rows to obs positions through one uniform lookup. For **multimodal** datasets the suffixed tables already exist (kept by the join); copy each joined obs row's `uid` onto the matching `{obs_class}_{feature_space}` row via `multimodal_barcode`. For **single-modality** datasets (staged bare, no suffixed table) materialize `{obs_class}_{feature_space}` from the bare obs `uid` column in its DATA-ordered order. Either way no new schema fields are introduced.
2. **Stamp `dataset_uid`** on obs tables (one constant per dataset; details below).
3. **Populate registry keys** — only after every *target_schema* table already has its `uid` assigned (below).
4. **Run `compute_auto_fields`** for `HoxBaseSchema` (obs) tables — fills derived columns. Must run *after* registry keys, because some derived columns may depend on them.
5. **Clean up** — two kinds of column removal, by who owns the column:
   - **`*_join` scaffolding** (finalization's own transient handoff columns) is dropped **directly to Lance**, unaudited: the referencing-side `{field}_{target}_join` columns as each registry key is filled, and the target-side `{target}_join` columns once the whole collection's registry keys are resolved (a collection-level step, since one target column is shared by every referrer).
   - **Non-schema leftovers** (original *source* columns never mapped to a schema field) are dropped through the **audited** `CurationApplicator.DropColumn` — removing source data is recorded, never silent. This runs last, after the `*_join` cleanup and the whole DAG pass, so only genuine leftovers remain.

6. **Ensure schema columns** — null-initialize any non-deferred schema field that harmonization never materialized (nullable fields with Python defaults are easy to miss but still required as columns for downstream writers). Skips pointer fields, ``has_<pointer>`` flags, ``SummaryField`` aggregates, and ``global_index``. Runs after cleanup so only genuine schema gaps remain.
7. **Validate** — constructs each row against the target schema class using the columns present (schema defaults apply to absent fields). Any value-level non-conformance is a hard error, not a silent fix. Missing ingestion-deferred columns are expected and are not an error.

## Whole-collection order is a DAG

Registry keys impose a dependency order: a target table must have its `uid` assigned before any table referencing it can be filled. Enumerate every `RegistryKeyField` / `PolymorphicRegistryKeyField` from the schema (the `target_schema` is declared on the field) to build the reference graph, then finalize tables in dependency order: targets first, referencing tables last. The order is derived from the schema, not hard-coded.

A single entrypoint drives the whole pass:

```bash
python scripts/finalize_collection.py <collection_root> --schema <schema.py> --dry-run
```

It joins multimodal obs tables (when present), resolves the DAG, runs the per-table steps in order, and reports what each step changed. The individual scripts below can also be run table-by-table for debugging.

### Joining feature-space obs tables

Multimodal datasets stage one obs table per feature space (`CellIndex_gene_expression`, …). After `multimodal-alignment` has written `multimodal_barcode` and harmonization has run on those tables, join them into a single obs table named after the schema class (`CellIndex`). The join is an outer merge on `multimodal_barcode`; `multimodal_barcode` must be **unique** in every source table and in the joined result (duplicate barcodes fail loud). Overlapping columns are coalesced (conflicting non-null values fail loud). Per-feature-space source tables are **kept** in Lance — they are not finalized, but preserve staged row order for DATA alignment at ingestion. Single-modality datasets are skipped.

After `assign_uids` on the joined obs table, `stamp_uid_on_feature_space_obs.py` copies each row's `uid` onto the matching feature-space table(s) by `multimodal_barcode`. For single-modality datasets it instead materializes `{obs_class}_{feature_space}` from the bare obs `uid` column (in staged DATA order), so the per-feature-space artifact exists uniformly. At ingestion, map each emitted DATA row `i` to the 0-based position, in the bare obs table, of `uid[i]` from `{obs_class}_{feature_space}`.

`finalize_collection.py` runs this automatically for every obs class in the schema (or one named with `--obs-class`). Run it standalone to inspect or dry-run first:

```bash
python scripts/join_feature_space_obs.py <collection_root> --obs-class CellIndex --dry-run
```

Table discovery during finalization matches **exact** schema class names only — not `_{feature_space}` suffixes. Harmonization still runs on the suffixed tables before the join. Suffixed tables do not pass through the rest of finalization (registry keys, validation, leftover drop); they only receive a stamped `uid` for ingestion lookup.

## Stamping `dataset_uid`

`dataset_uid` is the one automatic field set upstream within a Collection. Unlike `uid` (per-row) it is one constant for the whole dataset — the `Dataset.uid` assigned at creation and persisted in `collection.json` — so it is an auditable broadcast `AddColumn`. Stamping it links every obs row to its dataset record.

```bash
python scripts/set_dataset_uid.py \
  <collection_root> --dataset HepG2 --obs-class CellIndex --dry-run
```

Pass the obs **schema class name** — the concrete Lance table name (`CellIndex`). For multimodal datasets, run `join_feature_space_obs.py` first so feature-space tables are merged into that bare name. Only obs tables carry `dataset_uid`.

## Populating RegistryKeyField

A registry key links a row to a row in a *target_schema* table by that target's `uid`. The target row was assigned a `uid` (often random) during finalization, so the link cannot be recomputed from content — it must be resolved by **joining on a natural key** shared between the two tables.

### The join-column handoff (written by harmonization)

The natural key that links a registry key to its target is almost always some original column — but not necessarily a column that survives in the standardized schema, and the best key on either side is often a leftover that would otherwise be dropped. So harmonization, which has the most context about the raw data, is responsible for **recording the join key as a standardized column** rather than finalization rediscovering it:

- On the **referencing** table, harmonization writes a column named `{field_name}_{target_schema}_join` holding the natural-key value(s) that identify the target row(s), in the registry key's cardinality (a scalar for a scalar registry key, a list for a list registry key). It renames an existing column or adds one as needed, including any reshaping (splitting a delimited pair, exploding a combinatorial cell) so the join column already has the right shape.
- On the **target_schema** table, harmonization exposes the matching key under the same convention so the two can be equi-joined.

**Exception — one publication per collection.** The publication registry is staged with `{PubSchema}_join = 0` on the single publication row; the section table (when present) gets its referencing join at staging too. Other referencers (e.g. `publication_uid` on a `DatasetSchema` table) do not need harmonization to record a key — `populate_registry_keys.py` seeds `{field}_{PubSchema}_join = 0` on those tables before the usual fill, then the join proceeds like any other registry key.

Finalization discovers `*_join` columns by naming convention; it does not guess natural keys for non-publication registries. If a registry key has no recorded join column and cannot be left null, surface it to the user rather than inventing a link.

### Resolving and filling

```bash
python skills/finalize-tables/scripts/populate_registry_keys.py <collection_root> \
  --schema <schema.py> --table CellIndex --dry-run
```

For each registry key on the table the script:

0. **Publication referencers (collection-wide, once per run).** When a publication registry target is present (auto-detected from `{PubSchema}_join = 0` on the target table, or named with `--publication-schema`), seed any missing `{field}_{PubSchema}_join = 0` columns on tables that declare a scalar `RegistryKeyField` to that target.
1. Reads the referencing `*_join` column and the matching key on the (already uid-assigned) target table.
2. Equi-joins natural key → target `uid`, mapping each key to exactly one target row.
3. **Verifies coverage**: the join key is unique in the target, and every non-null source key matches exactly one target row. It reports matched / unmatched / total and **fails loud on any unmatched key** — never silently nulls (consistent with harmonization's nullable-field rule). Unmatched keys are investigated, not dropped.
4. Writes the resolved `uid`(s) into the registry key column (scalar, list, or — for `PolymorphicRegistryKeyField` — merged across the per-variant join columns into the aligned lists), directly to Lance.
5. Drops the transient `*_join` column once the fill is verified.

## Scripts

| Script | Input | Function |
|---|---|---|
| `finalize_collection.py` | collection root, schema | DAG-ordered entrypoint; joins multimodal obs tables, runs every step on every table in dependency order, then the target-join cleanup, the audited leftover-column drop, and the validation sweep. |
| `join_feature_space_obs.py` | `lance_db` or collection root, `--obs-class` | Outer-join per-feature-space obs tables on `multimodal_barcode` into the bare obs class name; require unique barcodes; keep suffixed source tables. |
| `stamp_uid_on_feature_space_obs.py` | `lance_db` or collection root, `--obs-class` | Produce the per-feature-space `uid` artifact for ingestion DATA row lookup: multimodal — stamp `uid` onto each existing `{obs_class}_{feature_space}` table via `multimodal_barcode`; single-modality — materialize `{obs_class}_{feature_space}` from the bare obs `uid` in DATA order. |
| `assign_uids.py` | collection root, schema, optional `--table` | Assign each table's automatic key — `uid` (stable vs random per the schema declaration), or `zarr_group` for `DatasetSchema` tables; idempotent — preserves existing keys. |
| `set_dataset_uid.py` | collection root, `--dataset`, `--obs-class` | Stamp the dataset's `uid` onto its obs table(s) as an audited broadcast. |
| `populate_registry_keys.py` | collection root, schema, optional `--table`, optional `--publication-schema` | Seed publication referencing join columns (`0`), resolve `*_join` columns against target `uid`s, verify coverage (fail-loud on any unmatched key), fill scalar and position-aligned polymorphic registry-key columns, drop the referencing-side join columns. |
| `drop_leftover_columns.py` | collection root, schema, optional `--table` | Drop every column that is not a field of its schema class through an audited `DropColumn` (source data; recorded, not silent). Run after registry keys and the `*_join` cleanup. |
| `ensure_schema_columns.py` | collection root, schema, optional `--table` | Null-init any missing non-deferred schema column so finalized tables carry the full declared column set. |
| `validate_tables.py` | collection root, schema, optional `--table`, `--limit` | Structural + per-row validation of every table against its schema class; exits non-zero and reports any column or value that does not conform. |

Shared logic (schema loading via `homeobox.parser`, table discovery, dependency ordering, Arrow read/mutate/overwrite) lives in the main library: the `SchemaInfo` / `TableRef` dataclasses in `auto_atlas.types` and the functions in `auto_atlas.util`. Table discovery matches Lance table names to schema classes **by exact name** — feature-space suffixes are a staging/harmonization convention only, resolved by `join_feature_space_obs.py` before finalization discovers tables. Registry keys are described by homeobox's own `RegistryKeyField` / `PolymorphicRegistryKeyField` markers, not local copies. The deterministic columns (uid / zarr_group / derived / registry-key fills) and finalization's own `*_join` scaffolding are written directly to Lance; the two writes that touch source-derived data — `set_dataset_uid` and `drop_leftover_columns` — go through the `CurationApplicator`.
