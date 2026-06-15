# Dataset table resolution

Every dataset directory carries a `DatasetSchema` table, staged with **one row per feature space**. Its identity columns (`dataset_uid`, `feature_space`) are already filled from `collection.json`. Harmonization fills the table's *descriptive* metadata and leaves automatic, publication-link, and ingestion-deferred columns alone.

The `DatasetSchema` row describes the dataset as a whole, so the same dataset-level value is written to every per-feature-space row. Its columns fall into three groups, each handled differently.

## 1. Descriptive metadata — fill it

The free-text and accession-style descriptors of the dataset — for example an accession database and id, and a dataset description — come from the dataset's own metadata: the collection/dataset manifest, an accession record (e.g. a GEO series or sample), or other coalesced package files. Fill them with audited ops, under the same discipline as any nullable field.

## 2. Automatic and summary columns — leave them

Two kinds of column on this table are not harmonization's to fill:

- **Automatic columns.** `dataset_uid` is stamped from `collection.json` at staging; `zarr_group` is assigned by finalization. Both are deterministic — no decision or source to record — so do not write them.
- **Summary columns.** A field the schema marks with `SummaryField` is an aggregate of a target table's column — for example a row count, or unique-value rollups of obs columns such as organism or tissue. These are computed at ingestion time, **after** the obs rows are final, not during harmonization — the staged scaffold does not even create them. Do not add or fill a `SummaryField`-marked column here; the marker in the schema is the signal to skip it.

## Rules

- Fill descriptive metadata; never fill `dataset_uid`, `zarr_group`, or any `SummaryField`-marked column.
- Write the same dataset-level value to every per-feature-space row of the table.
- Do not record publication registry join keys here — one publication per collection makes that a separate concern that is handled automatically.
- A `SummaryField`-marked column is filled downstream at ingestion; skipping it here is correct, not an omission.
- Apply every fill as an audited transaction, like all harmonization.
