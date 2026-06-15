# Publication table resolution

Collection-level publication tables live in `<collection_root>/lance_db/`. They are staged from `publication.json` by **prepare-package-for-resolution** — skip harmonization here if that step was not run.

Each collection has **one** publication row. A separate section table (when the schema defines one) holds one row per `text_data` entry. Staging already copied the JSON fields and seeded the `*_join` scaffolding (`{PubSchema}_join = 0` on the publication table; `{field}_{PubSchema}_join = 0` on the section table when applicable). Harmonization is mostly alignment, not enrichment.

## Publication table

One row. Staged columns typically match the schema field names already (`pmid`, `doi`, `title`, `journal`, `publication_date`, `authors`, `text_source`, …). Work is usually:

- **`RenameColumn`** — only when a staged name differs from the schema field.
- **`CastColumn`** — when a type must change (most often `publication_date`, staged as an ISO string from JSON).

Do not fill `uid`. Do not touch `{PubSchema}_join` — finalization reads it, resolves registry keys, and drops it.

## Publication section table

One row per text section (`section_title`, `section_text`). Same story: rename if needed, cast if needed. Do not fill `publication_uid` or the referencing `*_join` column.

In **denormalized** staging (section table only, publication fields repeated on each row), align those inline publication columns the same way — still no uid or join-key work.

## Rules

- Align column names and types; do not re-resolve metadata from PubMed. If you think it might be necessary, then this is an issue to raise to the user.
- Leave automatic and registry-key columns alone.
- Apply changes as audited transactions, like all harmonization.
