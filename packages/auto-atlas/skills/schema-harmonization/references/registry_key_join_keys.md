# Registry key join keys

A `RegistryKeyField` / `PolymorphicRegistryKeyField` links a row to a row in a *target_schema* table by that target's `uid`. Those `uid`s are assigned (often randomly) by `finalize-tables` after the whole collection is harmonized, so harmonization **does not fill the uid column**. What harmonization *does* own is recording **how the two tables connect** — the natural join key — as a standardized column that finalization later resolves to `uid`s.

This split exists because the join key is a *decision* that needs the raw-data context harmonization has, while the fill is a mechanical join finalization can run once every target has `uid`s. Identify and name the key here; let finalization apply it.

## What a natural join key is

A natural (business) key is a column, or small set of columns, whose value identifies the same entity in both the referencing table and the target table — a donor sample ID, a reagent ID, a guide-pair ID, an accession. It is whatever the data already used to mean "this row and that row are the same thing".

Two facts shape the search:

- **It may not be a schema field.** Often the natural key will be a column in the original dataframes that are not part of the schema. 
- **It lives on both sides.** You need a key present in the referencing table *and* a matching key in the target table. Confirm both before committing.

## Identifying the key

1. **Start from the target's identity.** What does a single target row uniquely represent, and which column(s) carry that identity? That column is the target-side key candidate.
2. **Find its counterpart in the referencing table.** Look across all columns for the value vocabulary that matches the target's identity. It may be encoded differently (a delimited pair, a prefixed string, a different case) — that is fine if a *deterministic* transform makes them equal (below).
3. **Mind cardinality.** A scalar FK takes one key per row; a list FK (e.g. combinatorial perturbations) takes a list of keys per row, which may require reshaping a delimited cell or a column family into a list first (see `ExplodeColumn` / `WideToLong` ops).
4. **Polymorphic FKs split by variant.** A `PolymorphicRegistryKeyField` targets several schemas; identify and record a separate join key per variant target, each against its own registry.

## Validating the key

A key is only valid if the join is **unambiguous and complete**. Check both before naming it — these checks are what keep the link correct; do not substitute fuzzy or approximate matching for a key that does not actually align.

- **Unique in the target.** Each target-side key value must map to exactly one target row (one `uid`). If the key repeats across target rows, it is not an identity — choose a more specific column or combine columns until it is unique.
- **Full coverage from the source.** Every non-null referencing key value should match exactly one target key value. Compute the match rate against the target's distinct keys before committing. **Investigate unmatched values** — a formatting difference to normalize, a subset that uses a different identifier, or genuinely missing entries — rather than accepting a partial join. A key that leaves rows unmatched is the wrong key (or needs normalization), not an acceptable approximation.
- **Normalize only with deterministic transforms.** Case-folding, trimming, type coercion, splitting a delimiter, stripping a prefix — recorded as ordinary harmonization ops — are fine because they are exact and auditable. Equality after such a transform is a real match. Approximate/semantic matching is not supported: if values do not align under a deterministic transform, the key is wrong.

If no valid join key exists and the registry key cannot be left null, surface that to the user rather than recording a key that does not validate.

## Naming for downstream

Record the validated key as a standardized column so finalization discovers it by convention (it does not guess). Write these columns through the `CurationApplicator` like any other harmonization op — recording the key *is* a curation decision, with a `reason` and `tool` (e.g. `"join_key"`) — using `AddColumn` / `RenameColumn` / `SetColumn` to land the key in the right shape.

- **Referencing table:** `{field_name}_{target_schema}_join` — holds the natural-key value(s) that identify the target row(s), in the registry key's cardinality (scalar for a scalar FK, list for a list FK). For a `PolymorphicRegistryKeyField`, write one column per variant, each suffixed with that variant's target schema.
- **Target table:** `{target_schema}_join` — exposes the matching key. Finalization pairs `{field_name}_{target_schema}_join` on the referencing side with `{target_schema}_join` on the target side and equi-joins them.

Because the target carries a single `{target_schema}_join` column, every referencing table that links to it must record its key in the **same vocabulary**. When a referencing table's raw identifier is in a different vocabulary than the target's key, the deterministic transform that maps it into that vocabulary belongs here, on the referencing side, where the raw context lives.

Leave the value columns these keys reference untouched otherwise — finalization reads the `*_join` columns, resolves them to `uid`s, fills the registry key fields, and drops the `*_join` columns as part of finalizing the table.
