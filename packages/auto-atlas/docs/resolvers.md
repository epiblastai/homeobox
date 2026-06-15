# Resolvers

Harmonization repeatedly needs to turn messy, free-text values into canonical identifiers: a gene symbol like `brca2` into an Ensembl ID, a cell-type string into a Cell Ontology term, a compound name into a PubChem CID, a guide sequence into genomic coordinates. The **resolver pipeline** is the shared machinery that does this. Every resolver follows the same staged flow — normalize, look up locally, disambiguate, enrich, fall back to external APIs, cache — so all entity types behave consistently and cache to the same LanceDB reference database.

Resolvers never mutate your tables. They return a `ResolutionReport`; the [curation](curation.md) layer is what applies the resulting changes under audit.

## The pipeline

`ResolverPipeline` composes a fixed sequence of stages. Each public entrypoint (`resolve_genes`, `resolve_cell_types`, …) is a thin wrapper that wires up the stages appropriate to its entity and calls `resolve(values, ...)`. A request flows through:

1. **Preprocess** — normalize each input value (case, whitespace, simple cleanup) into a lookup key.
2. **Deduplicate** — collapse duplicate keys, keeping a map back to the caller's original strings and order.
3. **Prescan fallbacks** — short-circuit values that should never hit the database (e.g. perturbation controls, hardcoded conventions).
4. **Local lookup** — a single batched query against the LanceDB reference tables (the cache).
5. **Disambiguate & build** — pick the best candidate among matches and build a typed resolution object; canonical aliases win over non-canonical.
6. **Enrich** — a batch second pass that fills secondary fields (e.g. attaching a symbol or protein metadata to an ID).
7. **Fallbacks** — for cache misses only, cascade to external APIs in order.
8. **Cache write** — persist freshly resolved values back into LanceDB so the next run is a local hit.
9. **Re-expand & report** — map unique results back to the caller's input order and aggregate statistics into a `ResolutionReport`.

The stages are defined as small protocols — `Preprocessor`, `LocalLookup`, `Disambiguator`, `ResultBuilder`, `Enricher`, `Fallback`, `CacheSink` — so a new entity type is assembled from these parts rather than written from scratch. The shared parts live in `auto_atlas.resolvers` (e.g. `AliasLookup`, `CanonicalAliasDisambiguator`).

A resolution carries the chosen identifier, a `confidence` (1.0 for an exact local hit, lower for fuzzy or API matches, 0.0 for unresolved), a `source` recording provenance (`lancedb`, `pubchem`, `reference_db_fts`, …), and `alternatives` — viable candidates not chosen. A result is **ambiguous** when `alternatives` is non-empty. Unresolved inputs are returned as stubs (`resolved_value=None`, `confidence=0.0`, `source="none"`) rather than dropped, so every input maps to an output.

## The reference database (LanceDB cache)

Resolution is backed by a LanceDB **reference database** — a set of tables that double as the lookup source and the cache. Local lookup is a single batched query against these tables (chunked into ~500-key `IN` queries), which is why bulk resolution is fast and mostly offline. The tables include:

| Table | Holds |
|-------|-------|
| `organisms` | Organism metadata (common/scientific name, Ensembl prefix) |
| `genomic_features` / `genomic_feature_aliases` | Gene records and their aliases |
| `proteins` / `protein_aliases` | Protein records and aliases |
| `compounds` / `compound_synonyms` | Molecule records and synonyms |
| `ontology_terms` | Unified ontology terms (cell types, tissues, diseases, …) |
| `cell_lines` / `cell_line_synonyms` | Cellosaurus cell-line records and aliases |
| `guide_rnas` | Guide-RNA resolution cache |

Ontology and cell-line lookups load their tables into in-memory name/synonym indices (cached with `lru_cache`) for fuzzy matching; the others query LanceDB directly.

The `guide_rnas` table is a true **read-write cache**, including a negative cache: guide resolution runs an expensive BLAT alignment, so both successful coordinates **and** misses (`resolved_value=None`) are written back, so a sequence is never aligned twice. After any resolution with external fallbacks, newly resolved values are written back to the reference DB via the pipeline's cache-write stage; a `CacheSink` returning `None` for a record skips persisting it.

## Fallback APIs

When a value misses the local cache, resolvers cascade to external services in a defined order, then cache the result. Calls are wrapped by `auto_atlas._rate_limit` (a per-endpoint token bucket with exponential-backoff retry on HTTP 429/503):

| Entity | Local source | Fallback order |
|--------|-------------|----------------|
| Genes | `genomic_features` / aliases | *(none — local only)* |
| Proteins | `proteins` / aliases | *(none — local only)* |
| Molecules (name) | `compound_synonyms` | PubChem → ChEMBL |
| Molecules (smiles/cid) | — | RDKit canonicalize → PubChem |
| Guide RNAs | `guide_rnas` cache | UCSC BLAT → Ensembl overlap annotation |
| Cell lines | `cell_lines` / synonyms | LanceDB full-text search |
| Ontology terms | `ontology_terms` indices | *(in-memory fuzzy match)* |

Default rate limits are set per endpoint (e.g. PubChem 5/s, Ensembl 15/s, UniProt 10/s, mygene 10/s, OLS 10/s, NCBI 3/s, UCSC BLAT 1/s). External libraries used include `pubchempy`, `gget` (BLAT), `mygene`, and REST calls to Ensembl, ChEMBL, OLS, and Cellosaurus.

## Resolver entrypoints

Each function takes a `list[str]` of values and returns a `ResolutionReport`:

| Function | Resolves to | Notes |
|----------|-------------|-------|
| `resolve_genes(values, organism="human", input_type="auto")` | Ensembl gene ID (+ symbol, NCBI ID) | `input_type` routes symbol vs. Ensembl-ID lanes; `auto` detects per value. |
| `resolve_proteins(values, organism="human")` | UniProt ID (+ name, gene, sequence) | Alias lookup with metadata enrichment. |
| `resolve_molecules(values, input_type="name")` | PubChem CID (+ SMILES, InChIKey, …) | `input_type` is `name`, `smiles`, or `cid`. |
| `resolve_guide_sequences(sequences, organism="human")` | Genomic coordinates + intended gene | BLAT + Ensembl overlap, cached in `guide_rnas`. |
| `resolve_cell_types(values)` | `CL:` term | |
| `resolve_tissues(values)` | `UBERON:` term | |
| `resolve_diseases(values)` | `MONDO:` term | |
| `resolve_organisms(values)` | `NCBITaxon:` term | |
| `resolve_assays(values)` | `EFO:` term | |
| `resolve_cell_lines(values)` | Cellosaurus ID (+ species, disease, …) | |

The ontology entrypoints are wrappers over a shared `resolve_ontology_terms(values, entity, ...)` that dispatches on the entity type.

## Registries

`auto_atlas/registry.py` defines the **authorities** that resolution targets and binds each one to the resolver that fills it. Two enums name the authorities:

- **`OntologyRegistry`** — ontology prefixes that share the unified `ontology_terms` table: `CL` (cell types), `UBERON` (tissues), `MONDO` (diseases), `NCBITaxon` (organisms), `EFO` (assays), `HANCESTRO` (ancestry), `HsapDv`/`MmusDv` (developmental stage).
- **`CrossReferenceDbRegistry`** — external identifier authorities: `ENSEMBL`, `GENCODE`, `NCBI Gene`, `UniProt`, `PubChem`, `Cellosaurus`, `ChEMBL`, `InChI`, plus reference-only ones (`DOI`, `PubMed`, `GenBank`, `RefSeq`, …) that have no resolver.

These map a schema field's declared authority to the right resolution behavior. `ONTOLOGY_BINDINGS` and `CROSSREF_BINDINGS` associate each registry value with a `ResolverBinding` — which resolver `tool` to call, which result field holds the identifier, extra resolver kwargs (e.g. `input_type="ensembl_id"` for `ENSEMBL`), whether organism context is required, and the resolution `mode` (`single` for automatic one-column passes, `custom` for manual handling, `none` for authorities with no resolver). `RESOLVER_TOOLS` is the registry of callable resolver functions keyed by name.

This binding is what lets harmonization auto-discover resolvable fields from a schema: the `apply_resolution_pass.py --from-schema` mode reads a table's `OntologyAlignedField` / `CrossReferenceField` markers, looks up each field's authority in these bindings, and runs the bound resolver — one pass per resolvable field. See [Curation](curation.md) for how the resulting resolutions become audited table mutations.
