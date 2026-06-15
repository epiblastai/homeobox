# auto-atlas

auto-atlas automates the ingestion of public datasets — unimodal and multimodal, across many modalities — into [homeobox](https://epiblast.ai/homeobox/) atlases.

Turning a public dataset into a finalized atlas means downloading and organizing files, aligning their ad-hoc columns and values to a target schema, and streaming the matrices into storage. auto-atlas provides:

- **Standardized data packages** — the [Collection API](collections.md) for assembling related datasets into a tagged, serialized `collection.json` package, the unit the rest of the pipeline operates on.
- **Schema alignment with resolution** — a [resolver pipeline](resolvers.md) that maps raw values to canonical identifiers: ontology terms (cell types, tissues, diseases), registries, and external database cross-references (Ensembl, UniProt, PubChem, Cellosaurus, …), cached in a LanceDB reference database with external-API fallbacks.
- **Auditable curation** — every table mutation flows through a [curation system](curation.md) that records each transformation of the original obs and var tables, with full provenance, in an audit log.
- **Agent skills** — a set of skills that guide an agent through the [end-to-end workflow](workflow.md) systematically, one stage at a time.

## Where to start

- **[Workflow](workflow.md)** — the end-to-end pipeline, the skill for each stage, the scripts, and the on-disk layout. Start here for the big picture.
- **[Collections](collections.md)** — the Collection API: tagging files, building datasets, coalescing, and the `collection.json` manifest.
- **[Resolvers](resolvers.md)** — the resolver pipeline, the LanceDB reference cache, fallback APIs, and the registries that bind schema fields to resolvers.
- **[Curation](curation.md)** — the curation operations, the applicator, and the audit log of transformations.
