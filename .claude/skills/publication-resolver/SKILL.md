---
name: publication-resolver
description: Fetch publication metadata (title, DOI, journal, date) from PubMed and full text sections from PMC for PublicationSchema and PublicationSectionSchema. Accepts PMID, DOI, or paper title. Use when a dataset needs publication metadata for its publication.json sidecar or when populating publication records in a LanceDB table.
---

# Publication Resolver

Fetch publication metadata from PubMed and full text from PMC Open Access. Populates `PublicationSchema` and `PublicationSectionSchema` records for downstream ingestion.

## Interface

**Input:** A publication identifier — PMID (numeric), DOI, or paper title.

**Output:** A `publication.json` file and/or populated schema fields for `PublicationSchema` + `PublicationSectionSchema`.

## Scripts

Run these via Bash. All paths are relative to this skill directory.

| Script | Usage | Purpose |
|--------|-------|---------|
| `scripts/write_publication_json.py` | `python scripts/write_publication_json.py <data_dir> [--pmid PMID] [--title TITLE]` | Fetch publication metadata from PubMed/PMC and write publication.json |

The script supports three modes:

1. **From metadata.json** (default): reads `<data_dir>/metadata.json` and extracts PMIDs from `series_metadata.pmids` or resolves GSE accessions.
2. **`--pmid PMID`**: fetch metadata for a specific PubMed ID.
3. **`--title TITLE`**: search PubMed by title, then fetch metadata.

## Imports

```python
from lancell.standardization import (
    fetch_publication,
    fetch_publication_text,
    fetch_publication_metadata,
    search_pubmed_by_title,
    PublicationMetadata,
    PublicationFullText,
    PublicationSection,
)
```

## Workflow

### 1. Identify the publication

Determine the identifier type. `fetch_publication()` auto-detects:
- Pure digits or `PMID:` prefix → PMID
- Starts with `10.` or contains `/10.` → DOI
- Otherwise → title search

```python
pub = fetch_publication("31806696")         # by PMID
pub = fetch_publication("10.1016/j.cell.2017.10.049")  # by DOI
pub = fetch_publication("Massively multiplex chemical transcriptomics")  # by title
```

### 2. Fetch metadata

`fetch_publication()` returns a `PublicationMetadata` dataclass:

```python
pub = fetch_publication(identifier)
print(f"PMID: {pub.pmid}")
print(f"DOI: {pub.doi}")
print(f"Title: {pub.title}")
print(f"Journal: {pub.journal}")
print(f"Date: {pub.publication_date}")
print(f"Authors: {pub.authors}")
print(f"PMC ID: {pub.pmc_id}")
```

### 3. Fetch full text (if needed by schema)

Only fetch full text if the target schema includes `PublicationSectionSchema`. If the schema only has `PublicationSchema`, the metadata from step 2 is sufficient.

```python
text = fetch_publication_text(pub.pmid, pub.pmc_id)
print(f"Source: {text.source}")  # "pmc" or "abstract_only"
for section in text.sections:
    print(f"  [{section.section_title}] {section.section_text[:100]}...")
```

PMC full text is attempted first. If the article is not in PMC Open Access, the abstract is returned as a single section (or multiple sections for structured abstracts with labeled parts like "Background", "Methods", "Results").

### 4. Map to schema

#### PublicationSchema

```python
from datetime import datetime

schema_record = {
    "doi": pub.doi or "",       # Required field — almost always present
    "pmid": pub.pmid,           # int | None
    "title": pub.title,         # Required field
    "journal": pub.journal,     # str | None
    "publication_date": pub.publication_date,  # datetime | None
}
```

#### PublicationSectionSchema (one row per section)

```python
for section in text.sections:
    section_record = {
        "publication_uid": publication_uid,  # FK from PublicationSchema
        "section_text": section.section_text,
        "section_title": section.section_title,
    }
```

### 5. Write publication.json (for CLI workflow)

When working within a data preparation pipeline, use the script:

```bash
python scripts/write_publication_json.py /tmp/geo_agent/GSE123456 --pmid 31806696
```

This writes a flat `publication.json` with keys: `pmid`, `doi`, `title`, `journal`, `publication_date`, `full_text`.

For programmatic access (e.g., in ingestion scripts), use `fetch_publication_metadata()` which returns the same dict:

```python
pub_dict = fetch_publication_metadata("31806696")
# {"pmid": 31806696, "doi": "10.1016/...", "title": "...", ...}
```

## Rules

- **PMC before abstract.** Always attempt PMC full text before falling back to abstract. Many biology papers have PMC full text (~40% of PubMed-indexed papers).
- **DOI is best-effort.** If no DOI is found in the PubMed record, the field will be None. Do not fail on missing DOI.
- **Multiple PMIDs.** For datasets with multiple associated publications (multiple PMIDs in metadata.json), process each one. The shared script uses the first PMID by default.
- **Identifier auto-detection.** `fetch_publication()` auto-detects PMID vs DOI vs title. No need to classify the identifier manually.
- **PubMed-only.** Resolution goes through PubMed. Papers not yet indexed in PubMed (e.g., very recent preprints with only a DOI) will fail with a clear error message.
- **Section titles for PMC.** PMC full text sections use the article's own headings (e.g., "Introduction", "Methods", "Results"). Nested subsections are flattened with `>` separators (e.g., "Methods > Cell Culture").
- **Structured abstracts.** Some PubMed abstracts have labeled sections (e.g., "Background", "Methods", "Conclusions"). These are returned as separate `PublicationSection` entries rather than a single block.
