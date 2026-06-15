# Collections

A **data package** is a collection of related datasets organized in a single root directory, with every file tagged by its role. It is the unit auto-atlas standardizes and aligns: once a collection is assembled and serialized to a `collection.json` manifest, the downstream skills ([staging](workflow.md), [harmonization](curation.md), finalization, ingestion) all read that manifest to discover datasets, feature spaces, and the files that back them.

The `auto_atlas.collection` module provides the API for assembling a package. A collection is built in memory by registering datasets and files, then **coalesced** — files are physically organized into a predictable on-disk layout — and written to `collection.json`.

A `Collection` commonly corresponds to one publication or experiment; each `Dataset` within it corresponds to a sample, condition, or cell line that will become one logical dataset in the homeobox atlas.

## File type tags

Every file in a package is tagged with a `FileTypeTag`, which tells the rest of the pipeline how to treat it:

| Tag | Role | Feature space |
|-----|------|---------------|
| `OBS` | Row-level (cell/sample) metadata table aligned with a matrix's rows | Required |
| `VAR` | Feature-level metadata table aligned with a matrix's columns | Required |
| `DATA` | Array data — the matrices themselves (`h5ad`, `mtx`, `zarr`, `npy`, …) | Required |
| `LIBRARY` | Shared reference/lookup tables (guide, reagent, donor libraries) referenced by obs/var | Omit |
| `OTHER` | Free-form informational files (READMEs, protocols, publication text, join tables) | Omit |

Tags matter because the pipeline routes files by them. `prepare-package-for-resolution` stages `OBS`/`VAR`/`LIBRARY` files into LanceDB tables of the corresponding kind; ingestion streams `DATA` matrices into zarr; `OTHER` files are kept for reference but not staged.

Two conventions are worth remembering:

- **One OBS and one VAR per feature space.** Within a dataset, each `feature_space` may have at most one primary OBS and one primary VAR — the tables aligned with that modality's matrix.
- **Join tables go to `OTHER`, not `OBS`.** Extra tabular metadata that joins to the main obs on an ID column (clinical data, cell-type calls, QC metrics) is tagged `OTHER`. Only the primary row table for a feature space is `OBS`.

The `feature_space` argument (e.g. `"gene_expression"`, `"protein_abundance"`, `"chromatin_accessibility"`) is **required** for `OBS`/`VAR`/`DATA` and **omitted** for shared `LIBRARY` and `OTHER` files. It is how a single multimodal dataset keeps its modalities distinct.

## Datasets

A `Dataset` groups the files for one logical dataset and carries a stable `uid`:

```python
from auto_atlas.collection import Dataset, FileTypeTag

hepg2 = Dataset("HepG2")  # uid auto-generated if not passed
hepg2.add_file("gex.h5ad",     FileTypeTag.DATA, "gene_expression")
hepg2.add_file("gex_obs.csv",  FileTypeTag.OBS,  "gene_expression")
hepg2.add_file("gex_var.csv",  FileTypeTag.VAR,  "gene_expression")
```

| Member | Signature | Purpose |
|--------|-----------|---------|
| `Dataset(dataset_name, uid=None)` | constructor | Name the dataset; `uid` is auto-generated via homeobox `make_uid()` if omitted and preserved across JSON round-trips. |
| `add_file(file_path, tag, feature_space=None)` | `-> None` | Register a file with its tag and (for obs/var/data) feature space. Raises if the path is already added. |
| `files_for(tag=None, feature_space=None)` | `-> list[str]` | Query files by tag and/or feature space (`None` means "no constraint on that axis"). |
| `files` | property | All file paths in the dataset. |
| `feature_spaces` | property | Sorted distinct feature spaces across the dataset's files. |

The `uid` is the logical identity of the dataset. It is referenced by `dataset_uid` on every obs row at finalization, so it must stay stable — which is why it survives a `collection.json` round-trip.

A multimodal dataset is just one `Dataset` with files in more than one feature space:

```python
cite = Dataset("sample1")
cite.add_file("rna.h5ad",      FileTypeTag.DATA, "gene_expression")
cite.add_file("rna_obs.csv",   FileTypeTag.OBS,  "gene_expression")
cite.add_file("rna_var.csv",   FileTypeTag.VAR,  "gene_expression")
cite.add_file("adt.tsv",       FileTypeTag.DATA, "protein_abundance")
cite.add_file("adt_obs.csv",   FileTypeTag.OBS,  "protein_abundance")
cite.add_file("adt_var.csv",   FileTypeTag.VAR,  "protein_abundance")
```

## Building a collection

A `Collection` owns the root directory, the datasets, and any collection-level shared files:

```python
from auto_atlas.collection import Collection, Dataset, FileTypeTag

collection = Collection(root_dir="/data/GSE264667")
collection.add_dataset(hepg2)

# Collection-level files are shared across datasets — no feature space.
collection.add_file("guide_library.csv", FileTypeTag.LIBRARY)
collection.add_file("publication.json",  FileTypeTag.OTHER)

collection.coalesce(copy=True)   # organize files on disk
collection.to_json()             # write <root_dir>/collection.json
```

| Member | Signature | Purpose |
|--------|-----------|---------|
| `Collection(root_dir)` | constructor | Create (or attach to) the package root. |
| `add_dataset(dataset)` | `-> None` | Add a `Dataset`. Raises on duplicate dataset name. |
| `add_file(file_path, tag, feature_space=None)` | `-> None` | Add a collection-level shared file (`LIBRARY` or `OTHER`). |
| `coalesce(copy=True)` | `-> None` | Physically organize files into the on-disk layout; rewrites tracked paths. `copy=False` moves instead of copying. Idempotent. |
| `dumps()` | `-> str` | Return the manifest JSON. Raises if anything is not yet coalesced. |
| `to_json()` | `-> None` | Coalesce (if needed) and write `collection.json` into the root. |
| `Collection.from_json(path)` | classmethod `-> Collection` | Rehydrate a collection from a manifest; everything is marked coalesced so re-coalescing is a no-op. |
| `datasets` | property | List of dataset names. |

### Coalescing and the on-disk layout

Before `coalesce()`, files may live anywhere on the filesystem — their paths are tracked as given. `coalesce()` moves or copies each file into a predictable structure under the root and rewrites the tracked paths to match:

```
root_dir/
  collection.json          # manifest, written by to_json()
  HepG2/                    # one directory per dataset (by dataset_name)
    gex.h5ad
    gex_obs.csv
    gex_var.csv
  guide_library.csv        # shared LIBRARY files at the root
  other_files/             # OTHER files collected here
    publication.json
```

`coalesce()` is idempotent: already-coalesced datasets and files are skipped, and it checks for basename collisions before moving anything. `to_json()` calls it for you, so a normal build is just `add_*` calls followed by `to_json()`.

### Serialization

`collection.json` is the manifest the rest of the pipeline consumes. It records the root, the shared files, and each dataset's `dataset_uid` plus its tagged files:

```json
{
  "root_dir": "/data/GSE264667",
  "shared_files": [
    {"path": ".../guide_library.csv", "tag": "library", "feature_space": null}
  ],
  "datasets": {
    "HepG2": {
      "dataset_uid": "<stable-uid>",
      "files": [
        {"path": ".../HepG2/gex.h5ad",    "tag": "data", "feature_space": "gene_expression"},
        {"path": ".../HepG2/gex_obs.csv", "tag": "obs",  "feature_space": "gene_expression"},
        {"path": ".../HepG2/gex_var.csv", "tag": "var",  "feature_space": "gene_expression"}
      ]
    }
  }
}
```

Round-tripping is lossless — `Collection.from_json(path).dumps()` reproduces the manifest — and `from_json` raises if any dataset is missing its `dataset_uid`, since that identity must never be regenerated.

## Helper utilities

When a source ships a single `h5ad`, `auto_atlas.util.extract_h5ad_obs_var` writes its obs and var tables out as CSVs so they can be tagged separately:

```python
from auto_atlas.util import extract_h5ad_obs_var

obs_csv, var_csv = extract_h5ad_obs_var("GSE264667_HepG2.h5ad")
hepg2.add_file("GSE264667_HepG2.h5ad", FileTypeTag.DATA, "gene_expression")
hepg2.add_file(obs_csv, FileTypeTag.OBS, "gene_expression")
hepg2.add_file(var_csv, FileTypeTag.VAR, "gene_expression")
```

The `create-data-package` skill bundles additional scripts for fetching GEO/PubMed metadata and downloading supplementary files; see the [Workflow](workflow.md) page.
