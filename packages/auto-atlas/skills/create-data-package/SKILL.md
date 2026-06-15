---
name: create-data-package
description: Describes the creation of a data package, which is a collection of datasets organized in a directory as a prerequisite for agent-based data standardization and alignment. Provides guidelines for tagging files and a reference for using the Collection API.
---

# Data package creation

This skill covers downloading and organizing files only. It should not be relied upon to modify the contents of the files.

## Expected input

Before following the workflow described in this skill, you will need a link or path to a dataset. Often this will be a URL to a public database like the Gene Expression Omnibus (GEO), Sequencing Read Archive (SRA), Single Cell Portal (SCP), or Cell Painting Gallery (CPG). At the URL or path you should find files available for download, and possibly other metadata.

If the user does not provide a link or path, or if the link or path does not contain data files, then raise to the user and do not proceed with the workflow.

## The Collection API

The main tool you'll be using is the Collection API defined in `auto_atlas`. A `Collection` is a set of related datasets, commonly part of the same publication or experiment. A `Dataset` is a set of files that go together. Following AnnData, files that contain row-level metadata are called `OBS` and those that contain feature-level metadata are called `VAR`. `DATA` is reserved for files with array data, which can come in many different formats. At the collection level, there can be `LIBRARY` tables and `OTHER` files that are relevant to all datasets.

Each `(dataset, feature_space)` pair has **at most one** primary `OBS` and **at most one** `VAR` — the tables that align with that feature space's matrix rows and columns. Additional row-level metadata that joins to the main OBS on a shared ID (clinical annotations, batch labels, separate QC exports, etc.) should still be included in the package, but tagged as `OTHER`, not `OBS`.

## Worflow

### 1. Locate files

Given the URL or path provided by the user, identify the files to download. If ambiguous because the user did not specify a file name or there are many, then stop and ask the user for clarification. A particularly important case to watch out for and flag is if there are files with suffixed like `_validated`, `_processed`, `_filtered`, etc.

Currently, we support the following file formats (which may be in `.tar` files):

| Format | Action |
|--------|--------|
| `.csv` / `.tsv` / `.parquet` | Tabular data delimited or in a columnar format |
| `.xlsx` | Excel workbook; may hold several sheets  |
| `.npy` | Numpy array data |
| `.zarr` | Zarr array data |
| `.h5ad` | AnnData |
| `.h5` (10x HDF5) | Set `matrix_file` field; can be read with `scanpy.read_10x_h5()` for validation |
| `.mtx` / `.mtx.gz` (Market Matrix) | Set `matrix_file` field; companions go to `cell_metadata`/`var_metadata` |
| `.tsv` / `.tsv.gz` | Sometimes used for protein abundance which is not sparse |
| `_fragments.tsv.gz` / `.bed.gz` / `.bed` | Fragment files — per-cell chromatin accessibility regions. Columns: `(chrom, start, end, barcode)` (4-col) or `(chrom, start, end, barcode, count)` (5-col, 10x format) |

### 2. Download files

Using CLI tools or a provided script, download the file or files necessary. You may save them in a temp directory. If files are compressed into a bundle, decompress them.

### 3. Split files into tables and arrays

Before tagging files as `OBS`, `VAR`, and `DATA`, it may be necessary to split them into components.

Among the supported file types, `h5ad` packs `OBS`, `VAR`, and `DATA` into a single file. Use the provided utility to make separate csv files for `OBS` and `VAR`:

```python
from auto_atlas.util import extract_h5ad_obs_var
h5ad_fpath = ".../GSE..._HepG2.h5ad"
# This saves the csv files at the same path and with the same name as h5ad, but
# a different suffix and file extension.
obs, var = extract_h5ad_obs_var(h5ad_fpath)  # -> ..._obs.csv, ..._var.csv
```

While less common, it's also possible for `DATA` and `VAR` to be inline with `OBS` in a `csv` (or other tabular format). Protein abundance (ADT) and Cell Profiler features are sometimes packaged this way. In that case, the column names may have a prefix to denote which columns are for features and which are for metadata. There are no provided tools or scripts and you may need to develop a custom solution or ask the user for help.

### 4. Download library and other files

If the user provided direct URLs or file others, you should download them now. Otherwise, we always strive to download two other informational files to add to the data package: record metadata and publication.

Record metadata might be in a README co-located with the dataset or accessible by querying a database API (like GEO). If neither a README nor database API are available, then this can be skipped.

Publications may be downloaded using the PubMed API when a PMID or title is available. The user may have provided this information or it might be found in Record Metadata. If neither, then this step can be skipped.

When completed with this workflow, be sure to tell the user if record metadata or the publication were not downloaded.

### 5. Create collection, coalesce, and save

Organize files with `auto_atlas.collection`. Create one `Dataset` per experiment, add each file with the appropriate `FileTypeTag` (and feature space for obs/var/data files), add the datasets to a `Collection`, then `coalesce()` to lay out the directory structure on disk and `to_json()` to write the manifest to the root directory.

- A `Dataset` is one experiment. Multimodal modalities from the same experiment go in the SAME dataset, distinguished by `feature_space` — do not split them.
- Tag files with `FileTypeTag`: `DATA` for matrices (h5ad, mtx, etc.), `OBS`/`VAR` for metadata tables, `LIBRARY` for reagent/guide/donor libraries, `OTHER` for free-form informational files (READMEs, protocols).
- **One OBS and one VAR per feature space.** Within a dataset, each `feature_space` may have at most one `OBS` and one `VAR` — the primary tables aligned with that modality's matrix. Do not tag a second row-level table as `OBS` for the same feature space, even in a single-modality dataset.
- **OBS-like join tables → `OTHER`.** Extra tabular metadata that joins to the main OBS on an ID column (clinical data, cell-type calls from a separate file, QC metrics, etc.) belongs in the package but should be tagged `OTHER`, not `OBS` or `VAR`.
- Set `feature_space` (e.g. `gene_expression`, `protein_abundance`, `chromatin_accessibility`) on obs/var/data files; omit it for shared libraries and informational files.
- Files shared across datasets (e.g. one guide library used by every experiment) are added to the `Collection` via `add_file`, not to an individual `Dataset`.

```python
from auto_atlas.collection import Collection, Dataset, FileTypeTag

# IMPORTANT: While the downloaded data might be in a temp directory, the root
# directory of a collection data package SHOULD NOT be in temp.
collection = Collection(root_dir="/home/ubuntu/auto_atlas_data_packages/<accession_id>")

hepg2 = Dataset("HepG2")

# h5ad: extracted obs/var to CSVs previously, now we are adding all 3.
gex_h5ad = ".../GSE..._HepG2.h5ad"
hepg2.add_file(gex_h5ad, FileTypeTag.DATA, "gene_expression")
hepg2.add_file(gex_obs, FileTypeTag.OBS, "gene_expression")
hepg2.add_file(gex_var, FileTypeTag.VAR, "gene_expression")

# shared, collection-level library referenced by multiple datasets
collection.add_file(".../guide_library.csv", FileTypeTag.LIBRARY)
collection.add_file(".../record_metadata.json", FileTypeTag.OTHER)
collection.add_file(".../publication.json", FileTypeTag.OTHER)

collection.coalesce(copy=False)  # moves dataset files under root/<name>/, OTHER files under root/other_files/
collection.to_json()  # write collection.json under root_dir
```

After `coalesce()`, dataset files live in `root/<dataset_name>/`, shared files in `root/`, and `OTHER` files in `root/other_files/`. The `collection.json` manifest records every file with its tag and feature space and is the source of truth for the steps that follow.

Before finalizing, ask the user whether any files are missing — especially `LIBRARY` tables (guide/reagent/donor libraries). These are usually user-provided and easy to forget, and a perturbation dataset that references a library without including it is a strong signal one is missing. If anything in the data suggests a file that isn't present, ask the user whether they have it rather than proceeding without it; just asking costs nothing.

## Scripts

Depending on the public source of the data, you may use the following scripts as appropriate:

| Script | Usage | Purpose |
|--------|-------|---------|
| `scripts/write_publication_json.py` | `python scripts/write_publication_json.py <data_dir> --pmid 40259084` | Download a publication archive from pubmed, parse it, and save it as json |

## References

More instructions for specific databases may be found in `references`:

| File | Purpose |
|------|---------|
| `references/geo_instructions.md` | Additional scripts and instructions for working with GEO accession records (GSE or GSM) |
