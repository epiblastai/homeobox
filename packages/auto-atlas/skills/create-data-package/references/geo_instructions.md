# GEO Database Instructions

## Scripts

You have access to scripts to help with the navigation and download for files and metadata from GEO (paths are relevant to the root of this skills):

| Script | Usage | Purpose |
|--------|-------|---------|
| `scripts/list_geo_files.py` | `python scripts/list_geo_files.py GSE123456` | List supplementary files for any GEO accession (GSE or GSM) |
| `scripts/download_geo_file.py` | `python scripts/download_geo_file.py GSE123456 file.h5ad [dest_dir]` | Download a supplementary file from GEO via FTP |
| `scripts/write_metadata_json.py` | `python scripts/write_metadata_json.py <collection_dir> <accession>` | Fetch and write GEO metadata to `<collection_dir>/<accession>_metadata.json` |

## Additional instructions

### List data files for the provided GEO accession

If the accession code is for a series or superseries (GSE prefix) series record, look a single large file that might be aggregated data, these are generally preferable over downloading files separately for each sample. However, if the series level has no files or only summary statistics, then you should check the sample-level for the real data. If there are many sample records its best to process them one at a time to avoid confusion. If very many, you should ask the user how they would like to proceed.

### Download a file

Download the metadata from the GEO series or sample records:

```
python scripts/write_metadata_json.py /tmp/geo_agent/<accession> <accession>
```

You may need to run this multiple times. Sometimes when the data is stored at the series level, it still references a sample record (e.g., the filename contains a GSM id). In this case, download the metadata from the series and from the referenced sample ids.