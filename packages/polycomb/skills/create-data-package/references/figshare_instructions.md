# figshare

figshare hosts datasets on figshare.com and on institutional / figshare+ portals
(`plus.figshare.com`, `<institution>.figshare.com`). Every portal is served by one
public API at `https://api.figshare.com/v2`, and public records need no token.

## Identifiers

A dataset page URL ends in the **article id**:

```
https://plus.figshare.com/articles/dataset/<slug>/27261219   ->  article 27261219
```

Download links on that page carry a **file id**, which is what a user typically
hands over:

```
https://plus.figshare.com/ndownloader/files/64650261         ->  file 64650261
```

The two are not interchangeable. Given only a file id, find its article by
searching the API for the dataset title, then confirm the file id appears in the
article's `files` list:

```bash
curl -s -X POST https://api.figshare.com/v2/articles/search -H "Content-Type: application/json" -d '{"search_for":"<title or keyword>","page_size":10}'
```

## Record metadata

`GET /v2/articles/<article_id>` returns the full record: title, DOI, description
(the per-file documentation authors usually write there), authors, funding,
references, license, and a `files` list with `id`, `name`, `size`, and
`supplied_md5` / `computed_md5`. Save it as the package's record metadata file:

```bash
python scripts/list_figshare_files.py 27261219 --write-metadata <dest_dir>
```

The `references` field often holds the paper DOI — use it to find the PMID for
`write_publication_json.py`. Read the `description` before tagging files: it is
where authors explain which file is the full dataset and which are subsets,
pilots, or reprocessed variants.

## Downloading

Do **not** fetch `https://<portal>.figshare.com/ndownloader/files/<file_id>`
programmatically — that host sits behind an AWS WAF that answers non-browser
clients with a 202 challenge and then 403.

Use the API download endpoint instead:

```
GET https://api.figshare.com/v2/file/download/<file_id>
  -> 302 https://ndownloader.figshare.com/files/<file_id>
  -> 302 presigned S3 URL (X-Amz-Expires=10)
```

The presigned URL lives ~10 seconds, so follow the redirects in one request and
re-request the whole chain for every retry or resume; a saved S3 URL is dead by
the time it is reused. `Range` requests survive the redirect chain, which makes
resume work:

```bash
python scripts/download_figshare_file.py 64650261 <dest_dir> --article-id 27261219 --verify-md5
```

With `--article-id` the script takes the file name and expected size from the
record and checks them after the transfer; `--verify-md5` additionally hashes the
result. Dataset files here routinely run to hundreds of GB, so check free disk
before starting and prefer the resumable script over a bare `curl`.

Equivalent one-liner, if a shell download is preferred (`-C -` resumes, and each
retry re-signs because the chain is refollowed):

```bash
curl -L -C - --retry 5 --retry-delay 10 -o <dest>/<name>.h5ad https://api.figshare.com/v2/file/download/64650261
```
