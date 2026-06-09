"""Readers: a source -> a stream of layer batches.

A reader maps source -> target layer names and streams row-batches. It is
fully spec-agnostic — layer-set conformance to the spec (required present,
whitelist respected) is the converter's job. One reader works for any
feature space.
"""

import subprocess
from collections.abc import Generator
from pathlib import Path
from typing import Any, Protocol

import anndata as ad
import numpy as np
import polars as pl
import scipy.sparse as sp


class Reader(Protocol):
    """A source adapter: streams a source as row-batches of layer arrays.

    Implementations yield ``{target_layer: array}`` dicts, one per row-batch,
    where every layer in a batch shares one sparsity structure. The array type
    (CSR or dense) selects the converter downstream, so a reader is free to
    decode whatever source it wants as long as it emits in row order.
    """

    def iter_layer_batches(
        self, batch_size: int, layer_mapping: dict[str, str]
    ) -> Generator[dict[str, Any]]: ...


class AnnDataReader:
    """Streams an AnnData as row-batches of layer arrays.

    The source mirrors ``add_from_anndata``: an in-memory :class:`AnnData`, or
    a path to an ``.h5ad`` file. ``open`` reads a path; an already-open AnnData
    is returned as-is.

    Backed sources (``backed="r"``) are streamed lazily: each row-batch is read
    straight from the open HDF5 file, so only ``batch_size`` rows are
    materialized at a time. Backed sparse layers must be CSR (row-major);
    backed dense layers are read as row slices.
    """

    def __init__(self, source: ad.AnnData | str | Path) -> None:
        self.source = source

    def open(self, backed: str | None = None, **kwargs) -> ad.AnnData:
        if isinstance(self.source, ad.AnnData):
            return self.source
        return ad.read_h5ad(self.source, backed=backed, **kwargs)

    def iter_layer_batches(
        self,
        batch_size: int,
        layer_mapping: dict[str, str],
        **open_kwargs,
    ) -> Generator[dict[str, Any]]:
        adata = self.open(**open_kwargs)
        if adata.isbacked:
            yield from self._iter_backed_batches(adata, batch_size, layer_mapping)
        else:
            yield from self._iter_in_memory_batches(adata, batch_size, layer_mapping)

    def _iter_in_memory_batches(
        self,
        adata: ad.AnnData,
        batch_size: int,
        layer_mapping: dict[str, str],
    ) -> Generator[dict[str, Any]]:
        """Yield ``{target_layer: array}`` per row-slice of an in-memory AnnData.

        ``layer_mapping`` maps a source layer name (``"X"`` or a key in
        ``adata.layers``) to a target layer name.
        """
        for start_idx in range(0, len(adata), batch_size):
            batch = adata[start_idx : start_idx + batch_size]
            batch_layers: dict[str, Any] = {}
            for src_name, tgt_name in layer_mapping.items():
                source = batch.X if src_name == "X" else batch.layers[src_name]
                if sp.issparse(source):
                    source = source.tocsr()
                else:
                    source = np.asarray(source)
                batch_layers[tgt_name] = source
            yield batch_layers

    def _iter_backed_batches(
        self,
        adata: ad.AnnData,
        batch_size: int,
        layer_mapping: dict[str, str],
    ) -> Generator[dict[str, Any]]:
        """Yield row-batches by reading the backing HDF5 file lazily.

        Each source layer is resolved to its HDF5 node once (``X`` or a member
        of the ``layers`` group), caching the CSR ``indptr`` so only the rows of
        the current batch are read. The per-batch arrays match the in-memory
        path: a ``csr_matrix`` for sparse layers, a dense ``ndarray`` otherwise.
        """
        import h5py

        n_rows = adata.n_obs
        n_vars = adata.n_vars
        h5file = adata.file._file

        nodes: dict[str, Any] = {}
        indptrs: dict[str, np.ndarray] = {}
        for src_name in layer_mapping:
            node = h5file["X"] if src_name == "X" else h5file["layers"][src_name]
            if isinstance(node, h5py.Group):
                encoding = node.attrs.get("encoding-type", "")
                if encoding and encoding != "csr_matrix":
                    raise ValueError(
                        f"Backed ingestion of source layer '{src_name}' requires CSR "
                        f"(row-major) storage, but it is encoded as '{encoding}'. "
                        f"Re-save the h5ad with a CSR X or load it in memory."
                    )
                indptrs[src_name] = node["indptr"][:]
            nodes[src_name] = node

        for r0 in range(0, n_rows, batch_size):
            r1 = min(r0 + batch_size, n_rows)
            batch_layers: dict[str, Any] = {}
            for src_name, tgt_name in layer_mapping.items():
                node = nodes[src_name]
                if isinstance(node, h5py.Group):
                    indptr = indptrs[src_name]
                    o0, o1 = int(indptr[r0]), int(indptr[r1])
                    batch_layers[tgt_name] = sp.csr_matrix(
                        (node["data"][o0:o1], node["indices"][o0:o1], indptr[r0 : r1 + 1] - o0),
                        shape=(r1 - r0, n_vars),
                    )
                else:
                    batch_layers[tgt_name] = np.asarray(node[r0:r1])
            yield batch_layers


class COOReader:
    """Streams a cell-sorted COO triplet file as CSR row-batches.

    The source is a gzipped or plain text file of
    ``(feature_idx, cell_idx, value)`` triplets that **must be sorted by cell
    index**. Each emitted batch is a ``csr_matrix`` of ``batch_size``
    consecutive cells spanning the full ``[0, n_rows)`` row space: cells absent
    from the file appear as empty rows, so the batch stream stays positionally
    aligned with the obs table. The existing :class:`CSRSparseConverter` then
    handles the sparse layout, exactly as for an AnnData source — no COO-aware
    converter or writer is needed.

    Because each batch is built directly from a contiguous file slice, the
    cell-sorted invariant is load-bearing; the reader checks monotonicity and
    index ranges and fails loudly rather than emitting a corrupt matrix. Only
    ``batch_size`` cells' worth of triplets (plus one in-flight read chunk) are
    materialized at a time, so peak memory is bounded regardless of file size.
    """

    def __init__(
        self,
        coo_path: str | Path,
        *,
        n_rows: int,
        n_features: int,
        separator: str = "\t",
        gene_col: int = 0,
        cell_col: int = 1,
        value_col: int = 2,
        one_indexed: bool = True,
        read_batch_rows: int = 5_000_000,
    ) -> None:
        self.coo_path = coo_path
        self.n_rows = n_rows
        self.n_features = n_features
        self.separator = separator
        self.gene_col = gene_col
        self.cell_col = cell_col
        self.value_col = value_col
        self.offset = 1 if one_indexed else 0
        self.read_batch_rows = read_batch_rows

    def _iter_triplet_chunks(
        self,
    ) -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield ``(cells, genes, values)`` per raw read chunk, 0-based indices.

        Validates each chunk: indices stay in range, and the cell column is
        non-decreasing both within and across chunks (the cell-sorted invariant
        the CSR-by-batch construction depends on).
        """
        cell_col_name = f"column_{self.cell_col + 1}"
        gene_col_name = f"column_{self.gene_col + 1}"
        value_col_name = f"column_{self.value_col + 1}"

        # gzip via subprocess is faster than the Python gzip module for the
        # large files this reader targets; plain files are opened directly.
        is_gzip = str(self.coo_path).endswith(".gz")
        if is_gzip:
            proc = subprocess.Popen(["gzip", "-dc", str(self.coo_path)], stdout=subprocess.PIPE)
            source = proc.stdout
        else:
            proc = None
            source = open(self.coo_path, "rb")

        prev_last_cell = -1
        try:
            reader = pl.read_csv_batched(
                source,
                has_header=False,
                separator=self.separator,
                batch_size=self.read_batch_rows,
                schema_overrides={
                    gene_col_name: pl.Int32,
                    cell_col_name: pl.Int32,
                    value_col_name: pl.Int32,
                },
            )
            while True:
                batches = reader.next_batches(1)
                if not batches:
                    break
                batch = batches[0]
                cells = batch[cell_col_name].to_numpy().astype(np.int64) - self.offset
                genes = batch[gene_col_name].to_numpy().astype(np.int64) - self.offset
                vals = batch[value_col_name].to_numpy()
                if cells.size == 0:
                    continue

                if not np.all(cells[1:] >= cells[:-1]) or cells[0] < prev_last_cell:
                    raise ValueError(
                        "COO file is not sorted by cell index; COOReader requires "
                        "cell-sorted input. Sort the file by the cell column first."
                    )
                prev_last_cell = int(cells[-1])

                cmin, cmax = int(cells[0]), int(cells[-1])
                if cmin < 0 or cmax >= self.n_rows:
                    raise ValueError(
                        f"cell index out of range [0, {self.n_rows}): saw [{cmin}, {cmax}]. "
                        f"Check one_indexed ({self.offset == 1}) and n_rows."
                    )
                gmin, gmax = int(genes.min()), int(genes.max())
                if gmin < 0 or gmax >= self.n_features:
                    raise ValueError(
                        f"feature index out of range [0, {self.n_features}): saw [{gmin}, {gmax}]. "
                        f"Check one_indexed ({self.offset == 1}) and n_features."
                    )
                yield cells, genes, vals
        finally:
            source.close()
            if proc is not None:
                proc.wait()

    def _emit(
        self,
        r0: int,
        r1: int,
        cells: np.ndarray,
        genes: np.ndarray,
        vals: np.ndarray,
    ) -> sp.csr_matrix:
        """Build a CSR for cell rows ``[r0, r1)`` from the (sorted) buffer.

        Cells absent from the slice become empty rows, so the matrix always has
        exactly ``r1 - r0`` rows regardless of which cells carried entries.
        """
        n_batch = r1 - r0
        lo = int(np.searchsorted(cells, r0, side="left"))
        hi = int(np.searchsorted(cells, r1, side="left"))
        local = cells[lo:hi] - r0
        indptr = np.zeros(n_batch + 1, dtype=np.int64)
        np.cumsum(np.bincount(local, minlength=n_batch), out=indptr[1:])
        return sp.csr_matrix((vals[lo:hi], genes[lo:hi], indptr), shape=(n_batch, self.n_features))

    def iter_layer_batches(
        self,
        batch_size: int,
        layer_mapping: dict[str, str],
    ) -> Generator[dict[str, Any]]:
        if len(layer_mapping) != 1:
            raise ValueError(
                f"COOReader writes a single value layer, but layer_mapping has "
                f"{len(layer_mapping)} entries: {sorted(layer_mapping)}"
            )
        (tgt_name,) = layer_mapping.values()

        buf_cells = np.empty(0, dtype=np.int64)
        buf_genes = np.empty(0, dtype=np.int64)
        buf_vals: np.ndarray | None = None
        next_row = 0

        for cells, genes, vals in self._iter_triplet_chunks():
            if buf_vals is None:
                buf_vals = np.empty(0, dtype=vals.dtype)
            buf_cells = np.concatenate([buf_cells, cells])
            buf_genes = np.concatenate([buf_genes, genes])
            buf_vals = np.concatenate([buf_vals, vals])

            # The highest cell in the buffer may continue in the next chunk, so
            # only cells strictly below it are complete and safe to emit.
            safe_upto = int(buf_cells[-1])
            while next_row + batch_size <= safe_upto:
                r1 = next_row + batch_size
                yield {tgt_name: self._emit(next_row, r1, buf_cells, buf_genes, buf_vals)}
                next_row = r1

            # Drop fully-emitted triplets (cells < next_row) from the buffer.
            keep = int(np.searchsorted(buf_cells, next_row, side="left"))
            buf_cells = buf_cells[keep:]
            buf_genes = buf_genes[keep:]
            buf_vals = buf_vals[keep:]

        # Flush the remainder, padding empty rows out to n_rows so the batch
        # stream covers every obs row even past the last cell with entries.
        if buf_vals is None:
            buf_vals = np.empty(0, dtype=np.int64)
        while next_row < self.n_rows:
            r1 = min(next_row + batch_size, self.n_rows)
            yield {tgt_name: self._emit(next_row, r1, buf_cells, buf_genes, buf_vals)}
            next_row = r1
