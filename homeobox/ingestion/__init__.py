"""Reference ingestion functions for writing AnnData into a RaggedAtlas.

These are extracted from the original ``RaggedAtlas`` write path and serve as a
reference implementation.  Downstream projects can write their own ingestion
that calls the lower-level ``var_df`` helpers directly.

The streaming ingestion path behind ``add_from_anndata``: a reader streams a
source as row-batches of layer arrays, a converter adapts each batch onto the
arrays a zarr group spec wants, and a writer owns the zarr group and the
running offsets. The trio is resolved from the spec, so a new feature space
generally needs no new ingestion code.
"""

from homeobox.ingestion.converters import (
    ArrayConverter,
    CSRSparseConverter,
    DenseConverter,
    converter_for,
    register_converter,
)
from homeobox.ingestion.feature_oriented import add_csc
from homeobox.ingestion.functions import (
    add_from_anndata,
    ingest_dataset,
    ingest_multimodal,
)
from homeobox.ingestion.ingestor import (
    _DEFAULT_BATCH_ROWS,
    Ingestor,
    _build_row_arrow_table,
    _check_var_no_duplicate_uids_pl,
    _make_sparse_pointer,
    _pointer_struct_from_columns,
    _validate_var_columns_against_registry,
    _writer_create_kwargs,
    insert_obs_records,
)
from homeobox.ingestion.readers import AnnDataReader, COOReader, Reader
from homeobox.ingestion.writers import (
    _CHUNK_ELEMS,
    _CHUNKS_PER_SHARD,
    _SHARD_ELEMS,
    DenseZarrWriter,
    SparseZarrWriter,
    write_feature_space,
    writer_for,
)

__all__ = [
    "ArrayConverter",
    "AnnDataReader",
    "COOReader",
    "CSRSparseConverter",
    "DenseConverter",
    "DenseZarrWriter",
    "Ingestor",
    "Reader",
    "SparseZarrWriter",
    "_CHUNK_ELEMS",
    "_CHUNKS_PER_SHARD",
    "_DEFAULT_BATCH_ROWS",
    "_SHARD_ELEMS",
    "_build_row_arrow_table",
    "_check_var_no_duplicate_uids_pl",
    "_make_sparse_pointer",
    "_pointer_struct_from_columns",
    "_validate_var_columns_against_registry",
    "_writer_create_kwargs",
    "add_csc",
    "add_from_anndata",
    "converter_for",
    "ingest_dataset",
    "ingest_multimodal",
    "insert_obs_records",
    "register_converter",
    "write_feature_space",
    "writer_for",
]
