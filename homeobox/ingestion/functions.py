"""High-level ingestion functions."""

from pathlib import Path

import anndata as ad
import pandas as pd

from homeobox.atlas import RaggedAtlas
from homeobox.ingestion.ingestor import _DEFAULT_BATCH_ROWS, Ingestor
from homeobox.ingestion.readers import AnnDataReader, Reader
from homeobox.schema import DatasetSchema


def ingest_dataset(
    atlas: RaggedAtlas,
    reader: Reader,
    *,
    obs_df: pd.DataFrame,
    field_name: str,
    layer_mapping: dict[str, str],
    dataset_record: DatasetSchema,
    n_vars: int,
    var_df: pd.DataFrame | None = None,
    required_pointer_type: type | None = None,
    batch_size: int = _DEFAULT_BATCH_ROWS,
    chunk_shape: tuple[int, ...] | None = None,
    shard_shape: tuple[int, ...] | None = None,
    obs_table_name: str | None = None,
) -> int:
    """Stream one feature space from a reader into the atlas and stamp obs rows.

    The shared spine behind :func:`add_from_anndata`: given any :class:`Reader`
    plus the obs (and, where the spec needs it, var) tables it was built from,
    this validates obs and var, registers the dataset, writes the matrix
    feature-space-first via :func:`write_feature_space`, and stamps the resulting
    pointer column onto the obs rows. The reader decides the source format; the
    converter and writer are resolved from the spec. One or many layers can be
    written in a single pass — they share the feature space's structure.

    This is the single-feature-space case of :class:`Ingestor`: one
    :meth:`~Ingestor.write_array` followed by :meth:`~Ingestor.write_obs_records`.
    Use :class:`Ingestor` directly (or :func:`ingest_multimodal`) when several
    matrices must populate different pointer fields on the same obs rows.

    Parameters
    ----------
    atlas:
        The atlas to ingest into. Features must already be registered.
    reader:
        A :class:`Reader` that streams the matrix as row-batches. It must emit
        exactly ``len(obs_df)`` rows, in obs order.
    obs_df:
        Validated obs DataFrame with schema-aligned columns, one row per cell.
    field_name:
        Obs-schema attribute name for the pointer column to populate.
    layer_mapping:
        Maps each source layer the reader should read to its destination layer
        name in the spec's ``layers/`` group, e.g. ``{"X": "counts"}`` or
        ``{"X": "counts", "spliced": "spliced"}`` for a multi-layer write. All
        layers share one structure (sparsity / row count); destination names
        must be unique.
    dataset_record:
        Dataset record to register; ``dataset_record.zarr_group`` is the zarr
        group path.
    n_vars:
        Number of features (matrix width). Only used to size dense-writer
        chunks/shards; ignored for sparse layouts.
    var_df:
        Pandas var table (one row per feature, in positional order); its index
        is dropped before use. Required for feature spaces whose spec sets
        ``has_var_df``; validated against the registry schema and checked for
        duplicate uids. Ignored otherwise.
    required_pointer_type:
        If given, fail fast unless the spec's pointer type matches — lets a
        reader that only feeds one layout family (e.g. ``COOReader`` → sparse)
        reject a mismatched feature space before any data is written.
    batch_size:
        Rows read and written per batch.
    chunk_shape, shard_shape:
        Optional zarr chunk/shard shapes (1-element for sparse, 2-element for
        dense). Default to this module's constants.
    obs_table_name:
        Which obs table to ingest into. May be ``None`` only when the atlas has
        exactly one obs table.

    Returns
    -------
    int
        Number of cells ingested.
    """
    ingestor = Ingestor(atlas, obs_df=obs_df, obs_table_name=obs_table_name)
    ingestor.write_array(
        reader,
        field_name=field_name,
        layer_mapping=layer_mapping,
        dataset_record=dataset_record,
        n_vars=n_vars,
        var_df=var_df,
        required_pointer_type=required_pointer_type,
        batch_size=batch_size,
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
    )
    return ingestor.write_obs_records()


def add_from_anndata(
    atlas: RaggedAtlas,
    adata: ad.AnnData | str | Path,
    *,
    field_name: str,
    zarr_layer: str,
    layer_mapping: dict[str, str] | None = None,
    dataset_record: DatasetSchema,
    batch_size: int = _DEFAULT_BATCH_ROWS,
    backed: str | None = None,
    chunk_shape: tuple[int, ...] | None = None,
    shard_shape: tuple[int, ...] | None = None,
    obs_table_name: str | None = None,
) -> int:
    """Ingest an AnnData (or .h5ad path) into the atlas.

    A thin source adapter over :func:`ingest_dataset`: it opens the AnnData
    (honoring ``backed``), extracts obs and var, and streams ``adata.X`` into
    ``zarr_layer`` (plus any extra ``layer_mapping`` entries) via an
    :class:`AnnDataReader`. The feature space is derived from ``field_name``'s
    registered ``PointerField``; the spec decides whether a var_df is needed and
    which converter/writer apply.

    Parameters
    ----------
    atlas:
        The atlas to ingest into. Features must already be registered.
    adata:
        In-memory AnnData or a path to an ``.h5ad`` file.
    field_name:
        Obs-schema attribute name for the pointer column to populate.
    zarr_layer:
        Destination layer name within the spec's ``layers/`` group for
        ``adata.X``.
    layer_mapping:
        Extra source→destination layer pairs to write alongside ``X``, e.g.
        ``{"spliced": "spliced"}`` to also write ``adata.layers["spliced"]``.
        Empty by default (only ``adata.X`` → ``zarr_layer``); a key of ``"X"``
        overrides the default X destination.
    dataset_record:
        Dataset record to register; ``dataset_record.zarr_group`` is the
        zarr group path.
    batch_size:
        Rows read and written per batch.
    backed:
        When ``adata`` is a path, the ``backed`` mode passed to
        ``anndata.read_h5ad`` (e.g. ``"r"``). In backed mode the matrix is
        streamed off disk one batch at a time instead of being read fully into
        memory; a backed sparse ``X`` must be CSR (row-major). Ignored when
        ``adata`` is already an in-memory AnnData.
    chunk_shape, shard_shape:
        Optional zarr chunk/shard shapes (1-element for sparse, 2-element for
        dense). Default to this module's constants.

    Returns
    -------
    int
        Number of cells ingested.
    """
    if not isinstance(adata, ad.AnnData):
        adata = ad.read_h5ad(adata, backed=backed)

    return ingest_dataset(
        atlas,
        AnnDataReader(adata),
        obs_df=adata.obs,
        field_name=field_name,
        layer_mapping={"X": zarr_layer, **(layer_mapping or {})},
        dataset_record=dataset_record,
        n_vars=adata.n_vars,
        var_df=adata.var,
        batch_size=batch_size,
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
        obs_table_name=obs_table_name,
    )


def ingest_multimodal(
    atlas: RaggedAtlas,
    modalities: dict[str, ad.AnnData],
    *,
    obs_df: pd.DataFrame,
    zarr_layer: str,
    dataset_records: dict[str, DatasetSchema],
    batch_size: int = _DEFAULT_BATCH_ROWS,
    obs_table_name: str | None = None,
) -> int:
    """Ingest aligned multimodal AnnData, creating one obs record per cell.

    Each modality's matrix is written to its own zarr group via :class:`Ingestor`,
    then a single obs record per cell is inserted with every modality's pointer
    field populated (pointer fields no modality wrote are null-filled).

    Parameters
    ----------
    atlas:
        The atlas to ingest into. Features must already be registered.
    modalities:
        ``{field_name: AnnData}`` — keyed by pointer field name in the obs
        schema. Each AnnData must have ``len(obs_df)`` cells in obs order.
        ``adata.var`` must validate against the registry for feature spaces
        with a var_df.
    obs_df:
        Shared obs DataFrame for all modalities (validated, schema-aligned).
    zarr_layer:
        Destination layer name (e.g. ``"counts"``) for each modality's ``X``.
    dataset_records:
        ``{field_name: DatasetSchema}`` — one per modality, keyed the same way
        as ``modalities``. All records must share a single ``dataset_uid``.
    batch_size:
        Rows read and written per batch when streaming each modality's matrix.
    obs_table_name:
        Which obs table to ingest into. May be ``None`` only when the atlas has
        exactly one obs table.

    Returns
    -------
    int
        Number of cells ingested.
    """
    if set(modalities) != set(dataset_records):
        raise ValueError(
            f"modalities and dataset_records must share keys; got "
            f"modalities={sorted(modalities)}, dataset_records={sorted(dataset_records)}"
        )

    # Check cell counts up front so a mismatch fails before any zarr is written
    # (write_array enforces the same per modality, but only once that modality's
    # matrix has been streamed).
    n_cells = len(obs_df)
    for field_name, adata in modalities.items():
        if adata.n_obs != n_cells:
            raise ValueError(
                f"Modality '{field_name}' has {adata.n_obs} cells, expected {n_cells} "
                f"(len(obs_df))."
            )

    ingestor = Ingestor(atlas, obs_df=obs_df, obs_table_name=obs_table_name)
    for field_name, adata in modalities.items():
        ingestor.write_array(
            AnnDataReader(adata),
            field_name=field_name,
            layer_mapping={"X": zarr_layer},
            dataset_record=dataset_records[field_name],
            n_vars=adata.n_vars,
            var_df=adata.var,
            batch_size=batch_size,
        )
    return ingestor.write_obs_records()
