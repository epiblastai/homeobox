"""Feature-oriented ingestion utilities (e.g. CSR-to-CSC transposition)."""

import numpy as np
import polars as pl
import scipy.sparse as sp

from homeobox.atlas import RaggedAtlas
from homeobox.group_specs import FeatureSpaceSpec, get_spec
from homeobox.ingestion.writers import _CHUNK_ELEMS, _SHARD_ELEMS
from homeobox.util import sql_escape


def add_csc(
    atlas: RaggedAtlas,
    zarr_group: str,
    field_name: str,
    layer_name: str = "counts",
    chunk_size: int = _CHUNK_ELEMS,
    shard_size: int = _SHARD_ELEMS,
    *,
    obs_table_name: str | None = None,
) -> None:
    """Read existing CSR group and write CSC alongside it.

    Reads the full CSR flat arrays from ``{zarr_group}/csr/``, transposes
    to CSC order sorted by feature index, writes ``{zarr_group}/csc/``, and
    stores the CSC ``indptr`` as a zarr array at ``{zarr_group}/csc/indptr``.

    After running, a new ``{zarr_group}/csc/`` subgroup appears alongside the
    existing ``{zarr_group}/csr/``, including an ``indptr`` array. Subsequent
    feature-filtered queries will automatically use the CSC path.

    Parameters
    ----------
    atlas:
        The atlas whose zarr store and obs table to use.
    zarr_group:
        Path of the zarr group to process (relative to atlas store root).
    field_name:
        Obs-schema attribute name for the pointer column that references
        *zarr_group*. The feature_space is derived from its ``PointerField``.
    layer_name:
        Which layer to transpose (e.g. ``"counts"``).
    chunk_size:
        Chunk size for the new CSC zarr arrays.
    shard_size:
        Shard size for the new CSC zarr arrays.

    Raises
    ------
    ValueError
        If no rows or no dataset record are found for this group, or if
        ``zarr_row`` is not sequential.
    """
    name, table = atlas._resolve_obs_table(obs_table_name=obs_table_name)
    pointer_fields = atlas.pointer_fields_for(name)
    pointer_field = pointer_fields[field_name]
    feature_space = pointer_field.feature_space

    # Look up layout_uid for this zarr_group + feature_space
    datasets_df = atlas.find_datasets(zarr_group, feature_space=feature_space).select(
        ["layout_uid"]
    )
    if datasets_df.is_empty():
        raise ValueError(
            f"No dataset record found for zarr_group='{zarr_group}', "
            f"feature_space='{feature_space}'"
        )
    layout_uid = datasets_df["layout_uid"][0]

    # Query all rows in this zarr group via the specified pointer column
    obs_df = (
        table.search()
        .where(f"{field_name}.zarr_group = '{sql_escape(zarr_group)}'", prefilter=True)
        .select([field_name])
        .to_polars()
    )
    ptr_struct = obs_df[field_name].struct.unnest()
    obs_df = pl.DataFrame(
        {
            "_zg": ptr_struct["zarr_group"],
            "_zarr_row": ptr_struct["zarr_row"],
            "_start": ptr_struct["start"],
            "_end": ptr_struct["end"],
        }
    )

    if obs_df.is_empty():
        raise ValueError(f"No rows found for zarr_group='{zarr_group}', field_name='{field_name}'")

    obs_df = obs_df.sort("_zarr_row")
    zarr_rows = obs_df["_zarr_row"].to_numpy()
    starts = obs_df["_start"].to_numpy()
    ends = obs_df["_end"].to_numpy()
    n_rows = len(zarr_rows)

    if len(zarr_rows) != len(np.unique(zarr_rows)):
        raise ValueError(
            f"zarr_rows for group '{zarr_group}' contain duplicate values. "
            f"Was zarr_row populated correctly during ingest?"
        )
    if not np.array_equal(zarr_rows, np.arange(n_rows)):
        raise ValueError(
            f"zarr_rows for group '{zarr_group}' are not sequential 0..{n_rows - 1}. "
            f"Was zarr_row populated correctly during ingest?"
        )

    # Get n_features from _feature_layouts
    rows = atlas.read_feature_layout(layout_uid)
    n_features = len(rows)

    spec = get_spec(feature_space)
    _add_csc_scipy(
        atlas,
        zarr_group,
        layer_name,
        starts,
        ends,
        n_rows,
        n_features,
        chunk_size,
        shard_size,
        feature_space,
        spec,
    )


def _add_csc_scipy(
    atlas: RaggedAtlas,
    zarr_group: str,
    layer_name: str,
    starts: np.ndarray,
    ends: np.ndarray,
    n_rows: int,
    n_features: int,
    chunk_size: int,
    shard_size: int,
    feature_space: str,
    spec: FeatureSpaceSpec,
) -> None:
    """CSR-to-CSC using scipy (fast, but loads full matrix into RAM)."""
    csr_prefix = spec.zarr_group_spec.layers.prefix
    csr_layers_path = spec.zarr_group_spec.find_layers_path()

    csr_group = atlas.open_zarr_group(zarr_group)
    csr_indices = csr_group[f"{csr_prefix}/indices"][:]
    csr_values = csr_group[f"{csr_layers_path}/{layer_name}"][:]

    # starts/ends are absolute offsets; since starts[0]==0 and ends are
    # cumulative, the CSR indptr is just [starts[0], *ends].
    indptr_csr = np.concatenate([[starts[0]], ends])

    csr = sp.csr_matrix(
        (csr_values, csr_indices.astype(np.int32), indptr_csr),
        shape=(n_rows, n_features),
    )
    csc = csr.tocsc()

    if spec.feature_oriented is None:
        raise ValueError(
            f"Feature space '{feature_space}' has no feature_oriented spec; "
            "cannot write a CSC copy."
        )
    csc_spec = spec.feature_oriented

    nnz = csc.nnz

    csc_indices_zarr = csc_spec.create_array(
        csr_group,
        "csc/indices",
        (nnz,),
        chunks=(chunk_size,),
        shards=(shard_size,),
    )
    csc_values_zarr = csc_spec.create_array(
        csr_group,
        layer_name,
        (nnz,),
        chunks=(chunk_size,),
        shards=(shard_size,),
    )

    # Write in shard-sized batches, casting to each array's own dtype (not a
    # literal) so non-integer layers like log_normalized/tpm aren't truncated.
    written = 0
    while written < nnz:
        end = min(written + shard_size, nnz)
        csc_indices_zarr[written:end] = csc.indices[written:end].astype(
            csc_indices_zarr.dtype, copy=False
        )
        csc_values_zarr[written:end] = csc.data[written:end].astype(
            csc_values_zarr.dtype, copy=False
        )
        written = end

    csc_indptr_zarr = csc_spec.create_array(csr_group, "csc/indptr", csc.indptr.shape)
    csc_indptr_zarr[:] = csc.indptr.astype(np.int64)

    # Cache invalidation
    atlas.invalidate_group_reader(zarr_group, feature_space)
