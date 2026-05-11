"""Tests for atlases with more than one obs table."""

import json
import os

import anndata as ad
import numpy as np
import obstore
import polars as pl
import pyarrow as pa
import pytest
import scipy.sparse as sp

from homeobox.atlas import RaggedAtlas
from homeobox.feature_layouts import reindex_registry
from homeobox.ingestion import add_from_anndata
from homeobox.obs_alignment import align_obs_to_schema
from homeobox.pointer_types import SparseZarrPointer
from homeobox.schema import (
    DatasetSchema,
    FeatureBaseSchema,
    HoxBaseSchema,
    PointerField,
)


class GeneFeatureSchema(FeatureBaseSchema):
    gene_name: str


class CellSchema(HoxBaseSchema):
    gene_expression: SparseZarrPointer | None = PointerField.declare(
        feature_space="gene_expression"
    )


class NucleusSchema(HoxBaseSchema):
    # Same field name + feature_space as CellSchema. Two obs tables can share a
    # modality under the same name as long as the declarations match.
    gene_expression: SparseZarrPointer | None = PointerField.declare(
        feature_space="gene_expression"
    )


def _ds(adata: ad.AnnData, zarr_group: str) -> DatasetSchema:
    return DatasetSchema(zarr_group=zarr_group, feature_space="gene_expression", n_rows=adata.n_obs)


def _make_adata(n_obs: int, gene_uids: list[str], seed: int) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X = sp.random(
        n_obs, len(gene_uids), density=0.3, format="csr", dtype=np.uint32, random_state=rng
    )
    var = pl.DataFrame({"global_feature_uid": gene_uids}).to_pandas()
    return ad.AnnData(X=X, var=var)


def _build_dual_atlas(tmp_path) -> tuple[RaggedAtlas, str, obstore.store.ObjectStore, list[str]]:
    """Atlas with two obs tables, both ingesting the same modality name."""
    atlas_dir = str(tmp_path / "atlas")
    os.makedirs(atlas_dir + "/zarr_store", exist_ok=True)
    store = obstore.store.LocalStore(prefix=atlas_dir + "/zarr_store")
    atlas = RaggedAtlas.create(
        db_uri=atlas_dir,
        obs_schemas={"cells": CellSchema, "nuclei": NucleusSchema},
        store=store,
        registry_schemas={"gene_expression": GeneFeatureSchema},
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
    )
    gene_uids = [f"gene_{i}" for i in range(10)]
    atlas.register_features(
        "gene_expression",
        [GeneFeatureSchema(uid=uid, gene_name=f"GENE{i}") for i, uid in enumerate(gene_uids)],
    )
    reindex_registry(atlas._registry_tables["gene_expression"])

    cell_adata = align_obs_to_schema(_make_adata(20, gene_uids, seed=0), CellSchema)
    add_from_anndata(
        atlas,
        cell_adata,
        field_name="gene_expression",
        zarr_layers={"counts": "X"},
        dataset_record=_ds(cell_adata, "cells/ds1"),
        obs_table_name="cells",
    )

    nuc_adata = align_obs_to_schema(_make_adata(15, gene_uids, seed=1), NucleusSchema)
    add_from_anndata(
        atlas,
        nuc_adata,
        field_name="gene_expression",
        zarr_layers={"counts": "X"},
        dataset_record=_ds(nuc_adata, "nuclei/ds1"),
        obs_table_name="nuclei",
    )
    return atlas, atlas_dir, store, gene_uids


# ---------------------------------------------------------------------------
# Resolver / accessor behavior
# ---------------------------------------------------------------------------


def test_pointer_fields_share_entry_when_definitions_match(tmp_path):
    """Same field_name + feature_space in two tables collapses to one PointerField."""
    atlas, *_ = _build_dual_atlas(tmp_path)
    assert set(atlas._pointer_fields) == {"gene_expression"}
    # Both tables advertise the field.
    assert atlas._field_to_tables == {"gene_expression": ["cells", "nuclei"]}


def test_obs_table_singular_raises_when_ambiguous(tmp_path):
    atlas, *_ = _build_dual_atlas(tmp_path)
    with pytest.raises(ValueError, match="multiple obs tables"):
        _ = atlas.obs_table


def test_pointer_fields_flat_view_collapses_shared_fields(tmp_path):
    atlas, *_ = _build_dual_atlas(tmp_path)
    flat = atlas.pointer_fields
    assert set(flat) == {"gene_expression"}
    assert flat["gene_expression"].feature_space == "gene_expression"


def test_pointer_fields_for_returns_per_table_view(tmp_path):
    atlas, *_ = _build_dual_atlas(tmp_path)
    assert set(atlas.pointer_fields_for("cells")) == {"gene_expression"}
    assert set(atlas.pointer_fields_for("nuclei")) == {"gene_expression"}


def test_pointer_fields_for_unknown_table_raises(tmp_path):
    atlas, *_ = _build_dual_atlas(tmp_path)
    with pytest.raises(KeyError, match="Unknown obs table"):
        atlas.pointer_fields_for("ghosts")


def test_resolve_obs_table_requires_name_when_multiple_tables(tmp_path):
    atlas, *_ = _build_dual_atlas(tmp_path)
    with pytest.raises(ValueError, match="Pass obs_table_name"):
        atlas._resolve_obs_table()


# ---------------------------------------------------------------------------
# Snapshot / checkout round-trip
# ---------------------------------------------------------------------------


def test_snapshot_round_trip_preserves_both_tables(tmp_path):
    atlas, atlas_dir, store, _ = _build_dual_atlas(tmp_path)
    v = atlas.snapshot()
    assert v == 0

    checked = RaggedAtlas.checkout_latest(
        atlas_dir,
        obs_schemas={"cells": CellSchema, "nuclei": NucleusSchema},
        store=store,
    )
    assert set(checked.obs_tables) == {"cells", "nuclei"}
    assert checked.obs_tables["cells"].count_rows() == 20
    assert checked.obs_tables["nuclei"].count_rows() == 15


def test_snapshot_record_uses_versions_json(tmp_path):
    """The version record stores names+versions as a single JSON dict."""
    atlas, atlas_dir, store, _ = _build_dual_atlas(tmp_path)
    atlas.snapshot()
    versions = RaggedAtlas.list_versions(atlas_dir)
    row = versions.row(0, named=True)

    decoded = json.loads(row["obs_table_versions"])
    assert set(decoded) == {"cells", "nuclei"}
    assert all(isinstance(v, int) and v >= 0 for v in decoded.values())


def test_single_table_snapshot_round_trips(tmp_path):
    atlas_dir = str(tmp_path / "atlas")
    os.makedirs(atlas_dir + "/zarr_store", exist_ok=True)
    store = obstore.store.LocalStore(prefix=atlas_dir + "/zarr_store")
    atlas = RaggedAtlas.create(
        db_uri=atlas_dir,
        obs_schemas={"cells": CellSchema},
        store=store,
        registry_schemas={"gene_expression": GeneFeatureSchema},
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
    )
    gene_uids = [f"gene_{i}" for i in range(10)]
    atlas.register_features(
        "gene_expression",
        [GeneFeatureSchema(uid=uid, gene_name=f"GENE{i}") for i, uid in enumerate(gene_uids)],
    )
    reindex_registry(atlas._registry_tables["gene_expression"])
    adata = align_obs_to_schema(_make_adata(7, gene_uids, seed=0), CellSchema)
    add_from_anndata(
        atlas,
        adata,
        field_name="gene_expression",
        zarr_layers={"counts": "X"},
        dataset_record=_ds(adata, "cells/ds1"),
    )
    atlas.snapshot()

    versions = RaggedAtlas.list_versions(atlas_dir)
    row = versions.row(0, named=True)
    decoded = json.loads(row["obs_table_versions"])
    assert set(decoded) == {"cells"}
    assert decoded["cells"] >= 0

    checked = RaggedAtlas.checkout_latest(atlas_dir, obs_schemas={"cells": CellSchema}, store=store)
    assert checked.obs_table.count_rows() == 7


# ---------------------------------------------------------------------------
# Query routing
# ---------------------------------------------------------------------------


def test_query_requires_obs_table_name_when_ambiguous(tmp_path):
    atlas, atlas_dir, store, _ = _build_dual_atlas(tmp_path)
    atlas.snapshot()
    checked = RaggedAtlas.checkout_latest(
        atlas_dir,
        obs_schemas={"cells": CellSchema, "nuclei": NucleusSchema},
        store=store,
    )
    with pytest.raises(ValueError, match="multiple obs tables"):
        checked.query()


def test_query_with_obs_table_name_targets_one_table(tmp_path):
    atlas, atlas_dir, store, _ = _build_dual_atlas(tmp_path)
    atlas.snapshot()
    checked = RaggedAtlas.checkout_latest(
        atlas_dir,
        obs_schemas={"cells": CellSchema, "nuclei": NucleusSchema},
        store=store,
    )

    cells_count = checked.query("cells").count()
    nuclei_count = checked.query("nuclei").count()
    assert cells_count == 20
    assert nuclei_count == 15


def test_query_obs_table_name_property_exposed(tmp_path):
    atlas, atlas_dir, store, _ = _build_dual_atlas(tmp_path)
    atlas.snapshot()
    checked = RaggedAtlas.checkout_latest(
        atlas_dir,
        obs_schemas={"cells": CellSchema, "nuclei": NucleusSchema},
        store=store,
    )
    q = checked.query("cells")
    assert q.obs_table_name == "cells"


# ---------------------------------------------------------------------------
# Ingestion routing
# ---------------------------------------------------------------------------


def test_add_from_anndata_requires_obs_table_name_when_field_shared(tmp_path):
    """When two tables share a field name, ingestion must specify obs_table_name."""
    atlas_dir = str(tmp_path / "atlas")
    os.makedirs(atlas_dir + "/zarr_store", exist_ok=True)
    store = obstore.store.LocalStore(prefix=atlas_dir + "/zarr_store")
    atlas = RaggedAtlas.create(
        db_uri=atlas_dir,
        obs_schemas={"cells": CellSchema, "nuclei": NucleusSchema},
        store=store,
        registry_schemas={"gene_expression": GeneFeatureSchema},
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
    )
    gene_uids = [f"gene_{i}" for i in range(10)]
    atlas.register_features(
        "gene_expression",
        [GeneFeatureSchema(uid=uid, gene_name=f"GENE{i}") for i, uid in enumerate(gene_uids)],
    )
    reindex_registry(atlas._registry_tables["gene_expression"])

    adata = align_obs_to_schema(_make_adata(5, gene_uids, seed=0), CellSchema)
    with pytest.raises(ValueError, match="Pass obs_table_name"):
        add_from_anndata(
            atlas,
            adata,
            field_name="gene_expression",
            zarr_layers={"counts": "X"},
            dataset_record=_ds(adata, "cells/ds1"),
        )


# ---------------------------------------------------------------------------
# Blessed obs-write path + stale-handle guard
# ---------------------------------------------------------------------------


def _build_bare_atlas(tmp_path, schemas: dict[str, type[HoxBaseSchema]]):
    """Atlas with the given obs schemas but no rows / datasets / zarr writes."""
    atlas_dir = str(tmp_path / "atlas")
    os.makedirs(atlas_dir + "/zarr_store", exist_ok=True)
    store = obstore.store.LocalStore(prefix=atlas_dir + "/zarr_store")
    atlas = RaggedAtlas.create(
        db_uri=atlas_dir,
        obs_schemas=schemas,
        store=store,
        registry_schemas={"gene_expression": GeneFeatureSchema},
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
    )
    return atlas, atlas_dir, store


def _empty_obs_arrow(schema_cls: type[HoxBaseSchema], uids: list[str]) -> pa.Table:
    """Build a pa.Table conforming to ``schema_cls`` with null pointer columns."""
    return pa.Table.from_pylist(
        [{"uid": u, "dataset_uid": "", "gene_expression": None} for u in uids],
        schema=schema_cls.to_arrow_schema(),
    )


def test_add_obs_records_with_arrow_table_round_trips(tmp_path):
    atlas, atlas_dir, store = _build_bare_atlas(tmp_path, {"cells": CellSchema})
    arrow = _empty_obs_arrow(CellSchema, ["c1", "c2", "c3"])

    atlas.add_obs_records(arrow)

    atlas.snapshot()
    checked = RaggedAtlas.checkout_latest(atlas_dir, obs_schemas={"cells": CellSchema}, store=store)
    assert checked.obs_table.count_rows() == 3


def test_add_obs_records_requires_obs_table_name_when_ambiguous(tmp_path):
    atlas, *_ = _build_bare_atlas(tmp_path, {"cells": CellSchema, "nuclei": NucleusSchema})
    arrow = _empty_obs_arrow(CellSchema, ["c1"])
    with pytest.raises(ValueError, match="Pass obs_table_name"):
        atlas.add_obs_records(arrow)


def test_add_obs_records_targets_named_table(tmp_path):
    atlas, atlas_dir, store = _build_bare_atlas(
        tmp_path, {"cells": CellSchema, "nuclei": NucleusSchema}
    )
    atlas.add_obs_records(_empty_obs_arrow(CellSchema, ["c1", "c2"]), obs_table_name="cells")
    atlas.add_obs_records(_empty_obs_arrow(NucleusSchema, ["n1"]), obs_table_name="nuclei")

    atlas.snapshot()
    checked = RaggedAtlas.checkout_latest(
        atlas_dir,
        obs_schemas={"cells": CellSchema, "nuclei": NucleusSchema},
        store=store,
    )
    assert checked.obs_tables["cells"].count_rows() == 2
    assert checked.obs_tables["nuclei"].count_rows() == 1


def test_snapshot_refuses_when_held_handle_is_stale(tmp_path):
    """Writing through a fresh handle leaves the held handle stale; snapshot must refuse."""
    atlas, atlas_dir, store = _build_bare_atlas(tmp_path, {"cells": CellSchema})

    # Footgun: bypass the held handle and write through a fresh one.
    fresh = atlas.db.open_table("cells")
    fresh.add(_empty_obs_arrow(CellSchema, ["c1", "c2"]))
    assert atlas._obs_tables["cells"].version != fresh.version

    with pytest.raises(RuntimeError) as excinfo:
        atlas.snapshot()
    msg = str(excinfo.value)
    assert "snapshot() refused" in msg
    assert "cells" in msg
    assert "atlas.refresh()" in msg

    atlas.refresh()
    assert atlas._obs_tables["cells"].version == fresh.version

    atlas.snapshot()
    checked = RaggedAtlas.checkout_latest(atlas_dir, obs_schemas={"cells": CellSchema}, store=store)
    assert checked.obs_table.count_rows() == 2


def test_snapshot_clean_when_writes_use_blessed_path(tmp_path):
    """Writes via add_obs_records keep the held handle in sync — no refresh needed."""
    atlas, atlas_dir, store = _build_bare_atlas(tmp_path, {"cells": CellSchema})
    atlas.add_obs_records(_empty_obs_arrow(CellSchema, ["c1"]))
    atlas.snapshot()  # must not raise

    checked = RaggedAtlas.checkout_latest(atlas_dir, obs_schemas={"cells": CellSchema}, store=store)
    assert checked.obs_table.count_rows() == 1
