"""Tests for the `DiscreteSpatialPointer` pointer type.

Core learns the type at the schema level *and* supports the read path in
:class:`MultimodalCellDataset` (via :class:`DiscreteSpatialBatch`). The
single-modality :class:`CellDataset`, ingestion, and DEx still raise
NotImplementedError — external code owns those.
"""

import os

import anndata as ad
import numpy as np
import obstore
import pyarrow as pa
import pytest
import scipy.sparse as sp
from pydantic import ValidationError

from homeobox.atlas import RaggedAtlas
from homeobox.dataloader import CellDataset, DiscreteSpatialBatch, MultimodalCellDataset
from homeobox.dex._dex import _compare, _extract_matrix, dex
from homeobox.group_specs import (
    LayersSpec,
    PointerKind,
    ZarrGroupSpec,
    register_spec,
    registered_feature_spaces,
)
from homeobox.ingestion import add_from_anndata
from homeobox.obs_alignment import (
    _DISCRETE_SPATIAL_SUBFIELDS,
    _extract_pointer_fields,
    _infer_pointer_fields_from_arrow,
)
from homeobox.reconstruction import DenseReconstructor
from homeobox.schema import (
    POINTER_FEATURE_SPACE_METADATA_KEY,
    DatasetRecord,
    DiscreteSpatialPointer,
    HoxBaseSchema,
    PointerField,
    SparseZarrPointer,
    make_uid,
)

# ---------------------------------------------------------------------------
# Register a test feature space with the new pointer kind (module-level so it
# is available before any schema subclass is declared below).
# ---------------------------------------------------------------------------


_DS_FS = "test_discrete_spatial_boxes"

if _DS_FS not in registered_feature_spaces():
    register_spec(
        ZarrGroupSpec(
            feature_space=_DS_FS,
            pointer_kind=PointerKind.DISCRETE_SPATIAL,
            has_var_df=False,
            layers=LayersSpec(),
            # reconstructor is never invoked in these tests; DenseReconstructor
            # is a convenient concrete Protocol implementation.
            reconstructor=DenseReconstructor(),
        )
    )


class DiscreteSpatialSchema(HoxBaseSchema):
    boxes: DiscreteSpatialPointer | None = PointerField.declare(feature_space=_DS_FS)
    cell_type: str | None = None


# ---------------------------------------------------------------------------
# LanceModel instantiation + validators
# ---------------------------------------------------------------------------


class TestPointerValidation:
    def test_valid_instance(self):
        p = DiscreteSpatialPointer(zarr_group="g", min_corner=[0, 5], max_corner=[10, 12])
        assert p.zarr_group == "g"
        assert p.min_corner == [0, 5]
        assert p.max_corner == [10, 12]

    def test_empty_corners_allowed(self):
        """The absent-pointer convention uses empty corner lists."""
        p = DiscreteSpatialPointer(zarr_group="", min_corner=[], max_corner=[])
        assert p.min_corner == []
        assert p.max_corner == []

    def test_mismatched_corner_lengths_rejected(self):
        with pytest.raises(ValidationError, match="same length"):
            DiscreteSpatialPointer(zarr_group="g", min_corner=[0, 1], max_corner=[10])

    def test_inverted_corner_rejected(self):
        with pytest.raises(ValidationError, match="exceeds max_corner"):
            DiscreteSpatialPointer(zarr_group="g", min_corner=[10], max_corner=[0])

    def test_equal_corners_allowed(self):
        """min == max is a legitimate degenerate box (single index)."""
        p = DiscreteSpatialPointer(zarr_group="g", min_corner=[5], max_corner=[5])
        assert p.min_corner == p.max_corner


# ---------------------------------------------------------------------------
# Schema introspection + arrow inference
# ---------------------------------------------------------------------------


class TestSchemaIntrospection:
    def test_extract_pointer_fields_resolves_kind(self):
        pfs = _extract_pointer_fields(DiscreteSpatialSchema)
        assert set(pfs.keys()) == {"boxes"}
        pf = pfs["boxes"]
        assert pf.field_name == "boxes"
        assert pf.feature_space == _DS_FS
        assert pf.pointer_kind is PointerKind.DISCRETE_SPATIAL

    def test_arrow_schema_has_discrete_struct(self):
        arrow_schema = DiscreteSpatialSchema.to_arrow_schema()
        field = arrow_schema.field("boxes")
        assert pa.types.is_struct(field.type)
        sub_names = {field.type.field(i).name for i in range(field.type.num_fields)}
        assert _DISCRETE_SPATIAL_SUBFIELDS <= sub_names

    def test_arrow_metadata_stamped(self):
        arrow_schema = DiscreteSpatialSchema.to_arrow_schema()
        field = arrow_schema.field("boxes")
        assert field.metadata is not None
        assert field.metadata[POINTER_FEATURE_SPACE_METADATA_KEY] == _DS_FS.encode("utf-8")

    def test_infer_from_arrow_recovers_kind(self):
        arrow_schema = DiscreteSpatialSchema.to_arrow_schema()
        pfs = _infer_pointer_fields_from_arrow(arrow_schema)
        assert set(pfs.keys()) == {"boxes"}
        assert pfs["boxes"].pointer_kind is PointerKind.DISCRETE_SPATIAL
        assert pfs["boxes"].feature_space == _DS_FS


# ---------------------------------------------------------------------------
# Atlas-level: empty-fill, roundtrip through lance
# ---------------------------------------------------------------------------


@pytest.fixture
def populated_atlas(tmp_path):
    """Atlas with DiscreteSpatialSchema and three manually-ingested cells."""
    atlas_dir = str(tmp_path / "atlas")
    os.makedirs(atlas_dir + "/zarr_store", exist_ok=True)
    store = obstore.store.LocalStore(prefix=atlas_dir + "/zarr_store")

    atlas = RaggedAtlas.create(
        db_uri=atlas_dir,
        cell_table_name="cells",
        cell_schema=DiscreteSpatialSchema,
        store=store,
        registry_schemas={},
        dataset_table_name="datasets",
        dataset_schema=DatasetRecord,
    )

    arrow_schema = DiscreteSpatialSchema.to_arrow_schema()
    n_cells = 3
    min_corners = [[0], [5], [10]]
    max_corners = [[2], [7], [15]]
    dataset_uid = make_uid()
    # Create a real (total_rows, n_features) zarr array at layers/raw so the
    # MultimodalCellDataset read path has something to slice into.
    n_features = 4
    total_rows = 15  # must cover max_corner[0] across all pointers
    _grp = atlas._root.create_group("ds0/boxes")
    _layers = _grp.create_group("layers")
    _arr = _layers.create_array(
        "raw",
        shape=(total_rows, n_features),
        chunks=(total_rows, n_features),
        shards=(total_rows, n_features),
        dtype=np.float32,
    )
    _arr[:] = np.arange(total_rows * n_features, dtype=np.float32).reshape(total_rows, n_features)
    atlas._dataset_table.add(
        pa.Table.from_pylist(
            [
                DatasetRecord(
                    dataset_uid=dataset_uid,
                    zarr_group="ds0/boxes",
                    feature_space=_DS_FS,
                    n_cells=n_cells,
                ).model_dump()
            ],
            schema=DatasetRecord.to_arrow_schema(),
        )
    )
    boxes = pa.StructArray.from_arrays(
        [
            pa.array(["ds0/boxes"] * n_cells, type=pa.string()),
            pa.array(min_corners, type=pa.list_(pa.int64())),
            pa.array(max_corners, type=pa.list_(pa.int64())),
        ],
        names=["zarr_group", "min_corner", "max_corner"],
    )
    columns = {
        "uid": pa.array([make_uid() for _ in range(n_cells)], type=pa.string()),
        "dataset_uid": pa.array([dataset_uid] * n_cells, type=pa.string()),
        "boxes": boxes,
        "cell_type": pa.array(["a", "b", "c"], type=pa.string()),
    }
    atlas.cell_table.add(pa.table(columns, schema=arrow_schema))
    atlas.snapshot()

    atlas = RaggedAtlas.checkout_latest(atlas_dir, DiscreteSpatialSchema, store=store)
    return atlas, atlas_dir, store, (min_corners, max_corners)


def _expected_slab(
    min_corners: list[list[int]], max_corners: list[list[int]], n_features: int, i: int
) -> np.ndarray:
    """Reconstruct the flat ``[min_corner[0]:max_corner[0], :]`` slab for cell *i*."""
    lo = min_corners[i][0]
    hi = max_corners[i][0]
    total_cols = n_features
    # mirrors the fixture's writer: np.arange((total_rows, n_features)).
    return np.arange(lo * total_cols, hi * total_cols, dtype=np.float32).reshape(
        hi - lo, total_cols
    )


class TestAtlasRoundtrip:
    def test_write_and_read_rows_preserves_corners(self, populated_atlas):
        atlas, _, _, (min_corners, max_corners) = populated_atlas
        df = atlas.cell_table.to_polars().collect().sort("cell_type")
        unnested = df["boxes"].struct.unnest()
        assert unnested["zarr_group"].to_list() == ["ds0/boxes"] * len(min_corners)
        assert unnested["min_corner"].to_list() == min_corners
        assert unnested["max_corner"].to_list() == max_corners

    def test_schemaless_open_recovers_pointer_kind(self, populated_atlas):
        atlas, _, store, _ = populated_atlas
        reopened = RaggedAtlas.open(
            db_uri=atlas._db_uri,
            cell_table_name=atlas.cell_table.name,
            cell_schema=None,
            store=store,
        )
        pf = reopened._pointer_fields["boxes"]
        assert pf.pointer_kind is PointerKind.DISCRETE_SPATIAL
        assert pf.feature_space == _DS_FS


class TestMultimodalDiscreteSpatialRoundtrip:
    def test_batch_slabs_match_zarr(self, populated_atlas):
        atlas, _, _, (min_corners, max_corners) = populated_atlas
        # Pull cell_type with _rowid so we can sort the requested order back
        # to the fixture's a/b/c insertion order.
        cells_pl = (
            atlas.cell_table.search()
            .where("boxes.zarr_group != ''")
            .with_row_id(True)
            .select(["boxes", "cell_type"])
            .to_polars()
        )
        assert cells_pl.height == len(min_corners)

        ds = MultimodalCellDataset(
            atlas,
            cells_pl,
            field_names=["boxes"],
            layers={"boxes": "raw"},
            metadata_columns=["cell_type"],
        )
        # Query order on the cell table isn't guaranteed to match our insert
        # order; sort by cell_type (a/b/c) to align with the fixture's
        # min_corners/max_corners lists.
        order = np.argsort(cells_pl["cell_type"].to_numpy())
        batch = ds.__getitems__(order.tolist())

        assert batch.n_cells == 3
        assert set(batch.modalities.keys()) == {"boxes"}
        assert batch.present["boxes"].all()

        sub = batch.modalities["boxes"]
        assert isinstance(sub, DiscreteSpatialBatch)
        n_features = sub.n_features
        assert sub.n_features == 4
        expected_lengths = [max_corners[i][0] - min_corners[i][0] for i in range(len(min_corners))]
        np.testing.assert_array_equal(
            np.diff(sub.offsets), np.array(expected_lengths, dtype=np.int64)
        )
        for i in range(3):
            lo, hi = int(sub.offsets[i]), int(sub.offsets[i + 1])
            np.testing.assert_array_equal(
                sub.data[lo:hi],
                _expected_slab(min_corners, max_corners, n_features, i),
            )
        assert batch.metadata["cell_type"].tolist() == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Core read/write paths must raise NotImplementedError for the new kind.
# ---------------------------------------------------------------------------


class TestCoreRaisesNotImplemented:
    def test_add_from_anndata_raises(self, populated_atlas):
        atlas, _, _, _ = populated_atlas
        adata = ad.AnnData(X=sp.random(2, 4, density=0.5, format="csr", dtype=np.float32))
        with pytest.raises(NotImplementedError, match="DiscreteSpatialPointer ingestion"):
            add_from_anndata(
                atlas,
                adata,
                field_name="boxes",
                zarr_layer="raw",
                dataset_record=DatasetRecord(
                    zarr_group="ds_extra/x",
                    feature_space=_DS_FS,
                    n_cells=2,
                ),
            )

    def test_cell_dataset_raises(self, populated_atlas):
        atlas, _, _, _ = populated_atlas
        cells_pl = atlas.query()._materialize_cells_for_dataset()
        with pytest.raises(NotImplementedError, match="not supported by the homeobox dataloader"):
            CellDataset(atlas, cells_pl, field_name="boxes")

    def test_dex_extract_matrix_raises(self):
        adata = ad.AnnData(X=np.zeros((2, 3), dtype=np.float32))
        with pytest.raises(NotImplementedError, match="DiscreteSpatialPointer"):
            _extract_matrix(adata, PointerKind.DISCRETE_SPATIAL)

    def test_dex_compare_raises(self):
        features = np.array(["f1", "f2"])
        dummy = np.zeros((1, 2))
        with pytest.raises(NotImplementedError, match="DiscreteSpatialPointer"):
            _compare(
                target_matrix=dummy,
                control_matrix=dummy,
                pointer_kind=PointerKind.DISCRETE_SPATIAL,
                test="ttest",
                target_sum=1e4,
                geometric_mean=False,
                feature_names=features,
            )

    def test_dex_top_level_raises(self, populated_atlas):
        atlas, _, _, _ = populated_atlas
        with pytest.raises(NotImplementedError, match="DiscreteSpatialPointer"):
            dex(
                atlas,
                feature_space=_DS_FS,
                groupby="cell_type",
                control="a",
                target=["b"],
                test="ttest",
            )


# ---------------------------------------------------------------------------
# Schema-level kind mismatch is caught at class-definition time.
# ---------------------------------------------------------------------------


class TestKindMismatch:
    def test_sparse_annotation_on_discrete_spec_raises(self):
        with pytest.raises(TypeError, match="does not match feature_space"):

            class Bad(HoxBaseSchema):
                boxes: SparseZarrPointer | None = PointerField.declare(feature_space=_DS_FS)
