"""Tests for feature spaces sharing one feature-registry table.

A registry table's name derives from its registry *schema class*, so two
feature spaces declaring the same schema resolve to one table, one set of
registrations, and one ``global_index`` space. ``RegistrySpec`` overrides the
derived name in either direction.
"""

import json
import os

import lancedb
import numpy as np
import obstore
import polars as pl
import pytest

from homeobox.atlas import RaggedAtlas, create_or_open_atlas
from homeobox.pointer_types import DenseZarrPointer
from homeobox.schema import (
    AtlasVersionRecord,
    DatasetSchema,
    FeatureBaseSchema,
    HoxBaseSchema,
    PointerField,
    RegistrySpec,
    default_registry_table_name,
)
from homeobox.schema.definitions import _snake_case

# Two feature spaces that carry the same kind of entity: the shape the shared
# registry exists for (a dense panel and a second view of the same panel).
SPACE_A = "protein_abundance"
SPACE_B = "image_features"


class ProteinSchema(FeatureBaseSchema):
    antibody_name: str


class OtherProteinSchema(FeatureBaseSchema):
    antibody_name: str


class TwoSpaceCellSchema(HoxBaseSchema):
    protein_abundance: DenseZarrPointer | None = PointerField.declare(
        feature_space=SPACE_A, feature_registry_schema=ProteinSchema
    )
    image_features: DenseZarrPointer | None = PointerField.declare(
        feature_space=SPACE_B, feature_registry_schema=ProteinSchema
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_atlas(tmp_path, registry_schemas, *, obs_schemas=None, name="atlas"):
    atlas_dir = str(tmp_path / name)
    os.makedirs(atlas_dir + "/zarr_store", exist_ok=True)
    store = obstore.store.LocalStore(prefix=atlas_dir + "/zarr_store")
    atlas = RaggedAtlas.create(
        db_uri=atlas_dir,
        obs_schemas=obs_schemas or {"cells": TwoSpaceCellSchema},
        store=store,
        registry_schemas=registry_schemas,
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
    )
    return atlas, atlas_dir, store


def _shared_atlas(tmp_path, **kwargs):
    """Atlas where both feature spaces declare ProteinSchema (so they share)."""
    return _make_atlas(tmp_path, {SPACE_A: ProteinSchema, SPACE_B: ProteinSchema}, **kwargs)


def _antibodies(n: int, prefix: str = "ab") -> list[ProteinSchema]:
    return [ProteinSchema(uid=f"{prefix}_{i}", antibody_name=f"CD{i}") for i in range(n)]


def _registry_names(atlas_dir: str) -> set[str]:
    db = lancedb.connect(atlas_dir + "/lance_db")
    return {t for t in db.list_tables().tables if t.endswith("_registry")}


# ---------------------------------------------------------------------------
# Name derivation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "class_name,expected",
    [
        ("Gene", "gene"),
        ("ProteinSchema", "protein_schema"),
        ("GeneFeatureSchema", "gene_feature_schema"),
        ("ATACPeak", "atac_peak"),
        ("RNASeqFeature", "rna_seq_feature"),
        ("Gene2Vec", "gene2_vec"),
        ("already_snake", "already_snake"),
        # Collapsing repeated underscores keeps these from becoming two tables.
        ("Protein_Schema", "protein_schema"),
    ],
)
def test_snake_case(class_name, expected):
    assert _snake_case(class_name) == expected


def test_default_registry_table_name_uses_schema_class():
    assert default_registry_table_name(ProteinSchema) == "protein_schema_registry"


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------


def test_default_table_name_comes_from_schema_not_feature_space(tmp_path):
    atlas, atlas_dir, _ = _shared_atlas(tmp_path)
    assert atlas._registry_tables[SPACE_A].name == "protein_schema_registry"
    assert _registry_names(atlas_dir) == {"protein_schema_registry"}


def test_two_spaces_with_same_schema_share_one_table(tmp_path):
    atlas, atlas_dir, _ = _shared_atlas(tmp_path)
    # One table on disk, and one handle in memory for both spaces.
    assert _registry_names(atlas_dir) == {"protein_schema_registry"}
    assert atlas._registry_tables[SPACE_A] is atlas._registry_tables[SPACE_B]


def test_registry_spec_table_name_opts_out_of_sharing(tmp_path):
    atlas, atlas_dir, _ = _make_atlas(
        tmp_path,
        {
            SPACE_A: ProteinSchema,
            SPACE_B: RegistrySpec(ProteinSchema, table_name="image_panel_registry"),
        },
    )
    assert _registry_names(atlas_dir) == {"protein_schema_registry", "image_panel_registry"}
    assert atlas._registry_tables[SPACE_A] is not atlas._registry_tables[SPACE_B]
    assert atlas._registry_tables[SPACE_B].name == "image_panel_registry"


def test_registry_spec_table_name_opts_in_to_sharing(tmp_path):
    """Different schema classes cannot share, but an explicit name can rename."""
    atlas, atlas_dir, _ = _make_atlas(
        tmp_path,
        {
            SPACE_A: RegistrySpec(ProteinSchema, table_name="antibodies_registry"),
            SPACE_B: RegistrySpec(ProteinSchema, table_name="antibodies_registry"),
        },
    )
    assert _registry_names(atlas_dir) == {"antibodies_registry"}
    assert atlas._registry_tables[SPACE_A] is atlas._registry_tables[SPACE_B]


def test_conflicting_schema_classes_for_one_table_raise(tmp_path):
    with pytest.raises(ValueError, match="different registry schemas"):
        _make_atlas(
            tmp_path,
            {
                SPACE_A: RegistrySpec(ProteinSchema, table_name="antibodies_registry"),
                SPACE_B: RegistrySpec(OtherProteinSchema, table_name="antibodies_registry"),
            },
        )


def test_registry_table_name_colliding_with_obs_table_raises(tmp_path):
    with pytest.raises(ValueError, match="collide with other atlas tables"):
        _make_atlas(
            tmp_path,
            {SPACE_A: RegistrySpec(ProteinSchema, table_name="cells")},
            obs_schemas={"cells": TwoSpaceCellSchema},
        )


def test_registry_spec_rejects_non_schema_class():
    # The swapped-argument call RegistrySpec("antibodies_registry").
    with pytest.raises(TypeError, match="FeatureBaseSchema subclass"):
        RegistrySpec("antibodies_registry")


# ---------------------------------------------------------------------------
# Registration and indexing
# ---------------------------------------------------------------------------


def test_features_registered_via_one_space_are_visible_from_the_other(tmp_path):
    atlas, _, _ = _shared_atlas(tmp_path)
    assert atlas.register_features(SPACE_A, _antibodies(3)) == 3

    from_a = atlas.feature_registry(SPACE_A)
    from_b = atlas.feature_registry(SPACE_B)
    assert from_a.sort("uid").equals(from_b.sort("uid"))
    assert sorted(from_b["uid"].to_list()) == ["ab_0", "ab_1", "ab_2"]

    # Registering the same entities through the other space is a no-op.
    assert atlas.register_features(SPACE_B, _antibodies(3)) == 0


def test_shared_registry_gives_both_spaces_one_global_index(tmp_path):
    atlas, _, _ = _shared_atlas(tmp_path)
    atlas.register_features(SPACE_A, _antibodies(4))
    atlas.optimize()

    def _index_map(feature_space: str) -> dict[str, int]:
        df = atlas.feature_registry(feature_space)
        return dict(zip(df["uid"], df["global_index"], strict=True))

    index_a = _index_map(SPACE_A)
    index_b = _index_map(SPACE_B)
    assert index_a == index_b
    assert sorted(index_a.values()) == [0, 1, 2, 3]


def test_optimize_is_idempotent_on_a_shared_registry(tmp_path):
    atlas, _, _ = _shared_atlas(tmp_path)
    atlas.register_features(SPACE_A, _antibodies(4))
    atlas.optimize()
    before = atlas.feature_registry(SPACE_A).sort("uid")

    atlas.optimize()
    after = atlas.feature_registry(SPACE_A).sort("uid")
    assert before.equals(after)


def test_validate_reports_one_error_naming_both_spaces(tmp_path):
    atlas, _, _ = _shared_atlas(tmp_path)
    # Register without reindexing so global_index stays null.
    atlas.register_features(SPACE_A, _antibodies(3))

    errors = atlas.validate()
    registry_errors = [e for e in errors if "global_index" in e]
    assert len(registry_errors) == 1
    assert "protein_schema_registry" in registry_errors[0]
    assert SPACE_A in registry_errors[0]
    assert SPACE_B in registry_errors[0]


def test_iter_managed_tables_yields_shared_registry_once(tmp_path):
    atlas, _, _ = _shared_atlas(tmp_path)
    names = [t.name for t in atlas._iter_managed_tables()]
    assert len(names) == len(set(names))
    assert names.count("protein_schema_registry") == 1


def test_constructor_rejects_independent_handles_for_one_table(tmp_path):
    atlas, atlas_dir, store = _shared_atlas(tmp_path)
    db = lancedb.connect(atlas_dir + "/lance_db")
    # Two separate handles on the same table would drift apart on write.
    with pytest.raises(ValueError, match="independent Table handles"):
        RaggedAtlas(
            db=db,
            obs_tables={"cells": db.open_table("cells")},
            obs_schemas={"cells": TwoSpaceCellSchema},
            root=atlas._root,
            registry_tables={
                SPACE_A: db.open_table("protein_schema_registry"),
                SPACE_B: db.open_table("protein_schema_registry"),
            },
            dataset_table=db.open_table("datasets"),
            version_table=db.open_table("atlas_versions"),
            feature_layouts_table=db.open_table("_feature_layouts"),
        )


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------


def test_snapshot_records_one_table_for_both_spaces(tmp_path):
    atlas, atlas_dir, _ = _shared_atlas(tmp_path)
    atlas.register_features(SPACE_A, _antibodies(3))
    atlas.optimize()
    atlas.snapshot()

    record = RaggedAtlas.list_versions(atlas_dir).row(-1, named=True)
    names = json.loads(record["registry_table_names"])
    versions = json.loads(record["registry_table_versions"])
    assert names == {SPACE_A: "protein_schema_registry", SPACE_B: "protein_schema_registry"}
    assert versions[SPACE_A] == versions[SPACE_B]


def test_checkout_shares_one_handle(tmp_path):
    atlas, atlas_dir, store = _shared_atlas(tmp_path)
    atlas.register_features(SPACE_A, _antibodies(3))
    atlas.optimize()
    version = atlas.snapshot()

    reopened = RaggedAtlas.checkout(
        atlas_dir, version, obs_schemas={"cells": TwoSpaceCellSchema}, store=store
    )
    assert reopened._registry_tables[SPACE_A] is reopened._registry_tables[SPACE_B]
    assert reopened.feature_registry(SPACE_B).height == 3


def test_restore_restores_shared_table_once(tmp_path):
    """Regression: restore() used to run once per feature space.

    The second restore committed on top of the first and left the earlier
    handle stale, so the next snapshot() refused with a stale-handle error.
    """
    atlas, atlas_dir, store = _shared_atlas(tmp_path)
    atlas.register_features(SPACE_A, _antibodies(3))
    atlas.optimize()
    v0 = atlas.snapshot()

    atlas.register_features(SPACE_A, _antibodies(6))
    atlas.optimize()
    atlas.snapshot()
    assert atlas.feature_registry(SPACE_A).height == 6

    restored = RaggedAtlas.restore(
        atlas_dir, v0, obs_schemas={"cells": TwoSpaceCellSchema}, store=store
    )
    assert restored.feature_registry(SPACE_A).height == 3
    assert restored._registry_tables[SPACE_A] is restored._registry_tables[SPACE_B]
    # The restored handles must not be stale relative to on-disk state.
    restored.snapshot()


def test_checkout_raises_on_divergent_recorded_versions(tmp_path):
    atlas, atlas_dir, store = _shared_atlas(tmp_path)
    atlas.register_features(SPACE_A, _antibodies(3))
    atlas.optimize()
    atlas.snapshot()

    # Hand-write an inconsistent record: one table, two recorded versions.
    db = lancedb.connect(atlas_dir + "/lance_db")
    version_table = db.open_table("atlas_versions")
    row = version_table.search().to_polars().sort("version").row(-1, named=True)
    version_table.add(
        [
            AtlasVersionRecord(
                version=row["version"] + 1,
                obs_table_versions=row["obs_table_versions"],
                dataset_table_name=row["dataset_table_name"],
                dataset_table_version=row["dataset_table_version"],
                registry_table_names=json.dumps(
                    {SPACE_A: "protein_schema_registry", SPACE_B: "protein_schema_registry"}
                ),
                registry_table_versions=json.dumps({SPACE_A: 1, SPACE_B: 2}),
                feature_layouts_table_version=row["feature_layouts_table_version"],
                total_rows=row["total_rows"],
            )
        ]
    )

    with pytest.raises(ValueError, match="different Lance versions"):
        RaggedAtlas.checkout(
            atlas_dir,
            row["version"] + 1,
            obs_schemas={"cells": TwoSpaceCellSchema},
            store=store,
        )


def test_open_infers_shared_registry_from_version_record(tmp_path):
    atlas, atlas_dir, store = _shared_atlas(tmp_path)
    atlas.register_features(SPACE_A, _antibodies(3))
    atlas.optimize()
    atlas.snapshot()

    reopened = RaggedAtlas.open(
        db_uri=atlas_dir, obs_schemas={"cells": TwoSpaceCellSchema}, store=store
    )
    assert reopened._registry_tables[SPACE_A] is reopened._registry_tables[SPACE_B]
    assert reopened._registry_tables[SPACE_B].name == "protein_schema_registry"


# ---------------------------------------------------------------------------
# create_or_open_atlas
# ---------------------------------------------------------------------------


def test_create_or_open_atlas_creates_one_shared_registry(tmp_path):
    atlas_path = str(tmp_path / "coa")
    atlas = create_or_open_atlas(
        atlas_path,
        obs_schemas={"cells": TwoSpaceCellSchema},
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
        registry_schemas={SPACE_A: ProteinSchema, SPACE_B: ProteinSchema},
    )
    assert atlas._registry_tables[SPACE_A] is atlas._registry_tables[SPACE_B]

    reopened = create_or_open_atlas(
        atlas_path,
        obs_schemas={"cells": TwoSpaceCellSchema},
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
        registry_schemas={SPACE_A: ProteinSchema, SPACE_B: ProteinSchema},
    )
    assert reopened._registry_tables[SPACE_A] is reopened._registry_tables[SPACE_B]
    assert reopened._registry_tables[SPACE_A].name == "protein_schema_registry"


def test_create_or_open_atlas_reopen_resolves_legacy_names(tmp_path):
    """An atlas created under the old {fs}_registry convention still reopens."""
    atlas_path = str(tmp_path / "legacy")
    create_or_open_atlas(
        atlas_path,
        obs_schemas={"cells": TwoSpaceCellSchema},
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
        registry_schemas={
            SPACE_A: RegistrySpec(ProteinSchema, table_name=f"{SPACE_A}_registry"),
            SPACE_B: RegistrySpec(ProteinSchema, table_name=f"{SPACE_B}_registry"),
        },
    )
    before = set(lancedb.connect(atlas_path + "/lance_db").list_tables().tables)

    # Reopened with bare classes, which would derive protein_schema_registry.
    reopened = create_or_open_atlas(
        atlas_path,
        obs_schemas={"cells": TwoSpaceCellSchema},
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
        registry_schemas={SPACE_A: ProteinSchema, SPACE_B: ProteinSchema},
    )
    assert reopened._registry_tables[SPACE_A].name == f"{SPACE_A}_registry"
    assert reopened._registry_tables[SPACE_B].name == f"{SPACE_B}_registry"
    assert set(lancedb.connect(atlas_path + "/lance_db").list_tables().tables) == before


def test_create_or_open_atlas_reopen_missing_registry_raises(tmp_path):
    atlas_path = str(tmp_path / "partial")
    create_or_open_atlas(
        atlas_path,
        obs_schemas={"cells": TwoSpaceCellSchema},
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
        registry_schemas={SPACE_A: ProteinSchema},
    )
    # Adding a feature space whose registry does not exist is not supported.
    with pytest.raises(ValueError, match="only created when the atlas is initialised"):
        create_or_open_atlas(
            atlas_path,
            obs_schemas={"cells": TwoSpaceCellSchema},
            dataset_table_name="datasets",
            dataset_schema=DatasetSchema,
            registry_schemas={
                SPACE_A: ProteinSchema,
                SPACE_B: RegistrySpec(ProteinSchema, table_name="never_created_registry"),
            },
        )


def test_create_or_open_atlas_reopen_schema_mismatch_raises(tmp_path):
    atlas_path = str(tmp_path / "mismatch")
    create_or_open_atlas(
        atlas_path,
        obs_schemas={"cells": TwoSpaceCellSchema},
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
        registry_schemas={SPACE_A: RegistrySpec(ProteinSchema, table_name="panel_registry")},
    )

    class DifferentSchema(FeatureBaseSchema):
        wavelength: int

    with pytest.raises(ValueError, match="does not match the declared schema"):
        create_or_open_atlas(
            atlas_path,
            obs_schemas={"cells": TwoSpaceCellSchema},
            dataset_table_name="datasets",
            dataset_schema=DatasetSchema,
            registry_schemas={SPACE_A: RegistrySpec(DifferentSchema, table_name="panel_registry")},
        )


# ---------------------------------------------------------------------------
# Ingestion through a shared registry
# ---------------------------------------------------------------------------


def test_ingest_both_spaces_against_a_shared_registry(tmp_path):
    """The payoff: one column ordering is valid for both feature spaces."""
    import anndata as ad

    from homeobox.ingestion import add_from_anndata
    from homeobox.obs_alignment import align_obs_to_schema

    atlas, atlas_dir, store = _shared_atlas(tmp_path)
    uids = [f"ab_{i}" for i in range(3)]
    atlas.register_features(SPACE_A, _antibodies(3))

    var = pl.DataFrame({"uid": uids, "antibody_name": [f"CD{i}" for i in range(3)]}).to_pandas()
    for field_name, group in ((SPACE_A, "ds/protein"), (SPACE_B, "ds/image")):
        adata = ad.AnnData(
            X=np.arange(6, dtype=np.float32).reshape(2, 3),
            obs={"placeholder": ["a", "b"]},
            var=var.copy(),
        )
        adata.X = adata.X.astype(np.uint32 if field_name == SPACE_A else np.float32)
        adata = align_obs_to_schema(adata, TwoSpaceCellSchema)
        add_from_anndata(
            atlas,
            adata,
            field_name=field_name,
            zarr_layer="counts" if field_name == SPACE_A else "ctrl_standardized",
            dataset_record=DatasetSchema(zarr_group=group, feature_space=field_name),
        )

    atlas.optimize()

    # Both datasets resolved their features against the same registry, so the
    # uid -> global_index map is identical from either feature space.
    layouts = atlas._feature_layouts_table.search().to_polars()
    per_uid = layouts.group_by("feature_uid").agg(pl.col("global_index").n_unique())
    assert per_uid["global_index"].to_list() == [1, 1, 1]
