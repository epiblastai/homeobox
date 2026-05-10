import polars as pl

from homeobox.reconstruction import _build_obs_df, _pointer_field_names_from_obs


def test_pointer_field_names_from_obs_matches_registered_pointer_structs():
    obs_pl = pl.DataFrame(
        {
            "uid": ["cell-1"],
            "sparse_ptr": [{"zarr_group": "zg1", "start": 0, "end": 3, "zarr_row": 0}],
            "dense_ptr": [{"zarr_group": "zg1", "position": 4}],
            "spatial_ptr": [{"zarr_group": "zg1", "min_corner": [0, 1], "max_corner": [2, 3]}],
            "metadata_struct": [{"zarr_group": "metadata", "start": 10}],
        }
    )

    assert _pointer_field_names_from_obs(obs_pl) == [
        "sparse_ptr",
        "dense_ptr",
        "spatial_ptr",
    ]


def test_build_obs_df_drops_pointer_and_internal_columns_only():
    obs_pl = pl.DataFrame(
        {
            "uid": ["cell-1"],
            "sample": ["sample-a"],
            "sparse_ptr": [{"zarr_group": "zg1", "start": 0, "end": 3, "zarr_row": 0}],
            "metadata_struct": [{"zarr_group": "metadata", "start": 10}],
            "_zg": ["zg1"],
        }
    )

    obs = _build_obs_df(obs_pl)

    assert obs.index.tolist() == ["cell-1"]
    assert obs.columns.tolist() == ["sample", "metadata_struct"]
