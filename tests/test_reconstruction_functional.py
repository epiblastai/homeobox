from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from homeobox.batch_types import DenseFeatureBatch, SparseBatch, SpatialTileBatch
from homeobox.group_reader import LayoutReader
from homeobox.group_specs import FeatureSpaceSpec, LayersSpec, ZarrGroupSpec, get_spec
from homeobox.pointer_types import DiscreteSpatialPointer, SparseZarrPointer
from homeobox.reconstruction_functional import (
    RowOrderMapping,
    concat_remapped_batches,
    get_array_paths_to_read,
    read_arrays_by_group,
    remap_sparse_indices_and_values,
    reorder_batch_rows,
)
from homeobox.reconstructor_base import Reconstructor

# ---------------------------------------------------------------------------
# Fakes for read_arrays_by_group tests
# ---------------------------------------------------------------------------


class _FakeArrayReader:
    """Records read calls and returns deterministic synthetic results."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.range_calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.box_calls: list[tuple[np.ndarray, np.ndarray, bool]] = []

    async def read_ranges(self, starts: np.ndarray, ends: np.ndarray):
        self.range_calls.append((starts.copy(), ends.copy()))
        lengths = (ends - starts).astype(np.int64)
        flat = (
            np.concatenate(
                [np.arange(s, e, dtype=np.int32) for s, e in zip(starts, ends, strict=True)]
            )
            if len(starts)
            else np.array([], dtype=np.int32)
        )
        return flat, lengths

    async def read_boxes(
        self,
        min_corners: np.ndarray,
        max_corners: np.ndarray,
        *,
        stack_uniform: bool = True,
    ):
        self.box_calls.append((min_corners.copy(), max_corners.copy(), stack_uniform))
        tiles = [
            np.full(tuple(int(d) for d in (mx - mn)), fill_value=int(mn[0]), dtype=np.int32)
            for mn, mx in zip(min_corners, max_corners, strict=True)
        ]
        if stack_uniform:
            return np.stack(tiles, axis=0) if tiles else np.empty((0,), dtype=np.int32)
        return tiles


class _FakeGroupReader:
    def __init__(self, readers: dict[str, _FakeArrayReader]) -> None:
        self.readers = readers
        self.layout_reader: LayoutReader | None = None

    def get_array_reader(self, name: str) -> _FakeArrayReader:
        return self.readers[name]


class _FakeSparseReconstructor(Reconstructor):
    read_method = "ranges"

    def build_group_batch(self, group_reader, group_rows, layer_names, results):
        flat_indices, lengths = results[0]
        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        layers = {ln: vals for ln, (vals, _) in zip(layer_names, results[1:], strict=True)}
        return SparseBatch(
            indices=flat_indices,
            offsets=offsets,
            layers=layers,
            n_features=0,
            metadata=group_rows,
        )


class _FakeBoxReconstructor(Reconstructor):
    read_method = "boxes"
    stack_uniform = False

    def build_group_batch(self, group_reader, group_rows, layer_names, results):
        layers: dict[str, list[np.ndarray]] = {}
        for ln, group_data in zip(layer_names, results, strict=True):
            if isinstance(group_data, list):
                layers[ln] = group_data
            else:
                layers[ln] = [group_data[i] for i in range(group_data.shape[0])]
        return SpatialTileBatch(layers=layers, metadata=group_rows)


def test_get_array_paths_to_read_uses_required_arrays_and_default_layers():
    spec = get_spec("gene_expression")

    required, layers = get_array_paths_to_read(spec)

    assert required == ["csr/indices"]
    assert layers == {"counts": "csr/layers/counts"}


def test_get_array_paths_to_read_respects_layer_overrides():
    spec = get_spec("gene_expression")

    required, layers = get_array_paths_to_read(spec, ["log_normalized", "tpm"])

    assert required == ["csr/indices"]
    assert layers == {
        "log_normalized": "csr/layers/log_normalized",
        "tpm": "csr/layers/tpm",
    }


def test_get_array_paths_to_read_rejects_specs_with_nothing_to_read():
    spec = FeatureSpaceSpec(
        feature_space="empty",
        pointer_type=object,
        reconstructor=Reconstructor(),
        zarr_group_spec=ZarrGroupSpec(layers=LayersSpec(required=[])),
    )

    with pytest.raises(Exception, match="cannot both be empty"):
        get_array_paths_to_read(spec)


def test_remap_sparse_indices_and_values_filters_missing_features_per_row():
    remapping_array = np.array([10, -1, 11, 12, -1], dtype=np.int32)
    flat_indices = np.array([0, 1, 2, 4, 3, 1], dtype=np.uint32)
    values = {
        "counts": np.array([1, 2, 3, 4, 5, 6], dtype=np.uint32),
        "log_normalized": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
    }
    lengths = np.array([3, 3], dtype=np.int64)

    remapped, filtered_values, filtered_lengths = remap_sparse_indices_and_values(
        remapping_array,
        flat_indices,
        values,
        lengths,
    )

    np.testing.assert_array_equal(remapped, np.array([10, 11, 12], dtype=np.int32))
    np.testing.assert_array_equal(filtered_values["counts"], np.array([1, 3, 5], dtype=np.uint32))
    np.testing.assert_allclose(
        filtered_values["log_normalized"],
        np.array([0.1, 0.3, 0.5], dtype=np.float32),
    )
    np.testing.assert_array_equal(filtered_lengths, np.array([2, 1], dtype=np.int64))


def test_remap_sparse_indices_and_values_keeps_lengths_when_all_features_are_present():
    remapping_array = np.array([4, 5, 6], dtype=np.int32)
    flat_indices = np.array([2, 0, 1], dtype=np.uint32)
    values = {"counts": np.array([7, 8, 9], dtype=np.uint32)}
    lengths = np.array([1, 2], dtype=np.int64)

    remapped, filtered_values, filtered_lengths = remap_sparse_indices_and_values(
        remapping_array,
        flat_indices,
        values,
        lengths,
    )

    np.testing.assert_array_equal(remapped, np.array([6, 4, 5], dtype=np.int32))
    assert filtered_values is values
    assert filtered_lengths is lengths


# ---------------------------------------------------------------------------
# read_arrays_by_group
# ---------------------------------------------------------------------------


def test_read_arrays_by_group_dispatches_ranges_per_group():
    obs = pl.DataFrame(
        {
            "_zg": ["g0", "g0", "g1"],
            "_start": [1, 4, 10],
            "_end": [3, 9, 14],
        }
    )
    g0_idx = _FakeArrayReader("indices")
    g0_lyr = _FakeArrayReader("counts")
    g1_idx = _FakeArrayReader("indices")
    g1_lyr = _FakeArrayReader("counts")
    group_readers = {
        "g0": _FakeGroupReader({"csr/indices": g0_idx, "csr/layers/counts": g0_lyr}),
        "g1": _FakeGroupReader({"csr/indices": g1_idx, "csr/layers/counts": g1_lyr}),
    }
    spec = SimpleNamespace(
        pointer_type=SparseZarrPointer,
        reconstructor=_FakeSparseReconstructor(),
    )

    batches = read_arrays_by_group(
        group_readers,
        obs.group_by("_zg", maintain_order=True),
        spec,
        ["csr/indices"],
        {"counts": "csr/layers/counts"},
    )

    assert [zg for zg, _ in batches] == ["g0", "g1"]
    np.testing.assert_array_equal(g0_idx.range_calls[0][0], np.array([1, 4]))
    np.testing.assert_array_equal(g0_idx.range_calls[0][1], np.array([3, 9]))
    np.testing.assert_array_equal(g1_idx.range_calls[0][0], np.array([10]))
    np.testing.assert_array_equal(g1_idx.range_calls[0][1], np.array([14]))

    g0_batch = batches[0][1]
    assert isinstance(g0_batch, SparseBatch)
    np.testing.assert_array_equal(g0_batch.offsets, np.array([0, 2, 7]))
    assert "counts" in g0_batch.layers
    # Metadata is the full group obs DataFrame (internal columns included).
    assert g0_batch.metadata is not None
    assert g0_batch.metadata.height == 2


def test_read_arrays_by_group_dispatches_boxes_with_stack_uniform_from_reconstructor():
    obs = pl.DataFrame(
        {
            "_zg": ["g0", "g1"],
            "_min_corner": [[1, 2], [5, 6]],
            "_max_corner": [[4, 8], [9, 11]],
        }
    )
    g0_raw = _FakeArrayReader("raw")
    g1_raw = _FakeArrayReader("raw")
    group_readers = {
        "g0": _FakeGroupReader({"layers/raw": g0_raw}),
        "g1": _FakeGroupReader({"layers/raw": g1_raw}),
    }
    spec = SimpleNamespace(
        pointer_type=DiscreteSpatialPointer,
        reconstructor=_FakeBoxReconstructor(),
    )

    batches = read_arrays_by_group(
        group_readers,
        obs.group_by("_zg", maintain_order=True),
        spec,
        [],
        {"raw": "layers/raw"},
    )

    assert [zg for zg, _ in batches] == ["g0", "g1"]
    # stack_uniform is read off the reconstructor (False), forwarded to read_boxes.
    assert g0_raw.box_calls[0][2] is False
    assert g1_raw.box_calls[0][2] is False
    np.testing.assert_array_equal(g0_raw.box_calls[0][0], np.array([[1, 2]]))
    np.testing.assert_array_equal(g0_raw.box_calls[0][1], np.array([[4, 8]]))

    g0_batch = batches[0][1]
    assert isinstance(g0_batch, SpatialTileBatch)
    assert len(g0_batch.layers["raw"]) == 1
    assert g0_batch.layers["raw"][0].shape == (3, 6)


# ---------------------------------------------------------------------------
# concat_remapped_batches
# ---------------------------------------------------------------------------


def _layout(remap: list[int]) -> LayoutReader:
    return LayoutReader.from_remap(layout_uid="t", remap=np.array(remap, dtype=np.int32))


def test_concat_remapped_batches_sparse_remaps_and_concatenates():
    # Two groups, joined feature space size 4; each group has its own local layout.
    g0 = SparseBatch(
        indices=np.array([0, 1, 0], dtype=np.int32),
        offsets=np.array([0, 2, 3], dtype=np.int64),
        layers={"counts": np.array([10, 20, 30], dtype=np.int32)},
        n_features=2,
        metadata=pl.DataFrame({"r": [0, 1]}),
    )
    g1 = SparseBatch(
        indices=np.array([1], dtype=np.int32),
        offsets=np.array([0, 1], dtype=np.int64),
        layers={"counts": np.array([40], dtype=np.int32)},
        n_features=2,
        metadata=pl.DataFrame({"r": [2]}),
    )
    layouts = {"g0": _layout([2, 0]), "g1": _layout([3, 1])}

    batch = concat_remapped_batches(
        [("g0", g0), ("g1", g1)],
        layouts_per_group=layouts,
        n_features=4,
    )

    assert isinstance(batch, SparseBatch)
    np.testing.assert_array_equal(batch.indices, np.array([2, 0, 2, 1], dtype=np.int32))
    np.testing.assert_array_equal(batch.offsets, np.array([0, 2, 3, 4], dtype=np.int64))
    np.testing.assert_array_equal(batch.layers["counts"], np.array([10, 20, 30, 40]))
    assert batch.metadata["r"].to_list() == [0, 1, 2]
    assert batch.n_features == 4


def test_concat_remapped_batches_dense_scatters_into_joined_columns():
    g0 = DenseFeatureBatch(
        layers={"x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)},
        n_features=2,
        metadata=pl.DataFrame({"r": [0, 1]}),
    )
    g1 = DenseFeatureBatch(
        layers={"x": np.array([[5.0, 6.0]], dtype=np.float32)},
        n_features=2,
        metadata=pl.DataFrame({"r": [2]}),
    )
    # g0 local cols [0, 1] -> joined [2, 0]; g1 local [0, 1] -> joined [1, 3]
    layouts = {"g0": _layout([2, 0]), "g1": _layout([1, 3])}

    batch = concat_remapped_batches(
        [("g0", g0), ("g1", g1)],
        layouts_per_group=layouts,
        n_features=4,
    )

    assert isinstance(batch, DenseFeatureBatch)
    np.testing.assert_array_equal(
        batch.layers["x"],
        np.array(
            [
                [2.0, 0.0, 1.0, 0.0],
                [4.0, 0.0, 3.0, 0.0],
                [0.0, 5.0, 0.0, 6.0],
            ],
            dtype=np.float32,
        ),
    )
    assert batch.metadata["r"].to_list() == [0, 1, 2]


def test_concat_remapped_batches_dense_intersection_drops_negative_one_columns():
    # Local col 1 has joined index -1 -> dropped from output.
    g0 = DenseFeatureBatch(
        layers={"x": np.array([[1.0, 2.0]], dtype=np.float32)},
        n_features=2,
        metadata=pl.DataFrame({"r": [0]}),
    )
    layouts = {"g0": _layout([0, -1])}

    batch = concat_remapped_batches(
        [("g0", g0)],
        layouts_per_group=layouts,
        n_features=2,
    )

    np.testing.assert_array_equal(batch.layers["x"], np.array([[1.0, 0.0]], dtype=np.float32))


def test_concat_remapped_batches_spatial_concatenates_lists_and_ignores_layouts():
    g0 = SpatialTileBatch(
        layers={"raw": [np.zeros((2, 2)), np.ones((3, 3))]},
        metadata=pl.DataFrame({"r": [0, 1]}),
    )
    g1 = SpatialTileBatch(
        layers={"raw": [np.full((1, 1), 5.0)]},
        metadata=pl.DataFrame({"r": [2]}),
    )

    batch = concat_remapped_batches(
        [("g0", g0), ("g1", g1)],
        layouts_per_group=None,
        n_features=0,
    )

    assert isinstance(batch, SpatialTileBatch)
    assert len(batch.layers["raw"]) == 3
    assert batch.layers["raw"][0].shape == (2, 2)
    assert batch.layers["raw"][2].shape == (1, 1)
    assert batch.metadata["r"].to_list() == [0, 1, 2]


def test_concat_remapped_batches_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one batch"):
        concat_remapped_batches([], layouts_per_group=None, n_features=0)


# ---------------------------------------------------------------------------
# RowOrderMapping / reorder_batch_rows
# ---------------------------------------------------------------------------


def test_row_order_mapping_length_mismatch_raises():
    batch = DenseFeatureBatch(
        layers={"x": np.zeros((2, 1), dtype=np.float32)},
        n_features=1,
    )
    mapping = RowOrderMapping(
        source_row_ids=np.array([10, 20]),
        target_row_ids=np.array([10, 20, 30]),
    )
    with pytest.raises(ValueError, match="length mismatch"):
        reorder_batch_rows(batch, mapping)


def test_reorder_batch_rows_dense_permutes_rows_and_metadata():
    batch = DenseFeatureBatch(
        layers={"x": np.array([[1.0], [2.0], [3.0]], dtype=np.float32)},
        n_features=1,
        metadata=pl.DataFrame({"_rowid": [10, 20, 30], "tag": ["a", "b", "c"]}),
    )
    # Want rows in order [30, 10, 20]
    mapping = RowOrderMapping(
        source_row_ids=np.array([10, 20, 30]),
        target_row_ids=np.array([30, 10, 20]),
    )

    out = reorder_batch_rows(batch, mapping)

    np.testing.assert_array_equal(
        out.layers["x"], np.array([[3.0], [1.0], [2.0]], dtype=np.float32)
    )
    assert out.metadata["tag"].to_list() == ["c", "a", "b"]


def test_reorder_batch_rows_sparse_permutes_offsets_indices_and_values():
    # Three rows: row0 has 1 nnz, row1 has 0, row2 has 2 nnz.
    batch = SparseBatch(
        indices=np.array([5, 7, 8], dtype=np.int32),
        offsets=np.array([0, 1, 1, 3], dtype=np.int64),
        layers={"counts": np.array([100, 200, 300], dtype=np.int32)},
        n_features=10,
        metadata=pl.DataFrame({"_rowid": [10, 20, 30]}),
    )
    # Want rows in order [30, 10, 20]
    mapping = RowOrderMapping(
        source_row_ids=np.array([10, 20, 30]),
        target_row_ids=np.array([30, 10, 20]),
    )

    out = reorder_batch_rows(batch, mapping)

    np.testing.assert_array_equal(out.offsets, np.array([0, 2, 3, 3], dtype=np.int64))
    np.testing.assert_array_equal(out.indices, np.array([7, 8, 5], dtype=np.int32))
    np.testing.assert_array_equal(out.layers["counts"], np.array([200, 300, 100]))
    assert out.metadata["_rowid"].to_list() == [30, 10, 20]


def test_reorder_batch_rows_spatial_permutes_per_layer_lists():
    a, b, c = np.full((1, 1), 0.0), np.full((2, 2), 1.0), np.full((3, 3), 2.0)
    batch = SpatialTileBatch(
        layers={"raw": [a, b, c]},
        metadata=pl.DataFrame({"_rowid": [10, 20, 30]}),
    )
    mapping = RowOrderMapping(
        source_row_ids=np.array([10, 20, 30]),
        target_row_ids=np.array([30, 10, 20]),
    )

    out = reorder_batch_rows(batch, mapping)

    assert [t.shape for t in out.layers["raw"]] == [(3, 3), (1, 1), (2, 2)]
    assert out.metadata["_rowid"].to_list() == [30, 10, 20]
