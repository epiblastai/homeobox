from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from homeobox.group_specs import FeatureSpaceSpec, LayersSpec, ZarrGroupSpec, get_spec
from homeobox.pointer_types import DenseZarrPointer, DiscreteSpatialPointer, SparseZarrPointer
from homeobox.reconstruction_functional import (
    get_array_paths_to_read,
    read_arrays_by_group,
    remap_sparse_indices_and_values,
)
from homeobox.reconstructor_base import Reconstructor


class FakeArrayReader:
    def __init__(self, name: str) -> None:
        self.name = name
        self.range_calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.box_calls: list[tuple[np.ndarray, np.ndarray, bool]] = []

    async def read_ranges(self, starts: np.ndarray, ends: np.ndarray):
        self.range_calls.append((starts.copy(), ends.copy()))
        lengths = ends - starts
        return starts + len(self.name), lengths

    async def read_boxes(
        self,
        min_corners: np.ndarray,
        max_corners: np.ndarray,
        *,
        stack_uniform: bool = True,
    ):
        self.box_calls.append((min_corners.copy(), max_corners.copy(), stack_uniform))
        return max_corners - min_corners


class FakeGroupReader:
    def __init__(self, readers: dict[str, FakeArrayReader]) -> None:
        self.readers = readers

    def get_array_reader(self, array_name: str) -> FakeArrayReader:
        return self.readers[array_name]


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


def test_read_arrays_by_group_reads_ranges_for_each_group_and_array():
    obs = pl.DataFrame(
        {
            "_zg": ["g0", "g0", "g1"],
            "_start": [1, 4, 10],
            "_end": [3, 9, 14],
        }
    )
    g0_indices = FakeArrayReader("indices")
    g0_counts = FakeArrayReader("counts")
    g1_indices = FakeArrayReader("indices")
    g1_counts = FakeArrayReader("counts")
    group_readers = {
        "g0": FakeGroupReader({"csr/indices": g0_indices, "csr/layers/counts": g0_counts}),
        "g1": FakeGroupReader({"csr/indices": g1_indices, "csr/layers/counts": g1_counts}),
    }
    spec = SimpleNamespace(pointer_type=SparseZarrPointer)

    group_obs_data, results = read_arrays_by_group(
        group_readers,
        obs.group_by("_zg", maintain_order=True),
        spec,
        ["csr/indices", "csr/layers/counts"],
        "ranges",
    )

    assert [zg for zg, _rows in group_obs_data] == ["g0", "g1"]
    np.testing.assert_array_equal(g0_indices.range_calls[0][0], np.array([1, 4]))
    np.testing.assert_array_equal(g0_indices.range_calls[0][1], np.array([3, 9]))
    np.testing.assert_array_equal(g1_counts.range_calls[0][0], np.array([10]))
    np.testing.assert_array_equal(g1_counts.range_calls[0][1], np.array([14]))

    g0_results, g1_results = results
    np.testing.assert_array_equal(g0_results[0][0], np.array([8, 11]))
    np.testing.assert_array_equal(g0_results[0][1], np.array([2, 5]))
    np.testing.assert_array_equal(g0_results[1][0], np.array([7, 10]))
    np.testing.assert_array_equal(g1_results[0][0], np.array([17]))
    np.testing.assert_array_equal(g1_results[1][1], np.array([4]))


def test_read_arrays_by_group_reads_boxes_and_forwards_stack_uniform():
    obs = pl.DataFrame(
        {
            "_zg": ["g0", "g1"],
            "_min_corner": [[1, 2], [5, 6]],
            "_max_corner": [[4, 8], [9, 11]],
        }
    )
    g0_raw = FakeArrayReader("raw")
    g1_raw = FakeArrayReader("raw")
    group_readers = {
        "g0": FakeGroupReader({"layers/raw": g0_raw}),
        "g1": FakeGroupReader({"layers/raw": g1_raw}),
    }
    spec = SimpleNamespace(pointer_type=DiscreteSpatialPointer)

    group_obs_data, results = read_arrays_by_group(
        group_readers,
        obs.group_by("_zg", maintain_order=True),
        spec,
        ["layers/raw"],
        "boxes",
        stack_uniform=False,
    )

    assert [zg for zg, _rows in group_obs_data] == ["g0", "g1"]
    np.testing.assert_array_equal(g0_raw.box_calls[0][0], np.array([[1, 2]]))
    np.testing.assert_array_equal(g0_raw.box_calls[0][1], np.array([[4, 8]]))
    assert g0_raw.box_calls[0][2] is False
    np.testing.assert_array_equal(g1_raw.box_calls[0][0], np.array([[5, 6]]))
    np.testing.assert_array_equal(results[0][0], np.array([[3, 6]]))
    np.testing.assert_array_equal(results[1][0], np.array([[4, 5]]))


def test_read_arrays_by_group_raises_for_unknown_read_method():
    obs = pl.DataFrame({"_zg": ["g0"], "_pos": [0]})
    group_readers = {"g0": FakeGroupReader({"layers/raw": FakeArrayReader("raw")})}
    spec = SimpleNamespace(pointer_type=DenseZarrPointer)

    with pytest.raises(ValueError, match="Unknown read_method"):
        read_arrays_by_group(
            group_readers,
            obs.group_by("_zg", maintain_order=True),
            spec,
            ["layers/raw"],
            "not-a-method",
        )


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
