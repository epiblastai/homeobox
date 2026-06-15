import json
import os
import tempfile

import pytest

from auto_atlas.collection import Collection, Dataset, FileTypeTag


def test_to_json_writes_collection_manifest_under_root_dir() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root_dir = os.path.join(tmp, "collection")
        source_dir = os.path.join(tmp, "source")
        os.makedirs(source_dir)

        obs_path = os.path.join(source_dir, "obs.csv")
        library_path = os.path.join(source_dir, "library.csv")
        with open(obs_path, "w") as f:
            f.write("cell_id\ncell-1\n")
        with open(library_path, "w") as f:
            f.write("guide\nA\n")

        dataset = Dataset("dataset_a")
        dataset.add_file(obs_path, FileTypeTag.OBS, "gene_expression")

        collection = Collection(root_dir)
        collection.add_dataset(dataset)
        collection.add_file(library_path, FileTypeTag.LIBRARY)
        collection.coalesce()

        collection.to_json()

        manifest_path = os.path.join(root_dir, "collection.json")
        assert os.path.exists(manifest_path)

        with open(manifest_path) as f:
            payload = json.load(f)

        assert payload == json.loads(collection.dumps())


def test_from_json_loads_collection_manifest_from_path() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root_dir = os.path.join(tmp, "collection")
        source_dir = os.path.join(tmp, "source")
        os.makedirs(source_dir)

        obs_path = os.path.join(source_dir, "obs.csv")
        with open(obs_path, "w") as f:
            f.write("cell_id\ncell-1\n")

        dataset = Dataset("dataset_a")
        dataset.add_file(obs_path, FileTypeTag.OBS, "gene_expression")

        collection = Collection(root_dir)
        collection.add_dataset(dataset)
        collection.coalesce()
        collection.to_json()

        loaded = Collection.from_json(os.path.join(root_dir, "collection.json"))

        assert json.loads(loaded.dumps()) == json.loads(collection.dumps())


def test_to_json_requires_collection_to_be_coalesced() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        collection = Collection(os.path.join(tmp, "collection"))
        dataset = Dataset("dataset_a")
        dataset.add_file(os.path.join(tmp, "obs.csv"), FileTypeTag.OBS)
        collection.add_dataset(dataset)

        with pytest.raises(ValueError, match="run coalesce"):
            collection.to_json()

        assert not os.path.exists(os.path.join(collection.root_dir, "collection.json"))
