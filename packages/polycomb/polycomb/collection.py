import json
import os
import shutil
from dataclasses import dataclass
from enum import StrEnum

from homeobox.schema import make_uid


class FileTypeTag(StrEnum):
    # Use this tag for files that contain row-level metadata
    OBS = "obs"
    # Use this tag for files that contain column-level metadata
    VAR = "var"
    # Use this tag for files that contain actual array data
    DATA = "data"
    # Use this tag for files that contain libraries like names of
    # small molecule perturbation, guide RNAs, or donor information.
    # Libraries have a column that can be linked to another column in
    # an OBS or VAR file.
    LIBRARY = "library"
    # Use this tag for free-form informational files that are not tables
    # or arrays (e.g. sample preparation protocols, dataset READMEs, or
    # publication texts). These are coalesced into an other_files/ subdir.
    OTHER = "other"


@dataclass(frozen=True)
class TaggedFile:
    """A file path together with its tag and (optionally) the feature space it
    belongs to. Feature space is set for obs/var/data files of a given modality
    (e.g. "gene_expression", "protein_abundance") and may be omitted for shared
    library files that are not specific to a single modality."""

    path: str
    tag: FileTypeTag
    feature_space: str | None = None

    def to_dict(self) -> dict:
        return {"path": self.path, "tag": str(self.tag), "feature_space": self.feature_space}


class Dataset:
    def __init__(self, dataset_name: str, uid: str | None = None) -> None:
        self.dataset_name = dataset_name
        # Stable logical identifier for this dataset, referenced by
        # HoxBaseSchema.dataset_uid on every obs row that belongs to it. Assigned
        # once on creation and preserved across JSON round-trips; never changes.
        self.uid = uid or make_uid()
        self._tagged_files: dict[str, TaggedFile] = {}

    @property
    def files(self) -> list[str]:
        return list(self._tagged_files.keys())

    @property
    def feature_spaces(self) -> list[str]:
        """Distinct, sorted feature spaces present across this dataset's files."""
        spaces = {tf.feature_space for tf in self._tagged_files.values()}
        spaces.discard(None)
        return sorted(spaces)

    def add_file(
        self,
        file_path: str,
        tag: FileTypeTag,
        feature_space: str | None = None,
    ) -> None:
        if file_path in self._tagged_files:
            raise ValueError(f"file_path {file_path} has already be added!")

        self._tagged_files[file_path] = TaggedFile(file_path, tag, feature_space)

    def files_for(
        self,
        tag: FileTypeTag | None = None,
        feature_space: str | None = None,
    ) -> list[str]:
        """List files, optionally filtered by tag and/or feature space.

        A filter left as None places no constraint on that axis.
        """
        return [
            tf.path
            for tf in self._tagged_files.values()
            if (tag is None or tf.tag == tag)
            and (feature_space is None or tf.feature_space == feature_space)
        ]

    def _rename_file(self, old_path: str, new_path: str) -> None:
        """Internal: rewrite a tracked path.

        Used by Collection.coalesce after physically moving/copying files.
        """
        tf = self._tagged_files.pop(old_path)
        self._tagged_files[new_path] = TaggedFile(new_path, tf.tag, tf.feature_space)

    def _to_dict(self) -> dict:
        return {
            "dataset_uid": self.uid,
            "files": [tf.to_dict() for tf in self._tagged_files.values()],
        }


class Collection:
    """A collection groups multiple datasets that share a root directory.

    Datasets hold tables and arrays. Collection-level files added with
    Collection.add_file are not tied to a single dataset: use the LIBRARY tag
    for files linked from obs/var tables, or the OTHER tag for free-form
    informational files (sample preparation protocols, dataset READMEs,
    publication texts).

    On coalesce, dataset files move under root_dir/<dataset_name>/, OTHER files
    move under root_dir/other_files/, and other shared files move into root_dir.
    """

    OTHER_FILES_DIR = "other_files"

    def __init__(self, root_dir: str):
        os.makedirs(root_dir, exist_ok=True)
        self.root_dir = root_dir
        # Files within each dataset are independent of each other
        self._datasets: dict[str, Dataset] = {}
        self._coalesced_datasets: set[str] = set()

        # Collection-level files that are shared across datasets (LIBRARY) or
        # are free-form informational files (OTHER). Not all datasets are
        # required to reference a shared file.
        self._shared_tagged_files: dict[str, TaggedFile] = {}
        self._coalesced_files: set[str] = set()

    @classmethod
    def from_json(cls, path: str) -> "Collection":
        """Rehydrate a Collection from a collection JSON file.

        Because collection JSON files only contain a fully-coalesced collection, the
        reconstructed datasets and shared files are marked coalesced so the
        manifest round-trips (``Collection.from_json(path).dumps()`` equals
        ``c.dumps()``) and re-coalescing is a no-op.
        """
        with open(path) as f:
            data = f.read()
        payload = json.loads(data)
        collection = cls(payload["root_dir"])

        for f in payload.get("shared_files", []):
            collection.add_file(f["path"], FileTypeTag(f["tag"]), f["feature_space"])
            collection._coalesced_files.add(f["path"])

        for name, dataset_payload in payload.get("datasets", {}).items():
            uid = dataset_payload.get("dataset_uid")
            if uid is None:
                raise ValueError(
                    f"Dataset {name!r} in {path} has no dataset_uid; the manifest "
                    "is wrong. Regenerate it (Collection.to_json)."
                )
            dataset = Dataset(name, uid=uid)
            for f in dataset_payload["files"]:
                dataset.add_file(f["path"], FileTypeTag(f["tag"]), f["feature_space"])
            collection.add_dataset(dataset)
            collection._coalesced_datasets.add(name)

        return collection

    @property
    def datasets(self) -> list[str]:
        return list(self._datasets.keys())

    def add_dataset(self, dataset: Dataset) -> None:
        if dataset.dataset_name in self._datasets:
            raise ValueError(f"dataset {dataset.dataset_name} has already be added!")

        self._datasets[dataset.dataset_name] = dataset

    def add_file(
        self,
        file_path: str,
        tag: FileTypeTag,
        feature_space: str | None = None,
    ) -> None:
        if file_path in self._shared_tagged_files:
            raise ValueError(f"file_path {file_path} has already be added!")

        self._shared_tagged_files[file_path] = TaggedFile(file_path, tag, feature_space)

    def _move_files(
        self,
        paths: list[str],
        dest_dir: str,
        copy: bool,
    ) -> list[tuple[str, str]]:
        """Copy or move files into dest_dir, keeping their original filenames.

        Returns the list of (old_path, new_path). Collisions (two files sharing
        a basename, or an existing destination) are checked up front so nothing
        is moved when the group is invalid. This is a local-filesystem operation
        (shutil does not handle s3 urls).
        """
        os.makedirs(dest_dir, exist_ok=True)

        moves: list[tuple[str, str]] = []
        seen_dests: set[str] = set()
        for src in paths:
            dest = os.path.join(dest_dir, os.path.basename(src))
            if dest in seen_dests:
                raise ValueError(f"basename collision: multiple files map to {dest}")
            if os.path.exists(dest):
                raise ValueError(f"destination {dest} already exists; refusing to overwrite")
            seen_dests.add(dest)
            moves.append((src, dest))

        for src, dest in moves:
            if copy:
                shutil.copy2(src, dest)
            else:
                shutil.move(src, dest)

        return moves

    def coalesce(self, copy: bool = True) -> None:
        """Organize files on disk into root_dir.

        Each not-yet-coalesced dataset's files go to root_dir/<dataset_name>/;
        collection-level OTHER files go to root_dir/other_files/ and other shared
        files go into root_dir. Tracked paths are rewritten to the new
        locations and re-running is a no-op for anything already coalesced.
        """
        for name, dataset in self._datasets.items():
            if name in self._coalesced_datasets:
                continue

            moves = self._move_files(dataset.files, os.path.join(self.root_dir, name), copy)
            for src, dest in moves:
                dataset._rename_file(src, dest)
            self._coalesced_datasets.add(name)

        pending = [
            tf
            for path, tf in self._shared_tagged_files.items()
            if path not in self._coalesced_files
        ]
        groups = (
            ([tf for tf in pending if tf.tag != FileTypeTag.OTHER], self.root_dir),
            (
                [tf for tf in pending if tf.tag == FileTypeTag.OTHER],
                os.path.join(self.root_dir, self.OTHER_FILES_DIR),
            ),
        )
        for group, dest_dir in groups:
            if not group:
                continue

            moves = self._move_files([tf.path for tf in group], dest_dir, copy)
            for src, dest in moves:
                tf = self._shared_tagged_files.pop(src)
                self._shared_tagged_files[dest] = TaggedFile(dest, tf.tag, tf.feature_space)
                self._coalesced_files.add(dest)

    def dumps(self) -> str:
        """Create a JSON string listing file paths and their tags.

        Includes dataset subdirectories. Everything must be coalesced first, so
        the manifest always reflects organized, on-disk locations.
        """
        uncoalesced = [name for name in self._datasets if name not in self._coalesced_datasets]
        uncoalesced += [p for p in self._shared_tagged_files if p not in self._coalesced_files]
        if uncoalesced:
            raise ValueError(
                f"{uncoalesced} have not been coalesced; run coalesce() before dumps()"
            )

        payload = {
            "root_dir": self.root_dir,
            "shared_files": [tf.to_dict() for tf in self._shared_tagged_files.values()],
            "datasets": {name: dataset._to_dict() for name, dataset in self._datasets.items()},
        }
        return json.dumps(payload, indent=2)

    def to_json(self) -> None:
        data = self.dumps()
        with open(os.path.join(self.root_dir, "collection.json"), "w") as f:
            f.write(data)
