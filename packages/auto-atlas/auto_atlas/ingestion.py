"""Add a finalized :class:`~auto_atlas.collection.Collection` to a homeobox atlas.

This is the final write step, after the collection has been harmonized and
``skills/finalize-tables`` has run. By then the on-disk state is fixed:

- ``<root>/lance_db/`` holds the collection-level registry-key target tables;
- ``<root>/<dataset>/lance_db/`` holds the finalized obs table, one per-feature
  -space obs artifact (``<ObsClass>_<feature_space>``, carrying ``uid`` in DATA
  row order), the per-dataset feature registries, and the dataset table;
- the raw matrix files live beside them, tagged ``DATA`` in ``collection.json``.

Everything here is schema-driven boilerplate *except one thing*: turning a
dataset's raw DATA files into a homeobox :class:`~homeobox.ingestion.Reader`.
That varies per source format (h5ad, COO, dense arrays, fragments, ...), so it
is the single pluggable hook — a ``Loader`` callable keyed by feature space (see
:data:`Loader`). Converter and writer selection is homeobox's job, resolved from
the feature-space spec; callers never pick them.

:func:`ingest_collection` drives the whole thing: open the atlas, copy registry
tables, register features, then per dataset assemble obs, align DATA rows to obs
positions, and let :class:`homeobox.ingestion.Ingestor` write the arrays and
stamp the pointers.

No ``pathlib``: paths are plain strings joined with ``os.path`` so s3 urls keep
working.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import NamedTuple

import lancedb
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
from homeobox.atlas import RaggedAtlas, create_or_open_atlas
from homeobox.group_specs import get_spec
from homeobox.ingestion import Ingestor, Reader
from homeobox.schema import PointerField, _extract_pointer_fields

from auto_atlas.collection import Collection, FileTypeTag
from auto_atlas.types import SchemaInfo
from auto_atlas.util import load_schema_info

LANCE_DB_DIR = "lance_db"
UID_COLUMN = "uid"
DATASET_UID_COLUMN = "dataset_uid"


# ===========================================================================
# The pluggable hook: a Loader produces a homeobox Reader from DATA files
# ===========================================================================


@dataclass(frozen=True)
class LoaderContext:
    """What a loader is handed to read one feature space of one dataset."""

    dataset_name: str
    feature_space: str
    # DATA files tagged for this feature space, from the collection manifest.
    data_files: list[str]
    # The per-dataset feature registry / var table, in finalized feature order,
    # or ``None`` for feature spaces with no registry (``has_var_df=False``).
    var_table: pa.Table | None


class LoaderResult(NamedTuple):
    """A homeobox source plus the per-write metadata it needs.

    The reader emits row-batches in DATA-file row order; auto-atlas maps that
    order onto finalized obs positions (see :func:`_obs_indices`). The array
    *type* the reader yields selects the converter, so the loader never names
    one.
    """

    reader: Reader
    # {source layer name the reader reads -> destination zarr layer}.
    layer_mapping: dict[str, str]
    n_vars: int
    # Required iff ``get_spec(feature_space).has_var_df``; must carry ``uid``.
    var_df: pd.DataFrame | None = None


# A loader turns one feature space's DATA files into a homeobox source.
Loader = Callable[[LoaderContext], LoaderResult]


# ===========================================================================
# Schema introspection
# ===========================================================================


@dataclass
class _SchemaModel:
    """Resolved schema facts the ingestor needs, derived from the schema module."""

    info: SchemaInfo
    obs_class: str
    dataset_class: str
    pointers: list[PointerField]
    registry_key_classes: list[str]

    @property
    def obs_cls(self) -> type:
        return self.info.live_class(self.obs_class)

    @property
    def dataset_cls(self) -> type:
        return self.info.live_class(self.dataset_class)

    def feature_space_registry(self) -> dict[str, type]:
        """``{feature_space: live registry class}`` for pointers that have a registry."""
        out: dict[str, type] = {}
        for p in self.pointers:
            if p.feature_registry_schema is None:
                continue
            cls = self.info.live_class(p.feature_registry_schema)
            if cls is not None:
                out[p.feature_space] = cls
        return out

    def pointer_for(self, feature_space: str) -> PointerField:
        for p in self.pointers:
            if p.feature_space == feature_space:
                return p
        raise KeyError(f"No obs pointer field for feature space {feature_space!r}")


def _resolve_schema(schema_path: str) -> _SchemaModel:
    info = load_schema_info(schema_path)

    obs_classes = [name for name, kind in info.kinds.items() if kind == "obs"]
    dataset_classes = [name for name, kind in info.kinds.items() if kind == "dataset"]
    if len(obs_classes) != 1 or len(dataset_classes) != 1:
        raise ValueError("schema must declare exactly one obs table and one dataset table")

    pointers = list(_extract_pointer_fields(info.live_class(obs_classes[0])).values())
    registry_key_classes = [
        name for name, kind in info.kinds.items() if kind in {"entity", "table"}
    ]

    return _SchemaModel(
        info=info,
        obs_class=obs_classes[0],
        dataset_class=dataset_classes[0],
        pointers=pointers,
        registry_key_classes=registry_key_classes,
    )


# ===========================================================================
# Entry point
# ===========================================================================


@dataclass
class IngestReport:
    datasets_ingested: list[str] = field(default_factory=list)
    datasets_skipped: list[str] = field(default_factory=list)
    rows_per_feature_space: dict[str, int] = field(default_factory=dict)
    features_registered: dict[str, int] = field(default_factory=dict)
    registry_tables_copied: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = ["Ingestion report:"]
        lines.append(f"  datasets ingested: {self.datasets_ingested}")
        if self.datasets_skipped:
            lines.append(f"  datasets skipped (already present): {self.datasets_skipped}")
        lines.append(f"  rows per feature space: {self.rows_per_feature_space}")
        lines.append(f"  features registered: {self.features_registered}")
        lines.append(f"  registry tables copied: {self.registry_tables_copied}")
        return "\n".join(lines)


def ingest_collection(
    collection_root: str,
    schema_path: str,
    atlas_path: str,
    loaders: Mapping[str, Loader],
    *,
    dataset_loaders: Mapping[str, Mapping[str, Loader]] | None = None,
    obs_table_name: str | None = None,
    store_kwargs: dict | None = None,
    skip_existing: bool = True,
) -> IngestReport:
    """Add a finalized collection and all its datasets to a homeobox atlas.

    Parameters
    ----------
    collection_root:
        Directory holding ``collection.json`` and the finalized lance tables.
    schema_path:
        The target homeobox schema module (one obs + one dataset class).
    atlas_path:
        Atlas location (local path or s3 url); created if absent.
    loaders:
        ``{feature_space: Loader}``. A loader turns that feature space's DATA
        files into a homeobox :class:`~homeobox.ingestion.Reader` (see
        :data:`Loader`).
    dataset_loaders:
        Optional ``{dataset_name: {feature_space: Loader}}`` overrides that win
        over ``loaders`` for that one dataset.
    obs_table_name:
        Obs lance table / atlas obs table name; defaults to the obs class name.
    skip_existing:
        Skip datasets whose ``dataset_uid`` is already in the atlas.
    """
    collection_root = os.fspath(collection_root)
    atlas_path = os.fspath(atlas_path)

    collection = Collection.from_json(os.path.join(collection_root, "collection.json"))
    schema = _resolve_schema(schema_path)
    obs_table_name = obs_table_name or schema.obs_class

    atlas = create_or_open_atlas(
        atlas_path,
        obs_schemas={obs_table_name: schema.obs_cls},
        dataset_table_name=schema.dataset_class,
        dataset_schema=schema.dataset_cls,
        registry_schemas=schema.feature_space_registry(),
        store_kwargs=store_kwargs,
    )

    report = IngestReport()
    report.registry_tables_copied = _copy_registry_key_tables(collection_root, atlas_path, schema)
    report.features_registered = _register_feature_registries(
        collection, collection_root, atlas, schema
    )

    existing = _existing_dataset_uids(atlas)
    for name in collection.datasets:
        dataset = collection._datasets[name]
        if skip_existing and dataset.uid in existing:
            print(f"== {name}: dataset_uid {dataset.uid} already in atlas, skipping ==")
            report.datasets_skipped.append(name)
            continue
        print(f"== ingesting {name} ==")
        rows = _ingest_dataset(
            collection_root, atlas, schema, obs_table_name, name, dataset, loaders, dataset_loaders
        )
        for fs, n in rows.items():
            report.rows_per_feature_space[fs] = report.rows_per_feature_space.get(fs, 0) + n
        report.datasets_ingested.append(name)

    print(report)
    return report


# ===========================================================================
# Collection-level setup
# ===========================================================================


def _existing_dataset_uids(atlas: RaggedAtlas) -> set[str]:
    try:
        df = atlas._dataset_table.search().select([DATASET_UID_COLUMN]).to_polars()
    except Exception:
        return set()
    return set() if df.is_empty() else set(df[DATASET_UID_COLUMN].to_list())


def _copy_registry_key_tables(
    collection_root: str, atlas_path: str, schema: _SchemaModel
) -> dict[str, int]:
    """Copy collection-level registry-key target tables into the atlas (dedup on uid).

    Homeobox has no helper for cross-db table copies, so auto-atlas owns this.
    """
    copied: dict[str, int] = {}
    src_path = os.path.join(collection_root, LANCE_DB_DIR)
    if not os.path.isdir(src_path):
        return copied
    src = lancedb.connect(src_path)
    src_names = set(src.list_tables().tables)
    dst = lancedb.connect(os.path.join(atlas_path, LANCE_DB_DIR))
    dst_names = set(dst.list_tables().tables)

    for cls in schema.registry_key_classes:
        if cls not in src_names:
            continue
        arrow = src.open_table(cls).to_arrow()
        print(f"  registry-key table {cls}: {arrow.num_rows} row(s)")
        copied[cls] = arrow.num_rows
        if cls not in dst_names:
            dst.create_table(cls, data=arrow)
        else:
            (
                dst.open_table(cls)
                .merge_insert(on=UID_COLUMN)
                .when_not_matched_insert_all()
                .execute(arrow)
            )
    return copied


def _register_feature_registries(
    collection: Collection, collection_root: str, atlas: RaggedAtlas, schema: _SchemaModel
) -> dict[str, int]:
    """Register each feature space's per-dataset var tables before ingestion.

    ``register_features`` accepts a DataFrame and dedupes on ``uid``, so each
    dataset's registry table is passed through whole.
    """
    registered: dict[str, int] = {}
    for feature_space, registry_cls in schema.feature_space_registry().items():
        for name in collection.datasets:
            table = _read_table(collection_root, name, registry_cls.__name__)
            if table is None:
                continue
            n_new = atlas.register_features(feature_space, pl.from_arrow(table))
            registered[feature_space] = registered.get(feature_space, 0) + n_new
            print(f"  register_features({feature_space}) from {name}: {n_new} new")
    return registered


# ===========================================================================
# Per-dataset ingestion
# ===========================================================================


class _Plan(NamedTuple):
    feature_space: str
    field_name: str
    result: LoaderResult
    dataset_record: object
    obs_indices: np.ndarray | None


def _ingest_dataset(
    collection_root: str,
    atlas: RaggedAtlas,
    schema: _SchemaModel,
    obs_table_name: str,
    name: str,
    dataset: object,
    loaders: Mapping[str, Loader],
    dataset_loaders: Mapping[str, Mapping[str, Loader]] | None,
) -> dict[str, int]:
    bare_obs = _read_table(collection_root, name, obs_table_name)
    if bare_obs is None:
        raise ValueError(f"{name}: no finalized obs table {obs_table_name!r}")

    plans: list[_Plan] = []
    for feature_space in dataset.feature_spaces:
        loader = _resolve_loader(loaders, dataset_loaders, name, feature_space)
        data_files = dataset.files_for(tag=FileTypeTag.DATA, feature_space=feature_space)
        if loader is None:
            raise ValueError(
                f"No loader for {name}/{feature_space}; data files: {data_files}. "
                f"Pass one in loaders={{{feature_space!r}: ...}} or dataset_loaders."
            )

        pointer = schema.pointer_for(feature_space)
        var_table = (
            _read_table(collection_root, name, pointer.feature_registry_schema)
            if pointer.feature_registry_schema
            else None
        )
        result = loader(
            LoaderContext(
                dataset_name=name,
                feature_space=feature_space,
                data_files=data_files,
                var_table=var_table,
            )
        )
        _validate_loader_result(name, feature_space, result)

        plans.append(
            _Plan(
                feature_space=feature_space,
                field_name=pointer.field_name,
                result=result,
                dataset_record=_build_dataset_record(
                    collection_root, name, schema, feature_space, bare_obs
                ),
                obs_indices=_obs_indices(
                    collection_root, name, feature_space, obs_table_name, bare_obs
                ),
            )
        )

    obs_df = _prepare_obs_df(bare_obs, schema, plans)
    ingestor = Ingestor(atlas, obs_df=obs_df, obs_table_name=obs_table_name)
    rows_per_feature_space: dict[str, int] = {}
    for plan in plans:
        n = ingestor.write_array(
            plan.result.reader,
            field_name=plan.field_name,
            layer_mapping=plan.result.layer_mapping,
            dataset_record=plan.dataset_record,
            n_vars=plan.result.n_vars,
            var_df=plan.result.var_df,
            required_pointer_type=get_spec(plan.feature_space).pointer_type,
            obs_indices=plan.obs_indices,
        )
        rows_per_feature_space[plan.feature_space] = (
            rows_per_feature_space.get(plan.feature_space, 0) + n
        )
    n_obs = ingestor.write_obs_records()
    print(f"  added {n_obs} obs row(s)")
    return rows_per_feature_space


def _resolve_loader(
    loaders: Mapping[str, Loader],
    dataset_loaders: Mapping[str, Mapping[str, Loader]] | None,
    dataset_name: str,
    feature_space: str,
) -> Loader | None:
    """Dataset override wins over the feature-space default."""
    override = (dataset_loaders or {}).get(dataset_name, {})
    return override.get(feature_space) or loaders.get(feature_space)


def _validate_loader_result(name: str, feature_space: str, result: LoaderResult) -> None:
    if not result.layer_mapping:
        raise ValueError(f"{name}/{feature_space}: loader returned an empty layer_mapping")

    has_var_df = get_spec(feature_space).has_var_df
    if has_var_df:
        if result.var_df is None:
            raise ValueError(f"{name}/{feature_space}: feature space requires a var_df")
        if len(result.var_df) != result.n_vars:
            raise ValueError(
                f"{name}/{feature_space}: loader reports {result.n_vars} variables, "
                f"but var_df has {len(result.var_df)} rows"
            )
        if UID_COLUMN not in result.var_df.columns:
            raise ValueError(f"{name}/{feature_space}: var_df is missing {UID_COLUMN!r}")


def _obs_indices(
    collection_root: str,
    name: str,
    feature_space: str,
    obs_table_name: str,
    bare_obs: pa.Table,
) -> np.ndarray | None:
    """Map emitted DATA rows onto positions in the finalized bare obs table.

    Finalization keeps a ``<ObsClass>_<feature_space>`` artifact whose ``uid``
    column is in DATA-file row order. The reader emits in that same order, so
    emitted row ``i`` belongs at the position of that artifact's ``uid[i]`` in
    the bare obs table. When no artifact exists (the reader is expected to emit
    one row per bare obs row, in order), return ``None``.
    """
    artifact = _read_table(collection_root, name, f"{obs_table_name}_{feature_space}")
    if artifact is None:
        return None

    if UID_COLUMN not in bare_obs.column_names:
        raise ValueError(f"{name}: bare obs table is missing {UID_COLUMN!r}")
    bare_pos = {str(uid): i for i, uid in enumerate(bare_obs.column(UID_COLUMN).to_pylist())}

    if UID_COLUMN not in artifact.column_names:
        raise ValueError(
            f"{name}/{feature_space}: obs artifact is missing {UID_COLUMN!r}; "
            f"available: {artifact.column_names}"
        )
    fs_uids = [str(uid) for uid in artifact.column(UID_COLUMN).to_pylist()]
    missing = [uid for uid in fs_uids if uid not in bare_pos]
    if missing:
        raise ValueError(
            f"{name}/{feature_space}: {len(missing)} artifact uid(s) absent from the bare "
            f"obs table; examples: {missing[:5]}"
        )
    return np.array([bare_pos[uid] for uid in fs_uids], dtype=np.int64)


def _prepare_obs_df(bare_obs: pa.Table, schema: _SchemaModel, plans: list[_Plan]) -> pd.DataFrame:
    """Build the obs frame for homeobox.Ingestor, stamping ``has_<field>`` flags.

    Homeobox null-fills unwritten pointer columns but does not derive presence
    flags, so auto-atlas sets them: ``False`` everywhere, then ``True`` at the
    rows each feature space actually covers.
    """
    obs_df = bare_obs.to_pandas()
    schema_names = set(schema.obs_cls.to_arrow_schema().names)

    for pointer in schema.pointers:
        flag = f"has_{pointer.field_name}"
        if flag in schema_names:
            obs_df[flag] = False

    for plan in plans:
        flag = f"has_{plan.field_name}"
        if flag not in obs_df.columns:
            continue
        values = np.asarray(obs_df[flag].fillna(False), dtype=bool)
        covered = plan.obs_indices if plan.obs_indices is not None else slice(None)
        values[covered] = True
        obs_df[flag] = values

    return obs_df


def _build_dataset_record(
    collection_root: str, name: str, schema: _SchemaModel, feature_space: str, bare_obs: pa.Table
) -> object:
    """Reuse the finalized dataset row for this fs, filling SummaryFields from obs."""
    table = _read_table(collection_root, name, schema.dataset_class)
    if table is None:
        raise ValueError(f"{name}: no dataset table {schema.dataset_class!r}")
    df = table.to_pandas()
    rows = df[df["feature_space"] == feature_space]
    if rows.empty:
        raise ValueError(f"{name}: dataset table has no row for feature_space={feature_space!r}")
    record = _fill_summary_fields(rows.iloc[0].to_dict(), schema, bare_obs)
    return schema.dataset_cls(**record)


def _fill_summary_fields(record: dict, schema: _SchemaModel, obs: pa.Table) -> dict:
    """Fill the dataset record's SummaryFields (count / unique) from the obs table."""
    for s in schema.info.summary_fields.get(schema.dataset_class, []):
        if s.target_field not in obs.column_names:
            continue
        values = obs.column(s.target_field).to_pylist()
        if s.op == "count":
            record[s.field_name] = len(values)
        elif s.op == "unique":
            record[s.field_name] = sorted({v for v in values if v is not None})
    return record


def _read_table(collection_root: str, dataset_name: str, table_name: str) -> pa.Table | None:
    """Read a named per-dataset lance table to Arrow, or ``None`` if absent."""
    db_path = os.path.join(collection_root, dataset_name, LANCE_DB_DIR)
    if not os.path.isdir(db_path):
        return None
    db = lancedb.connect(db_path)
    if table_name not in db.list_tables().tables:
        return None
    return db.open_table(table_name).to_arrow()
