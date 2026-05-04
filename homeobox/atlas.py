"""RaggedAtlas: user-facing API for writing, querying, and streaming homeobox data."""

import json
import os
from collections import OrderedDict, defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from homeobox.group_reader import GroupReader, LayoutReader
    from homeobox.query import AtlasQuery

import lancedb
import obstore
import polars as pl
import pyarrow as pa
import zarr

from homeobox.feature_layouts import (
    build_feature_layout_df,
    layout_exists,
    read_feature_layout,
    reindex_registry,
    sync_layouts_global_index,
    validate_feature_layout,
)
from homeobox.group_specs import get_spec
from homeobox.obs_alignment import _extract_pointer_fields, _infer_pointer_fields_from_arrow
from homeobox.schema import (
    AtlasVersionRecord,
    DatasetSchema,
    FeatureBaseSchema,
    FeatureLayout,
    HoxBaseSchema,
    PointerField,
)
from homeobox.util import sql_escape

# Maximum number of GroupReader objects kept alive in the RaggedAtlas cache.
# Each reader holds BatchAsyncArray handles and a Rust shard-index cache, so
# this bounds how much memory stays pinned across a long batch-read loop.
_MAX_GROUP_READERS: int = 128

# ---------------------------------------------------------------------------
# Store URI helpers
# ---------------------------------------------------------------------------


def _store_from_uri(
    uri: str,
    **store_kwargs,
) -> obstore.store.ObjectStore:
    """Construct an obstore ObjectStore from a URI string."""
    return obstore.store.from_url(uri, **store_kwargs)


def _store_kwargs_to_storage_options(store_kwargs: dict | None) -> dict[str, str] | None:
    """Convert obstore-style kwargs to lancedb ``storage_options``.

    obstore accepts ``{"config": {"skip_signature": True, "region": "us-east-2"}}``
    while lancedb expects a flat dict with string values:
    ``{"skip_signature": "true", "region": "us-east-2"}``.
    """
    if not store_kwargs:
        return store_kwargs
    options = store_kwargs.get("config", store_kwargs)
    return {k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in options.items()}


def _resolve_db_uri(uri: str) -> str:
    """Normalize an atlas path or db_uri to a LanceDB connection URI.

    If *uri* already ends with ``lance_db`` it is returned as-is.
    Otherwise ``/lance_db`` is appended (mirrors the convention used by
    :func:`create_or_open_atlas`).  Local relative paths are resolved to
    absolute paths so that ``obstore.store.from_url`` can handle them.
    """
    stripped = uri.rstrip("/")
    is_remote = stripped.startswith(("s3://", "gs://", "az://"))
    if not is_remote:
        stripped = os.path.abspath(stripped)
    if not stripped.endswith("/lance_db") and not stripped.endswith("\\lance_db"):
        stripped += "/lance_db"
    return stripped


def _check_atlas_exists(db_uri: str) -> None:
    """Raise :class:`FileNotFoundError` if a local atlas does not exist.

    *db_uri* must already be resolved (ending with ``/lance_db``).
    Remote URIs are not checked.
    """
    if db_uri.startswith(("s3://", "gs://", "az://")):
        return
    if not os.path.isdir(db_uri):
        atlas_root = db_uri.rsplit("/lance_db", 1)[0]
        raise FileNotFoundError(
            f"No atlas found at '{atlas_root}'. Create one first with create_or_open_atlas()."
        )


def _zarr_uri_from_db_uri(db_uri: str) -> str:
    """Derive a zarr store URI from a db_uri using naming convention.

    Replaces the last path segment with ``zarr_store``.
    Works for local paths (``/a/b/lance_db`` -> ``/a/b/zarr_store``)
    and cloud URIs (``s3://bucket/prefix/lance_db`` -> ``s3://bucket/prefix/zarr_store``).
    """
    # Handle both "/" and trailing-slash-stripped URIs
    uri = db_uri.rstrip("/")
    last_sep = uri.rfind("/")
    if last_sep == -1:
        return "zarr_store"
    return uri[: last_sep + 1] + "zarr_store"


def _derive_store_from_db_uri(db_uri: str, **store_kwargs) -> obstore.store.ObjectStore:
    """Build a zarr ObjectStore from the ``{atlas_root}/zarr_store`` convention."""
    zarr_uri = _zarr_uri_from_db_uri(db_uri)
    if zarr_uri.startswith(("s3://", "gs://", "az://")):
        return _store_from_uri(zarr_uri, **store_kwargs)
    return obstore.store.LocalStore(zarr_uri)


# ---------------------------------------------------------------------------
# RaggedAtlas
# ---------------------------------------------------------------------------


class RaggedAtlas:
    """Main entry point for reading and writing homeobox atlases.

    The atlas is backed by a LanceDB database (obs table + feature registries)
    and a zarr-compatible object store for array data.
    """

    def __init__(
        self,
        db: lancedb.DBConnection,
        obs_table: lancedb.table.Table,
        obs_schema: type[HoxBaseSchema] | None,
        root: zarr.Group,
        registry_tables: dict[str, lancedb.table.Table],
        dataset_table: lancedb.table.Table,
        *,
        version_table: lancedb.table.Table,
        feature_layouts_table: lancedb.table.Table,
    ) -> None:
        # REVIEW: Add a docstring that __init__ should not be called
        # directly, use create, open or checkout classmethods instead.
        self.db = db
        self._db_uri = db.uri
        self.obs_table = obs_table
        self._obs_schema = obs_schema
        self._root = root
        self._store = root.store.store
        # Pointer fields are keyed by the Python attribute name on the schema
        # class (``field_name``), which is unique per class. Multiple keys may
        # share the same ``feature_space`` when a schema declares several
        # columns in the same modality.
        self._pointer_fields: dict[str, PointerField]
        if obs_schema is not None:
            self._pointer_fields = _extract_pointer_fields(obs_schema)
        else:
            self._pointer_fields = _infer_pointer_fields_from_arrow(obs_table.schema)
        self._registry_tables = registry_tables
        self._dataset_table = dataset_table
        self._version_table = version_table
        self._feature_layouts_table = feature_layouts_table

        self._checked_out_version: int | None = None

        # Instance-level LRU cache: one GroupReader per (zarr_group, feature_space).
        # Capped at _MAX_GROUP_READERS to prevent unbounded memory growth during
        # long batch-read loops that touch many distinct zarr groups.
        self._group_readers: OrderedDict[tuple[str, str], GroupReader] = OrderedDict()
        # LayoutReaders are shared across all GroupReaders that reference the
        # same layout_uid. Bounded by the number of distinct layouts in the
        # atlas (typically O(few)–O(few hundred)) so no eviction is applied;
        # entries outlive their GroupReaders so a re-fetched group keeps the
        # already-materialized remap.
        self._layout_readers: dict[str, LayoutReader] = {}

    # -- Construction -------------------------------------------------------

    @classmethod
    def create(
        cls,
        db_uri: str,
        obs_table_name: str,
        obs_schema: type[HoxBaseSchema],
        dataset_table_name: str,
        dataset_schema: type[DatasetSchema],
        *,
        store: obstore.store.ObjectStore,
        registry_schemas: dict[str, type[FeatureBaseSchema]],
        version_table_name: str = "atlas_versions",
        store_kwargs: dict | None = None,
    ) -> "RaggedAtlas":
        """Create a new atlas, initialising the LanceDB tables.

        Parameters
        ----------
        db_uri:
            LanceDB connection URI (local path or remote).
        obs_table_name:
            Name for the obs table.
        obs_schema:
            A :class:`HoxBaseSchema` subclass declaring the pointer fields.
        dataset_table_name:
            Name for the dataset metadata table.
        dataset_schema:
            A :class:`DatasetSchema` subclass for the dataset schema.
        store:
            An obstore ObjectStore for zarr I/O.
        registry_schemas:
            Mapping of feature space names to their registry schema classes.
            Table names default to ``"{feature_space}_registry"``.
        version_table_name:
            Name for the version tracking table.
        store_kwargs:
            Extra keyword arguments forwarded to ``lancedb.connect`` as
            ``storage_options`` (e.g. ``region``, ``skip_signature``).
        """
        db_uri = _resolve_db_uri(db_uri)
        db = lancedb.connect(db_uri, storage_options=_store_kwargs_to_storage_options(store_kwargs))
        obs_table = db.create_table(obs_table_name, schema=obs_schema)
        dataset_table = db.create_table(dataset_table_name, schema=dataset_schema)

        registry_tables: dict[str, lancedb.table.Table] = {}
        for fs, schema_cls in registry_schemas.items():
            table_name = f"{fs}_registry"
            registry_tables[fs] = db.create_table(table_name, schema=schema_cls)

        version_table = db.create_table(version_table_name, schema=AtlasVersionRecord)

        feature_layouts_table = db.create_table("_feature_layouts", schema=FeatureLayout)
        feature_layouts_table.create_fts_index("feature_uid")
        feature_layouts_table.create_fts_index("layout_uid")

        root = zarr.open_group(zarr.storage.ObjectStore(store), mode="w")

        return cls(
            db=db,
            obs_table=obs_table,
            obs_schema=obs_schema,
            root=root,
            registry_tables=registry_tables,
            dataset_table=dataset_table,
            version_table=version_table,
            feature_layouts_table=feature_layouts_table,
        )

    @classmethod
    def open(
        cls,
        db_uri: str,
        obs_table_name: str,
        obs_schema: type[HoxBaseSchema] | None = None,
        dataset_table_name: str = "datasets",
        *,
        store: obstore.store.ObjectStore,
        registry_tables: dict[str, str] | None = None,
        version_table_name: str = "atlas_versions",
        store_kwargs: dict | None = None,
    ) -> "RaggedAtlas":
        """Open an existing atlas.

        Parameters
        ----------
        db_uri:
            LanceDB connection URI.
        obs_table_name:
            Name of the obs table.
        obs_schema:
            The schema class.  If ``None``, pointer fields are inferred
            from the obs table's Arrow schema (sufficient for read-only use).
        dataset_table_name:
            Name of the dataset metadata table.
        store:
            An obstore ObjectStore for zarr I/O.
        registry_tables:
            Mapping of feature space names to LanceDB table names.
            If ``None``, inferred from the dataset table using the naming
            convention ``{feature_space}_registry``.
        version_table_name:
            Name of the version tracking table.
        store_kwargs:
            Extra keyword arguments forwarded to ``lancedb.connect`` as
            ``storage_options`` (e.g. ``region``, ``skip_signature``).
        """
        db_uri = _resolve_db_uri(db_uri)
        db = lancedb.connect(db_uri, storage_options=_store_kwargs_to_storage_options(store_kwargs))
        obs_table = db.open_table(obs_table_name)
        dataset_table = db.open_table(dataset_table_name)

        if registry_tables is None:
            datasets_df = dataset_table.search().select(["feature_space"]).to_polars()
            feature_spaces = (
                datasets_df["feature_space"].unique().to_list()
                if not datasets_df.is_empty()
                else []
            )
            # Not all feature spaces are guaranteed to have registries, only
            # load the ones that do
            all_tables = set(db.list_tables().tables)
            registry_tables = {
                fs: f"{fs}_registry" for fs in feature_spaces if f"{fs}_registry" in all_tables
            }

        resolved_registries: dict[str, lancedb.table.Table] = {}
        for fs, table_name in registry_tables.items():
            resolved_registries[fs] = db.open_table(table_name)

        version_table = db.open_table(version_table_name)
        feature_layouts_table = db.open_table("_feature_layouts")

        root = zarr.open_group(zarr.storage.ObjectStore(store), mode="a")

        return cls(
            db=db,
            obs_table=obs_table,
            obs_schema=obs_schema,
            root=root,
            registry_tables=resolved_registries,
            dataset_table=dataset_table,
            version_table=version_table,
            feature_layouts_table=feature_layouts_table,
        )

    # -- Store helpers ------------------------------------------------------

    def _pointer_fields_for(self, feature_space: str) -> list[PointerField]:
        """Return all pointer fields that reference *feature_space*.

        A schema may declare multiple pointer columns in the same feature
        space (e.g. ``cycle1_image_tiles``, ``cycle2_image_tiles``).
        """
        return [pf for pf in self._pointer_fields.values() if pf.feature_space == feature_space]

    def get_group_reader(self, zarr_group: str, feature_space: str) -> "GroupReader":
        """Return (cached) GroupReader for the given zarr_group + feature_space.

        Uses an LRU policy capped at _MAX_GROUP_READERS entries. The least recently
        used GroupReader is evicted when the cap is exceeded, releasing its zarr
        handles and Rust shard-index cache.
        """
        from homeobox.group_reader import GroupReader, LayoutReader

        key = (zarr_group, feature_space)
        if key in self._group_readers:
            self._group_readers.move_to_end(key)
            return self._group_readers[key]

        datasets_df = (
            self._dataset_table.search()
            .where(
                f"zarr_group = '{sql_escape(zarr_group)}' AND feature_space = '{sql_escape(feature_space)}'",
                prefilter=True,
            )
            .select(["dataset_uid", "layout_uid"])
            .to_polars()
        )
        layout_uid: str | None = None
        if not datasets_df.is_empty():
            layout_uid = datasets_df["layout_uid"][0]
            if layout_uid == "":
                layout_uid = None

        layout_reader = None
        if layout_uid is not None:
            layout_reader = self._layout_readers.get(layout_uid)
            if layout_reader is None:
                layout_reader = LayoutReader(
                    layout_uid=layout_uid,
                    feature_layouts_table=self._feature_layouts_table,
                )
                self._layout_readers[layout_uid] = layout_reader

        reader = GroupReader.from_atlas_root(
            zarr_group=zarr_group,
            feature_space=feature_space,
            store=self._store,
            layout_reader=layout_reader,
        )
        self._group_readers[key] = reader
        if len(self._group_readers) > _MAX_GROUP_READERS:
            self._group_readers.popitem(last=False)
        return reader

    @property
    def schemas(self) -> str:
        """Print a summary of tables and their Arrow schemas."""
        lines: list[str] = []

        def _fmt_table(label: str, table: lancedb.table.Table) -> None:
            schema = table.schema
            rows = table.count_rows()
            lines.append(f"  {label} ({table.name!r}, {rows} rows)")
            for field in schema:
                lines.append(f"    {field.name}: {field.type}")

        lines.append("Atlas tables:")
        _fmt_table("Obs table", self.obs_table)
        _fmt_table("Dataset table", self._dataset_table)
        for fs, reg_table in sorted(self._registry_tables.items()):
            _fmt_table(f"Registry [{fs}]", reg_table)

        summary = "\n".join(lines)
        print(summary)
        return summary

    # -- Public accessors ---------------------------------------------------

    @property
    def obs_schema(self) -> type[HoxBaseSchema] | None:
        """Obs schema class, or ``None`` if the atlas was opened without one."""
        return self._obs_schema

    @property
    def pointer_fields(self) -> dict[str, PointerField]:
        """Pointer fields declared on the obs schema, keyed by field name."""
        return self._pointer_fields

    @property
    def registry_tables(self) -> dict[str, lancedb.table.Table]:
        """Feature registry tables, keyed by feature space."""
        return self._registry_tables

    @property
    def store(self) -> obstore.store.ObjectStore:
        """Underlying obstore ObjectStore for zarr I/O."""
        return self._store

    @property
    def db_uri(self) -> str:
        """LanceDB connection URI this atlas was opened from."""
        return self._db_uri

    # -- Dataset / zarr helpers --------------------------------------------

    def register_dataset(self, dataset_record: DatasetSchema) -> None:
        """Insert a ``DatasetSchema`` into the dataset table."""
        arrow_table = pa.Table.from_pylist(
            [dataset_record.model_dump()],
            schema=type(dataset_record).to_arrow_schema(),
        )
        self._dataset_table.add(arrow_table)

    def find_datasets(
        self,
        zarr_group: str,
        feature_space: str | None = None,
    ) -> pl.DataFrame:
        """Return dataset rows matching ``zarr_group`` (and optionally ``feature_space``)."""
        where = f"zarr_group = '{sql_escape(zarr_group)}'"
        if feature_space is not None:
            where += f" AND feature_space = '{sql_escape(feature_space)}'"
        return self._dataset_table.search().where(where, prefilter=True).to_polars()

    def create_zarr_group(self, zarr_group: str) -> zarr.Group:
        """Create a new zarr group at ``zarr_group`` under the atlas root."""
        return self._root.create_group(zarr_group)

    def open_zarr_group(self, zarr_group: str) -> zarr.Group:
        """Return the existing zarr group at ``zarr_group`` under the atlas root."""
        return self._root[zarr_group]

    def require_zarr_group(self, zarr_group: str) -> zarr.Group:
        """Return (creating if missing) the zarr group at ``zarr_group``."""
        return self._root.require_group(zarr_group)

    def read_feature_layout(self, layout_uid: str) -> pl.DataFrame:
        """Read all feature-layout rows for ``layout_uid``, sorted by ``local_index``."""
        return read_feature_layout(self._feature_layouts_table, layout_uid)

    def invalidate_group_reader(self, zarr_group: str, feature_space: str) -> None:
        """Drop the cached GroupReader for ``(zarr_group, feature_space)``."""
        self._group_readers.pop((zarr_group, feature_space), None)

    # -- Query entry point --------------------------------------------------

    def query(self) -> "AtlasQuery":
        """Start building a query against this atlas."""
        from homeobox.query import AtlasQuery

        if self._checked_out_version is None:
            raise RuntimeError(
                "query() is only available on a versioned atlas. "
                "After ingestion, call atlas.snapshot() then "
                "RaggedAtlas.checkout(db_uri, version, schema, store) to pin to a "
                "validated snapshot. For convenience, use RaggedAtlas.checkout_latest(...)."
            )
        return AtlasQuery(self)

    # -- Feature registration -----------------------------------------------

    def register_features(
        self,
        feature_space: str,
        features: list[FeatureBaseSchema] | pl.DataFrame,
    ) -> int:
        """Register features in a feature registry.

        Must be called before ingestion for feature spaces that
        have a registry (``has_var_df=True``).

        Features are inserted with ``global_index = None``.  The index is
        assigned later by ``optimize()`` / ``reindex_registry()`` which is
        designed to run after ingestion completes.

        Parameters
        ----------
        feature_space:
            Which feature space to register features for.
        features:
            Either a list of ``FeatureBaseSchema`` records or a Polars
            DataFrame with at minimum a ``uid`` column.

        Returns
        -------
        int
            Number of newly registered features.
        """
        if feature_space not in self._registry_tables:
            raise ValueError(
                f"No registry table for feature space '{feature_space}'. "
                f"Ensure a registry schema was provided at create() time."
            )
        registry_table = self._registry_tables[feature_space]

        if isinstance(features, pl.DataFrame):
            if "uid" not in features.columns:
                raise ValueError("features DataFrame must have a 'uid' column")
            features_df = features
        else:
            features_df = pl.DataFrame([f.model_dump() for f in features])

        # Deduplicate within the input batch; merge_insert(on="uid") with
        # when_not_matched_insert_all handles skipping rows that already
        # exist in the registry.
        new_records = features_df.unique(subset=["uid"], keep="first")

        n_before = registry_table.count_rows()
        (registry_table.merge_insert(on="uid").when_not_matched_insert_all().execute(new_records))
        return registry_table.count_rows() - n_before

    # -- Feature layouts ----------------------------------------------------

    def add_or_reuse_layout(
        self,
        var_df: pl.DataFrame,
        zarr_group: str,
        feature_space: str,
    ) -> str:
        """Compute or reuse a feature layout for a dataset.

        Computes the layout_uid from the feature ordering in var_df. If
        the layout already exists in the table, skips insertion. Otherwise
        inserts the layout rows. Updates the DatasetSchema to set layout_uid.

        Parameters
        ----------
        var_df:
            One row per local feature in local feature order.
            Must have a ``global_feature_uid`` column.
        zarr_group:
            The DatasetSchema zarr_group (per-row primary key) for this dataset.
        feature_space:
            Which feature space this dataset belongs to (used to look up registry).

        Returns
        -------
        str
            The layout_uid assigned to this dataset.
        """
        registry_table = self._registry_tables.get(feature_space)
        if registry_table is None:
            raise ValueError(
                f"No registry table for feature space '{feature_space}'. "
                f"Ensure a registry schema was provided at create() time."
            )
        layout_uid, layout_df = build_feature_layout_df(var_df, registry_table)

        if not layout_exists(self._feature_layouts_table, layout_uid):
            # Use merge_insert for concurrency safety: if two parallel
            # ingestions compute the same layout, the second is a no-op.
            (
                self._feature_layouts_table.merge_insert(on=["layout_uid", "feature_uid"])
                .when_not_matched_insert_all()
                .execute(layout_df)
            )

        # Update DatasetSchema with layout_uid
        (
            self._dataset_table.merge_insert(on="zarr_group")
            .when_matched_update_all()
            .execute(
                self._dataset_table.search()
                .where(f"zarr_group = '{sql_escape(zarr_group)}'", prefilter=True)
                .to_polars()
                .with_columns(pl.lit(layout_uid).alias("layout_uid"))
            )
        )

        return layout_uid

    # -- Maintenance --------------------------------------------------------

    def optimize(self) -> None:
        """Compact tables and reindex feature registries.

        Calls ``table.optimize()`` on the obs, dataset, and registry tables
        to compact small Lance fragments, then assigns ``global_index`` to any
        unindexed registry features via
        :func:`~homeobox.feature_layouts.reindex_registry`, and propagates
        updated indices to ``_feature_layouts`` via
        :func:`~homeobox.feature_layouts.sync_layouts_global_index`.
        """
        self.obs_table.optimize()
        self._dataset_table.optimize()
        self._deduplicate_new_rows(
            self._feature_layouts_table, subset=["layout_uid", "feature_uid"]
        )
        for table in self._registry_tables.values():
            self._deduplicate_new_rows(table, subset=["uid"])
            reindex_registry(table)
            table.create_scalar_index("uid", replace=True)
            table.optimize()
            sync_layouts_global_index(self._feature_layouts_table, table)

        # FTS index creates an inverted table that makes it easy to find which
        # layouts have a given feature
        self._feature_layouts_table.create_fts_index("feature_uid", replace=True)
        self._feature_layouts_table.create_scalar_index("layout_uid", replace=True)
        self._feature_layouts_table.optimize()

    @staticmethod
    def _deduplicate_new_rows(table: lancedb.table.Table, subset: list[str]) -> None:
        """Remove duplicate rows introduced since the last optimize/snapshot.

        Only reads rows where ``global_index IS NULL`` (i.e. newly added),
        deduplicates on *subset*, then deletes all new rows and re-adds the
        unique set.  This avoids rewriting the entire table.
        """
        new_rows = table.search().where("global_index IS NULL", prefilter=True).to_polars()
        if new_rows.is_empty():
            return
        deduped = new_rows.unique(subset=subset, keep="first")
        n_removed = len(new_rows) - len(deduped)
        if n_removed == 0:
            return
        # Delete all new rows, then add back the deduplicated set
        table.delete("global_index IS NULL")
        arrow_table = deduped.to_arrow().cast(table.schema)
        table.add(arrow_table)
        print(f"  Deduplicated {table.name}: removed {n_removed} duplicate rows")

    # -- Validation ---------------------------------------------------------

    def validate(
        self,
        *,
        check_zarr: bool = True,
        check_var_dfs: bool = True,
        check_registries: bool = True,
    ) -> list[str]:
        """Validate atlas consistency. Returns a list of error strings.

        Parameters
        ----------
        check_zarr:
            Open each unique zarr group and validate against its spec.
        check_var_dfs:
            For feature spaces with var_df, validate _feature_layouts rows.
        check_registries:
            Check that all registry rows have a global_index assigned.
        """
        errors: list[str] = []

        # Schema validation
        for pf in self._pointer_fields.values():
            spec = get_spec(pf.feature_space)
            if pf.pointer_kind is not spec.pointer_kind:
                errors.append(
                    f"Field '{pf.field_name}': pointer_kind {pf.pointer_kind.value} "
                    f"doesn't match spec {spec.pointer_kind.value}"
                )

        if check_registries:
            errors.extend(self._validate_registries())

        # Collect unique zarr groups from dataset table
        zarr_groups_by_space = self._collect_zarr_groups()

        if check_zarr:
            errors.extend(self._validate_zarr_groups(zarr_groups_by_space))

        if check_var_dfs:
            errors.extend(self._validate_feature_layouts(zarr_groups_by_space))

        return errors

    def _collect_zarr_groups(self) -> dict[str, set[str]]:
        """Collect unique zarr groups per feature space from the dataset table."""
        result: dict[str, set[str]] = defaultdict(set)
        datasets_df = (
            self._dataset_table.search().select(["feature_space", "zarr_group"]).to_polars()
        )
        if datasets_df.is_empty():
            return result
        for row in datasets_df.iter_rows(named=True):
            result[row["feature_space"]].add(row["zarr_group"])
        return result

    def list_datasets(self) -> pl.DataFrame:
        """Return a Polars DataFrame of all ingested datasets."""
        return self._dataset_table.search().to_polars()

    def feature_registry(self, feature_space: str) -> pl.DataFrame:
        """Return the feature registry for a feature space as a Polars DataFrame.

        Parameters
        ----------
        feature_space : str
            Feature space name (e.g. ``"gene_expression"``).

        Returns
        -------
        pl.DataFrame
            All rows from the registry table (uid, global_index, plus
            modality-specific columns such as gene_name, gene_id, etc.).
        """
        if feature_space not in self._registry_tables:
            raise KeyError(
                f"No registry for feature space '{feature_space}'. "
                f"Available: {sorted(self._registry_tables.keys())}"
            )
        return self._registry_tables[feature_space].search().to_polars()

    def join_feature_metadata(
        self,
        df: pl.DataFrame,
        feature_space: str,
        columns: list[str],
        on: str = "feature",
    ) -> pl.DataFrame:
        """Join feature registry columns onto a DataFrame.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame with a column of feature UIDs to join on.
        feature_space : str
            Feature space name (e.g. ``"gene_expression"``).
        columns : list[str]
            Registry columns to join (e.g. ``["gene_name", "gene_id"]``).
        on : str
            Column in *df* that contains feature UIDs. Default ``"feature"``.

        Returns
        -------
        pl.DataFrame
            *df* with the requested registry columns added via a left join.
        """
        registry_df = self.feature_registry(feature_space)
        available = [c for c in registry_df.columns if c not in ("uid", "global_index")]
        unknown = set(columns) - set(available)
        if unknown:
            raise ValueError(
                f"Unknown registry columns: {sorted(unknown)}. Available: {sorted(available)}"
            )
        registry_df = registry_df.select(["uid"] + columns)
        return df.join(registry_df, left_on=on, right_on="uid", how="left")

    def find_datasets_with_features(
        self,
        feature_uids: str | list[str],
        feature_space: str,
    ) -> pl.DataFrame:
        """Find datasets that measured specific features.

        Uses a two-step join: FTS on ``_feature_layouts.feature_uid`` to find
        layout_uids, then join against ``_dataset_table.layout_uid``.
        Note: rows added after the last FTS index build are not yet indexed;
        call ``optimize()`` after bulk ingestion to refresh.

        Parameters
        ----------
        feature_uids:
            One or more ``global_feature_uid`` values to search for.
        feature_space:
            Which feature space to search within (e.g. ``"gene_expression"``).

        Returns
        -------
        polars.DataFrame
            One row per (zarr_group, feature) match with columns
            ``zarr_group``, ``global_feature_uid``, plus dataset metadata.
        """
        from lancedb.query import MatchQuery

        if isinstance(feature_uids, str):
            feature_uids = [feature_uids]

        query_str = " ".join(feature_uids)
        # Estimate limit: each feature could appear in every layout
        n_layouts = self._feature_layouts_table.count_rows()
        limit = max(len(feature_uids) * max(n_layouts, 1), 1)

        pairs = (
            self._feature_layouts_table.search(MatchQuery(query_str), query_type="fts")
            .select(["feature_uid", "layout_uid"])
            .limit(limit)
            .to_polars()
        )
        if pairs.is_empty():
            return pl.DataFrame(schema={"zarr_group": pl.Utf8, "global_feature_uid": pl.Utf8})

        pairs = pairs.unique(subset=["feature_uid", "layout_uid"])

        # TODO: `layout_uid` is already in the datasets table, this should be an AND where
        # clause and no join is necessary.
        datasets_df = (
            self._dataset_table.search()
            .where(f"feature_space = '{sql_escape(feature_space)}'", prefilter=True)
            .to_polars()
        )

        return (
            pairs.rename({"feature_uid": "global_feature_uid"})
            .join(datasets_df, on="layout_uid", how="inner")
            .drop("layout_uid")
        )

    def _validate_registries(self) -> list[str]:
        errors: list[str] = []
        for fs, table in self._registry_tables.items():
            df = table.search().select(["global_index"]).to_polars()
            if df.is_empty():
                continue
            null_count = df["global_index"].null_count()
            if null_count > 0:
                errors.append(
                    f"Registry '{fs}': {null_count} row(s) have no global_index. "
                    f"Run reindex_registry(table) to fix."
                )
        return errors

    def _validate_zarr_groups(self, zarr_groups_by_space: dict[str, set[str]]) -> list[str]:
        errors: list[str] = []
        for fs, groups in zarr_groups_by_space.items():
            spec = get_spec(fs)
            for zg in groups:
                group = self._root[zg]
                group_errors = spec.zarr_group_spec.validate_group(group)
                for e in group_errors:
                    errors.append(f"zarr group '{zg}': {e}")
        return errors

    def _validate_feature_layouts(self, zarr_groups_by_space: dict[str, set[str]]) -> list[str]:
        errors: list[str] = []
        datasets_df = (
            self._dataset_table.search()
            .select(["dataset_uid", "zarr_group", "feature_space", "layout_uid"])
            .to_polars()
        )
        # Validate per unique layout_uid (not per dataset)
        validated_layouts: set[str] = set()
        for fs, groups in zarr_groups_by_space.items():
            spec = get_spec(fs)
            if not spec.has_var_df:
                continue
            registry = self._registry_tables.get(fs)
            for zg in groups:
                matched = datasets_df.filter(
                    (pl.col("zarr_group") == zg) & (pl.col("feature_space") == fs)
                )
                if matched.is_empty():
                    errors.append(f"No dataset record for zarr_group='{zg}', feature_space='{fs}'")
                    continue
                lid = matched["layout_uid"][0]
                if not lid or lid in validated_layouts:
                    continue
                validated_layouts.add(lid)
                group = self._root[zg]
                fl_errors = validate_feature_layout(
                    self._feature_layouts_table,
                    lid,
                    spec=spec,
                    group=group,
                    registry_table=registry,
                )
                for e in fl_errors:
                    errors.append(f"_feature_layouts '{lid}': {e}")
        return errors

    # -- Versioning ---------------------------------------------------------

    def snapshot(self) -> int:
        """Record a consistent snapshot of all table versions.

        Returns the new atlas version number (0-indexed, monotonically increasing).
        Raises ``ValueError`` if the atlas was created without a version table, or if
        validation errors are found.

        """
        errors = self.validate()
        if errors:
            raise ValueError(
                "Atlas validation failed — fix errors before snapshotting:\n"
                + "\n".join(f"  • {e}" for e in errors)
            )

        existing = self._version_table.search().select(["version"]).to_polars()
        if existing.is_empty():
            next_version = 0
        else:
            next_version = existing["version"].max() + 1

        registry_names = {fs: t.name for fs, t in self._registry_tables.items()}
        registry_versions = {fs: t.version for fs, t in self._registry_tables.items()}

        record = AtlasVersionRecord(
            version=next_version,
            obs_table_name=self.obs_table.name,
            obs_table_version=self.obs_table.version,
            dataset_table_name=self._dataset_table.name,
            dataset_table_version=self._dataset_table.version,
            registry_table_names=json.dumps(registry_names),
            registry_table_versions=json.dumps(registry_versions),
            feature_layouts_table_version=self._feature_layouts_table.version,
            total_rows=self.obs_table.count_rows(),
        )
        self._version_table.add([record])
        return next_version

    @classmethod
    def list_versions(
        cls,
        db_uri: str,
        *,
        store_kwargs: dict | None = None,
        version_table_name: str = "atlas_versions",
    ) -> pl.DataFrame:
        """Return a DataFrame of all recorded snapshots, sorted by version.

        Parameters
        ----------
        db_uri:
            LanceDB connection URI.
        store_kwargs:
            Extra keyword arguments forwarded to ``lancedb.connect`` as
            ``storage_options`` (e.g. ``region``, ``skip_signature``).
        version_table_name:
            Name of the version tracking table.
        """
        db_uri = _resolve_db_uri(db_uri)
        _check_atlas_exists(db_uri)
        db = lancedb.connect(db_uri, storage_options=_store_kwargs_to_storage_options(store_kwargs))
        version_table = db.open_table(version_table_name)
        return version_table.search().to_polars().sort("version")

    @classmethod
    def checkout(
        cls,
        db_uri: str,
        version: int,
        obs_schema: type[HoxBaseSchema] | None = None,
        store: obstore.store.ObjectStore | None = None,
        *,
        store_kwargs: dict | None = None,
        version_table_name: str = "atlas_versions",
    ) -> "RaggedAtlas":
        """Open a read-only atlas pinned to a specific snapshot version.

        Parameters
        ----------
        db_uri:
            LanceDB connection URI.
        version:
            Atlas version number (as returned by :meth:`snapshot`).
        obs_schema:
            The schema class used when the atlas was created.  If ``None``,
            pointer fields are inferred from the obs table's Arrow schema.
        store:
            An obstore ObjectStore for zarr I/O.  If ``None``, constructed
            from ``db_uri`` using the ``{atlas_root}/zarr_store`` convention.
        store_kwargs:
            Extra keyword arguments forwarded to ``obstore.store.from_url``
            when constructing the store from a URI (e.g. ``region``,
            ``skip_signature``, ``credential_provider``).
        version_table_name:
            Name of the version tracking table.
        """
        db_uri = _resolve_db_uri(db_uri)
        _check_atlas_exists(db_uri)
        db = lancedb.connect(db_uri, storage_options=_store_kwargs_to_storage_options(store_kwargs))
        version_table = db.open_table(version_table_name)

        records = version_table.search().where(f"version = {version}", prefilter=True).to_polars()
        if records.is_empty():
            raise ValueError(
                f"Atlas version {version} not found. "
                f"Use RaggedAtlas.list_versions('{db_uri}') to see available versions."
            )
        row = records.row(0, named=True)

        if store is None:
            store = _derive_store_from_db_uri(db_uri, **(store_kwargs or {}))

        obs_table = db.open_table(row["obs_table_name"])
        obs_table.checkout(row["obs_table_version"])

        dataset_table = db.open_table(row["dataset_table_name"])
        dataset_table.checkout(row["dataset_table_version"])

        registry_names: dict[str, str] = json.loads(row["registry_table_names"])
        registry_versions: dict[str, int] = json.loads(row["registry_table_versions"])
        resolved_registries: dict[str, lancedb.table.Table] = {}
        for fs, table_name in registry_names.items():
            t = db.open_table(table_name)
            t.checkout(registry_versions[fs])
            resolved_registries[fs] = t

        feature_layouts_table = db.open_table("_feature_layouts")
        feature_layouts_table.checkout(row["feature_layouts_table_version"])

        root = zarr.open_group(zarr.storage.ObjectStore(store), mode="r")

        atlas = cls(
            db=db,
            obs_table=obs_table,
            obs_schema=obs_schema,
            root=root,
            registry_tables=resolved_registries,
            dataset_table=dataset_table,
            version_table=version_table,
            feature_layouts_table=feature_layouts_table,
        )
        atlas._checked_out_version = version
        return atlas

    @classmethod
    def restore(
        cls,
        db_uri: str,
        version: int,
        obs_schema: type[HoxBaseSchema] | None = None,
        store: obstore.store.ObjectStore | None = None,
        *,
        store_kwargs: dict | None = None,
        version_table_name: str = "atlas_versions",
    ) -> "RaggedAtlas":
        """Restore all tables to a previous snapshot and return a writable atlas.

        Each managed table (obs, datasets, registries, feature layouts) is
        restored to its version at the given snapshot. This creates new Lance
        table versions whose data matches the snapshot — no data is deleted,
        but subsequent reads see only the restored state.

        Orphaned zarr groups (written after the snapshot) are not removed
        automatically; use a separate cleanup pass if needed.

        Parameters
        ----------
        db_uri:
            LanceDB connection URI.
        version:
            Atlas version number to restore to (as returned by :meth:`snapshot`).
        obs_schema:
            The schema class used when the atlas was created.  If ``None``,
            pointer fields are inferred from the obs table's Arrow schema.
        store:
            An obstore ObjectStore for zarr I/O.  If ``None``, constructed
            from ``db_uri`` using the ``{atlas_root}/zarr_store`` convention.
        store_kwargs:
            Extra keyword arguments forwarded to ``obstore.store.from_url``
            when constructing the store from a URI.
        version_table_name:
            Name of the version tracking table.
        """
        db_uri = _resolve_db_uri(db_uri)
        _check_atlas_exists(db_uri)
        db = lancedb.connect(db_uri, storage_options=_store_kwargs_to_storage_options(store_kwargs))
        version_table = db.open_table(version_table_name)

        records = version_table.search().where(f"version = {version}", prefilter=True).to_polars()
        if records.is_empty():
            raise ValueError(
                f"Atlas version {version} not found. "
                f"Use RaggedAtlas.list_versions('{db_uri}') to see available versions."
            )
        row = records.row(0, named=True)

        if store is None:
            store = _derive_store_from_db_uri(db_uri, **(store_kwargs or {}))

        # Restore each table to its snapshot version
        obs_table = db.open_table(row["obs_table_name"])
        obs_table.checkout(row["obs_table_version"])
        obs_table.restore()

        dataset_table = db.open_table(row["dataset_table_name"])
        dataset_table.checkout(row["dataset_table_version"])
        dataset_table.restore()

        registry_names: dict[str, str] = json.loads(row["registry_table_names"])
        registry_versions: dict[str, int] = json.loads(row["registry_table_versions"])
        resolved_registries: dict[str, lancedb.table.Table] = {}
        for fs, table_name in registry_names.items():
            t = db.open_table(table_name)
            t.checkout(registry_versions[fs])
            t.restore()
            resolved_registries[fs] = t

        feature_layouts_table = db.open_table("_feature_layouts")
        feature_layouts_table.checkout(row["feature_layouts_table_version"])
        feature_layouts_table.restore()

        root = zarr.open_group(zarr.storage.ObjectStore(store), mode="a")

        return cls(
            db=db,
            obs_table=obs_table,
            obs_schema=obs_schema,
            root=root,
            registry_tables=resolved_registries,
            dataset_table=dataset_table,
            version_table=version_table,
            feature_layouts_table=feature_layouts_table,
        )

    @classmethod
    def checkout_latest(
        cls,
        db_uri: str,
        obs_schema: type[HoxBaseSchema] | None = None,
        store: obstore.store.ObjectStore | None = None,
        *,
        store_kwargs: dict | None = None,
        version_table_name: str = "atlas_versions",
    ) -> "RaggedAtlas":
        """Open the most recent validated snapshot.

        Convenience wrapper around :meth:`checkout` that automatically selects
        the highest recorded version number.

        Parameters
        ----------
        db_uri:
            LanceDB connection URI.
        obs_schema:
            The schema class used when the atlas was created.  If ``None``,
            pointer fields are inferred from the obs table's Arrow schema.
        store:
            An obstore ObjectStore for zarr I/O.  If ``None``, reconstructed
            from the version record or inferred from ``db_uri``.
        store_kwargs:
            Extra keyword arguments forwarded to ``obstore.store.from_url``
            when constructing the store from a URI.
        version_table_name:
            Name of the version tracking table.
        """
        db_uri = _resolve_db_uri(db_uri)
        versions = cls.list_versions(
            db_uri, store_kwargs=store_kwargs, version_table_name=version_table_name
        )
        if versions.is_empty():
            raise ValueError(
                f"No snapshots found in atlas at '{db_uri}'. "
                "Call atlas.snapshot() after ingestion to create one."
            )
        latest_version = int(versions["version"].max())
        return cls.checkout(
            db_uri,
            version=latest_version,
            obs_schema=obs_schema,
            store=store,
            store_kwargs=store_kwargs,
            version_table_name=version_table_name,
        )


# ---------------------------------------------------------------------------
# Atlas create-or-open helper
# ---------------------------------------------------------------------------


def create_or_open_atlas(
    atlas_path: str,
    obs_table_name: str,
    obs_schema: type[HoxBaseSchema],
    dataset_table_name: str,
    dataset_schema: type[DatasetSchema],
    *,
    registry_schemas: dict[str, type[FeatureBaseSchema]],
    version_table_name: str = "atlas_versions",
    store_kwargs: dict | None = None,
) -> RaggedAtlas:
    """Create a new atlas or open an existing one.

    Accepts both local filesystem paths and cloud URIs (e.g.
    ``s3://bucket/prefix/my_atlas``).  For local paths, the required
    directories are created automatically.

    The LanceDB database is stored at ``{atlas_path}/lance_db`` and the
    zarr object store at ``{atlas_path}/zarr_store``.

    Parameters
    ----------
    atlas_path:
        Root directory or URI for the atlas.  Local paths and ``s3://``
        URIs are both supported.
    obs_table_name:
        Name for the obs table.
    obs_schema:
        A :class:`HoxBaseSchema` subclass declaring the pointer fields.
    dataset_table_name:
        Name for the dataset metadata table.
    dataset_schema:
        A :class:`DatasetSchema` subclass for the dataset schema.
    registry_schemas:
        Mapping of feature space names to their registry schema classes.
    version_table_name:
        Name for the version tracking table.
    store_kwargs:
        Extra keyword arguments forwarded to ``obstore.store.from_url``
        when constructing the zarr store (e.g. ``region``,
        ``skip_signature``, ``credential_provider``).
    """
    atlas_path = atlas_path.rstrip("/")
    is_local = not atlas_path.startswith(("s3://", "gs://", "az://"))

    db_uri = atlas_path + "/lance_db"
    zarr_uri = atlas_path + "/zarr_store"

    if is_local:
        os.makedirs(atlas_path, exist_ok=True)
        os.makedirs(zarr_uri, exist_ok=True)
        store = obstore.store.LocalStore(zarr_uri)
    else:
        store = _store_from_uri(zarr_uri, **(store_kwargs or {}))

    db = lancedb.connect(db_uri, storage_options=_store_kwargs_to_storage_options(store_kwargs))
    existing_tables = set(db.list_tables().tables)

    if obs_table_name in existing_tables:
        # Explicitly pass registry table names so open() doesn't rely on
        # the datasets table (which may be empty for a freshly-initialised atlas).
        registry_tables = {
            fs: f"{fs}_registry" for fs in registry_schemas if f"{fs}_registry" in existing_tables
        }
        return RaggedAtlas.open(
            db_uri=db_uri,
            obs_table_name=obs_table_name,
            obs_schema=obs_schema,
            dataset_table_name=dataset_table_name,
            store=store,
            registry_tables=registry_tables,
            version_table_name=version_table_name,
            store_kwargs=store_kwargs,
        )
    else:
        return RaggedAtlas.create(
            db_uri=db_uri,
            obs_table_name=obs_table_name,
            obs_schema=obs_schema,
            dataset_table_name=dataset_table_name,
            dataset_schema=dataset_schema,
            store=store,
            registry_schemas=registry_schemas,
            version_table_name=version_table_name,
            store_kwargs=store_kwargs,
        )
