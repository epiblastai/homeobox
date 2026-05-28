"""RaggedAtlas: user-facing API for writing, querying, and streaming homeobox data."""

import json
import os
from collections import OrderedDict, defaultdict
from collections.abc import Iterator
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
from homeobox.schema import (
    AtlasVersionRecord,
    DatasetSchema,
    FeatureBaseSchema,
    FeatureLayout,
    ForeignKeyField,
    HoxBaseSchema,
    PointerField,
    _extract_foreign_key_fields,
    _extract_pointer_fields,
    _infer_foreign_key_fields_from_arrow,
    _infer_pointer_fields_from_arrow,
)
from homeobox.util import sql_escape

# Maximum number of GroupReader objects kept alive in the RaggedAtlas cache.
# Each reader holds BatchAsyncArray handles and a Rust shard-index cache, so
# this bounds how much memory stays pinned across a long batch-read loop.
_MAX_GROUP_READERS: int = 128


def _foreign_key_record(source_table: str, fk: ForeignKeyField) -> dict[str, str]:
    return {
        "source_table": source_table,
        "source_field": fk.field_name,
        "target_table": fk.target_table,
        "target_field": fk.target_field,
    }


def _validate_foreign_key_source_schemas(
    schema_by_table: dict[str, type],
    *,
    registry_table_names: dict[str, str],
) -> None:
    """Validate FK declarations on schemas registered with create()."""
    for _source_table, schema_cls in schema_by_table.items():
        for fk in _extract_foreign_key_fields(schema_cls).values():
            if fk.target_table in registry_table_names:
                raise TypeError(
                    f"{schema_cls.__name__}.{fk.field_name}: foreign key target_table "
                    f"must be the actual LanceDB table name {registry_table_names[fk.target_table]!r}, "
                    f"not feature space alias {fk.target_table!r}"
                )
            target_schema = schema_by_table.get(fk.target_table)
            if target_schema is not None and fk.target_field not in target_schema.model_fields:
                raise TypeError(
                    f"{schema_cls.__name__}.{fk.field_name}: foreign key target field "
                    f"{fk.target_table}.{fk.target_field} does not exist"
                )


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
        obs_tables: dict[str, lancedb.table.Table],
        obs_schemas: dict[str, type[HoxBaseSchema] | None],
        root: zarr.Group,
        registry_tables: dict[str, lancedb.table.Table],
        dataset_table: lancedb.table.Table,
        *,
        version_table: lancedb.table.Table,
        feature_layouts_table: lancedb.table.Table,
        foreign_keys: list[dict[str, str]] | None = None,
    ) -> None:
        # REVIEW: Add a docstring that __init__ should not be called
        # directly, use create, open or checkout classmethods instead.
        self.db = db
        self._db_uri = db.uri
        self._obs_tables = obs_tables
        self._obs_schemas: dict[str, type[HoxBaseSchema] | None] = {
            name: obs_schemas.get(name) for name in obs_tables
        }
        self._root = root
        self._store = root.store.store
        # Pointer-field names may repeat across obs tables when the declarations
        # match (same feature_space). The shared PointerField is stored once;
        # _field_to_tables records every table that exposes that field.
        # Conflicting declarations of the same name (different feature_space)
        # raise at construction.
        self._pointer_fields: dict[str, PointerField] = {}
        self._field_to_tables: dict[str, list[str]] = {}
        for tbl_name, table in obs_tables.items():
            schema = self._obs_schemas.get(tbl_name)
            if schema is not None:
                pfs = _extract_pointer_fields(schema)
            else:
                pfs = _infer_pointer_fields_from_arrow(table.schema)

            for fn, pf in pfs.items():
                existing = self._pointer_fields.get(fn)
                if existing is None:
                    self._pointer_fields[fn] = pf
                elif existing != pf:
                    other = self._field_to_tables[fn][0]
                    raise ValueError(
                        f"Pointer field {fn!r} is declared with conflicting "
                        f"definitions in obs tables {other!r} ({existing}) and "
                        f"{tbl_name!r} ({pf})."
                    )
                self._field_to_tables.setdefault(fn, []).append(tbl_name)

        self._registry_tables = registry_tables
        self._dataset_table = dataset_table
        self._version_table = version_table
        self._feature_layouts_table = feature_layouts_table
        self._foreign_keys: list[dict[str, str]] = (
            list(foreign_keys)
            if foreign_keys is not None
            else self._compile_foreign_key_manifest_from_tables()
        )

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
        obs_schemas: dict[str, type[HoxBaseSchema]],
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
        obs_schemas:
            Mapping of ``{obs_table_name: HoxBaseSchema subclass}`` declaring
            one or more obs tables and their pointer-field schemas. Each obs
            table is independent. Pointer fields are keyed by ``field_name``
            across the atlas: the same field name may appear in multiple obs
            tables only if every declaration shares the same ``feature_space``
            (the shared declaration then collapses to one entry); conflicting
            declarations raise at construction.
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
        if not obs_schemas:
            raise ValueError("obs_schemas must contain at least one obs table.")
        db_uri = _resolve_db_uri(db_uri)
        registry_table_names = {fs: f"{fs}_registry" for fs in registry_schemas}
        _validate_foreign_key_source_schemas(
            {
                **obs_schemas,
                dataset_table_name: dataset_schema,
                **{
                    registry_table_names[fs]: schema_cls
                    for fs, schema_cls in registry_schemas.items()
                },
            },
            registry_table_names=registry_table_names,
        )
        db = lancedb.connect(db_uri, storage_options=_store_kwargs_to_storage_options(store_kwargs))
        obs_tables: dict[str, lancedb.table.Table] = {}
        for name, schema_cls in obs_schemas.items():
            obs_tables[name] = db.create_table(name, schema=schema_cls)
        dataset_table = db.create_table(dataset_table_name, schema=dataset_schema)

        registry_tables: dict[str, lancedb.table.Table] = {}
        for fs, schema_cls in registry_schemas.items():
            table_name = registry_table_names[fs]
            registry_tables[fs] = db.create_table(table_name, schema=schema_cls)

        version_table = db.create_table(version_table_name, schema=AtlasVersionRecord)

        feature_layouts_table = db.create_table("_feature_layouts", schema=FeatureLayout)
        feature_layouts_table.create_fts_index("feature_uid")
        feature_layouts_table.create_fts_index("layout_uid")

        root = zarr.open_group(zarr.storage.ObjectStore(store), mode="w")

        return cls(
            db=db,
            obs_tables=obs_tables,
            obs_schemas=dict(obs_schemas),
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
        obs_table_names: list[str] | None = None,
        obs_schemas: dict[str, type[HoxBaseSchema] | None] | None = None,
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
        obs_table_names:
            Names of the obs tables to open. For single-obs-table atlases pass
            a one-element list.
        obs_schemas:
            Optional mapping of ``{obs_table_name: schema_cls | None}``. For
            tables with no entry, pointer fields are inferred from the obs
            table's Arrow schema (sufficient for read-only use).
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
        if obs_table_names is None and not obs_schemas:
            raise ValueError(
                "open() requires obs_table_names or obs_schemas to identify "
                "which obs tables to open."
            )
        if obs_table_names is None:
            obs_table_names = list(obs_schemas)
        if obs_schemas is None:
            obs_schemas = {name: None for name in obs_table_names}
        if set(obs_schemas) != set(obs_table_names):
            raise ValueError(
                f"obs_table_names ({sorted(obs_table_names)}) and obs_schemas "
                f"({sorted(obs_schemas)}) must have identical keys."
            )
        if not obs_table_names:
            raise ValueError("obs_table_names must contain at least one name.")
        db_uri = _resolve_db_uri(db_uri)
        db = lancedb.connect(db_uri, storage_options=_store_kwargs_to_storage_options(store_kwargs))
        obs_tables = {name: db.open_table(name) for name in obs_table_names}
        obs_schemas_full: dict[str, type[HoxBaseSchema] | None] = dict(obs_schemas)
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
            obs_tables=obs_tables,
            obs_schemas=obs_schemas_full,
            root=root,
            registry_tables=resolved_registries,
            dataset_table=dataset_table,
            version_table=version_table,
            feature_layouts_table=feature_layouts_table,
        )

    # -- Store helpers ------------------------------------------------------

    def _resolve_obs_table(
        self,
        obs_table_name: str | None = None,
    ) -> tuple[str, lancedb.table.Table]:
        """Resolve a (obs_table_name, obs_table) pair.

        - If ``obs_table_name`` is given, use it (must exist).
        - Else, only valid when exactly one obs table is registered.
        """
        if obs_table_name is not None:
            if obs_table_name not in self._obs_tables:
                raise KeyError(
                    f"Unknown obs table {obs_table_name!r}. Available: {sorted(self._obs_tables)}"
                )
            return obs_table_name, self._obs_tables[obs_table_name]

        if len(self._obs_tables) == 1:
            name = next(iter(self._obs_tables))
            return name, self._obs_tables[name]
        raise ValueError(
            f"Atlas has multiple obs tables ({sorted(self._obs_tables)}); "
            "Pass obs_table_name= to select one."
        )

    # TODO: Can't recall why we need to specify feature_space
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
            # TODO: I don't think that specifying the feature_space is necessary
            # this is probably legacy. Each zarr_group is exactly 1 feature space, always
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
        for name, table in self._obs_tables.items():
            _fmt_table(f"Obs table [{name}]", table)
        _fmt_table("Dataset table", self._dataset_table)
        for fs, reg_table in sorted(self._registry_tables.items()):
            _fmt_table(f"Registry [{fs}]", reg_table)

        summary = "\n".join(lines)
        print(summary)
        return summary

    # -- Public accessors ---------------------------------------------------

    # TODO: I think we should remove this and stick to `obs_tables` instead
    @property
    def obs_table(self) -> lancedb.table.Table:
        """The obs table, when the atlas has exactly one. Raises otherwise.

        Read-only consumers can use the returned handle freely. For writes,
        prefer :meth:`add_obs_records` — calling ``.add(...)`` on a fresh
        handle obtained via ``atlas.db.open_table(name)`` would leave this
        held handle stale and cause :meth:`snapshot` to refuse.
        """
        _, table = self._resolve_obs_table()
        return table

    # TODO: I think we should remove this and stick to `obs_schemas` instead
    @property
    def obs_schema(self) -> type[HoxBaseSchema] | None:
        """Obs schema class for the atlas's single obs table.

        Raises if the atlas has multiple obs tables; use ``obs_schemas`` for
        the full mapping in that case.
        """
        name, _ = self._resolve_obs_table()
        return self._obs_schemas[name]

    # TODO: Tiny, but an `obs_table_names` property that just returns the keys
    # would be convenient
    @property
    def obs_tables(self) -> dict[str, lancedb.table.Table]:
        """All obs tables, keyed by obs_table_name.

        For writes, prefer :meth:`add_obs_records` so the held handle stays
        in sync with the on-disk version (otherwise :meth:`snapshot` will
        refuse with a stale-handle error).
        """
        return self._obs_tables

    @property
    def obs_schemas(self) -> dict[str, type[HoxBaseSchema] | None]:
        """All obs schemas, keyed by obs_table_name (value may be ``None``)."""
        return self._obs_schemas

    @property
    def pointer_fields(self) -> dict[str, PointerField]:
        """Flat ``{field_name: PointerField}`` view across all obs tables.

        Pointer fields with the same name in multiple obs tables collapse to a
        single entry — they are required to share the same definition (same
        feature_space). Use :meth:`pointer_fields_for` to get a per-table view.
        """
        return self._pointer_fields

    def pointer_fields_for(self, obs_table_name: str) -> dict[str, PointerField]:
        """Return the pointer-field mapping for a specific obs table."""
        if obs_table_name not in self._obs_tables:
            raise KeyError(
                f"Unknown obs table {obs_table_name!r}. Available: {sorted(self._obs_tables)}"
            )
        return {
            fn: self._pointer_fields[fn]
            for fn, tables in self._field_to_tables.items()
            if obs_table_name in tables
        }

    @property
    def foreign_keys(self) -> list[dict[str, str]]:
        """Compiled scalar foreign-key manifest for this atlas view."""
        return list(self._foreign_keys)

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

    def _compile_foreign_key_manifest_from_tables(self) -> list[dict[str, str]]:
        """Compile FK declarations from managed Lance table schemas."""
        records: list[dict[str, str]] = []
        for table_name, table in self._obs_tables.items():
            for fk in _infer_foreign_key_fields_from_arrow(table.schema).values():
                records.append(_foreign_key_record(table_name, fk))
        for fk in _infer_foreign_key_fields_from_arrow(self._dataset_table.schema).values():
            records.append(_foreign_key_record(self._dataset_table.name, fk))
        for table in self._registry_tables.values():
            for fk in _infer_foreign_key_fields_from_arrow(table.schema).values():
                records.append(_foreign_key_record(table.name, fk))
        return records

    def _managed_table_by_name(self, table_name: str) -> lancedb.table.Table | None:
        if table_name in self._obs_tables:
            return self._obs_tables[table_name]
        if table_name == self._dataset_table.name:
            return self._dataset_table
        if table_name == self._feature_layouts_table.name:
            return self._feature_layouts_table
        for table in self._registry_tables.values():
            if table.name == table_name:
                return table
        return None

    def _validate_foreign_key_rows(
        self,
        *,
        source_table_name: str,
        source_arrow: pa.Table | None = None,
    ) -> list[str]:
        """Validate FK constraints for one source table or pending Arrow batch.

        Returns one human-readable error string per violated or unresolvable
        constraint (empty when valid), so callers can aggregate alongside other
        validators. When *source_arrow* is given, only those rows are checked;
        otherwise the full source table column is read.
        """
        errors: list[str] = []
        for fk in self._foreign_keys:
            if fk["source_table"] != source_table_name:
                continue
            source_field = fk["source_field"]
            target_table_name = fk["target_table"]
            target_field = fk["target_field"]
            label = (
                f"FOREIGN KEY ({source_table_name}.{source_field}) "
                f"REFERENCES {target_table_name} ({target_field})"
            )

            # Resolve the source values: a pending Arrow batch, or the full table.
            if source_arrow is not None:
                if source_arrow.schema.get_field_index(source_field) < 0:
                    errors.append(f"{label}: source field cannot be resolved")
                    continue
                src_values = pl.from_arrow(source_arrow.select([source_field]))[source_field]
            else:
                source_table = self._managed_table_by_name(source_table_name)
                if source_table is None:
                    errors.append(f"{label}: source table cannot be opened")
                    continue
                if source_table.schema.get_field_index(source_field) < 0:
                    errors.append(f"{label}: source field cannot be resolved")
                    continue
                src_values = source_table.search().select([source_field]).to_polars()[source_field]

            # The target table may live outside the atlas; opening it can fail.
            target_table = self._managed_table_by_name(target_table_name)
            if target_table is None:
                try:
                    target_table = self.db.open_table(target_table_name)
                except Exception as exc:
                    errors.append(f"{label}: target table cannot be opened/resolved: {exc}")
                    continue
            if target_table.schema.get_field_index(target_field) < 0:
                errors.append(f"{label}: target field cannot be resolved")
                continue
            tgt_values = target_table.search().select([target_field]).to_polars()[target_field]

            # Anti-join: non-null source values absent from the target column.
            non_null = src_values.drop_nulls()
            invalid = non_null.filter(~non_null.is_in(tgt_values.implode()))
            if invalid.is_empty():
                continue
            sample = invalid.unique().head(5).to_list()
            errors.append(
                f"{label}: {invalid.len()} invalid non-null value(s); "
                f"sample missing values={sample!r}"
            )
        return errors

    def _validate_foreign_key_rows_or_raise(
        self,
        *,
        source_table_name: str,
        source_arrow: pa.Table,
    ) -> None:
        errors = self._validate_foreign_key_rows(
            source_table_name=source_table_name,
            source_arrow=source_arrow,
        )
        if errors:
            raise ValueError(
                "Foreign key validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
            )

    # -- Dataset / zarr helpers --------------------------------------------

    def register_dataset(
        self,
        dataset_record: DatasetSchema,
        *,
        var_df: pl.DataFrame | None = None,
    ) -> None:
        """Insert a ``DatasetSchema`` into the dataset table.

        When ``var_df`` is provided, computes the feature layout from
        ``var_df`` against the registry for ``dataset_record.feature_space``,
        writes any new layout rows into ``_feature_layouts``, and sets
        ``dataset_record.layout_uid`` before insertion. The dataset row is
        therefore born with its final ``layout_uid``; no read-modify-write
        on the dataset table is needed.

        Parameters
        ----------
        dataset_record:
            The dataset row to insert. ``feature_space`` and ``zarr_group``
            must already be set; ``layout_uid`` is overwritten when ``var_df``
            is given.
        var_df:
            One row per local feature in local feature order. Must have a
            ``uid`` column whose values match registry uids. Pass ``None`` for
            feature spaces without a per-dataset feature layout.
        """
        if var_df is not None:
            feature_space = dataset_record.feature_space
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

            dataset_record.layout_uid = layout_uid

        arrow_table = pa.Table.from_pylist(
            [dataset_record.model_dump()],
            schema=type(dataset_record).to_arrow_schema(),
        )
        self._validate_foreign_key_rows_or_raise(
            source_table_name=self._dataset_table.name,
            source_arrow=arrow_table,
        )
        self._dataset_table.add(arrow_table)

    def add_obs_records(
        self,
        records: pa.Table | list[HoxBaseSchema],
        *,
        obs_table_name: str | None = None,
    ) -> None:
        """Append rows to an obs table via the atlas's held handle.

        Always prefer this over ``atlas.obs_table.add(...)`` or
        ``atlas.db.open_table(name).add(...)``. A fresh handle commits a
        new on-disk version that the atlas's held handle never observes,
        so ``snapshot()`` would record a stale version. ``add_obs_records``
        routes through ``self._obs_tables[name]`` so the held handle stays
        in sync.
        """
        name, table = self._resolve_obs_table(obs_table_name)
        if isinstance(records, pa.Table):
            arrow = records
        else:
            if not records:
                return
            schema_cls = self._obs_schemas.get(name)
            if schema_cls is None:
                raise ValueError(
                    f"Cannot convert HoxBaseSchema records for obs table {name!r}: "
                    "no schema was supplied at open() time. Pass a pa.Table instead, "
                    f"or reopen the atlas with obs_schemas={{{name!r}: YourSchema}}."
                )
            arrow = pa.Table.from_pylist(
                [r.model_dump() for r in records],
                schema=schema_cls.to_arrow_schema(),
            )
        self._validate_foreign_key_rows_or_raise(
            source_table_name=name,
            source_arrow=arrow,
        )
        table.add(arrow)

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

    def _iter_managed_tables(self) -> Iterator[lancedb.table.Table]:
        """Yield every LanceDB table the atlas tracks for snapshotting.

        ``_version_table`` is intentionally excluded: ``snapshot()`` is its
        only writer, so it cannot drift relative to itself.
        """
        yield from self._obs_tables.values()
        yield self._dataset_table
        yield from self._registry_tables.values()
        yield self._feature_layouts_table

    def refresh(self) -> None:
        """Advance every held LanceDB table handle to the latest on-disk version.

        Call this after writes that bypassed the atlas's held handles —
        e.g. another process committed, or in-process code used
        ``atlas.db.open_table(...)`` instead of the blessed
        ``add_obs_records`` / ``register_*`` paths. ``snapshot()`` raises
        if any handle is stale; ``refresh()`` clears that.
        """
        for table in self._iter_managed_tables():
            table.checkout_latest()

    # -- Query entry point --------------------------------------------------

    def query(self, obs_table_name: str | None = None) -> "AtlasQuery":
        """Start building a query against one of this atlas's obs tables.

        Parameters
        ----------
        obs_table_name:
            Which obs table to query. May be ``None`` only when the atlas has
            exactly one obs table.
        """
        from homeobox.query import AtlasQuery

        if self._checked_out_version is None:
            raise RuntimeError(
                "query() is only available on a versioned atlas. "
                "After ingestion, call atlas.snapshot() then "
                "RaggedAtlas.checkout(db_uri, version, obs_schemas, store) to pin to a "
                "validated snapshot. For convenience, use RaggedAtlas.checkout_latest(...)."
            )
        return AtlasQuery(self, obs_table_name=obs_table_name)

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
        for table in self._obs_tables.values():
            table.optimize()
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
            get_spec(pf.feature_space)

        if check_registries:
            errors.extend(self._validate_registries())

        # Collect unique zarr groups from dataset table
        zarr_groups_by_space = self._collect_zarr_groups()

        if check_zarr:
            errors.extend(self._validate_zarr_groups(zarr_groups_by_space))

        if check_var_dfs:
            errors.extend(self._validate_feature_layouts(zarr_groups_by_space))

        errors.extend(self._validate_foreign_keys())

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

    def _validate_foreign_keys(self) -> list[str]:
        errors: list[str] = []
        for source_table_name in sorted({fk["source_table"] for fk in self._foreign_keys}):
            errors.extend(self._validate_foreign_key_rows(source_table_name=source_table_name))
        return errors

    # -- Versioning ---------------------------------------------------------

    def snapshot(self) -> int:
        """Record a consistent snapshot of all table versions.

        Returns the new atlas version number (0-indexed, monotonically increasing).
        Raises ``ValueError`` if validation errors are found, or ``RuntimeError``
        if any held table handle is behind the on-disk state (call
        :meth:`refresh` and retry).
        """
        stale: list[tuple[str, int, int]] = []
        for table in self._iter_managed_tables():
            held = table.version
            fresh = self.db.open_table(table.name).version
            if held != fresh:
                stale.append((table.name, held, fresh))
        if stale:
            details = "\n".join(f"  • {name}: held=v{h}, on-disk=v{f}" for name, h, f in stale)
            raise RuntimeError(
                "snapshot() refused: the following table handles are behind "
                "the on-disk state, likely because writes have been performed "
                "since the tables on this atlas were opened. Call atlas.refresh() and retry."
                + details
            )

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

        obs_table_versions = json.dumps({n: t.version for n, t in self._obs_tables.items()})
        total_rows = sum(t.count_rows() for t in self._obs_tables.values())

        record = AtlasVersionRecord(
            version=next_version,
            obs_table_versions=obs_table_versions,
            dataset_table_name=self._dataset_table.name,
            dataset_table_version=self._dataset_table.version,
            registry_table_names=json.dumps(registry_names),
            registry_table_versions=json.dumps(registry_versions),
            feature_layouts_table_version=self._feature_layouts_table.version,
            foreign_keys=json.dumps(self._foreign_keys, sort_keys=True),
            total_rows=total_rows,
        )
        self._version_table.add([record])
        return next_version

    @staticmethod
    def _foreign_keys_from_version_row(row: dict) -> list[dict[str, str]]:
        """Read the FK manifest persisted by :meth:`snapshot` for one version row."""
        return json.loads(row.get("foreign_keys") or "[]")

    @staticmethod
    def _read_obs_tables_from_record(
        db: lancedb.DBConnection, row: dict, *, restore: bool = False
    ) -> dict[str, lancedb.table.Table]:
        """Open + checkout (and optionally restore) the obs tables for a snapshot row."""
        versions: dict[str, int] = json.loads(row["obs_table_versions"])
        out: dict[str, lancedb.table.Table] = {}
        for tbl_name, version in versions.items():
            t = db.open_table(tbl_name)
            t.checkout(version)
            if restore:
                t.restore()
            out[tbl_name] = t
        return out

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
        obs_schemas: dict[str, type[HoxBaseSchema] | None] | None = None,
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
        obs_schemas:
            Optional ``{obs_table_name: schema_cls | None}`` mapping. For tables
            with no entry, pointer fields are inferred from the obs table's
            Arrow schema (sufficient for read-only use).
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

        obs_tables = cls._read_obs_tables_from_record(db, row)
        if obs_schemas is None:
            obs_schemas = {name: None for name in obs_tables}
        if set(obs_schemas) != set(obs_tables):
            raise ValueError(
                f"obs_schemas keys {sorted(obs_schemas)} do not match the obs "
                f"tables recorded in the snapshot ({sorted(obs_tables)})."
            )
        resolved_schemas: dict[str, type[HoxBaseSchema] | None] = dict(obs_schemas)

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
        foreign_keys = cls._foreign_keys_from_version_row(row)

        atlas = cls(
            db=db,
            obs_tables=obs_tables,
            obs_schemas=resolved_schemas,
            root=root,
            registry_tables=resolved_registries,
            dataset_table=dataset_table,
            version_table=version_table,
            feature_layouts_table=feature_layouts_table,
            foreign_keys=foreign_keys,
        )
        atlas._checked_out_version = version
        return atlas

    @classmethod
    def restore(
        cls,
        db_uri: str,
        version: int,
        obs_schemas: dict[str, type[HoxBaseSchema] | None] | None = None,
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
        obs_schemas:
            Optional ``{obs_table_name: schema_cls | None}`` mapping.
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

        obs_tables = cls._read_obs_tables_from_record(db, row, restore=True)
        if obs_schemas is None:
            obs_schemas = {name: None for name in obs_tables}
        if set(obs_schemas) != set(obs_tables):
            raise ValueError(
                f"obs_schemas keys {sorted(obs_schemas)} do not match the obs "
                f"tables recorded in the snapshot ({sorted(obs_tables)})."
            )
        resolved_schemas: dict[str, type[HoxBaseSchema] | None] = dict(obs_schemas)

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
        foreign_keys = cls._foreign_keys_from_version_row(row)

        return cls(
            db=db,
            obs_tables=obs_tables,
            obs_schemas=resolved_schemas,
            root=root,
            registry_tables=resolved_registries,
            dataset_table=dataset_table,
            version_table=version_table,
            feature_layouts_table=feature_layouts_table,
            foreign_keys=foreign_keys,
        )

    @classmethod
    def checkout_latest(
        cls,
        db_uri: str,
        obs_schemas: dict[str, type[HoxBaseSchema] | None] | None = None,
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
        obs_schemas:
            Optional ``{obs_table_name: schema_cls | None}`` mapping.
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
            obs_schemas=obs_schemas,
            store=store,
            store_kwargs=store_kwargs,
            version_table_name=version_table_name,
        )


# ---------------------------------------------------------------------------
# Atlas create-or-open helper
# ---------------------------------------------------------------------------


def create_or_open_atlas(
    atlas_path: str,
    obs_schemas: dict[str, type[HoxBaseSchema]],
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
    obs_schemas:
        Mapping of ``{obs_table_name: HoxBaseSchema subclass}`` declaring one
        or more obs tables and their pointer-field schemas.
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
    if not obs_schemas:
        raise ValueError("obs_schemas must contain at least one obs table.")
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

    obs_names = list(obs_schemas)
    present = [n for n in obs_names if n in existing_tables]
    missing = [n for n in obs_names if n not in existing_tables]
    if present and missing:
        raise ValueError(
            f"Atlas at {atlas_path!r} is in a mixed state: obs tables {present} "
            f"already exist but {missing} do not. Either drop the existing tables "
            "or remove the missing ones from obs_schemas."
        )

    if present:
        # Explicitly pass registry table names so open() doesn't rely on
        # the datasets table (which may be empty for a freshly-initialised atlas).
        registry_tables = {
            fs: f"{fs}_registry" for fs in registry_schemas if f"{fs}_registry" in existing_tables
        }
        return RaggedAtlas.open(
            db_uri=db_uri,
            obs_table_names=obs_names,
            obs_schemas=dict(obs_schemas),
            dataset_table_name=dataset_table_name,
            store=store,
            registry_tables=registry_tables,
            version_table_name=version_table_name,
            store_kwargs=store_kwargs,
        )
    else:
        return RaggedAtlas.create(
            db_uri=db_uri,
            obs_schemas=dict(obs_schemas),
            dataset_table_name=dataset_table_name,
            dataset_schema=dataset_schema,
            store=store,
            registry_schemas=registry_schemas,
            version_table_name=version_table_name,
            store_kwargs=store_kwargs,
        )
