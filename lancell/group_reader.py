"""GroupReader: per-(zarr_group, feature_space) read state."""

import lancedb
import numpy as np
import obstore
import polars as pl
import zarr

from lancell.batch_array import BatchAsyncArray
from lancell.var_df import (
    build_remap,
    has_csc,
    read_remap_if_fresh,
    read_var_df,
    write_remap,
)


class GroupReader:
    """Encapsulates all per-(zarr_group, feature_space) read state.

    Used by both the reconstruction path (RaggedAtlas) and the ML training
    path (CellDataset worker processes). Zarr handles are opened lazily and
    zeroed on pickling; all other state is picklable.

    Create via the two factories:
    - :meth:`from_atlas_root` — atlas reconstruction path
    - :meth:`for_worker` — DataLoader worker path
    """

    def __init__(
        self,
        zarr_group: str,
        feature_space: str,
        store: obstore.store.ObjectStore,
        registry_table: lancedb.table.Table | None,
        read_only: bool,
        remap_cache: tuple[int, np.ndarray] | None = None,
        var_df_cache: pl.DataFrame | None = None,
        zarr_group_handle: zarr.Group | None = None,
    ) -> None:
        self.zarr_group = zarr_group
        self.feature_space = feature_space
        self._store = store
        self._registry_table = registry_table
        self._read_only = read_only
        self._remap_cache = remap_cache
        self._var_df_cache = var_df_cache
        self._zarr_group_handle = zarr_group_handle
        self._array_reader_cache: dict[str, BatchAsyncArray] = {}

    @classmethod
    def from_atlas_root(
        cls,
        zarr_group: str,
        feature_space: str,
        root: zarr.Group,
        store: obstore.store.ObjectStore,
        registry_table: lancedb.table.Table | None,
        read_only: bool,
    ) -> "GroupReader":
        """Create a GroupReader from an open atlas root.

        The zarr group handle is obtained immediately from the root.
        Used by ``RaggedAtlas._get_group_reader``.
        ``registry_table`` may be ``None`` for feature spaces with
        ``has_var_df=False``.
        """
        return cls(
            zarr_group=zarr_group,
            feature_space=feature_space,
            store=store,
            registry_table=registry_table,
            read_only=read_only,
            zarr_group_handle=root[zarr_group],
        )

    @classmethod
    def for_worker(
        cls,
        zarr_group: str,
        feature_space: str,
        store: obstore.store.ObjectStore,
        remap: np.ndarray,
    ) -> "GroupReader":
        """Create a GroupReader for a DataLoader worker.

        Accepts a pre-resolved remap (already version-checked at CellDataset
        init time). Sets ``_registry_table=None`` — workers never re-check
        registry version. The zarr group handle is ``None`` until first use.
        """
        return cls(
            zarr_group=zarr_group,
            feature_space=feature_space,
            store=store,
            registry_table=None,
            read_only=True,
            remap_cache=(0, remap),
        )

    def get_remap(self) -> np.ndarray:
        """Return the local-to-global-index remap array.

        If ``_registry_table`` is ``None`` (worker path), returns the frozen
        remap directly. Otherwise performs a version-aware cache check and
        rebuilds if stale.
        """
        if self._registry_table is None:
            assert self._remap_cache is not None, (
                f"GroupReader for {self.zarr_group!r} has no remap. "
                "The for_worker path requires a remap at construction time."
            )
            return self._remap_cache[1]

        current_version = self._registry_table.version
        if self._remap_cache is not None:
            cached_version, cached_remap = self._remap_cache
            if cached_version == current_version:
                return cached_remap

        # In-memory cache miss or stale — try the on-disk remap
        group = self._zarr_group_handle
        disk_remap = read_remap_if_fresh(self._store, group, current_version)
        if disk_remap is not None:
            self._remap_cache = (current_version, disk_remap)
            return disk_remap

        # Rebuild from var_df + registry
        remap = build_remap(self.var_df, self._registry_table)
        if not self._read_only:
            write_remap(self._store, group, remap, registry_version=current_version)
        self._remap_cache = (current_version, remap)
        return remap

    @property
    def var_df(self) -> pl.DataFrame:
        """Load and cache var_df for this zarr group."""
        if self._var_df_cache is None:
            self._var_df_cache = read_var_df(self._store, self.zarr_group)
        return self._var_df_cache

    @property
    def has_csc(self) -> bool:
        """Return True if this zarr group has CSC data."""
        return has_csc(self.var_df)

    def get_array_reader(self, array_name: str) -> BatchAsyncArray:
        """Return a cached BatchAsyncArray reader for a zarr array."""
        self._ensure_initialized()
        reader = self._array_reader_cache.get(array_name)
        if reader is None:
            reader = BatchAsyncArray.from_array(self._zarr_group_handle[array_name])
            self._array_reader_cache[array_name] = reader
        return reader

    def _ensure_initialized(self) -> None:
        """Open the zarr group handle lazily if not yet done."""
        if self._zarr_group_handle is None:
            root = zarr.open_group(zarr.storage.ObjectStore(self._store), mode="r")
            self._zarr_group_handle = root[self.zarr_group]
            self._array_reader_cache = {}

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # Zero transient zarr state so the object is safely picklable.
        state["_zarr_group_handle"] = None
        state["_array_reader_cache"] = {}
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
