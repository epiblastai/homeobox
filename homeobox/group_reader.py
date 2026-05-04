"""GroupReader: per-(zarr_group, feature_space) read state."""

import lancedb
import numpy as np
import obstore
import polars as pl
import zarr

from homeobox.batch_array import BatchAsyncArray
from homeobox.feature_layouts import read_feature_layout


class LayoutReader:
    """Shared read state for a feature layout.

    ``GroupReader`` is per zarr group, but ``remap`` and ``var_df`` are
    determined only by ``layout_uid``.  This object keeps those layout-derived
    arrays shareable across many group readers.

    Construct lazily from a feature-layouts table by passing
    ``feature_layouts_table`` (the atlas path), or eagerly from a pre-resolved
    array via :meth:`from_remap` (the dataloader-worker path).
    """

    def __init__(
        self,
        layout_uid: str | None,
        feature_layouts_table: lancedb.table.Table | None = None,
        remap: np.ndarray | None = None,
        var_df: pl.DataFrame | None = None,
    ) -> None:
        self.layout_uid = layout_uid
        self._feature_layouts_table = feature_layouts_table
        self._remap = self._freeze_remap(remap) if remap is not None else None
        self._var_df = var_df
        self._rows: pl.DataFrame | None = None

    @classmethod
    def from_remap(
        cls,
        layout_uid: str | None,
        remap: np.ndarray,
        var_df: pl.DataFrame | None = None,
    ) -> "LayoutReader":
        return cls(layout_uid=layout_uid, remap=remap, var_df=var_df)

    @staticmethod
    def _freeze_remap(remap: np.ndarray) -> np.ndarray:
        if remap.dtype != np.int32:
            remap = remap.astype(np.int32, copy=False)
        remap.flags.writeable = False
        return remap

    def _read_rows(self) -> pl.DataFrame:
        if self._rows is not None:
            return self._rows
        if self._feature_layouts_table is None or self.layout_uid is None:
            raise ValueError("LayoutReader has no layout table to load from.")
        self._rows = read_feature_layout(self._feature_layouts_table, self.layout_uid)
        return self._rows

    def get_remap(self) -> np.ndarray:
        """Return the local-to-global-index remap array (load-once)."""
        if self._remap is not None:
            return self._remap
        rows = self._read_rows()
        if rows["global_index"].null_count() > 0:
            raise ValueError(
                f"Layout '{self.layout_uid}' has null global_index values; run optimize() first."
            )
        self._remap = self._freeze_remap(rows["global_index"].to_numpy())
        return self._remap

    @property
    def var_df(self) -> pl.DataFrame:
        """Return var_df for this layout in local feature order (load-once)."""
        if self._var_df is not None:
            return self._var_df
        rows = self._read_rows()
        self._var_df = rows.select(
            [
                pl.col("feature_uid").alias("global_feature_uid"),
            ]
        )
        return self._var_df

    def __setstate__(self, state: dict) -> None:
        # numpy's pickle does not preserve the writeable flag — re-freeze.
        self.__dict__.update(state)
        if self._remap is not None:
            self._remap.flags.writeable = False


_EMPTY_VAR_DF = pl.DataFrame(schema={"global_feature_uid": pl.Utf8})


class GroupReader:
    """Encapsulates all per-(zarr_group, feature_space) read state.

    Used by both the reconstruction path (RaggedAtlas) and the ML training
    path (UnimodalHoxDataset worker processes).

    ``zarr_group`` is the string path within the object store (e.g.
    ``"datasets/abc123/rna"``).  It is the durable, picklable identity of
    this reader.  ``_zarr_group_handle`` is the live ``zarr.Group`` object
    derived from that path; it is opened lazily on first array access and
    **zeroed out on pickling** (see ``__getstate__``).  The two are kept
    separate because ``zarr.Group`` handles are not safely picklable across
    process boundaries — when a ``GroupReader`` is sent to a DataLoader worker
    the handle is stripped and re-opened fresh inside the worker on first use.

    Create via the two factories:
    - :meth:`from_atlas_root` — atlas reconstruction path
    - :meth:`for_worker` — DataLoader worker path
    """

    def __init__(
        self,
        zarr_group: str,
        feature_space: str,
        store: obstore.store.ObjectStore,
        layout_reader: LayoutReader | None = None,
        zarr_group_handle: zarr.Group | None = None,
    ) -> None:
        self.zarr_group = zarr_group
        self.feature_space = feature_space
        self._store = store
        self._layout_reader = layout_reader
        self._zarr_group_handle = zarr_group_handle
        self._csc_indptr: np.ndarray | None = None
        self._array_reader_cache: dict[str, BatchAsyncArray] = {}

    @classmethod
    def from_atlas_root(
        cls,
        zarr_group: str,
        feature_space: str,
        store: obstore.store.ObjectStore,
        layout_reader: LayoutReader | None = None,
    ) -> "GroupReader":
        """Create a GroupReader for an atlas.

        The zarr group handle is opened lazily on first array access.
        Used by ``RaggedAtlas.get_group_reader``.  ``layout_reader`` may be
        ``None`` for feature spaces with ``has_var_df=False``.
        """
        return cls(
            zarr_group=zarr_group,
            feature_space=feature_space,
            store=store,
            layout_reader=layout_reader,
        )

    @classmethod
    def for_worker(
        cls,
        zarr_group: str,
        feature_space: str,
        store: obstore.store.ObjectStore,
        layout_reader: LayoutReader | None = None,
    ) -> "GroupReader":
        """Create a GroupReader for a DataLoader worker.

        Accepts a pre-resolved ``LayoutReader`` (already version-checked at
        UnimodalHoxDataset init time). The zarr group handle is ``None`` until
        first use.
        """
        return cls(
            zarr_group=zarr_group,
            feature_space=feature_space,
            store=store,
            layout_reader=layout_reader,
        )

    @property
    def layout_reader(self) -> LayoutReader | None:
        """Return the shared layout reader, if this group has a feature layout."""
        return self._layout_reader

    def get_remap(self) -> np.ndarray:
        """Return the local-to-global-index remap array (load-once)."""
        if self._layout_reader is None:
            raise ValueError(
                f"GroupReader for {self.zarr_group!r} has no remap and no table to load from."
            )
        return self._layout_reader.get_remap()

    @property
    def var_df(self) -> pl.DataFrame:
        """Load and cache var_df for this zarr group (load-once).

        Returns a DataFrame with column ``global_feature_uid`` in local
        feature order (row i = local feature i).
        """
        if self._layout_reader is None:
            return _EMPTY_VAR_DF
        return self._layout_reader.var_df

    @property
    def has_csc(self) -> bool:
        """Return True if this zarr group has a feature-oriented (CSC) copy.

        Resolved against the registered ``FeatureSpaceSpec.feature_oriented``:
        if the spec declares no feature-oriented copy this always returns
        False, otherwise the on-disk subgroup is validated against the spec.
        """
        from homeobox.group_specs import get_spec

        spec = get_spec(self.feature_space)
        if spec.feature_oriented is None:
            return False
        self._ensure_initialized()
        return spec.has_feature_oriented_copy(self._zarr_group_handle)

    def get_csc_indptr(self) -> np.ndarray:
        """Lazily load and cache the CSC indptr array from zarr."""
        if self._csc_indptr is not None:
            return self._csc_indptr
        self._ensure_initialized()
        self._csc_indptr = np.asarray(self._zarr_group_handle["csc"]["indptr"][:])
        return self._csc_indptr

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
