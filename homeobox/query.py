"""AtlasQuery: fluent query builder for reading cells from a RaggedAtlas."""

from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import mudata as mu
    import pandas
    from lancedb.query import LanceQueryBuilder

    from homeobox.dataloader import MultimodalHoxDataset, UnimodalHoxDataset
    from homeobox.fragments.reconstruction import FragmentResult
    from homeobox.multimodal import MultimodalResult

import anndata as ad
import numpy as np
import polars as pl

from homeobox.atlas import RaggedAtlas
from homeobox.group_specs import get_spec
from homeobox.reconstruction import (
    _build_obs_only_anndata,
    _get_pointer_columns,
)
from homeobox.schema import PointerField
from homeobox.util import sql_escape


class AtlasQuery:
    """Fluent query builder for reading cells from a RaggedAtlas."""

    def __init__(self, atlas: RaggedAtlas) -> None:
        self._atlas = atlas
        self._search_query: np.ndarray | list[float] | str | None = None
        self._search_kwargs: dict = {}
        self._where_clause: str | None = None
        self._offset_n: int | None = None
        self._limit_n: int | None = None
        self._select_columns: list[str] | None = None
        self._feature_spaces: list[str] | None = None
        self._field_names: list[str] | None = None
        self._layer_overrides: dict[str, list[str]] = {}
        self._feature_join: Literal["union", "intersection"] = "union"
        self._feature_filter: dict[str, list[str]] = {}
        self._balanced_limit_n: int | None = None
        self._balanced_limit_column: str | None = None

    def search(
        self,
        query: "np.ndarray | list[float] | str | None" = None,
        *,
        vector_column_name: str | None = None,
        query_type: str = "auto",
        fts_columns: str | list[str] | None = None,
    ) -> "AtlasQuery":
        """Add a vector or full-text search to the query.

        Parameters are forwarded to ``lancedb.Table.search()``.

        Parameters
        ----------
        query:
            A vector (ndarray / list), full-text search string, or ``None``
            for a full scan.
        vector_column_name:
            Which vector column to search against.
        query_type:
            One of ``"auto"``, ``"vector"``, ``"fts"``, or ``"hybrid"``.
        fts_columns:
            Column(s) to search for full-text queries.
        """
        self._search_query = query
        self._search_kwargs = {
            "vector_column_name": vector_column_name,
            "query_type": query_type,
            "fts_columns": fts_columns,
        }
        return self

    def select(self, columns: list[str]) -> "AtlasQuery":
        """Select specific metadata columns to return.

        Pointer columns required for AnnData reconstruction are always
        loaded internally, even if not listed here.

        Parameters
        ----------
        columns:
            Column names to include in the results.
        """
        if not isinstance(columns, list):
            raise ValueError("Columns must be a list")
        self._select_columns = columns
        return self

    def where(self, condition: str) -> "AtlasQuery":
        """Add a SQL WHERE filter (LanceDB syntax)."""
        self._where_clause = condition
        return self

    def offset(self, n: int) -> "AtlasQuery":
        """Skip the first *n* cells before returning results."""
        self._offset_n = n
        return self

    def limit(self, n: int) -> "AtlasQuery":
        """Limit the number of cells returned."""
        if self._balanced_limit_n is not None:
            raise ValueError("Cannot use both limit() and balanced_limit() on the same query")
        self._limit_n = n
        return self

    def balanced_limit(self, n: int, column: str) -> "AtlasQuery":
        """Limit cells, drawing equally from each unique value of *column*.

        The result contains at most *n* cells, split evenly across each
        unique value of *column* that passes any ``.where()`` filter.

        Cannot be combined with ``.limit()``.
        """
        if self._limit_n is not None:
            raise ValueError("Cannot use both limit() and balanced_limit() on the same query")
        self._balanced_limit_n = n
        self._balanced_limit_column = column
        return self

    def feature_spaces(self, *spaces: str) -> "AtlasQuery":
        """Restrict reconstruction to pointer fields in the listed feature spaces.

        Acts as a modality-level filter: every pointer field whose
        ``feature_space`` matches is included. Use :meth:`select_fields` for
        exact-field selection when multiple fields share a feature space.
        """
        known = {pf.feature_space for pf in self._atlas.pointer_fields.values()}
        unknown = set(spaces) - known
        if unknown:
            raise ValueError(
                f"Unknown feature space(s): {sorted(unknown)}. Available: {sorted(known)}"
            )
        self._feature_spaces = list(spaces)
        return self

    def select_fields(self, *field_names: str) -> "AtlasQuery":
        """Restrict reconstruction to specific pointer-field attribute names.

        Unlike :meth:`feature_spaces` (which selects by modality), this
        selects exact pointer columns — useful when a schema declares
        multiple columns in the same feature space (e.g. ``cycle1_image_tiles``
        and ``cycle2_image_tiles``).
        """
        known = set(self._atlas.pointer_fields.keys())
        unknown = set(field_names) - known
        if unknown:
            raise ValueError(
                f"Unknown pointer field(s): {sorted(unknown)}. Available: {sorted(known)}"
            )
        self._field_names = list(field_names)
        return self

    def layers(self, feature_space: str, names: list[str]) -> "AtlasQuery":
        """Specify which layers to read for a given feature space."""
        self._layer_overrides[feature_space] = names
        return self

    def features(self, uids: list[str], feature_space: str) -> "AtlasQuery":
        """Filter output to specific features by global UID.

        When set, reconstruction for this feature space returns only the
        requested features. The ``feature_join`` setting is ignored for
        filtered feature spaces; intersection semantics are used.
        """
        if feature_space not in self._atlas.registry_tables:
            known = sorted(self._atlas.registry_tables.keys())
            raise ValueError(f"No registry for feature space '{feature_space}'. Available: {known}")
        self._feature_filter[feature_space] = list(uids)
        return self

    def feature_join(self, join: Literal["union", "intersection"]) -> "AtlasQuery":
        """Set how features are joined across zarr groups.

        ``"union"`` (default) includes all features from any group.
        ``"intersection"`` includes only features present in every group.
        """
        self._feature_join = join
        return self

    # -- Execution ----------------------------------------------------------

    def _build_base_query(self) -> "LanceQueryBuilder":
        """Build a query with search, where, and limit applied (no column selection)."""
        q = self._atlas.obs_table.search(self._search_query, **self._search_kwargs)
        if self._where_clause is not None:
            q = q.where(self._where_clause)
        if self._offset_n is not None:
            q = q.offset(self._offset_n)
        if self._limit_n is not None:
            q = q.limit(self._limit_n)
        elif self._search_query is not None:
            # lancedb search() defaults to 10 rows when no limit is set.
            # For FTS MatchQuery this is safe: it only returns actual matches.
            q = q.limit(1_000_000)
        return q

    def _build_scanner(self) -> "LanceQueryBuilder":
        """Build a LanceDB query from the current state."""
        q = self._build_base_query()
        if self._select_columns is not None:
            pointer_cols = list(self._atlas.pointer_fields.keys())
            columns = list(dict.fromkeys(self._select_columns + pointer_cols))
            q = q.select(columns)
        return q

    @staticmethod
    def _drop_score(df: pl.DataFrame) -> pl.DataFrame:
        """Drop the ``_score`` column injected by lancedb search, if present."""
        if "_score" in df.columns:
            return df.drop("_score")
        return df

    def _materialize_cells(self) -> pl.DataFrame:
        """Materialise the cell DataFrame, respecting balanced_limit if set."""
        if self._balanced_limit_n is not None:
            return self._drop_score(self._materialize_balanced())
        return self._drop_score(self._build_scanner().to_polars())

    def _materialize_cells_for_dataset(self) -> pl.DataFrame:
        """Materialise a lightweight cell DataFrame with row IDs for UnimodalHoxDataset.

        Returns only pointer columns + ``_rowid`` (lance's built-in row ID).
        Metadata is loaded lazily per batch via ``take_row_ids``.
        """
        if self._balanced_limit_n is not None:
            return self._drop_score(self._materialize_balanced_for_dataset())

        q = self._build_base_query().with_row_id(True)
        pointer_cols = list(self._atlas.pointer_fields.keys())
        q = q.select(pointer_cols)
        return self._drop_score(q.to_polars())

    def _discover_balanced_groups(self, column: str) -> list:
        """Discover unique values of *column* across all matching rows."""
        q = self._atlas.obs_table.search(self._search_query, **self._search_kwargs)
        if self._where_clause is not None:
            q = q.where(self._where_clause)
        if self._search_query is not None:
            # lancedb search() defaults to 10 rows; override so discovery
            # sees all matching rows (MatchQuery returns only real matches).
            q = q.limit(1_000_000)
        return q.select([column]).to_polars()[column].unique().to_list()

    def _materialize_balanced_for_dataset(self) -> pl.DataFrame:
        """Two-phase balanced materialisation returning only pointers + _rowid."""
        column = self._balanced_limit_column
        assert column is not None

        unique_values = self._discover_balanced_groups(column)
        n_groups = len(unique_values)
        if n_groups == 0:
            pointer_cols = list(self._atlas.pointer_fields.keys())
            q = self._build_base_query().with_row_id(True).select(pointer_cols)
            return q.to_polars().head(0)

        per_group = self._balanced_limit_n // n_groups
        pointer_cols = list(self._atlas.pointer_fields.keys())

        frames: list[pl.DataFrame] = []
        for val in unique_values:
            escaped = sql_escape(str(val))
            group_filter = f"{column} = '{escaped}'"
            if self._where_clause is not None:
                combined = f"({self._where_clause}) AND ({group_filter})"
            else:
                combined = group_filter

            q = self._atlas.obs_table.search(self._search_query, **self._search_kwargs)
            q = q.where(combined).limit(per_group).with_row_id(True)
            q = q.select(pointer_cols)
            frames.append(q.to_polars())

        return pl.concat(frames)

    def _materialize_balanced(self) -> pl.DataFrame:
        """Two-phase balanced materialisation."""
        column = self._balanced_limit_column
        assert column is not None

        unique_values = self._discover_balanced_groups(column)
        n_groups = len(unique_values)
        if n_groups == 0:
            return self._build_scanner().to_polars().head(0)

        per_group = self._balanced_limit_n // n_groups

        # Phase 2: one sub-query per group
        frames: list[pl.DataFrame] = []
        for val in unique_values:
            escaped = sql_escape(str(val))
            group_filter = f"{column} = '{escaped}'"
            if self._where_clause is not None:
                combined = f"({self._where_clause}) AND ({group_filter})"
            else:
                combined = group_filter

            q = self._atlas.obs_table.search(self._search_query, **self._search_kwargs)
            q = q.where(combined).limit(per_group)
            if self._select_columns is not None:
                pointer_cols = list(self._atlas.pointer_fields.keys())
                columns = list(dict.fromkeys(self._select_columns + pointer_cols))
                q = q.select(columns)
            frames.append(q.to_polars())

        return pl.concat(frames)

    def _active_pointer_fields(self) -> dict[str, PointerField]:
        """Return pointer fields filtered by ``feature_spaces`` / ``select_fields``."""
        pfs = self._atlas.pointer_fields
        if self._field_names is not None:
            return {k: pfs[k] for k in self._field_names}
        if self._feature_spaces is not None:
            return {k: v for k, v in pfs.items() if v.feature_space in self._feature_spaces}
        return pfs

    def count(self, group_by: str | list[str] | None = None) -> "pl.DataFrame | int":
        """Count cells, optionally grouped by metadata columns.

        Only the grouping columns are fetched from LanceDB, so this is much
        cheaper than ``to_polars()`` for large atlases.

        Parameters
        ----------
        group_by:
            Column name(s) to group by.  If ``None``, returns a scalar count.

        Returns
        -------
        int
            Total cell count when ``group_by`` is ``None``.
        pl.DataFrame
            DataFrame with one row per group and a ``count`` column otherwise.
        """
        q = self._build_base_query()

        if group_by is None:
            # Fetch only a single cheap column to count rows
            any_col = self._atlas.obs_table.schema.names[0]
            return len(q.select([any_col]).to_arrow())

        cols = [group_by] if isinstance(group_by, str) else list(group_by)
        result = q.select(cols).to_polars()
        return result.group_by(cols).agg(pl.len().alias("count")).sort(cols)

    def to_polars(self) -> pl.DataFrame:
        """Execute the query and return a Polars DataFrame of cell metadata."""
        result = self._materialize_cells()
        pointer_cols = _get_pointer_columns(result)
        if pointer_cols:
            keep = [c for c in result.columns if c not in pointer_cols]
            result = result.select(keep)
        return result

    def to_anndata(self) -> ad.AnnData:
        """Execute the query and reconstruct an AnnData.

        If multiple feature spaces are active, only the first sparse feature
        space is used for X. Use :meth:`to_mudata` for multi-modal.
        """
        cells_pl = self._materialize_cells()
        if cells_pl.is_empty():
            return _build_obs_only_anndata(cells_pl)

        active_pfs = self._active_pointer_fields()
        if not active_pfs:
            return _build_obs_only_anndata(cells_pl)

        # Pick the first feature space that has data for the queried cells
        # TODO: Add a warning on this behavior. Should be clear to the user
        # that if they want a specific field they should use `select_fields`
        # TODO: We also shouldn't assume that all reconstructors have an `as_anndata`
        # method. This is something we should be able to figure out and warning or error.
        # if the wrong kind of feature space is selected.
        for pf in active_pfs.values():
            zg = cells_pl[pf.field_name].struct.field("zarr_group")
            if zg.is_not_null().any():
                return self._reconstruct_single_space_anndata(cells_pl, pf)

        return _build_obs_only_anndata(cells_pl)

    def to_mudata(self) -> "mu.MuData":
        """Execute the query and return a MuData with one modality per feature space.

        Non-AnnData modalities (fragments, raw arrays) are silently dropped
        with a warning. Use :meth:`to_multimodal` for full heterogeneous access.
        """
        return self.to_multimodal().to_mudata()

    def to_multimodal(self) -> "MultimodalResult":
        """Execute the query and return all active modalities in their native format.

        Each modality is reconstructed as its natural type:

        - AnnData for matrix-based modalities (gene expression, protein abundance, etc.)
        - :class:`~homeobox.fragments.reconstruction.FragmentResult` for chromatin accessibility
        - :class:`numpy.ndarray` for raw dense arrays without feature annotations (image tiles)

        Returns
        -------
        MultimodalResult
            Container with shared ``obs``, per-modality data in ``mod``,
            and boolean presence masks in ``present``.
        """
        from homeobox.multimodal import MultimodalResult
        from homeobox.reconstruction import _build_obs_df

        cells_pl = self._materialize_cells()
        obs = _build_obs_df(cells_pl)

        if cells_pl.is_empty():
            return MultimodalResult(obs=obs)

        active_pfs = self._active_pointer_fields()
        mod: dict[str, ad.AnnData | np.ndarray] = {}
        present: dict[str, np.ndarray] = {}

        for pf in active_pfs.values():
            spec = get_spec(pf.feature_space)
            reconstructor = spec.reconstructor
            endpoints = spec.valid_endpoints()

            # Compute presence mask from pointer column
            ptr_col = cells_pl[pf.field_name]
            zg_series = ptr_col.struct.field("zarr_group")
            mask = zg_series.is_not_null().to_numpy()

            if not mask.any():
                continue

            # TODO: Might make sense to not just have `endpoint` but `default_endpoint`
            # instead. That specifies what the natural preferred type is. This isn't perfect
            # when there are cases where a reconstructor isn't oriented toward 1 specific feature
            # space (i.e., DenseReconstructor can be for image tiles or image features).
            # Dispatch to the appropriate endpoint:
            # fragments first (modality-native), then raw array for var-less
            # dense feature spaces, otherwise an AnnData.
            if "as_fragments" in endpoints:
                result = reconstructor.as_fragments(self._atlas, cells_pl, pf, spec)
            elif "as_array" in endpoints and not spec.has_var_df:
                result = reconstructor.as_array(self._atlas, cells_pl, pf, spec)
            else:
                result = self._reconstruct_single_space_anndata(cells_pl, pf)

            mod[pf.field_name] = result
            present[pf.field_name] = mask

        return MultimodalResult(obs=obs, mod=mod, present=present)

    def to_fragments(self, field_name: str = "chromatin_accessibility") -> "FragmentResult":
        """Reconstruct a single fragment-based pointer field.

        Parameters
        ----------
        field_name
            Pointer-field attribute name whose feature_space exposes an
            ``as_fragments`` endpoint (e.g. chromatin accessibility).
        """
        pf = self._atlas.pointer_fields[field_name]
        spec = get_spec(pf.feature_space)
        endpoints = spec.valid_endpoints()
        if "as_fragments" not in endpoints:
            raise TypeError(
                f"Field '{field_name}' (feature_space='{pf.feature_space}') does not "
                f"support as_fragments. Valid endpoints: {endpoints}"
            )

        cells_pl = self._materialize_cells()
        return spec.reconstructor.as_fragments(self._atlas, cells_pl, pf, spec)

    def to_array(self, field_name: str) -> "tuple[np.ndarray, pandas.DataFrame]":
        """Reconstruct a single dense pointer field as a raw array.

        Returns the full-dimensionality array (e.g. 4D for image tiles)
        alongside the obs DataFrame for present cells.

        Parameters
        ----------
        field_name
            Pointer-field attribute name whose feature_space exposes an
            ``as_array`` endpoint (e.g. dense modalities like image tiles).

        Returns
        -------
        (array, obs)
            The raw NumPy array and a DataFrame of cell metadata for the
            cells present in this modality.
        """
        from homeobox.reconstruction import _build_obs_df

        pf = self._atlas.pointer_fields[field_name]
        spec = get_spec(pf.feature_space)
        endpoints = spec.valid_endpoints()
        if "as_array" not in endpoints:
            raise TypeError(
                f"Field '{field_name}' (feature_space='{pf.feature_space}') does not "
                f"support as_array. Valid endpoints: {endpoints}"
            )

        cells_pl = self._materialize_cells()
        array = spec.reconstructor.as_array(self._atlas, cells_pl, pf, spec)
        obs = _build_obs_df(cells_pl)
        return array, obs

    def to_batches(self, batch_size: int = 1024) -> Iterator[ad.AnnData]:
        """Stream results as AnnData batches.

        Each batch contains up to ``batch_size`` cells. BatchArray readers
        and remap arrays are cached on the atlas for reuse across batches.

        When ``balanced_limit`` is active, the full balanced result is
        materialised first and then chunked in Python.
        """
        active_pfs = self._active_pointer_fields()
        pf = next(iter(active_pfs.values())) if active_pfs else None

        if self._balanced_limit_n is not None:
            cells_pl = self._materialize_cells()
            for offset in range(0, len(cells_pl), batch_size):
                chunk = cells_pl.slice(offset, batch_size)
                if chunk.is_empty():
                    continue
                if pf is None:
                    yield _build_obs_only_anndata(chunk)
                else:
                    yield self._reconstruct_single_space_anndata(chunk, pf)
            return

        q = self._build_scanner()
        reader = q.to_batches(batch_size=batch_size)

        if pf is None:
            for batch in reader:
                if batch.num_rows == 0:
                    continue
                yield _build_obs_only_anndata(pl.from_arrow(batch))
            return

        for batch in reader:
            if batch.num_rows == 0:
                continue
            yield self._reconstruct_single_space_anndata(pl.from_arrow(batch), pf)

    def to_cell_dataset(
        self,
        field_name: str,
        layer: str | None = None,
        metadata_columns: list[str] | None = None,
        stack_dense: bool = True,
    ) -> "UnimodalHoxDataset":
        """Create a UnimodalHoxDataset for fast ML training iteration.

        Unlike :meth:`to_batches` (which reconstructs full AnnData per batch),
        this returns a :class:`~homeobox.dataloader.UnimodalHoxDataset` that yields
        lightweight :class:`~homeobox.dataloader.SparseBatch` or
        :class:`~homeobox.dataloader.DenseBatch` objects via
        :meth:`~homeobox.dataloader.UnimodalHoxDataset.__getitems__`.

        Use :func:`~homeobox.dataloader.make_loader` to wrap the dataset
        in a ``torch.utils.data.DataLoader``.

        Parameters
        ----------
        field_name:
            Pointer-field attribute name on the cell schema.
        layer:
            Which layer to read within the pointer field's feature space.
            When ``None``, auto-resolved to the first required layer for
            layered specs or ignored for layer-less specs (e.g. ``image_tiles``).
        metadata_columns:
            Obs column names to include as metadata on each batch.
        stack_dense:
            Whether dense batches should be stacked into a single ndarray. Set
            to ``False`` to return one array per cell, which allows variable-size
            image tiles.

        Notes
        -----
        If a feature filter was set on this query (via
        :meth:`~homeobox.query.AtlasQuery.features`), the returned dataset's
        feature space is automatically restricted to those features
        (``wanted_globals`` is derived from the filter; ``n_features``
        reflects the filtered count).
        """
        from homeobox.dataloader import UnimodalHoxDataset
        from homeobox.group_specs import get_spec

        pf = self._atlas.pointer_fields[field_name]
        feature_space = pf.feature_space
        spec = get_spec(feature_space)
        if layer is None:
            zgs_layers = spec.zarr_group_spec.layers
            layer = zgs_layers.required[0].array_name if zgs_layers.required else ""

        cells_pl = self._materialize_cells_for_dataset()

        wanted_globals = None
        if feature_space in self._feature_filter and spec.has_var_df:
            from homeobox.feature_layouts import resolve_feature_uids_to_global_indices

            wanted_globals = resolve_feature_uids_to_global_indices(
                self._atlas.registry_tables[feature_space],
                self._feature_filter[feature_space],
            )

        return UnimodalHoxDataset(
            atlas=self._atlas,
            cells_pl=cells_pl,
            field_name=field_name,
            layer=layer,
            metadata_columns=metadata_columns,
            wanted_globals=wanted_globals,
            stack_dense=stack_dense,
        )

    def to_multimodal_dataset(
        self,
        field_names: list[str],
        layers: dict[str, str] | None = None,
        metadata_columns: list[str] | None = None,
        stack_dense: bool | dict[str, bool] = True,
    ) -> "MultimodalHoxDataset":
        """Create a MultimodalHoxDataset for within-cell multimodal training.

        Each yielded :class:`~homeobox.dataloader.MultimodalBatch` contains
        one sub-batch per modality with only the cells that have that
        modality present. A ``present`` mask tracks membership. No fill
        values are added.

        Use :func:`~homeobox.dataloader.make_loader` to wrap the dataset
        in a ``torch.utils.data.DataLoader``.

        Parameters
        ----------
        field_names:
            Ordered list of pointer-field attribute names to include.
        layers:
            ``{field_name: layer_name}`` mapping.  Defaults to the first
            required layer of each pointer field's feature space (or ``""``
            for layer-less specs) when omitted.
        metadata_columns:
            Obs column names to include as metadata on each batch.
        stack_dense:
            Whether dense batches should be stacked into a single ndarray.
            May be a single bool for all dense modalities or a mapping by
            field name, e.g. ``{"image_tiles": False}``.
        """
        from homeobox.dataloader import MultimodalHoxDataset
        from homeobox.group_specs import get_spec

        cells_pl = self._materialize_cells_for_dataset()

        resolved_pfs = {fn: self._atlas.pointer_fields[fn] for fn in field_names}

        if layers is None:
            layers = {}
            for fn, pf in resolved_pfs.items():
                fs_spec = get_spec(pf.feature_space)
                zgs_layers = fs_spec.zarr_group_spec.layers
                layers[fn] = zgs_layers.required[0].array_name if zgs_layers.required else ""

        wanted_globals: dict[str, np.ndarray] | None = None
        for fn, pf in resolved_pfs.items():
            fs_spec = get_spec(pf.feature_space)
            if pf.feature_space in self._feature_filter and fs_spec.has_var_df:
                from homeobox.feature_layouts import resolve_feature_uids_to_global_indices

                wg = resolve_feature_uids_to_global_indices(
                    self._atlas.registry_tables[pf.feature_space],
                    self._feature_filter[pf.feature_space],
                )
                if wanted_globals is None:
                    wanted_globals = {}
                wanted_globals[fn] = wg

        return MultimodalHoxDataset(
            atlas=self._atlas,
            cells_pl=cells_pl,
            field_names=field_names,
            layers=layers,
            metadata_columns=metadata_columns,
            wanted_globals=wanted_globals,
            stack_dense=stack_dense,
        )

    # -- Reconstruction internals -------------------------------------------

    def _reconstruct_single_space_anndata(
        self,
        cells_pl: pl.DataFrame,
        pf: PointerField,
    ) -> ad.AnnData:
        """Reconstruct an AnnData for a single feature space."""
        spec = get_spec(pf.feature_space)
        endpoints = spec.valid_endpoints()
        if "as_anndata" not in endpoints:
            raise TypeError(
                f"Feature space '{pf.feature_space}' does not support as_anndata. "
                f"Valid endpoints: {endpoints}"
            )

        wanted_globals = None
        if pf.feature_space in self._feature_filter:
            from homeobox.feature_layouts import resolve_feature_uids_to_global_indices

            wanted_globals = resolve_feature_uids_to_global_indices(
                self._atlas.registry_tables[pf.feature_space],
                self._feature_filter[pf.feature_space],
            )

        return spec.reconstructor.as_anndata(
            self._atlas,
            cells_pl,
            pf,
            spec,
            layer_overrides=self._layer_overrides.get(pf.feature_space),
            feature_join=self._feature_join,
            wanted_globals=wanted_globals,
        )
