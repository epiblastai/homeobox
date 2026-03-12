"""AtlasQuery: fluent query builder for reading cells from a RaggedAtlas."""

from collections.abc import Iterator

import anndata as ad
import lancedb
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp

from lancell.atlas import PointerFieldInfo, RaggedAtlas
from lancell.group_specs import ZARR_SPECS, FeatureSpace, LayerName, PointerKind, ZarrGroupSpec
from lancell.reconstruction import (
    _build_obs_df,
    _build_obs_only_anndata,
    _load_remaps_and_union,
    _prepare_dense_cells,
    _prepare_sparse_cells,
)


class AtlasQuery:
    """Fluent query builder for reading cells from a RaggedAtlas."""

    def __init__(self, atlas: RaggedAtlas) -> None:
        self._atlas = atlas
        self._search_query: np.ndarray | list[float] | str | None = None
        self._search_kwargs: dict = {}
        self._where_clause: str | None = None
        self._limit_n: int | None = None
        self._feature_spaces: list[FeatureSpace] | None = None
        self._layer_overrides: dict[FeatureSpace, list[LayerName]] = {}

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

    def where(self, condition: str) -> "AtlasQuery":
        """Add a SQL WHERE filter (LanceDB syntax)."""
        self._where_clause = condition
        return self

    def limit(self, n: int) -> "AtlasQuery":
        """Limit the number of cells returned."""
        self._limit_n = n
        return self

    def feature_spaces(self, *spaces: FeatureSpace) -> "AtlasQuery":
        """Restrict reconstruction to specific feature spaces."""
        self._feature_spaces = list(spaces)
        return self

    def layers(self, feature_space: FeatureSpace, names: list[LayerName]) -> "AtlasQuery":
        """Specify which layers to read for a given feature space."""
        self._layer_overrides[feature_space] = names
        return self

    # -- Execution ----------------------------------------------------------

    def _build_scanner(self) -> lancedb.table.Table:
        """Build a LanceDB query from the current state."""
        q = self._atlas.cell_table.search(self._search_query, **self._search_kwargs)
        if self._where_clause is not None:
            q = q.where(self._where_clause)
        if self._limit_n is not None:
            q = q.limit(self._limit_n)
        return q

    def _active_pointer_fields(self) -> dict[str, PointerFieldInfo]:
        """Return pointer fields filtered by requested feature spaces."""
        pfs = self._atlas._pointer_fields
        if self._feature_spaces is None:
            return pfs
        return {k: v for k, v in pfs.items() if v.feature_space in self._feature_spaces}

    def to_polars(self) -> pl.DataFrame:
        """Execute the query and return a Polars DataFrame of cell metadata."""
        return self._build_scanner().to_polars()

    def to_anndata(self) -> ad.AnnData:
        """Execute the query and reconstruct an AnnData.

        If multiple feature spaces are active, only the first sparse feature
        space is used for X. Use :meth:`to_mudata` for multi-modal.
        """
        cells_pl = self._build_scanner().to_polars()
        if cells_pl.is_empty():
            return ad.AnnData()

        active_pfs = self._active_pointer_fields()
        # Pick the first feature space for X
        if not active_pfs:
            return _build_obs_only_anndata(cells_pl)

        # Use first pointer field
        pf = next(iter(active_pfs.values()))
        return self._reconstruct_single_space(cells_pl, pf)

    def to_mudata(self) -> "mu.MuData":
        """Execute the query and return a MuData with one modality per feature space."""
        import mudata as mu

        cells_pl = self._build_scanner().to_polars()
        if cells_pl.is_empty():
            return mu.MuData({})

        active_pfs = self._active_pointer_fields()
        modalities: dict[str, ad.AnnData] = {}
        for pf in active_pfs.values():
            adata = self._reconstruct_single_space(cells_pl, pf)
            if adata.n_obs > 0:
                modalities[pf.feature_space.value] = adata

        return mu.MuData(modalities)

    def to_batches(self, batch_size: int = 1024) -> Iterator[ad.AnnData]:
        """Stream results as AnnData batches.

        Each batch contains up to ``batch_size`` cells. BatchArray readers
        and remap arrays are cached on the atlas for reuse across batches.
        """
        q = self._build_scanner()
        arrow_table = q.to_arrow()
        n_total = arrow_table.num_rows
        if n_total == 0:
            return

        active_pfs = self._active_pointer_fields()
        if not active_pfs:
            # Obs-only batches
            for start in range(0, n_total, batch_size):
                batch_arrow = arrow_table.slice(start, batch_size)
                batch_pl = pl.from_arrow(batch_arrow)
                yield _build_obs_only_anndata(batch_pl)
            return

        pf = next(iter(active_pfs.values()))
        for start in range(0, n_total, batch_size):
            batch_arrow = arrow_table.slice(start, batch_size)
            batch_pl = pl.from_arrow(batch_arrow)
            yield self._reconstruct_single_space(batch_pl, pf)

    # -- Reconstruction internals -------------------------------------------

    def _reconstruct_single_space(
        self,
        cells_pl: pl.DataFrame,
        pf: PointerFieldInfo,
    ) -> ad.AnnData:
        """Reconstruct an AnnData for a single feature space."""
        spec = ZARR_SPECS[pf.feature_space]
        if pf.pointer_kind is PointerKind.SPARSE:
            return self._reconstruct_sparse(cells_pl, pf, spec)
        else:
            return self._reconstruct_dense(cells_pl, pf, spec)

    def _reconstruct_sparse(
        self,
        cells_pl: pl.DataFrame,
        pf: PointerFieldInfo,
        spec: ZarrGroupSpec,
    ) -> ad.AnnData:
        """Reconstruct sparse data (e.g. gene expression) across zarr groups."""
        # Determine index array name from spec's required_arrays
        if len(spec.required_arrays) != 1:
            raise NotImplementedError(
                f"Sparse reconstruction for feature space '{pf.feature_space.value}' "
                f"is not yet supported (requires {len(spec.required_arrays)} "
                f"primary arrays: {[a.array_name for a in spec.required_arrays]})"
            )
        index_array_name = spec.required_arrays[0].array_name

        cells_pl, groups = _prepare_sparse_cells(cells_pl, pf)
        if not groups:
            return ad.AnnData()

        _, union_globals, group_remap_to_union, n_features = _load_remaps_and_union(
            self._atlas, groups, spec
        )

        # Determine which layers to read
        layer_names = self._layer_overrides.get(pf.feature_space)
        if layer_names is None:
            layer_names = list(spec.required_layers)
            if not layer_names:
                raise ValueError(
                    f"No layers specified and spec for '{pf.feature_space.value}' "
                    f"has no required layers"
                )
        layers_to_read = [ln.value for ln in layer_names]

        # Process each zarr group
        all_csrs: dict[str, list[sp.csr_matrix]] = {ln: [] for ln in layers_to_read}
        obs_parts: list[pl.DataFrame] = []

        for zg in groups:
            group_cells = cells_pl.filter(pl.col("_zg") == zg)
            starts = group_cells["_start"].to_numpy().astype(np.int64)
            ends = group_cells["_end"].to_numpy().astype(np.int64)
            n_cells_group = len(starts)

            # Batch-read index array via Rust reader
            indices_reader = self._atlas._get_batch_reader(zg, index_array_name)
            flat_indices, lengths = indices_reader.read_ranges(starts, ends)

            # Remap local indices -> union positions
            if zg in group_remap_to_union:
                union_remap = group_remap_to_union[zg]
                union_indices = union_remap[flat_indices.astype(np.intp)]
            else:
                union_indices = flat_indices.astype(np.int32)

            # Build indptr from lengths
            indptr = np.zeros(n_cells_group + 1, dtype=np.int64)
            np.cumsum(lengths, out=indptr[1:])

            # Batch-read each layer
            for ln in layers_to_read:
                layer_reader = self._atlas._get_batch_reader(
                    zg, f"layers/{ln}"
                )
                flat_values, _ = layer_reader.read_ranges(starts, ends)

                csr = sp.csr_matrix(
                    (flat_values, union_indices, indptr),
                    shape=(n_cells_group, n_features),
                )
                all_csrs[ln].append(csr)

            obs_parts.append(group_cells)

        # Stack CSRs
        stacked: dict[str, sp.csr_matrix] = {}
        for ln, csr_list in all_csrs.items():
            if csr_list:
                stacked[ln] = sp.vstack(csr_list, format="csr")

        # Build obs
        obs_pl = pl.concat(obs_parts, how="diagonal_relaxed")
        obs = _build_obs_df(obs_pl)

        # Build var from registry
        var = self._build_var(pf.feature_space, union_globals)

        # First layer becomes X, rest go to layers
        first_layer = layers_to_read[0]
        X = stacked.get(first_layer)
        extra_layers = {ln: stacked[ln] for ln in layers_to_read[1:] if ln in stacked}

        adata = ad.AnnData(X=X, obs=obs, var=var, layers=extra_layers if extra_layers else None)
        return adata

    def _reconstruct_dense(
        self,
        cells_pl: pl.DataFrame,
        pf: PointerFieldInfo,
        spec: ZarrGroupSpec,
    ) -> ad.AnnData:
        """Reconstruct dense data (e.g. protein abundance) across zarr groups."""
        cells_pl, groups = _prepare_dense_cells(cells_pl, pf)
        if not groups:
            return ad.AnnData()

        _, union_globals, group_remap_to_union, n_union_features = _load_remaps_and_union(
            self._atlas, groups, spec
        )

        # Determine which layers to read
        layer_names = self._layer_overrides.get(pf.feature_space)
        if layer_names is None:
            layer_names = list(spec.required_layers)
        layers_to_read = [ln.value for ln in layer_names] if layer_names else []

        # Resolve array names: "layers/{ln}" for layered specs, "data" for plain
        array_names = [f"layers/{ln}" for ln in layers_to_read] if layers_to_read else ["data"]
        output_keys = layers_to_read if layers_to_read else ["data"]

        n_total_cells = cells_pl.height
        all_layers: dict[str, np.ndarray] = {
            k: np.zeros((n_total_cells, n_union_features), dtype=np.float32)
            for k in output_keys
        }

        obs_parts: list[pl.DataFrame] = []
        offset = 0

        for zg in groups:
            group_cells = cells_pl.filter(pl.col("_zg") == zg)
            positions = group_cells["_pos"].to_numpy().astype(np.int64)
            n_cells_group = len(positions)

            # Build axis-0 ranges: each position is a single row [pos, pos+1)
            starts = positions
            ends = positions + 1

            for array_name, out_key in zip(array_names, output_keys):
                reader = self._atlas._get_batch_reader(zg, array_name)
                flat_data, _ = reader.read_ranges(starts, ends)
                n_local_features = flat_data.shape[0] // n_cells_group
                local_data = flat_data.reshape(n_cells_group, n_local_features)

                if zg in group_remap_to_union:
                    union_cols = group_remap_to_union[zg]
                    all_layers[out_key][offset : offset + n_cells_group][:, union_cols] = local_data
                else:
                    all_layers[out_key][offset : offset + n_cells_group, :n_local_features] = local_data

            obs_parts.append(group_cells)
            offset += n_cells_group

        obs_pl = pl.concat(obs_parts, how="diagonal_relaxed")
        obs = _build_obs_df(obs_pl)
        var = self._build_var(pf.feature_space, union_globals)

        # First layer/array -> X, rest -> adata.layers
        first_key = output_keys[0]
        X = all_layers[first_key]
        extra_layers = {k: all_layers[k] for k in output_keys[1:]}

        return ad.AnnData(X=X, obs=obs, var=var, layers=extra_layers if extra_layers else None)

    def _build_var(
        self, feature_space: FeatureSpace, union_globals: np.ndarray
    ) -> pd.DataFrame:
        """Build a var DataFrame from the feature registry."""
        if feature_space not in self._atlas._registry_tables or len(union_globals) == 0:
            return pd.DataFrame(index=pd.RangeIndex(len(union_globals)))

        registry_table = self._atlas._registry_tables[feature_space]
        registry_df = registry_table.search().to_polars()

        # Filter to union globals
        registry_df = registry_df.filter(
            pl.col("global_index").is_in(union_globals.tolist())
        ).sort("global_index")

        var = registry_df.to_pandas()
        if "uid" in var.columns:
            var = var.set_index("uid")
        return var
