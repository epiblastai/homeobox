"""Tests for homeobox.dex pure math/stats functions (no atlas needed)."""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.stats import mannwhitneyu as scipy_mwu
from scipy.stats import ttest_ind

from homeobox.dex._dex import _benjamini_hochberg, _compare, _extract_matrix, _group_where
from homeobox.dex._math import fold_change, mwu, normalize_log1p_sparse, percent_change, pseudobulk
from homeobox.dex._numba_mwu import (
    MannWhitneyUResult,
    mannwhitneyu_dense,
    mannwhitneyu_sparse,
    sparse_column_index,
)
from homeobox.dex._ttest import welch_ttest
from homeobox.group_specs import PointerKind

# ---------------------------------------------------------------------------
# TestBenjaminiHochberg
# ---------------------------------------------------------------------------


class TestBenjaminiHochberg:
    def test_empty(self):
        result = _benjamini_hochberg(np.array([], dtype=np.float64))
        assert result.dtype == np.float64
        assert len(result) == 0

    def test_single_pvalue(self):
        result = _benjamini_hochberg(np.array([0.05]))
        np.testing.assert_allclose(result, [0.05])

    def test_all_ones(self):
        result = _benjamini_hochberg(np.ones(10))
        np.testing.assert_allclose(result, np.ones(10))

    def test_clipped_to_01(self):
        result = _benjamini_hochberg(np.array([0.9, 0.95, 0.99]))
        assert np.all(result <= 1.0)
        assert np.all(result >= 0.0)


# ---------------------------------------------------------------------------
# TestFoldChange
# ---------------------------------------------------------------------------


class TestFoldChange:
    def test_basic(self):
        result = fold_change(np.array([4.0]), np.array([2.0]))
        np.testing.assert_allclose(result, [1.0])

    def test_no_change(self):
        result = fold_change(np.array([5.0]), np.array([5.0]))
        np.testing.assert_allclose(result, [0.0])

    def test_inverse_symmetry(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(1, 100, size=20)
        y = rng.uniform(1, 100, size=20)
        np.testing.assert_allclose(fold_change(x, y), -fold_change(y, x), atol=1e-10)

    def test_vectorized(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(1, 100, size=100)
        y = rng.uniform(1, 100, size=100)
        np.testing.assert_allclose(fold_change(x, y), np.log2(x / y), atol=1e-10)


# ---------------------------------------------------------------------------
# TestPercentChange
# ---------------------------------------------------------------------------


class TestPercentChange:
    def test_basic(self):
        result = percent_change(np.array([6.0]), np.array([3.0]))
        np.testing.assert_allclose(result, [1.0])

    def test_no_change(self):
        result = percent_change(np.array([5.0]), np.array([5.0]))
        np.testing.assert_allclose(result, [0.0])

    def test_vectorized(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(1, 100, size=100)
        y = rng.uniform(1, 100, size=100)
        np.testing.assert_allclose(percent_change(x, y), (x - y) / y, atol=1e-10)


# ---------------------------------------------------------------------------
# TestNormalizeLog1pSparse
# ---------------------------------------------------------------------------


class TestNormalizeLog1pSparse:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = sp.random(10, 5, density=0.5, format="csr", dtype=np.float32, random_state=rng)
        X.data[:] = rng.integers(1, 100, size=X.nnz).astype(np.float32)
        target_sum = 1e4
        normalize_log1p_sparse(X, target_sum)
        # After normalization, expm1(row).sum() ≈ target_sum for non-empty rows
        for i in range(X.shape[0]):
            row = X[i].toarray().flatten()
            if np.any(row != 0):
                np.testing.assert_allclose(np.expm1(row).sum(), target_sum, rtol=1e-4)

    def test_empty_rows(self):
        indptr = np.array([0, 0, 2, 2], dtype=np.int32)
        indices = np.array([0, 1], dtype=np.int32)
        data = np.array([1.0, 2.0], dtype=np.float32)
        X = sp.csr_matrix((data, indices, indptr), shape=(3, 3))
        normalize_log1p_sparse(X, 1e4)
        # Empty rows remain all-zero
        assert X[0].nnz == 0
        assert X[2].nnz == 0

    def test_single_nonzero(self):
        indptr = np.array([0, 1], dtype=np.int32)
        indices = np.array([0], dtype=np.int32)
        data = np.array([5.0], dtype=np.float32)
        X = sp.csr_matrix((data, indices, indptr), shape=(1, 3))
        normalize_log1p_sparse(X, 1e4)
        expected = np.log1p(1e4)
        np.testing.assert_allclose(X.data[0], expected, rtol=1e-4)

    def test_preserves_sparsity_pattern(self):
        rng = np.random.default_rng(42)
        X = sp.random(10, 5, density=0.3, format="csr", dtype=np.float32, random_state=rng)
        X.data[:] = rng.integers(1, 100, size=X.nnz).astype(np.float32)
        orig_indices = X.indices.copy()
        orig_indptr = X.indptr.copy()
        normalize_log1p_sparse(X, 1e4)
        np.testing.assert_array_equal(X.indices, orig_indices)
        np.testing.assert_array_equal(X.indptr, orig_indptr)


# ---------------------------------------------------------------------------
# TestPseudobulk
# ---------------------------------------------------------------------------


class TestPseudobulk:
    def test_arithmetic_mean_dense(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 100, size=(20, 5)).astype(np.float64)
        result = pseudobulk(X, geometric_mean=False, is_log1p=False)
        expected = X.mean(axis=0)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_arithmetic_mean_sparse_log1p(self):
        rng = np.random.default_rng(42)
        X = sp.random(20, 5, density=0.5, format="csr", dtype=np.float32, random_state=rng)
        X.data[:] = rng.uniform(0.1, 5.0, size=X.nnz).astype(np.float32)
        # arithmetic mean with is_log1p: expm1(X).mean(axis=0)
        expected = np.array(np.expm1(X.toarray()).mean(axis=0)).flatten()
        result = pseudobulk(X, geometric_mean=False, is_log1p=True)
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_geometric_mean_dense(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 100, size=(20, 5)).astype(np.float64)
        # geometric mean: expm1(mean(log1p(X), axis=0))
        expected = np.expm1(np.log1p(X).mean(axis=0))
        result = pseudobulk(X, geometric_mean=True, is_log1p=False)
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_geometric_mean_sparse_log1p(self):
        rng = np.random.default_rng(42)
        X = sp.random(20, 5, density=0.5, format="csr", dtype=np.float32, random_state=rng)
        X.data[:] = rng.uniform(0.1, 5.0, size=X.nnz).astype(np.float32)
        # geometric mean with is_log1p: expm1(X.mean(axis=0))
        expected = np.expm1(np.array(X.mean(axis=0)).flatten())
        result = pseudobulk(X, geometric_mean=True, is_log1p=True)
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_single_row(self):
        X = np.array([[1.0, 2.0, 3.0]])
        result = pseudobulk(X, geometric_mean=False, is_log1p=False)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# TestWelchTTest
# ---------------------------------------------------------------------------


class TestWelchTTest:
    def test_matches_scipy(self):
        rng = np.random.default_rng(42)
        x = rng.normal(5, 2, size=(20, 10))
        y = rng.normal(3, 2, size=(15, 10))
        result = welch_ttest(x, y)
        for j in range(10):
            ref = ttest_ind(x[:, j], y[:, j], equal_var=False)
            np.testing.assert_allclose(result.statistic[j], ref.statistic, atol=1e-6)
            np.testing.assert_allclose(result.pvalue[j], ref.pvalue, atol=1e-6)

    def test_zero_variance(self):
        x = np.full((10, 1), 5.0)
        y = np.full((10, 1), 5.0)
        result = welch_ttest(x, y)
        assert result.statistic[0] == 0.0
        assert np.isfinite(result.pvalue[0])

    def test_single_feature(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, size=(10, 1))
        y = rng.normal(0, 1, size=(10, 1))
        result = welch_ttest(x, y)
        assert result.statistic.shape == (1,)
        assert result.pvalue.shape == (1,)

    def test_min_2_rows_required(self):
        x = np.array([[1.0, 2.0]])
        y = np.array([[3.0, 4.0], [5.0, 6.0]])
        with pytest.raises(AssertionError):
            welch_ttest(x, y)

    def test_shape_mismatch(self):
        x = np.random.randn(10, 5)
        y = np.random.randn(10, 3)
        with pytest.raises(AssertionError):
            welch_ttest(x, y)


# ---------------------------------------------------------------------------
# TestMannWhitneyUSparse
# ---------------------------------------------------------------------------


class TestMannWhitneyUSparse:
    def test_matches_scipy(self):
        rng = np.random.default_rng(42)
        X = sp.random(30, 10, density=0.5, format="csr", dtype=np.float32, random_state=rng)
        X.data[:] = rng.integers(1, 100, size=X.nnz).astype(np.float32)
        Y = sp.random(25, 10, density=0.5, format="csr", dtype=np.float32, random_state=rng)
        Y.data[:] = rng.integers(1, 100, size=Y.nnz).astype(np.float32)
        result = mannwhitneyu_sparse(X, Y)
        x_dense = X.toarray()
        y_dense = Y.toarray()
        for j in range(10):
            ref = scipy_mwu(x_dense[:, j], y_dense[:, j], alternative="two-sided")
            np.testing.assert_allclose(result.pvalue[j], ref.pvalue, atol=1e-4)

    def test_all_zeros(self):
        X = sp.csr_matrix((30, 5), dtype=np.float32)
        Y = sp.csr_matrix((25, 5), dtype=np.float32)
        result = mannwhitneyu_sparse(X, Y)
        np.testing.assert_allclose(result.statistic, 30 * 25 / 2.0)
        np.testing.assert_allclose(result.pvalue, 1.0)

    def test_clear_separation(self):
        rng = np.random.default_rng(42)
        X = sp.csr_matrix(np.zeros((30, 5), dtype=np.float32))
        Y_dense = rng.integers(100, 200, size=(25, 5)).astype(np.float32)
        Y = sp.csr_matrix(Y_dense)
        result = mannwhitneyu_sparse(X, Y)
        assert np.all(result.pvalue < 0.01)

    def test_column_mismatch(self):
        X = sp.csr_matrix((10, 5), dtype=np.float32)
        Y = sp.csr_matrix((10, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            mannwhitneyu_sparse(X, Y)

    def test_negative_values_raises(self):
        X = sp.csr_matrix(np.array([[-1.0, 2.0]], dtype=np.float32))
        Y = sp.csr_matrix(np.array([[1.0, 2.0]], dtype=np.float32))
        with pytest.raises(ValueError):
            mannwhitneyu_sparse(X, Y)


# ---------------------------------------------------------------------------
# TestMannWhitneyUDense
# ---------------------------------------------------------------------------


class TestMannWhitneyUDense:
    def test_matches_scipy(self):
        rng = np.random.default_rng(42)
        x = rng.normal(5, 2, size=(30, 10))
        y = rng.normal(3, 2, size=(25, 10))
        result = mannwhitneyu_dense(x, y)
        for j in range(10):
            ref = scipy_mwu(x[:, j], y[:, j], alternative="two-sided")
            np.testing.assert_allclose(result.pvalue[j], ref.pvalue, atol=1e-6)

    def test_all_equal(self):
        x = np.full((10, 3), 5.0)
        y = np.full((10, 3), 5.0)
        result = mannwhitneyu_dense(x, y)
        np.testing.assert_allclose(result.pvalue, 1.0)

    def test_single_feature(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, size=(10, 1))
        y = rng.normal(0, 1, size=(10, 1))
        result = mannwhitneyu_dense(x, y)
        assert result.statistic.shape == (1,)
        assert result.pvalue.shape == (1,)

    def test_empty_raises(self):
        x = np.empty((0, 3))
        y = np.ones((5, 3))
        with pytest.raises(ValueError):
            mannwhitneyu_dense(x, y)

    def test_not_2d_raises(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        with pytest.raises(ValueError):
            mannwhitneyu_dense(x, y)


# ---------------------------------------------------------------------------
# TestSparseColumnIndex
# ---------------------------------------------------------------------------


class TestSparseColumnIndex:
    def test_round_trip(self):
        rng = np.random.default_rng(42)
        X = sp.random(20, 5, density=0.4, format="csr", dtype=np.float32, random_state=rng)
        X.data[:] = rng.integers(1, 50, size=X.nnz).astype(np.float32)
        idx = sparse_column_index(X)
        for j in range(5):
            start, end = idx.col_indptr[j], idx.col_indptr[j + 1]
            gathered = np.array([idx.data[idx.col_order[k]] for k in range(start, end)])
            expected = X[:, j].toarray().flatten()
            expected_nz = expected[expected != 0]
            np.testing.assert_array_equal(np.sort(gathered), np.sort(expected_nz))

    def test_reuse_matches_raw(self):
        rng = np.random.default_rng(42)
        X = sp.random(20, 5, density=0.4, format="csr", dtype=np.float32, random_state=rng)
        X.data[:] = rng.integers(1, 50, size=X.nnz).astype(np.float32)
        Y = sp.random(15, 5, density=0.4, format="csr", dtype=np.float32, random_state=rng)
        Y.data[:] = rng.integers(1, 50, size=Y.nnz).astype(np.float32)
        # Compare raw CSR vs precomputed index
        result_raw = mannwhitneyu_sparse(X, Y)
        idx_x = sparse_column_index(X)
        idx_y = sparse_column_index(Y)
        result_idx = mannwhitneyu_sparse(idx_x, idx_y)
        np.testing.assert_allclose(result_raw.pvalue, result_idx.pvalue, atol=1e-10)
        np.testing.assert_allclose(result_raw.statistic, result_idx.statistic, atol=1e-10)


# ---------------------------------------------------------------------------
# TestMWURouter
# ---------------------------------------------------------------------------


class TestMWURouter:
    def test_routes_sparse(self):
        rng = np.random.default_rng(42)
        X = sp.random(20, 5, density=0.4, format="csr", dtype=np.float32, random_state=rng)
        X.data[:] = rng.integers(1, 50, size=X.nnz).astype(np.float32)
        Y = sp.random(15, 5, density=0.4, format="csr", dtype=np.float32, random_state=rng)
        Y.data[:] = rng.integers(1, 50, size=Y.nnz).astype(np.float32)
        result = mwu(X, Y)
        assert isinstance(result, MannWhitneyUResult)

    def test_routes_dense(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, size=(20, 5))
        y = rng.normal(0, 1, size=(15, 5))
        result = mwu(x, y)
        assert isinstance(result, MannWhitneyUResult)

    def test_routes_sparse_column_index(self):
        rng = np.random.default_rng(42)
        X = sp.random(20, 5, density=0.4, format="csr", dtype=np.float32, random_state=rng)
        X.data[:] = rng.integers(1, 50, size=X.nnz).astype(np.float32)
        Y = sp.random(15, 5, density=0.4, format="csr", dtype=np.float32, random_state=rng)
        Y.data[:] = rng.integers(1, 50, size=Y.nnz).astype(np.float32)
        idx_x = sparse_column_index(X)
        result = mwu(idx_x, Y)
        assert isinstance(result, MannWhitneyUResult)


# ---------------------------------------------------------------------------
# TestGroupWhere
# ---------------------------------------------------------------------------


class TestGroupWhere:
    def test_simple(self):
        assert _group_where("tissue", "brain") == "tissue = 'brain'"

    def test_apostrophe_escaping(self):
        assert _group_where("tissue", "O'Brien") == "tissue = 'O''Brien'"

    def test_spaces(self):
        assert _group_where("tissue", "spinal cord") == "tissue = 'spinal cord'"


# ---------------------------------------------------------------------------
# TestExtractMatrix
# ---------------------------------------------------------------------------


class TestExtractMatrix:
    def _make_adata(self, X):
        import anndata as ad

        return ad.AnnData(X=X)

    def test_sparse_returns_csr(self):
        X = sp.random(10, 5, density=0.3, format="csr", dtype=np.float32)
        adata = self._make_adata(X)
        result = _extract_matrix(adata, PointerKind.SPARSE)
        assert sp.issparse(result)
        assert result.format == "csr"

    def test_csc_converts_to_csr(self):
        X = sp.random(10, 5, density=0.3, format="csc", dtype=np.float32)
        adata = self._make_adata(X)
        result = _extract_matrix(adata, PointerKind.SPARSE)
        assert result.format == "csr"

    def test_dense_returns_ndarray(self):
        X = np.random.randn(10, 5).astype(np.float64)
        adata = self._make_adata(X)
        result = _extract_matrix(adata, PointerKind.DENSE)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    def test_sparse_to_dense_converts(self):
        X = sp.random(10, 5, density=0.3, format="csr", dtype=np.float32)
        adata = self._make_adata(X)
        result = _extract_matrix(adata, PointerKind.DENSE)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# TestCompare
# ---------------------------------------------------------------------------


class TestCompare:
    def test_sparse_mwu(self):
        rng = np.random.default_rng(42)
        target = sp.random(20, 5, density=0.5, format="csr", dtype=np.float32, random_state=rng)
        target.data[:] = rng.integers(1, 100, size=target.nnz).astype(np.float32)
        control = sp.random(15, 5, density=0.5, format="csr", dtype=np.float32, random_state=rng)
        control.data[:] = rng.integers(1, 100, size=control.nnz).astype(np.float32)
        features = np.array([f"gene_{i}" for i in range(5)])
        df = _compare(target, control, PointerKind.SPARSE, "mwu", 1e4, True, features)
        assert df.shape[0] == 5
        assert set(df.columns) == {
            "feature",
            "target_mean",
            "ref_mean",
            "target_n",
            "ref_n",
            "fold_change",
            "percent_change",
            "p_value",
            "statistic",
            "fdr",
        }
        assert all(0 <= p <= 1 for p in df["p_value"].to_list())
        assert all(0 <= f <= 1 for f in df["fdr"].to_list())

    def test_dense_ttest(self):
        rng = np.random.default_rng(42)
        target = rng.normal(5, 2, size=(20, 5))
        control = rng.normal(3, 2, size=(15, 5))
        features = np.array([f"feat_{i}" for i in range(5)])
        df = _compare(target, control, PointerKind.DENSE, "ttest", 1e4, True, features)
        assert df.shape[0] == 5
        assert all(0 <= p <= 1 for p in df["p_value"].to_list())

    def test_with_caches(self):
        rng = np.random.default_rng(42)
        target = sp.random(20, 5, density=0.5, format="csr", dtype=np.float32, random_state=rng)
        target.data[:] = rng.integers(1, 100, size=target.nnz).astype(np.float32)
        control = sp.random(15, 5, density=0.5, format="csr", dtype=np.float32, random_state=rng)
        control.data[:] = rng.integers(1, 100, size=control.nnz).astype(np.float32)
        features = np.array([f"gene_{i}" for i in range(5)])

        # Normalize control for cache computation (same as _compare does internally)
        normalize_log1p_sparse(control, 1e4)
        control_mean = pseudobulk(control, geometric_mean=True, is_log1p=True)
        control_idx = sparse_column_index(control)

        df_cached = _compare(
            target,
            control,
            PointerKind.SPARSE,
            "mwu",
            1e4,
            True,
            features,
            control_mean_cache=control_mean,
            control_idx_cache=control_idx,
        )
        assert df_cached.shape[0] == 5
        assert all(0 <= p <= 1 for p in df_cached["p_value"].to_list())

    def test_all_zeros(self):
        target = sp.csr_matrix((20, 5), dtype=np.float32)
        control = sp.csr_matrix((15, 5), dtype=np.float32)
        features = np.array([f"gene_{i}" for i in range(5)])
        df = _compare(target, control, PointerKind.SPARSE, "mwu", 1e4, True, features)
        # fold_change: log2((0+eps)/(0+eps)) = 0
        np.testing.assert_allclose(df["fold_change"].to_numpy(), 0.0, atol=1e-6)
        np.testing.assert_allclose(df["p_value"].to_numpy(), 1.0)
