"""Group-aware batch sampler (homeobox) + no-op mapping strategy (cell-load).

Kept in one file because they're conceptually paired: both make the two
systems run the same group-aware batching workload with control pairing
disabled, so the comparison reduces to backend speed on single-group reads.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import torch
from torch.utils.data import Sampler


def _ensure_cell_load_importable() -> None:
    """Add /home/ubuntu/cell-load/src to sys.path if cell_load isn't installed.

    cell-load isn't a homeobox dependency; this benchmark imports it from a
    sibling checkout. Falls back silently if cell_load is already on the path.
    """
    try:
        import cell_load  # noqa: F401

        return
    except ModuleNotFoundError:
        pass
    candidate = os.path.expanduser("~/cell-load/src")
    if os.path.isdir(os.path.join(candidate, "cell_load")):
        sys.path.insert(0, candidate)


_ensure_cell_load_importable()
from cell_load.mapping_strategies.mapping_strategies import BaseMappingStrategy  # noqa: E402

if TYPE_CHECKING:
    from cell_load.dataset import PerturbationDataset


# ---------------------------------------------------------------------------
# Homeobox: GroupBatchSampler
# ---------------------------------------------------------------------------


class GroupBatchSampler(Sampler[list[int]]):
    """Yield dataset-index batches where every batch is from one group.

    Mirrors cell-load's ``cell_sentence_len == batch_size, batch_size_param == 1``
    mode: one "sentence" per meta-batch, single (cell_type, gene) per batch.
    Use this with ``torch.utils.data.DataLoader(batch_sampler=...)`` or
    ``homeobox.make_loader(dataset, batch_sampler=...)``.

    Parameters
    ----------
    obs:
        Polars DataFrame whose row order matches the dataset's index order
        (row ``i`` of ``obs`` == dataset index ``i``). Must contain the columns
        listed in ``group_cols``.
    group_cols:
        Columns whose joint value defines a group. All cells in a yielded
        batch share these values.
    batch_size:
        Cells per batch.
    seed:
        Per-instance RNG seed. Each ``__iter__`` reseeds so reruns are
        reproducible.
    drop_last:
        Drop the trailing partial batch within each group when its size is
        less than ``batch_size``. Default True (matches cell-load's
        per-sentence handling when no upsampling-by-replacement is requested).
    cycle:
        If True (default), ``__iter__`` never exhausts: it reshuffles and
        keeps yielding indefinitely. Necessary for benchmarking, where the
        measurement window controls run length — without it, small/large
        ``batch_size`` configs would exit the loader before warmup+measure
        finishes (e.g. 500 batches × 2000 cells/group / 1024 bs ≈ 5s of work
        instead of 30s). cell-load's underlying iterator behaves the same
        way when wrapped by a PyTorch DataLoader for multi-epoch training.
    """

    def __init__(
        self,
        obs: pl.DataFrame,
        group_cols: list[str],
        batch_size: int,
        seed: int,
        drop_last: bool = True,
        cycle: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        missing = [c for c in group_cols if c not in obs.columns]
        if missing:
            raise ValueError(f"group_cols missing from obs: {missing}; have {obs.columns}")

        self._batch_size = int(batch_size)
        self._seed = int(seed)
        self._drop_last = bool(drop_last)
        self._group_cols = list(group_cols)
        self._cycle = bool(cycle)

        idx = obs.with_row_index("_idx").group_by(self._group_cols).agg(pl.col("_idx"))
        self._groups: list[np.ndarray] = [row.to_numpy().astype(np.int64) for row in idx["_idx"]]

        if self._drop_last:
            self._n_batches = sum(len(g) // self._batch_size for g in self._groups)
        else:
            self._n_batches = sum(
                (len(g) + self._batch_size - 1) // self._batch_size for g in self._groups
            )

    def __len__(self) -> int:
        return self._n_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self._seed)
        epoch = 0
        while True:
            group_order = rng.permutation(len(self._groups))
            batches: list[list[int]] = []
            for gi in group_order:
                idxs = self._groups[gi].copy()
                rng.shuffle(idxs)
                n = len(idxs)
                stop = (n // self._batch_size) * self._batch_size if self._drop_last else n
                for start in range(0, stop, self._batch_size):
                    batches.append(idxs[start : start + self._batch_size].tolist())

            rng.shuffle(batches)
            yield from batches
            epoch += 1
            if not self._cycle:
                return


# ---------------------------------------------------------------------------
# cell-load: NoOpMappingStrategy
# ---------------------------------------------------------------------------


class NoOpMappingStrategy(BaseMappingStrategy):
    """Bench-only mapping that uses the perturbed cell as its own control.

    Short-circuits control lookup so the comparison reduces to backend speed
    on single-group reads. Also opts into cell-load's batched-fetch path
    (``use_consecutive_loading=True``) so ``__getitems__`` routes into the
    single h5 slice read at ``_perturbation.py:432`` instead of the per-cell
    ``__getitem__`` fallback at line 363 — without this, cell-load pays N
    Python-dispatch + h5-call costs per batch and the bench measures Python
    overhead, not the backend.

    Side note: opting into the batched path means cell-load re-issues a
    second batched read for ``ctrl_expr`` using the same indices (small
    redundant cost, but still single-call, not per-cell).

    Not suitable for any real training run — the control "comes from" the
    perturbed cell. Bench harness only.
    """

    # Read by cell-load's `_use_batched_fetch` gate at
    # cell_load/dataset/_perturbation.py:539.
    use_consecutive_loading: bool = True

    def __init__(
        self,
        name: str = "noop",
        random_state: int = 42,
        n_basal_samples: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            random_state=random_state,
            n_basal_samples=n_basal_samples,
            **kwargs,
        )

    def register_split_indices(
        self,
        dataset: PerturbationDataset,
        split: str,
        perturbed_indices: np.ndarray,
        control_indices: np.ndarray,
    ) -> None:
        return None

    def get_control_indices(
        self, dataset: PerturbationDataset, split: str, perturbed_idx: int
    ) -> np.ndarray:
        return np.array([perturbed_idx], dtype=np.int64)

    def get_control_index(
        self, dataset: PerturbationDataset, split: str, perturbed_idx: int
    ) -> int:
        return int(perturbed_idx)

    def get_mapped_expressions(
        self, dataset: PerturbationDataset, split: str, perturbed_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if dataset.embed_key:
            pert_expr = dataset.fetch_obsm_expression(perturbed_idx, dataset.embed_key)
        else:
            pert_expr = dataset.fetch_gene_expression(perturbed_idx)
        return pert_expr, pert_expr, int(perturbed_idx)
