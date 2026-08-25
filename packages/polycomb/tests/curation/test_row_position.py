"""The row_position invariant: curation must never permute a positional table.

Obs and feature-registry tables are positionally aligned to their DATA file, so
physical row order is a join key. These tests pin that down at the applicator
boundary, where every curation op passes through.

The fixture order matters more than it looks. Rows are laid out so that the
staged order is neither sorted by any content column nor alphabetical by key --
a table that is accidentally sorted, or "restored" by a lexicographic sort, is
distinguishable from one that genuinely kept its order. A fixture with
ascending keys would pass under every bug these tests exist to catch.
"""

from __future__ import annotations

import lancedb
import pyarrow as pa
import pytest

from polycomb.curation import (
    AddColumn,
    CurationApplicator,
    CurationTransaction,
    DropColumn,
    ExplodeColumn,
    MergeColumns,
    RenameColumn,
    ReplaceValue,
    SetColumn,
    TransactionStatus,
)
from polycomb.curation.types import ROW_POSITION_COLUMN, CastColumn, WideToLong

# Deliberately unsorted keys: alphabetical order would be zebra, alpha, mike...
# so any sort-based "restore" reorders this and any order-preserving op does not.
KEYS = ["zebra", "mike", "alpha", "mike", "quebec", "alpha"]
VALUES = ["old", "keep", "old", "other", "keep", "old"]


def _staged_table(tmp_path, *, with_anchor: bool = True) -> tuple[str, str]:
    """Write a staged obs-like table; returns (lance_db_path, table_name)."""
    columns = {
        "obs_key": pa.array(KEYS, type=pa.string()),
        "value": pa.array(VALUES, type=pa.string()),
        "count": pa.array([5, None, 3, 9, None, 1], type=pa.int64()),
    }
    if with_anchor:
        columns[ROW_POSITION_COLUMN] = pa.array(range(len(KEYS)), type=pa.int64())
    path = str(tmp_path / "lance_db")
    lancedb.connect(path).create_table("CellIndex", data=pa.table(columns), mode="overwrite")
    return path, "CellIndex"


def _apply(path: str, table_name: str, *changes, **kwargs):
    applicator = CurationApplicator(path, audit_db_path=str(path) + "/audit.sqlite")
    try:
        txn = CurationTransaction(table_name=table_name, changes=list(changes))
        return applicator.apply(txn, **kwargs)
    finally:
        applicator.close()


def _read(path: str, table_name: str) -> pa.Table:
    return lancedb.connect(path).open_table(table_name).to_arrow()


# --- order preservation, one case per op kind ------------------------------


def _replace_value() -> ReplaceValue:
    return ReplaceValue(column="value", old_value="old", new_value="new", tool="t")


def _set_column() -> SetColumn:
    return SetColumn(column="value", new_value="flat", tool="t")


def _add_column() -> AddColumn:
    return AddColumn(column="added", value="x", tool="t")


def _rename_column() -> RenameColumn:
    return RenameColumn(column="value", new_name="value_renamed", tool="t")


def _drop_column() -> DropColumn:
    return DropColumn(column="count", tool="t")


def _cast_column() -> CastColumn:
    return CastColumn(column="count", data_type="string", tool="t")


def _merge_columns() -> MergeColumns:
    # Source order is deliberately unrelated to table order, and coverage is
    # partial: 'quebec' is absent, so its row must be left untouched.
    return MergeColumns(
        column="value",
        key_column="obs_key",
        rows=[
            {"obs_key": "alpha", "value": "A"},
            {"obs_key": "zebra", "value": "Z"},
        ],
        tool="t",
    )


ORDER_PRESERVING = [
    ("replace_value", _replace_value),
    ("set_column", _set_column),
    ("add_column", _add_column),
    ("rename_column", _rename_column),
    ("drop_column", _drop_column),
    ("cast_column", _cast_column),
    ("merge_columns", _merge_columns),
]


@pytest.mark.parametrize("name,build", ORDER_PRESERVING, ids=[n for n, _ in ORDER_PRESERVING])
def test_op_preserves_row_order(tmp_path, name, build):
    """Every op declaring preserves_row_order must leave the anchor untouched."""
    path, table_name = _staged_table(tmp_path)
    result = _apply(path, table_name, build())

    assert result.status is TransactionStatus.APPLIED, result.error
    after = _read(path, table_name)
    assert after.column(ROW_POSITION_COLUMN).to_pylist() == list(range(len(KEYS)))
    # The anchor could be right while the payload was permuted underneath it, so
    # check a content column that the op did not target.
    if name not in ("rename_column", "set_column", "replace_value", "merge_columns"):
        assert after.column("value").to_pylist() == VALUES
    assert after.column("obs_key").to_pylist() == KEYS


# --- MergeColumns, the op that used to reorder ------------------------------


def test_merge_columns_keeps_order_and_fills_every_duplicate(tmp_path):
    """Lance's merge_insert grouped duplicate keys and moved matched rows to the tail."""
    path, table_name = _staged_table(tmp_path)
    result = _apply(path, table_name, _merge_columns())

    assert result.status is TransactionStatus.APPLIED, result.error
    after = _read(path, table_name)

    assert after.column("obs_key").to_pylist() == KEYS
    assert after.column(ROW_POSITION_COLUMN).to_pylist() == list(range(len(KEYS)))
    # Both 'alpha' rows filled (indices 2 and 5), not collapsed into one;
    # 'quebec' (index 4) unmatched and so untouched.
    assert after.column("value").to_pylist() == ["Z", "keep", "A", "other", "keep", "A"]
    # rows_updated counts matched table rows, not source rows.
    assert result.applied_changes[0].rows_updated == 3


def test_merge_columns_reports_no_match(tmp_path):
    path, table_name = _staged_table(tmp_path)
    change = MergeColumns(
        column="value",
        key_column="obs_key",
        rows=[{"obs_key": "nobody", "value": "X"}],
        tool="t",
    )
    result = _apply(path, table_name, change)

    assert result.status is TransactionStatus.APPLIED, result.error
    assert result.applied_changes[0].rows_updated == 0
    assert _read(path, table_name).column("value").to_pylist() == VALUES


# --- the reservation --------------------------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        SetColumn(column=ROW_POSITION_COLUMN, new_value=0, tool="t"),
        DropColumn(column=ROW_POSITION_COLUMN, tool="t"),
        RenameColumn(column=ROW_POSITION_COLUMN, new_name="whatever", tool="t"),
        RenameColumn(column="value", new_name=ROW_POSITION_COLUMN, tool="t"),
        ReplaceValue(column=ROW_POSITION_COLUMN, old_value=0, new_value=1, tool="t"),
        MergeColumns(
            column=ROW_POSITION_COLUMN,
            key_column="obs_key",
            rows=[{"obs_key": "alpha", ROW_POSITION_COLUMN: 99}],
            tool="t",
        ),
    ],
    ids=["set", "drop", "rename-from", "rename-to", "replace", "merge-target"],
)
def test_reserved_column_is_rejected(tmp_path, change):
    path, table_name = _staged_table(tmp_path)
    with pytest.raises(ValueError, match="reserved column"):
        _apply(path, table_name, change)
    # Validation runs before anything is recorded or mutated.
    assert ROW_POSITION_COLUMN in _read(path, table_name).column_names


def test_finalization_may_drop_the_anchor(tmp_path):
    """The leftover sweep is the one caller allowed past the reservation."""
    path, table_name = _staged_table(tmp_path)
    result = _apply(
        path, table_name, DropColumn(column=ROW_POSITION_COLUMN, tool="t"), allow_reserved=True
    )

    assert result.status is TransactionStatus.APPLIED, result.error
    assert ROW_POSITION_COLUMN not in _read(path, table_name).column_names


# --- reshape ops on a positional table --------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        ExplodeColumn(column="value", delimiter=r"\|", tool="t"),
        WideToLong(
            column="value",
            groups={"merged": ["value", "obs_key"]},
            slot_labels=["a", "b"],
            tool="t",
        ),
    ],
    ids=["explode", "wide_to_long"],
)
def test_reshape_rejected_on_positional_table(tmp_path, change):
    path, table_name = _staged_table(tmp_path)
    with pytest.raises(ValueError, match="multiplies rows"):
        _apply(path, table_name, change)


def test_reshape_allowed_without_anchor(tmp_path):
    """A table with no row_position is not positionally bound, so reshape is fine."""
    path, table_name = _staged_table(tmp_path, with_anchor=False)
    result = _apply(path, table_name, ExplodeColumn(column="value", delimiter=r"\|", tool="t"))
    assert result.status is TransactionStatus.APPLIED, result.error


# --- the post-condition -----------------------------------------------------


def test_permuting_op_fails_the_transaction(tmp_path, monkeypatch):
    """If a write primitive reorders despite the op's declaration, refuse to commit.

    Simulates a Lance release changing behaviour underneath us, which is exactly
    the case the declaration alone cannot protect against.
    """
    path, table_name = _staged_table(tmp_path)

    def permuting_execute(self, change, table, name, field_types):
        arrow = table.to_arrow()
        reversed_rows = arrow.take(list(reversed(range(arrow.num_rows))))
        return 1, self._overwrite_table(name, reversed_rows)

    monkeypatch.setattr(CurationApplicator, "_execute", permuting_execute)
    result = _apply(path, table_name, _replace_value())

    assert result.status is TransactionStatus.FAILED
    assert "must preserve row order" in (result.error or "")
    # The permutation is still on disk -- rolling it back is the caller's job via
    # lance_version_before -- but the transaction is not recorded as applied, and
    # the version to roll back to is reported.
    assert result.applied_changes == []
    assert result.lance_version_before is not None


def test_unanchored_table_is_not_checked(tmp_path):
    """Tables with no anchor (joined obs, library tables) skip the check entirely."""
    path, table_name = _staged_table(tmp_path, with_anchor=False)
    result = _apply(path, table_name, _replace_value())
    assert result.status is TransactionStatus.APPLIED, result.error
    assert _read(path, table_name).column("value").to_pylist() == [
        "new" if v == "old" else v for v in VALUES
    ]
