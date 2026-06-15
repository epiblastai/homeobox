"""Datatypes for auditable Lance table curation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, ClassVar
from uuid import uuid4


class TransactionStatus(StrEnum):
    """Lifecycle of a curation transaction in the audit store."""

    PENDING = "pending"
    APPLIED = "applied"
    PARTIAL = "partial"  # some operations applied before failure
    FAILED = "failed"


class OpKind(StrEnum):
    """Discriminator for the column operations the applicator supports."""

    REPLACE_VALUE = "replace_value"  # find-and-replace specific cell values
    SET_COLUMN = "set_column"  # overwrite every row of a column
    ADD_COLUMN = "add_column"  # introduce a new column
    RENAME_COLUMN = "rename_column"  # rename a column (e.g. raw name -> schema field)
    DROP_COLUMN = "drop_column"  # remove a column (e.g. non-schema raw columns)
    CAST_COLUMN = "cast_column"  # change a column's data type
    MERGE_COLUMNS = "merge_columns"  # fill many columns from a keyed resolution batch

    # The ops below are mechanical reshapes, not resolution/curation decisions:
    # they restructure the table (multiplying rows) but record no per-cell
    # provenance worth auditing the way a resolution does -- their audit value is
    # mostly reproducibility, which Lance versioning already provides. They are
    # included only because splitting combinatorial perturbations into one record
    # per row is common and well-defined. WARNING: this is exactly where the
    # CurationOp framework risks becoming a worse reimplementation of a dataframe
    # library (pandas/polars). If we find ourselves reaching for a another reshape
    # op, stop and reconsider whether CurationOps is the right abstraction for
    # structural transforms at all, rather than enumerating the long tail here.
    EXPLODE_COLUMN = "explode_column"  # split a delimited cell into multiple rows
    WIDE_TO_LONG = "wide_to_long"  # melt parallel column families into multiple rows


@dataclass(kw_only=True)
class CurationOp:
    """Base for one auditable column operation.

    Carries the provenance shared by every operation. Subclasses add the
    payload specific to their :class:`OpKind`. ``column`` is the column the
    operation is *about* (the operated column, or the new column for an add).
    """

    # Class-level discriminator; set by each subclass.
    kind: ClassVar[OpKind]

    # Target column and justification metadata (shared by all ops).
    column: str
    tool: str
    reason: str = ""
    confidence: float | None = None
    source: str | None = None
    alternatives: list[str] = field(default_factory=list)
    # When using a resolution tool, this is the value that was given
    # to the tool to resolve. It might be different from old value if
    # some transformation like stripping a prefix or suffix occurred prior
    # to resolution.
    input_value: str | None = None


@dataclass(kw_only=True)
class ReplaceValue(CurationOp):
    """Find-and-replace specific cell values in a column (matched on old_value)."""

    kind: ClassVar[OpKind] = OpKind.REPLACE_VALUE

    old_value: Any
    new_value: Any


@dataclass(kw_only=True)
class SetColumn(CurationOp):
    """Overwrite every row of an existing column.

    Provide either ``new_value`` (a constant applied to all rows) or
    ``value_sql`` (a SQL expression evaluated per row, may reference other
    columns). Useful when a resolver replaces a whole raw column wholesale
    (e.g. resolved ``organism`` overwrites the raw ``organism`` column).
    """

    kind: ClassVar[OpKind] = OpKind.SET_COLUMN

    new_value: Any = None
    value_sql: str | None = None


@dataclass(kw_only=True)
class AddColumn(CurationOp):
    """Add a new column to the table.

    Exactly one of three modes:
    - ``value``: a constant applied to all rows.
    - ``value_sql``: a SQL expression evaluated per row (may reference columns).
    - neither, with ``data_type`` set: a null-initialized column of that type.
    """

    kind: ClassVar[OpKind] = OpKind.ADD_COLUMN

    value: Any = None
    value_sql: str | None = None
    # Serialized Arrow type alias (e.g. "int64", "string"). Optional for a
    # constant/expression add; required when null-initializing.
    data_type: str | None = None


@dataclass(kw_only=True)
class RenameColumn(CurationOp):
    """Rename a column. ``column`` is the source name; ``new_name`` the target."""

    kind: ClassVar[OpKind] = OpKind.RENAME_COLUMN

    new_name: str


@dataclass(kw_only=True)
class DropColumn(CurationOp):
    """Remove a column from the table."""

    kind: ClassVar[OpKind] = OpKind.DROP_COLUMN


@dataclass(kw_only=True)
class CastColumn(CurationOp):
    """Coerce a column to a new data type (e.g. on finalization to parquet)."""

    kind: ClassVar[OpKind] = OpKind.CAST_COLUMN

    # Serialized Arrow type alias (e.g. "int64", "double", "string", "bool").
    data_type: str


@dataclass(kw_only=True)
class MergeColumns(CurationOp):
    """Fill many columns at once from a keyed resolution batch (update-only).

    One expensive resolver call (e.g. ``resolve_guide_sequences``) yields many
    correlated fields per input value, destined for several different schema
    columns. This op applies them in a single keyed join: ``rows`` is the source
    table (one dict per *distinct resolved* key), and rows in the target table
    whose ``key_column`` matches have the remaining columns updated. Unmatched
    target rows are left untouched; no rows are inserted or deleted.

    ``rows`` keys are column names; every row carries ``key_column`` plus the
    target columns to fill. The target columns must already exist (add them with
    ``AddColumn`` first if needed). Provenance here is batch-level -- one op for
    the whole fan-out -- rather than per-value like :class:`ReplaceValue`; the
    per-key mapping is preserved in ``rows``. ``column`` is required by the base
    op and used only as an audit anchor; set it to a representative target column.

    Note: the underlying merge reorders rows (matched rows are rewritten at the
    end). It is row-count preserving and reversible via the Lance version, but
    do not rely on row order across it.
    """

    kind: ClassVar[OpKind] = OpKind.MERGE_COLUMNS

    # Join key column, present in both the target table and every source row.
    key_column: str
    # Source records: one dict per distinct key, each with key_column + targets.
    rows: list[dict[str, Any]]


# --- Row-multiplying (shape-changing) ops -----------------------------------
#
# Unlike every op above, these change the table's row count: one input row
# becomes many. They reshape wide -> long ("normalize to one record per row"),
# differing only in where the multiplicity is encoded -- inside a single
# delimited cell (ExplodeColumn) or across parallel column families
# (WideToLong). They cannot be expressed as a per-row SQL update, so the
# applicator runs them as a whole-table rewrite. Run a reshape as its own
# transaction, before any value resolution: row-matched ops like ReplaceValue
# match on values in the pre-reshape shape and must not share a transaction
# with a reshape.


@dataclass(kw_only=True)
class ExplodeColumn(CurationOp):
    """Split a delimited cell into multiple rows, repeating the other columns.

    Each value in ``column`` is split on ``delimiter`` (a regular expression);
    the row is replicated once per fragment, with ``column`` holding the
    individual fragment. Non-string cells (e.g. nulls) pass through as a single
    fragment. Use for combinatorial perturbations encoded in one cell
    (``"guideA|guideB"`` -> two rows).
    """

    kind: ClassVar[OpKind] = OpKind.EXPLODE_COLUMN

    # Regex delimiter the cell is split on (e.g. r"\s*[+&;|,]\s*").
    delimiter: str
    # Optional new column recording each fragment's 0-based position in its
    # original cell (provenance for the split).
    position_column: str | None = None
    # Drop empty fragments produced by the split (e.g. trailing delimiters).
    drop_empty: bool = True


@dataclass(kw_only=True)
class WideToLong(CurationOp):
    """Melt parallel column families into multiple rows (wide -> long).

    Each entry in ``groups`` maps an output column name to the list of source
    columns that feed it, one per slot, aligned by index with ``slot_labels``.
    Every other (unconsumed) column is treated as an id column and repeated for
    each slot. One input row becomes ``len(slot_labels)`` rows.

    Example (dual-guide pairs -> one guide per row)::

        WideToLong(
            column="targeting_sequence",  # representative output; audit anchor
            groups={
                "sgID": ["sgID_A", "sgID_B"],
                "targeting_sequence": ["targeting sequence A", "targeting sequence B"],
            },
            slot_labels=["A", "B"],
            slot_label_column="guide_slot",
            tool="schema_align",
            reason="dual-guide pairs -> one guide per row",
        )

    ``column`` is required by the base op and used purely as an audit anchor;
    set it to the most representative output column.
    """

    kind: ClassVar[OpKind] = OpKind.WIDE_TO_LONG

    # Output column -> source columns (one per slot), aligned with slot_labels.
    groups: dict[str, list[str]]
    # Label for each slot; length must equal each group's source-list length.
    slot_labels: list[str]
    # Optional new column recording which slot each long row came from.
    slot_label_column: str | None = None
    # Drop a produced row whose every output column is null (an absent slot).
    drop_null_slots: bool = True


@dataclass
class CurationTransaction:
    """Batch of column operations applied in a single apply() call."""

    # Target Lance table and planned operations
    table_name: str
    changes: list[CurationOp]

    # Assigned when the transaction is created; used by the audit store
    transaction_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    # Updated during apply(); optional caller context (organism, dry_run, etc.)
    status: TransactionStatus = TransactionStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AppliedChange:
    """Result of applying a single :class:`CurationOp`."""

    # Intent and link to the curation_changes audit row. ``-1`` on a dry run,
    # which writes no audit rows and so has no row to link to.
    operation: CurationOp
    change_id: int

    # Rows affected by row-level ops (replace/set); None for schema-only ops
    # (add/rename/drop/cast).
    rows_updated: int | None
    # Lance table version after this step. This can help during
    # debugging to see what the state of a table was immediately before
    # applying a change instead of what it was at the start of a whole
    # transaction.
    lance_version: int | None


@dataclass
class ApplyResult:
    """Result of applying a CurationTransaction."""

    transaction_id: str  # registry key to CurationTransaction
    status: TransactionStatus

    # Checkout this Lance version to undo the entire transaction
    lance_version_before: int | None

    # One entry per successful operation (shorter than changes if apply failed)
    applied_changes: list[AppliedChange] = field(default_factory=list)

    # True when the transaction was validated and reported only: neither Lance
    # nor the audit DB was written. ``applied_changes`` carries sentinel
    # change_ids of -1.
    dry_run: bool = False
    # Set when apply stops on the first exception (see status PARTIAL/FAILED)
    error: str | None = None
