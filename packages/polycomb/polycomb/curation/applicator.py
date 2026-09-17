"""Apply audited column operations to Lance tables."""

from __future__ import annotations

import os
import re
from typing import Any

import lancedb
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from polycomb.curation.audit import CurationAuditStore, default_audit_db_path
from polycomb.curation.sql import (
    arrow_alias_to_sql_cast,
    arrow_type_from_alias,
    build_add_column_expr,
    infer_arrow_type,
)
from polycomb.curation.types import (
    ROW_POSITION_COLUMN,
    AddColumn,
    AppliedChange,
    ApplyResult,
    CastColumn,
    CurationOp,
    CurationTransaction,
    DropColumn,
    ExplodeColumn,
    MergeColumns,
    OpKind,
    RenameColumn,
    ReplaceValue,
    SetColumn,
    TransactionStatus,
    WideToLong,
)

# Ops that change the table's row count via a whole-table rewrite. After one
# runs, the cached table handle and field types are stale and must be refreshed.
_RESHAPE_KINDS = frozenset({OpKind.EXPLODE_COLUMN, OpKind.WIDE_TO_LONG})

# Columns curation may never touch. ``row_position`` is the positional anchor
# staging writes onto tables aligned to a DATA file axis; ingestion's row
# alignment reads it, so an op that renamed, retyped, or refilled it would break
# that alignment as surely as a reorder. Finalization drops it once the
# alignment artifact has been built -- that one path passes allow_reserved.
_RESERVED_COLUMNS = frozenset({ROW_POSITION_COLUMN})


class CurationApplicator:
    """Apply curation transactions to Lance tables with SQLite audit logging."""

    def __init__(
        self,
        lance_db_path: str | os.PathLike[str],
        audit_db_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.lance_db_path = os.fspath(lance_db_path)
        self.audit_db_path = (
            os.fspath(audit_db_path) if audit_db_path else default_audit_db_path(lance_db_path)
        )
        self._audit = CurationAuditStore(self.audit_db_path)
        self._db = lancedb.connect(self.lance_db_path)

    def close(self) -> None:
        self._audit.close()

    def get_revert_version(self, transaction_id: str) -> int | None:
        return self._audit.get_revert_version(transaction_id)

    def apply(
        self,
        transaction: CurationTransaction,
        *,
        dry_run: bool = False,
        allowed_columns: set[str] | None = None,
        allow_reserved: bool = False,
    ) -> ApplyResult:
        """Apply a transaction's ops in order, validating all of them up front.

        ``allow_reserved`` lifts the :data:`_RESERVED_COLUMNS` guard for the one
        caller entitled to it -- finalization's leftover sweep, which drops
        ``row_position`` after the ingestion alignment artifact has been built.
        Nothing else should pass it.
        """
        table_name = transaction.table_name

        table = self._db.open_table(table_name)
        self._validate(transaction, table, allowed_columns, allow_reserved=allow_reserved)

        lance_version_before = table.version
        transaction.status = TransactionStatus.PENDING

        if dry_run:
            # A dry run validates only — it must not touch Lance or the audit DB.
            # Validation has already run above, so any error surfaced; here we
            # just report the ops that *would* apply. No audit rows are written,
            # so there is no change_id to link to (sentinel -1).
            applied = [
                AppliedChange(
                    operation=op,
                    change_id=-1,
                    rows_updated=None,
                    lance_version=None,
                )
                for op in transaction.changes
            ]
            return ApplyResult(
                transaction_id=transaction.transaction_id,
                status=TransactionStatus.PENDING,
                lance_version_before=lance_version_before,
                applied_changes=applied,
                dry_run=True,
            )

        change_ids = self._audit.insert_pending_transaction(
            transaction,
            lance_version_before=lance_version_before,
        )

        applied_changes: list[AppliedChange] = []
        error: str | None = None
        field_types = self._field_types(table)

        try:
            for change_id, change in zip(change_ids, transaction.changes, strict=True):
                # An op that is allowed to operate on the anchor itself (only
                # finalization's leftover sweep, dropping it) cannot be checked
                # against it.
                touches_anchor = bool(set(self._named_columns(change)) & _RESERVED_COLUMNS)
                anchor_before = (
                    self._row_anchor(table)
                    if change.preserves_row_order and not touches_anchor
                    else None
                )
                rows_updated, version = self._execute(change, table, table_name, field_types)
                if change.kind in _RESHAPE_KINDS:
                    # A reshape rewrites the table; the old handle is stale.
                    table = self._db.open_table(table_name)
                if change.kind is not OpKind.SET_COLUMN:
                    # Schema-altering ops change columns/types, and the value ops
                    # that rewrite through Arrow to keep row order (ReplaceValue,
                    # MergeColumns) replace the table; either way the handle is stale.
                    table = self._db.open_table(table_name)
                    field_types = self._field_types(table)
                if anchor_before is not None:
                    self._assert_row_order(table_name, change, anchor_before, table)
                self._audit.record_applied_change(
                    change_id,
                    rows_updated=rows_updated,
                    lance_version=version,
                )
                applied_changes.append(
                    AppliedChange(
                        operation=change,
                        change_id=change_id,
                        rows_updated=rows_updated,
                        lance_version=version,
                    )
                )
            status = TransactionStatus.APPLIED
        except Exception as exc:
            error = str(exc)
            status = TransactionStatus.PARTIAL if applied_changes else TransactionStatus.FAILED

        self._audit.finalize_transaction(
            transaction.transaction_id,
            status=status,
        )

        return ApplyResult(
            transaction_id=transaction.transaction_id,
            status=status,
            lance_version_before=lance_version_before,
            applied_changes=applied_changes,
            dry_run=False,
            error=error,
        )

    @staticmethod
    def _field_types(table: Any) -> dict[str, pa.DataType]:
        schema = table.schema
        return {name: schema.field(name).type for name in schema.names}

    @staticmethod
    def _row_anchor(table: Any) -> list[Any] | None:
        """The table's ``row_position`` column, or None when it carries none.

        Tables not bound to a DATA file axis (the joined multimodal obs table,
        library tables, registry-key targets) have no anchor and are not checked.
        """
        if ROW_POSITION_COLUMN not in table.schema.names:
            return None
        return table.to_arrow().column(ROW_POSITION_COLUMN).to_pylist()

    @classmethod
    def _assert_row_order(
        cls, table_name: str, change: CurationOp, before: list[Any], table: Any
    ) -> None:
        """Fail the transaction if an order-preserving op moved any row.

        The op *declares* ``preserves_row_order``, but the guarantee actually
        comes from Lance's write primitives, which differ per op and can change
        between releases. Checking is one int64 column read; the alternative is
        a permutation that no downstream step can detect.
        """
        after = cls._row_anchor(table)
        if after == before:
            return
        if after is None:
            raise RuntimeError(
                f"{table_name}: operation {change.kind.value} on column "
                f"{change.column!r} removed the {ROW_POSITION_COLUMN!r} anchor."
            )
        if len(after) != len(before):
            detail = f"row count changed from {len(before)} to {len(after)}"
        else:
            first = next(
                (i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b),
                None,
            )
            detail = f"row order changed (first difference at index {first})"
        raise RuntimeError(
            f"{table_name}: operation {change.kind.value} on column {change.column!r} "
            f"must preserve row order but {detail}. The table is positionally "
            f"aligned to its DATA file; the transaction has been failed rather "
            f"than committing a permutation."
        )

    @staticmethod
    def _named_columns(change: CurationOp) -> list[str]:
        """Every column name an op references, whatever role it plays in the op."""
        names = [change.column]
        if isinstance(change, RenameColumn):
            names.append(change.new_name)
        elif isinstance(change, MergeColumns):
            names.append(change.key_column)
            names.extend(c for row in change.rows for c in row)
        elif isinstance(change, ExplodeColumn):
            if change.position_column is not None:
                names.append(change.position_column)
        elif isinstance(change, WideToLong):
            names.extend(change.groups)
            names.extend(c for srcs in change.groups.values() for c in srcs)
            if change.slot_label_column is not None:
                names.append(change.slot_label_column)
        return names

    def _validate(
        self,
        transaction: CurationTransaction,
        table: Any,
        allowed_columns: set[str] | None,
        *,
        allow_reserved: bool = False,
    ) -> None:
        """Check every op up front against the (simulated) evolving schema.

        Walking changes in order lets intra-transaction dependencies validate
        correctly (e.g. add a column then set it). Nothing is recorded or
        mutated if validation fails. Drops are exempt from ``allowed_columns``
        since finalization must be free to remove any non-schema column -- but
        not from the reserved-column guard, which only ``allow_reserved`` lifts.
        """
        columns = set(self._field_types(table))
        positional = ROW_POSITION_COLUMN in columns

        for change in transaction.changes:
            kind = change.kind

            if not allow_reserved:
                reserved = sorted(set(self._named_columns(change)) & _RESERVED_COLUMNS)
                if reserved:
                    raise ValueError(
                        f"Operation {kind.value} references reserved column(s) {reserved} "
                        f"on table '{transaction.table_name}'. {ROW_POSITION_COLUMN!r} is "
                        f"the positional anchor ingestion aligns DATA rows on and is "
                        f"maintained by staging, not by curation."
                    )

            if positional and not change.preserves_row_order:
                raise ValueError(
                    f"Operation {kind.value} multiplies rows, which breaks the "
                    f"correspondence between table '{transaction.table_name}' and its "
                    f"DATA file (it carries {ROW_POSITION_COLUMN!r}). Reshape the raw "
                    f"table before staging, or run the reshape on a table that is not "
                    f"positionally aligned."
                )

            if kind is OpKind.EXPLODE_COLUMN:
                if change.column not in columns:
                    raise ValueError(
                        f"Column '{change.column}' not found in table "
                        f"'{transaction.table_name}'. Available: {sorted(columns)}"
                    )
                created = []
                if change.position_column is not None:
                    if change.position_column in columns:
                        raise ValueError(
                            f"position_column '{change.position_column}' already exists."
                        )
                    created.append(change.position_column)
                self._check_allowed([change.column, *created], allowed_columns)
                columns.update(created)
                continue

            if kind is OpKind.WIDE_TO_LONG:
                self._validate_wide_to_long(change, columns, transaction.table_name)
                consumed = {c for srcs in change.groups.values() for c in srcs}
                created = list(change.groups)
                if change.slot_label_column is not None:
                    created.append(change.slot_label_column)
                self._check_allowed(created, allowed_columns)
                columns = (columns - consumed) | set(created)
                continue

            if kind is OpKind.MERGE_COLUMNS:
                # Update-only keyed merge: schema is unchanged, so no simulation.
                self._validate_merge_columns(change, columns, transaction.table_name)
                targets = [c for c in self._merge_columns_targets(change) if c != change.key_column]
                self._check_allowed(targets, allowed_columns)
                continue

            if kind is OpKind.ADD_COLUMN:
                if change.column in columns:
                    raise ValueError(
                        f"Column '{change.column}' already exists in table "
                        f"'{transaction.table_name}'; use SetColumn to overwrite it."
                    )
            else:
                if change.column not in columns:
                    raise ValueError(
                        f"Column '{change.column}' not found in table "
                        f"'{transaction.table_name}'. Available: {sorted(columns)}"
                    )

            if kind is OpKind.RENAME_COLUMN and change.new_name in columns:
                raise ValueError(
                    f"Cannot rename '{change.column}' to '{change.new_name}': "
                    f"a column with that name already exists."
                )

            gated = change.new_name if kind is OpKind.RENAME_COLUMN else change.column
            if (
                allowed_columns is not None
                and kind is not OpKind.DROP_COLUMN
                and gated not in allowed_columns
            ):
                raise ValueError(
                    f"Column '{gated}' is not in allowed_columns: {sorted(allowed_columns)}"
                )

            # Simulate the schema change for subsequent ops.
            if kind is OpKind.ADD_COLUMN:
                columns.add(change.column)
            elif kind is OpKind.DROP_COLUMN:
                columns.discard(change.column)
            elif kind is OpKind.RENAME_COLUMN:
                columns.discard(change.column)
                columns.add(change.new_name)

    @staticmethod
    def _validate_wide_to_long(change: WideToLong, columns: set[str], table_name: str) -> None:
        """Validate a WideToLong reshape against the current column set."""
        if not change.groups:
            raise ValueError("WideToLong requires at least one group.")
        n_slots = len(change.slot_labels)
        if n_slots < 1:
            raise ValueError("WideToLong requires at least one slot label.")

        missing = [c for srcs in change.groups.values() for c in srcs if c not in columns]
        if missing:
            raise ValueError(
                f"WideToLong source columns not found in table '{table_name}': "
                f"{missing}. Available: {sorted(columns)}"
            )
        for out_col, srcs in change.groups.items():
            if len(srcs) != n_slots:
                raise ValueError(
                    f"WideToLong group '{out_col}' has {len(srcs)} source columns "
                    f"but there are {n_slots} slot labels; they must match."
                )

        consumed = {c for srcs in change.groups.values() for c in srcs}
        survivors = columns - consumed
        created = list(change.groups)
        if change.slot_label_column is not None:
            created.append(change.slot_label_column)
        collisions = [c for c in created if c in survivors]
        if collisions:
            raise ValueError(
                f"WideToLong output columns collide with columns that survive the "
                f"reshape: {collisions}."
            )
        duplicates = [c for c in set(created) if created.count(c) > 1]
        if duplicates:
            raise ValueError(f"WideToLong output column names are not unique: {duplicates}.")

    @staticmethod
    def _merge_columns_targets(change: MergeColumns) -> list[str]:
        """All column names referenced across a MergeColumns source batch."""
        seen: dict[str, None] = {}
        for row in change.rows:
            for col in row:
                seen.setdefault(col, None)
        return list(seen)

    @classmethod
    def _validate_merge_columns(
        cls, change: MergeColumns, columns: set[str], table_name: str
    ) -> None:
        """Validate an update-only keyed merge against the current column set."""
        if not change.rows:
            raise ValueError("MergeColumns requires at least one source row.")
        if change.key_column not in columns:
            raise ValueError(
                f"MergeColumns key_column '{change.key_column}' not found in table "
                f"'{table_name}'. Available: {sorted(columns)}"
            )

        referenced = cls._merge_columns_targets(change)
        missing = [c for c in referenced if c not in columns]
        if missing:
            raise ValueError(
                f"MergeColumns target columns not found in table '{table_name}': "
                f"{missing}. Add them with AddColumn first. Available: {sorted(columns)}"
            )
        if change.key_column not in referenced:
            raise ValueError(
                f"MergeColumns key_column '{change.key_column}' must appear in every "
                f"source row so rows can be matched to the table."
            )

        key_values = [row.get(change.key_column) for row in change.rows]
        if len(set(key_values)) != len(key_values):
            raise ValueError("MergeColumns source rows must have unique key_column values.")

    @staticmethod
    def _check_allowed(cols: list[str], allowed_columns: set[str] | None) -> None:
        if allowed_columns is None:
            return
        for col in cols:
            if col not in allowed_columns:
                raise ValueError(
                    f"Column '{col}' is not in allowed_columns: {sorted(allowed_columns)}"
                )

    def _execute(
        self,
        change: CurationOp,
        table: Any,
        table_name: str,
        field_types: dict[str, pa.DataType],
    ) -> tuple[int | None, int | None]:
        """Run one op against the Lance table; return (rows_updated, version)."""
        if isinstance(change, ReplaceValue):
            return self._replace_value_rewrite(
                table_name, table, change, field_types[change.column]
            )

        if isinstance(change, SetColumn):
            if change.value_sql is not None:
                result = table.update(values_sql={change.column: change.value_sql})
            else:
                value = self._coerce_update_value(change.new_value, field_types[change.column])
                result = table.update(values={change.column: value})
            return result.rows_updated, result.version

        if isinstance(change, AddColumn):
            if change.value_sql is not None:
                if change.data_type is not None:
                    field_type = arrow_type_from_alias(change.data_type)
                    if pa.types.is_nested(field_type):
                        field = pa.field(change.column, field_type)
                        return self._append_expression_rewrite(
                            table_name, table, field, change.value_sql
                        )
                result = table.add_columns({change.column: change.value_sql})
            elif change.value is not None:
                field_type = (
                    arrow_type_from_alias(change.data_type)
                    if change.data_type
                    else infer_arrow_type(change.value)
                )
                if pa.types.is_nested(field_type):
                    field = pa.field(change.column, field_type)
                    return self._append_column_rewrite(table_name, table, field, change.value)
                expr = build_add_column_expr(change.value, change.data_type)
                result = table.add_columns({change.column: expr})
            elif change.data_type is not None:
                field = pa.field(change.column, arrow_type_from_alias(change.data_type))
                if pa.types.is_nested(field.type):
                    return self._append_column_rewrite(table_name, table, field, None)
                result = table.add_columns(field)
            else:
                raise ValueError(
                    f"AddColumn for '{change.column}' needs value, value_sql, or data_type."
                )
            return None, self._version_after(result, table)

        if isinstance(change, RenameColumn):
            result = table.alter_columns({"path": change.column, "rename": change.new_name})
            return None, self._version_after(result, table)

        if isinstance(change, CastColumn):
            # Lance's alter_columns only re-types within a family, so coerce via
            # a SQL cast into a temp column, then drop the original and rename.
            # The recast column moves to the end of the schema.
            sql_type = arrow_alias_to_sql_cast(change.data_type)
            tmp = f"__cast_{change.column}"
            table.add_columns({tmp: f"cast({change.column} as {sql_type})"})
            table.drop_columns([change.column])
            result = table.alter_columns({"path": tmp, "rename": change.column})
            return None, self._version_after(result, table)

        if isinstance(change, DropColumn):
            result = table.drop_columns([change.column])
            return None, self._version_after(result, table)

        if isinstance(change, MergeColumns):
            return self._merge_columns_rewrite(table_name, table, change)

        # lancedb's Table.to_pandas() routes on-disk tables through
        # LanceDataset.to_pandas(), which lance no longer provides; go through
        # Arrow, which both storage paths support.
        if isinstance(change, ExplodeColumn):
            new_df = self._explode_frame(change, table.to_arrow().to_pandas())
            return self._rewrite(table_name, new_df)

        if isinstance(change, WideToLong):
            new_df = self._wide_to_long_frame(change, table.to_arrow().to_pandas())
            return self._rewrite(table_name, new_df)

        raise ValueError(f"Unsupported operation: {type(change).__name__}")

    def _merge_columns_rewrite(
        self, table_name: str, table: Any, change: MergeColumns
    ) -> tuple[int, int | None]:
        """Fill many columns from a keyed batch, keeping row order.

        Lance's ``merge_insert`` would rewrite matched rows at the end of the
        table and group duplicate keys together, so a fan-out resolution pass
        over a feature registry would scramble the feature axis ingestion aligns
        on. Locating each table row in the source batch and filling in place
        keeps every row where it was, and fills *all* rows sharing a key rather
        than collapsing them.

        ``rows`` carries Python scalars; building each incoming array with the
        target field's type makes coordinates land as ints, strands as strings,
        and keeps all-null columns typed rather than Arrow's ``null`` placeholder.
        """
        arrow = table.to_arrow()
        keys = pa.array(
            [row[change.key_column] for row in change.rows],
            type=arrow.schema.field(change.key_column).type,
        )
        # Position of each table row within the source batch; null when the row's
        # key is absent, which leaves that row untouched (update-only semantics).
        index = pc.index_in(arrow.column(change.key_column), value_set=keys)
        matched_mask = pc.is_valid(index)
        matched = pc.sum(pc.cast(matched_mask, pa.int64())).as_py() or 0
        if not matched:
            return 0, getattr(table, "version", None)

        # take() needs a null-free index; the mask decides which results survive.
        gather = pc.fill_null(index, 0)
        targets = [c for c in self._merge_columns_targets(change) if c != change.key_column]
        for target in targets:
            position = arrow.schema.get_field_index(target)
            field = arrow.schema.field(position)
            incoming = pa.array([row.get(target) for row in change.rows], type=field.type)
            arrow = arrow.set_column(
                position,
                field,
                pc.if_else(matched_mask, incoming.take(gather), arrow.column(target)),
            )
        return matched, self._overwrite_table(table_name, arrow)

    @staticmethod
    def _explode_frame(change: ExplodeColumn, df: pd.DataFrame) -> pd.DataFrame:
        """Split a delimited column into one row per fragment."""
        splitter = re.compile(change.delimiter)

        def to_fragments(value: Any) -> list[Any]:
            if not isinstance(value, str):
                return [value]
            parts = splitter.split(value)
            if change.drop_empty:
                parts = [p for p in parts if p.strip() != ""]
            return parts

        fragments = df[change.column].map(to_fragments)
        df = df.copy()
        df[change.column] = fragments
        explode_cols = [change.column]
        if change.position_column is not None:
            df[change.position_column] = fragments.map(lambda parts: list(range(len(parts))))
            explode_cols.append(change.position_column)
        return df.explode(explode_cols, ignore_index=True)

    @staticmethod
    def _wide_to_long_frame(change: WideToLong, df: pd.DataFrame) -> pd.DataFrame:
        """Melt parallel column families into one row per slot."""
        consumed = {c for srcs in change.groups.values() for c in srcs}
        id_cols = [c for c in df.columns if c not in consumed]

        blocks: list[pd.DataFrame] = []
        for slot_index, label in enumerate(change.slot_labels):
            block = df[id_cols].copy()
            for out_col, srcs in change.groups.items():
                block[out_col] = df[srcs[slot_index]].to_numpy()
            if change.slot_label_column is not None:
                block[change.slot_label_column] = label
            blocks.append(block)

        long_df = pd.concat(blocks, ignore_index=True)
        if change.drop_null_slots:
            out_cols = list(change.groups)
            all_null = long_df[out_cols].isna().all(axis=1)
            long_df = long_df[~all_null].reset_index(drop=True)
        return long_df

    def _rewrite(self, table_name: str, new_df: pd.DataFrame) -> tuple[int, int | None]:
        """Overwrite a table with reshaped data; a new Lance version preserves undo.

        ``rows_updated`` is reported as the resulting row count (the "after" of a
        reshape); the pre-reshape count is recoverable from the prior version.
        """
        new_table = self._db.create_table(table_name, data=new_df, mode="overwrite")
        return len(new_df), getattr(new_table, "version", None)

    def _append_column_rewrite(
        self, table_name: str, table: Any, field: pa.Field, value: Any
    ) -> tuple[None, int | None]:
        """Append nested columns via Arrow, bypassing Lance's nested add_columns path."""
        arrow = table.to_arrow()
        values = pa.array([value] * arrow.num_rows, type=field.type)
        return self._append_column_values(table_name, arrow, field, values)

    def _append_expression_rewrite(
        self, table_name: str, table: Any, field: pa.Field, value_sql: str
    ) -> tuple[None, int | None]:
        """Append a nested column computed per row by a SQL expression.

        Lance evaluates the expression during a scan but cannot materialize a nested
        result through ``add_columns`` -- it writes an all-null column and reports
        success -- so scan the expression and append the values as Arrow instead. Both
        scans run against one pinned dataset version, so the rows stay aligned and in
        order, which the obs tables depend on.
        """
        dataset = table.to_lance()
        arrow = dataset.to_table()
        values = dataset.to_table(columns={field.name: value_sql}).column(field.name)
        return self._append_column_values(table_name, arrow, field, values.cast(field.type))

    def _append_column_values(
        self, table_name: str, arrow: pa.Table, field: pa.Field, values: Any
    ) -> tuple[None, int | None]:
        return None, self._overwrite_table(table_name, arrow.append_column(field, values))

    def _replace_value_rewrite(
        self, table_name: str, table: Any, change: ReplaceValue, field_type: pa.DataType
    ) -> tuple[int, int | None]:
        """Find-and-replace one value, keeping row order.

        Lance's predicated ``update`` rewrites matched rows at the end of the table, so
        running a resolution pass over an obs or var table would silently permute it and
        break the positional alignment ingestion relies on. Rewriting through Arrow keeps
        every row where it was.

        TODO: this rewrites the whole table once per op, which is fine for feature
        registries and library tables but expensive for a multi-million-row obs table
        with many replacements. Batch the ReplaceValue ops of a transaction into one
        rewrite, keeping the per-op matched counts.
        """
        arrow = table.to_arrow()
        index = arrow.schema.get_field_index(change.column)
        column = arrow.column(change.column)

        if change.old_value is None:
            mask = pc.is_null(column)
        else:
            mask = pc.fill_null(pc.equal(column, pa.scalar(change.old_value, field_type)), False)
        matched = pc.sum(pc.cast(mask, pa.int64())).as_py() or 0
        if not matched:
            return 0, getattr(table, "version", None)

        replaced = pc.if_else(mask, pa.scalar(change.new_value, field_type), column)
        arrow = arrow.set_column(index, arrow.schema.field(index), replaced)
        return matched, self._overwrite_table(table_name, arrow)

    def _overwrite_table(self, table_name: str, arrow: pa.Table) -> int | None:
        db = lancedb.connect(self.lance_db_path)
        new_table = db.create_table(table_name, data=arrow, mode="overwrite")
        self._db = db
        return getattr(new_table, "version", None)

    @staticmethod
    def _version_after(result: Any, table: Any) -> int | None:
        # add_columns/drop_columns expose .version; alter_columns does not, so
        # fall back to the table's current version.
        version = getattr(result, "version", None)
        return version if version is not None else table.version

    @staticmethod
    def _coerce_update_value(value: Any, field_type: Any) -> Any:
        if value is None:
            return None
        if pa.types.is_boolean(field_type):
            return bool(value)
        if pa.types.is_integer(field_type):
            return int(value)
        if pa.types.is_floating(field_type):
            return float(value)
        return str(value)
