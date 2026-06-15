"""SQLite audit store for curation transactions."""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

from auto_atlas.curation.types import (
    AddColumn,
    AppliedChange,
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

_SCHEMA = """
CREATE TABLE IF NOT EXISTS curation_transactions (
    transaction_id       TEXT PRIMARY KEY,
    table_name           TEXT NOT NULL,
    created_at           TEXT NOT NULL,
    lance_version_before INTEGER,
    status               TEXT NOT NULL,
    metadata_json        TEXT
);

CREATE TABLE IF NOT EXISTS curation_changes (
    change_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id      TEXT NOT NULL REFERENCES curation_transactions(transaction_id),
    table_name          TEXT NOT NULL,
    op_kind             TEXT NOT NULL,

    -- column the op is about (operated column; new column name for add)
    column_name         TEXT NOT NULL,

    -- value-bearing ops (replace_value, set_column, add_column constant)
    old_value           TEXT,
    new_value           TEXT,

    -- structural op payloads
    target_column       TEXT,
    value_sql           TEXT,
    data_type           TEXT,

    -- JSON payload for ops whose parameters don't fit the columns above
    -- (e.g. the groups/slot_labels of a wide_to_long reshape)
    payload_json        TEXT,

    -- provenance (all ops)
    tool                TEXT NOT NULL,
    reason              TEXT,
    confidence          REAL,
    source              TEXT,
    alternatives_json   TEXT,
    input_value         TEXT,

    -- outcome
    rows_updated        INTEGER,
    lance_version       INTEGER,
    apply_order         INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_changes_txn ON curation_changes(transaction_id);
CREATE INDEX IF NOT EXISTS idx_changes_column ON curation_changes(table_name, column_name);
"""

_JSON_VALUE_KEY = "__auto_atlas_json_value__"


def _serialize_value(value: Any) -> Any:
    """Store SQLite-unsupported structured values without changing scalar audit rows."""
    if isinstance(value, list | dict):
        return json.dumps({_JSON_VALUE_KEY: value})
    return value


def _deserialize_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return value
    if isinstance(decoded, dict) and set(decoded) == {_JSON_VALUE_KEY}:
        return decoded[_JSON_VALUE_KEY]
    return value


def _deserialize_change_row(row: Any) -> dict[str, Any]:
    data = dict(row)
    data["old_value"] = _deserialize_value(data["old_value"])
    data["new_value"] = _deserialize_value(data["new_value"])
    return data


def _stored_new_value(change: CurationOp) -> Any:
    """The constant recorded in the new_value audit column, if the op has one."""
    if isinstance(change, ReplaceValue | SetColumn):
        return change.new_value
    if isinstance(change, AddColumn):
        return change.value
    return None


def _op_payload(change: CurationOp) -> dict[str, Any] | None:
    """Structured parameters for ops that don't fit the fixed audit columns."""
    if isinstance(change, ExplodeColumn):
        return {
            "delimiter": change.delimiter,
            "position_column": change.position_column,
            "drop_empty": change.drop_empty,
        }
    if isinstance(change, WideToLong):
        return {
            "groups": change.groups,
            "slot_labels": change.slot_labels,
            "slot_label_column": change.slot_label_column,
            "drop_null_slots": change.drop_null_slots,
        }
    if isinstance(change, MergeColumns):
        return {"key_column": change.key_column, "rows": change.rows}
    return None


def _row_to_op(row: Any) -> CurationOp:
    """Reconstruct the concrete CurationOp subclass from an audit row."""
    alternatives = json.loads(row["alternatives_json"]) if row["alternatives_json"] else []
    shared = dict(
        column=row["column_name"],
        tool=row["tool"],
        reason=row["reason"] or "",
        confidence=row["confidence"],
        source=row["source"],
        alternatives=alternatives,
        input_value=row["input_value"],
    )
    kind = OpKind(row["op_kind"])
    if kind is OpKind.REPLACE_VALUE:
        return ReplaceValue(
            old_value=_deserialize_value(row["old_value"]),
            new_value=_deserialize_value(row["new_value"]),
            **shared,
        )
    if kind is OpKind.SET_COLUMN:
        return SetColumn(
            new_value=_deserialize_value(row["new_value"]),
            value_sql=row["value_sql"],
            **shared,
        )
    if kind is OpKind.ADD_COLUMN:
        return AddColumn(
            value=_deserialize_value(row["new_value"]),
            value_sql=row["value_sql"],
            data_type=row["data_type"],
            **shared,
        )
    if kind is OpKind.RENAME_COLUMN:
        return RenameColumn(new_name=row["target_column"], **shared)
    if kind is OpKind.DROP_COLUMN:
        return DropColumn(**shared)
    if kind is OpKind.CAST_COLUMN:
        return CastColumn(data_type=row["data_type"], **shared)
    if kind is OpKind.EXPLODE_COLUMN:
        payload = json.loads(row["payload_json"])
        return ExplodeColumn(
            delimiter=payload["delimiter"],
            position_column=payload["position_column"],
            drop_empty=payload["drop_empty"],
            **shared,
        )
    if kind is OpKind.WIDE_TO_LONG:
        payload = json.loads(row["payload_json"])
        return WideToLong(
            groups=payload["groups"],
            slot_labels=payload["slot_labels"],
            slot_label_column=payload["slot_label_column"],
            drop_null_slots=payload["drop_null_slots"],
            **shared,
        )
    if kind is OpKind.MERGE_COLUMNS:
        payload = json.loads(row["payload_json"])
        return MergeColumns(key_column=payload["key_column"], rows=payload["rows"], **shared)
    raise ValueError(f"Unknown op_kind: {row['op_kind']}")


def _is_remote_path(path: str) -> bool:
    return path.startswith(("s3://", "gs://", "az://"))


def default_audit_db_path(lance_db_path: str | os.PathLike[str]) -> str:
    """Return the default audit DB path co-located with a Lance DB directory."""
    lance_dir = os.fspath(lance_db_path).rstrip(os.sep) or os.fspath(lance_db_path)
    if not _is_remote_path(lance_dir):
        lance_dir = os.path.abspath(lance_dir)
    return os.path.join(os.path.dirname(lance_dir), "curation_audit.db")


class CurationAuditStore:
    """Persistent SQLite store for curation audit records."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = os.fspath(path)
        if not _is_remote_path(self.path):
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def insert_pending_transaction(
        self,
        transaction: CurationTransaction,
        *,
        lance_version_before: int | None,
    ) -> list[int]:
        """Insert a pending transaction and its changes. Returns change row ids."""
        self._conn.execute(
            """
            INSERT INTO curation_transactions (
                transaction_id, table_name, created_at,
                lance_version_before, status, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                transaction.transaction_id,
                transaction.table_name,
                transaction.created_at,
                lance_version_before,
                transaction.status.value,
                json.dumps(transaction.metadata) if transaction.metadata else None,
            ),
        )
        change_ids: list[int] = []
        for order, change in enumerate(transaction.changes):
            cur = self._conn.execute(
                """
                INSERT INTO curation_changes (
                    transaction_id, table_name, op_kind, column_name,
                    old_value, new_value, target_column, value_sql, data_type,
                    payload_json,
                    tool, reason, confidence, source, alternatives_json, input_value,
                    rows_updated, lance_version, apply_order
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    transaction.transaction_id,
                    transaction.table_name,
                    change.kind.value,
                    change.column,
                    _serialize_value(getattr(change, "old_value", None)),
                    _serialize_value(_stored_new_value(change)),
                    getattr(change, "new_name", None),
                    getattr(change, "value_sql", None),
                    getattr(change, "data_type", None),
                    json.dumps(_op_payload(change)) if _op_payload(change) else None,
                    change.tool,
                    change.reason,
                    change.confidence,
                    change.source,
                    json.dumps(change.alternatives) if change.alternatives else None,
                    change.input_value,
                    None,
                    None,
                    order,
                ),
            )
            change_ids.append(cur.lastrowid)
        self._conn.commit()
        return change_ids

    def record_applied_change(
        self,
        change_id: int,
        *,
        rows_updated: int,
        lance_version: int | None,
    ) -> None:
        self._conn.execute(
            """
            UPDATE curation_changes
            SET rows_updated = ?, lance_version = ?
            WHERE change_id = ?
            """,
            (rows_updated, lance_version, change_id),
        )
        self._conn.commit()

    def finalize_transaction(
        self,
        transaction_id: str,
        *,
        status: TransactionStatus,
    ) -> None:
        self._conn.execute(
            """
            UPDATE curation_transactions
            SET status = ?
            WHERE transaction_id = ?
            """,
            (status.value, transaction_id),
        )
        self._conn.commit()

    def get_revert_version(self, transaction_id: str) -> int | None:
        row = self._conn.execute(
            "SELECT lance_version_before FROM curation_transactions WHERE transaction_id = ?",
            (transaction_id,),
        ).fetchone()
        if row is None:
            return None
        return row["lance_version_before"]

    def get_transaction(self, transaction_id: str) -> dict[str, Any] | None:
        txn = self._conn.execute(
            "SELECT * FROM curation_transactions WHERE transaction_id = ?",
            (transaction_id,),
        ).fetchone()
        if txn is None:
            return None
        changes = self._conn.execute(
            """
            SELECT * FROM curation_changes
            WHERE transaction_id = ?
            ORDER BY apply_order
            """,
            (transaction_id,),
        ).fetchall()
        return {
            "transaction": dict(txn),
            "changes": [_deserialize_change_row(c) for c in changes],
        }

    def list_transactions(
        self,
        *,
        table_name: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if table_name is not None:
            clauses.append("table_name = ?")
            params.append(table_name)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self._conn.execute(
            f"""
            SELECT * FROM curation_transactions
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def load_pending_changes(self, transaction_id: str) -> list[tuple[int, CurationOp]]:
        rows = self._conn.execute(
            """
            SELECT * FROM curation_changes
            WHERE transaction_id = ?
            ORDER BY apply_order
            """,
            (transaction_id,),
        ).fetchall()
        return [(row["change_id"], _row_to_op(row)) for row in rows]

    def build_applied_change(self, change_id: int, operation: CurationOp) -> AppliedChange:
        row = self._conn.execute(
            "SELECT rows_updated, lance_version FROM curation_changes WHERE change_id = ?",
            (change_id,),
        ).fetchone()
        return AppliedChange(
            operation=operation,
            change_id=change_id,
            rows_updated=row["rows_updated"],
            lance_version=row["lance_version"],
        )
