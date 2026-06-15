"""Tests for the curation audit system."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import lancedb
import pyarrow as pa
import pytest

from auto_atlas.curation import (
    AddColumn,
    CastColumn,
    CurationApplicator,
    CurationAuditStore,
    CurationTransaction,
    DropColumn,
    MergeColumns,
    RenameColumn,
    ReplaceValue,
    SetColumn,
    TransactionStatus,
    default_audit_db_path,
)
from auto_atlas.curation.sql import arrow_type_from_alias, build_where_clause
from auto_atlas.registry import RESOLVER_TOOLS, ResolverTool
from auto_atlas.types import GeneResolution, ResolutionReport


@pytest.fixture
def atlas_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        lance_dir = Path(tmp) / "lance_db"
        lance_dir.mkdir()
        audit_path = default_audit_db_path(str(lance_dir))
        yield str(lance_dir), audit_path


def _make_gene_table(db_uri: str) -> lancedb.table.Table:
    db = lancedb.connect(db_uri)
    data = pa.table(
        {
            "uid": ["aaa", "bbb", "ccc"],
            "gene_symbol": ["BRCA2", "BRCA2", "TP53"],
            "ensembl_id": ["ENS1", "ENS2", "ENS3"],
        }
    )
    return db.create_table("gene_expression", data=data, mode="overwrite")


def test_default_audit_db_path_sibling_to_lance_db():
    path = default_audit_db_path("/data/atlas/lance_db")
    assert path == "/data/atlas/curation_audit.db"


def test_default_audit_db_path_preserves_s3_uri():
    path = default_audit_db_path("s3://bucket/data/atlas/lance_db")
    assert path == "s3://bucket/data/atlas/curation_audit.db"


def test_propose_dedupes_shared_old_value():
    report = ResolutionReport(
        tool="resolve_genes",
        total=2,
        resolved=2,
        unresolved=0,
        ambiguous=0,
        results=[
            GeneResolution(
                input_value="brca2",
                resolved_value="BRCA2",
                confidence=1.0,
                source="lancedb",
                symbol="BRCA2",
                ensembl_gene_id="ENS0001",
            ),
            GeneResolution(
                input_value="TP53",
                resolved_value="TP53",
                confidence=1.0,
                source="lancedb",
                symbol="TP53",
                ensembl_gene_id="ENS0003",
            ),
        ],
    )
    replacements = report.propose_column_replacements(
        ["brca2", "TP53"],
        column="gene_symbol",
        reason="test",
        resolution_field_name="symbol",
    )
    assert len(replacements) == 1
    assert replacements[0].old_value == "brca2"
    assert replacements[0].new_value == "BRCA2"
    assert replacements[0].confidence == 1.0
    assert replacements[0].tool == "resolve_genes"


def _load_apply_resolution_pass_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "apply_resolution_pass",
        "skills/schema-harmonization/scripts/apply_resolution_pass.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_resolve_distinct_values_returns_tool_report():
    mod = _load_apply_resolution_pass_module()

    def fake_resolve(values, organism="human"):
        return ResolutionReport(
            tool="fake",
            total=len(values),
            resolved=len(values),
            unresolved=0,
            ambiguous=0,
            results=[
                GeneResolution(
                    input_value=v,
                    resolved_value=v.upper(),
                    confidence=1.0,
                    source="test",
                    symbol=v.upper(),
                )
                for v in values
            ],
        )

    with patch.dict(RESOLVER_TOOLS, {"fake": ResolverTool(fake_resolve)}):
        report = mod.resolve_distinct_values(["brca2", None, "brca2"], "fake")
    assert len(report.results) == 1
    assert report.results[0].input_value == "brca2"
    assert report.results[0].symbol == "BRCA2"


def test_apply_resolution_pass_updates_column(atlas_dirs):
    mod = _load_apply_resolution_pass_module()
    db_uri, _audit_path = atlas_dirs
    db = lancedb.connect(db_uri)
    db.create_table(
        "gene_expression",
        data=pa.table(
            {
                "uid": ["aaa", "bbb", "ccc"],
                "gene_symbol": ["brca2", "brca2", "TP53"],
                "ensembl_id": ["ENS1", "ENS2", "ENS3"],
            }
        ),
        mode="overwrite",
    )

    def fake_resolve(values, organism="human"):
        return ResolutionReport(
            tool="resolve_genes",
            total=len(values),
            resolved=len(values),
            unresolved=0,
            ambiguous=0,
            results=[
                GeneResolution(
                    input_value=v,
                    resolved_value=v.upper(),
                    confidence=1.0,
                    source="test",
                    symbol=v.upper(),
                )
                for v in values
            ],
        )

    with patch.dict(RESOLVER_TOOLS, {"resolve_genes": ResolverTool(fake_resolve)}):
        result = mod.apply_resolution_pass(
            db_uri,
            table_name="gene_expression",
            tool="resolve_genes",
            column="gene_symbol",
            resolution_field_name="symbol",
            reason="test",
        )
    assert result is not None
    assert result.status == TransactionStatus.APPLIED
    updated = lancedb.connect(db_uri).open_table("gene_expression").to_arrow().to_pydict()
    assert updated["gene_symbol"].count("BRCA2") == 2
    assert updated["gene_symbol"].count("TP53") == 1


def test_apply_round_trip(atlas_dirs):
    db_uri, audit_path = atlas_dirs
    _make_gene_table(db_uri)

    txn = CurationTransaction(
        table_name="gene_expression",
        changes=[
            ReplaceValue(
                column="gene_symbol",
                old_value="BRCA2",
                new_value="BRCA2_CANON",
                tool="resolve_genes",
                reason="standardize symbols",
                confidence=1.0,
                source="lancedb",
            )
        ],
    )

    applicator = CurationApplicator(db_uri, audit_db_path=audit_path)
    try:
        result = applicator.apply(txn, allowed_columns={"gene_symbol"})
    finally:
        applicator.close()

    assert result.status == TransactionStatus.APPLIED
    assert result.applied_changes[0].rows_updated == 2
    assert result.applied_changes[0].lance_version is not None
    assert result.applied_changes[0].lance_version > result.lance_version_before

    db = lancedb.connect(db_uri)
    updated = db.open_table("gene_expression").to_arrow().to_pydict()
    assert updated["gene_symbol"].count("BRCA2_CANON") == 2
    assert updated["gene_symbol"].count("TP53") == 1

    store = CurationAuditStore(audit_path)
    try:
        record = store.get_transaction(result.transaction_id)
        assert record is not None
        assert record["transaction"]["status"] == "applied"
        assert record["changes"][0]["rows_updated"] == 2
        assert record["changes"][0]["lance_version"] == result.applied_changes[0].lance_version
    finally:
        store.close()


def test_apply_null_old_value(atlas_dirs):
    db_uri, audit_path = atlas_dirs
    db = lancedb.connect(db_uri)
    data = pa.table({"gene_symbol": ["A", None, "B"]})
    db.create_table("features", data=data, mode="overwrite")

    txn = CurationTransaction(
        table_name="features",
        changes=[
            ReplaceValue(
                column="gene_symbol",
                old_value=None,
                new_value="UNKNOWN",
                tool="resolve_genes",
                reason="fill nulls",
            )
        ],
    )

    applicator = CurationApplicator(db_uri, audit_db_path=audit_path)
    try:
        result = applicator.apply(txn)
    finally:
        applicator.close()

    assert result.applied_changes[0].rows_updated == 1
    table = db.open_table("features")
    symbols = table.to_arrow()["gene_symbol"].to_pylist()
    assert symbols.count("UNKNOWN") == 1
    assert symbols.count("A") == 1
    assert symbols.count("B") == 1


def test_build_where_clause_null():
    field_type = pa.string()
    assert build_where_clause("gene_symbol", None, field_type) == "gene_symbol IS NULL"


def test_dry_run_does_not_mutate_lance(atlas_dirs):
    db_uri, audit_path = atlas_dirs
    _make_gene_table(db_uri)
    db = lancedb.connect(db_uri)
    version_before = db.open_table("gene_expression").version

    txn = CurationTransaction(
        table_name="gene_expression",
        changes=[
            ReplaceValue(
                column="gene_symbol",
                old_value="TP53",
                new_value="TP53_CANON",
                tool="resolve_genes",
            )
        ],
    )

    applicator = CurationApplicator(db_uri, audit_db_path=audit_path)
    try:
        result = applicator.apply(txn, dry_run=True)
    finally:
        applicator.close()

    assert result.dry_run is True
    assert db.open_table("gene_expression").version == version_before
    symbols = db.open_table("gene_expression").to_arrow()["gene_symbol"].to_pylist()
    assert symbols.count("BRCA2") == 2
    assert symbols.count("TP53") == 1

    # A dry run must not pollute the audit DB: only operations that actually
    # mutate Lance are recorded, so nothing is persisted here.
    store = CurationAuditStore(audit_path)
    try:
        assert store.get_transaction(result.transaction_id) is None
        assert store.list_transactions(table_name="gene_expression") == []
    finally:
        store.close()


def test_parallel_columns_on_same_table(atlas_dirs):
    db_uri, audit_path = atlas_dirs
    db = lancedb.connect(db_uri)
    data = pa.table(
        {
            "gene_symbol": ["OLD_G", "OLD_G"],
            "uniprot_id": ["OLD_P", "OLD_P"],
        }
    )
    db.create_table("features", data=data, mode="overwrite")

    gene_txn = CurationTransaction(
        table_name="features",
        changes=[
            ReplaceValue(
                column="gene_symbol",
                old_value="OLD_G",
                new_value="NEW_G",
                tool="resolve_genes",
            )
        ],
    )
    protein_txn = CurationTransaction(
        table_name="features",
        changes=[
            ReplaceValue(
                column="uniprot_id",
                old_value="OLD_P",
                new_value="NEW_P",
                tool="resolve_proteins",
            )
        ],
    )

    applicator = CurationApplicator(db_uri, audit_db_path=audit_path)
    try:
        gene_result = applicator.apply(gene_txn, allowed_columns={"gene_symbol"})
        protein_result = applicator.apply(protein_txn, allowed_columns={"uniprot_id"})
    finally:
        applicator.close()

    assert gene_result.status == TransactionStatus.APPLIED
    assert protein_result.status == TransactionStatus.APPLIED

    table = db.open_table("features")
    rows = table.to_arrow().to_pydict()
    assert rows["gene_symbol"] == ["NEW_G", "NEW_G"]
    assert rows["uniprot_id"] == ["NEW_P", "NEW_P"]

    store = CurationAuditStore(audit_path)
    try:
        txns = store.list_transactions(table_name="features")
        assert len(txns) == 2
    finally:
        store.close()


def test_allowed_columns_rejects_foreign_column(atlas_dirs):
    db_uri, audit_path = atlas_dirs
    _make_gene_table(db_uri)

    txn = CurationTransaction(
        table_name="gene_expression",
        changes=[
            ReplaceValue(
                column="ensembl_id",
                old_value="ENS1",
                new_value="ENSX",
                tool="resolve_genes",
            )
        ],
    )

    applicator = CurationApplicator(db_uri, audit_db_path=audit_path)
    try:
        with pytest.raises(ValueError, match="not in allowed_columns"):
            applicator.apply(txn, allowed_columns={"gene_symbol"})
    finally:
        applicator.close()


def test_get_revert_version(atlas_dirs):
    db_uri, audit_path = atlas_dirs
    table = _make_gene_table(db_uri)
    version_before = table.version

    txn = CurationTransaction(
        table_name="gene_expression",
        changes=[
            ReplaceValue(
                column="gene_symbol",
                old_value="TP53",
                new_value="TP53_CANON",
                tool="resolve_genes",
            )
        ],
    )

    applicator = CurationApplicator(db_uri, audit_db_path=audit_path)
    try:
        result = applicator.apply(txn)
        revert_version = applicator.get_revert_version(result.transaction_id)
    finally:
        applicator.close()

    assert revert_version == version_before
    assert result.lance_version_before == version_before


def test_add_column_constant(atlas_dirs):
    db_uri, audit_path = atlas_dirs
    _make_gene_table(db_uri)

    txn = CurationTransaction(
        table_name="gene_expression",
        changes=[
            AddColumn(
                column="organism",
                value="human",
                tool="geo_metadata",
                reason="from series metadata",
                source="GSE123456",
            )
        ],
    )

    applicator = CurationApplicator(db_uri, audit_db_path=audit_path)
    try:
        result = applicator.apply(txn, allowed_columns={"organism"})
    finally:
        applicator.close()

    assert result.status == TransactionStatus.APPLIED
    assert result.applied_changes[0].rows_updated is None

    db = lancedb.connect(db_uri)
    rows = db.open_table("gene_expression").to_arrow().to_pydict()
    assert rows["organism"] == ["human", "human", "human"]

    store = CurationAuditStore(audit_path)
    try:
        record = store.get_transaction(result.transaction_id)
        change = record["changes"][0]
        assert change["op_kind"] == "add_column"
        assert change["new_value"] == "human"
        assert change["source"] == "GSE123456"
    finally:
        store.close()


def test_arrow_type_from_alias_supports_list():
    assert arrow_type_from_alias("list<string>") == pa.list_(pa.string())
    assert arrow_type_from_alias("list<item: string>") == pa.list_(pa.string())


def test_add_column_constant_list(atlas_dirs):
    db_uri, audit_path = atlas_dirs
    _make_gene_table(db_uri)

    txn = CurationTransaction(
        table_name="gene_expression",
        changes=[
            AddColumn(
                column="perturbation_types",
                value=["genetic_perturbation"],
                tool="schema_align",
                reason="single perturbation type for all cells",
            )
        ],
    )

    applicator = CurationApplicator(db_uri, audit_db_path=audit_path)
    try:
        result = applicator.apply(txn, allowed_columns={"perturbation_types"})
    finally:
        applicator.close()

    assert result.status == TransactionStatus.APPLIED
    table = lancedb.connect(db_uri).open_table("gene_expression").to_arrow()
    assert table.schema.field("perturbation_types").type == pa.list_(pa.string())
    assert table.to_pydict()["perturbation_types"] == [
        ["genetic_perturbation"],
        ["genetic_perturbation"],
        ["genetic_perturbation"],
    ]

    store = CurationAuditStore(audit_path)
    try:
        change = store.get_transaction(result.transaction_id)["changes"][0]
        assert change["new_value"] == ["genetic_perturbation"]
    finally:
        store.close()


def test_add_column_null_list_and_merge_fill(atlas_dirs):
    db_uri, audit_path = atlas_dirs
    _make_gene_table(db_uri)

    txn = CurationTransaction(
        table_name="gene_expression",
        changes=[
            AddColumn(
                column="perturbation_types",
                data_type="list<string>",
                tool="schema_align",
            ),
            MergeColumns(
                column="perturbation_types",
                key_column="uid",
                rows=[
                    {"uid": "aaa", "perturbation_types": ["genetic_perturbation"]},
                    {"uid": "bbb", "perturbation_types": ["small_molecule"]},
                ],
                tool="schema_align",
            ),
        ],
    )

    applicator = CurationApplicator(db_uri, audit_db_path=audit_path)
    try:
        result = applicator.apply(txn, allowed_columns={"perturbation_types"})
    finally:
        applicator.close()

    assert result.status == TransactionStatus.APPLIED
    table = lancedb.connect(db_uri).open_table("gene_expression").to_arrow()
    assert table.schema.field("perturbation_types").type == pa.list_(pa.string())
    rows = {
        row["uid"]: row["perturbation_types"]
        for row in table.select(["uid", "perturbation_types"]).to_pylist()
    }
    assert rows == {
        "aaa": ["genetic_perturbation"],
        "bbb": ["small_molecule"],
        "ccc": None,
    }


def test_add_column_rejects_existing(atlas_dirs):
    db_uri, audit_path = atlas_dirs
    _make_gene_table(db_uri)

    txn = CurationTransaction(
        table_name="gene_expression",
        changes=[AddColumn(column="gene_symbol", value="x", tool="t")],
    )

    applicator = CurationApplicator(db_uri, audit_db_path=audit_path)
    try:
        with pytest.raises(ValueError, match="already exists"):
            applicator.apply(txn)
    finally:
        applicator.close()


def test_add_then_set_in_one_transaction(atlas_dirs):
    db_uri, audit_path = atlas_dirs
    _make_gene_table(db_uri)

    txn = CurationTransaction(
        table_name="gene_expression",
        changes=[
            AddColumn(column="organism", data_type="string", tool="t"),
            SetColumn(column="organism", new_value="mouse", tool="t"),
        ],
    )

    applicator = CurationApplicator(db_uri, audit_db_path=audit_path)
    try:
        result = applicator.apply(txn)
    finally:
        applicator.close()

    assert result.status == TransactionStatus.APPLIED
    db = lancedb.connect(db_uri)
    rows = db.open_table("gene_expression").to_arrow().to_pydict()
    assert rows["organism"] == ["mouse", "mouse", "mouse"]


def test_rename_column(atlas_dirs):
    db_uri, audit_path = atlas_dirs
    _make_gene_table(db_uri)

    txn = CurationTransaction(
        table_name="gene_expression",
        changes=[
            RenameColumn(
                column="ensembl_id",
                new_name="ensembl_gene_id",
                tool="schema_align",
            )
        ],
    )

    applicator = CurationApplicator(db_uri, audit_db_path=audit_path)
    try:
        result = applicator.apply(txn, allowed_columns={"ensembl_gene_id"})
    finally:
        applicator.close()

    assert result.status == TransactionStatus.APPLIED
    db = lancedb.connect(db_uri)
    names = db.open_table("gene_expression").schema.names
    assert "ensembl_gene_id" in names
    assert "ensembl_id" not in names

    store = CurationAuditStore(audit_path)
    try:
        change = store.get_transaction(result.transaction_id)["changes"][0]
        assert change["op_kind"] == "rename_column"
        assert change["target_column"] == "ensembl_gene_id"
    finally:
        store.close()


def test_rename_rejects_existing_target(atlas_dirs):
    db_uri, audit_path = atlas_dirs
    _make_gene_table(db_uri)

    txn = CurationTransaction(
        table_name="gene_expression",
        changes=[RenameColumn(column="ensembl_id", new_name="gene_symbol", tool="t")],
    )

    applicator = CurationApplicator(db_uri, audit_db_path=audit_path)
    try:
        with pytest.raises(ValueError, match="already exists"):
            applicator.apply(txn)
    finally:
        applicator.close()


def test_drop_column_exempt_from_allowed_columns(atlas_dirs):
    db_uri, audit_path = atlas_dirs
    _make_gene_table(db_uri)

    txn = CurationTransaction(
        table_name="gene_expression",
        changes=[DropColumn(column="ensembl_id", tool="finalize", reason="not in schema")],
    )

    applicator = CurationApplicator(db_uri, audit_db_path=audit_path)
    try:
        # ensembl_id is NOT in allowed_columns, but drops bypass the gate.
        result = applicator.apply(txn, allowed_columns={"gene_symbol"})
    finally:
        applicator.close()

    assert result.status == TransactionStatus.APPLIED
    db = lancedb.connect(db_uri)
    assert "ensembl_id" not in db.open_table("gene_expression").schema.names


def test_cast_column(atlas_dirs):
    db_uri, audit_path = atlas_dirs
    db = lancedb.connect(db_uri)
    db.create_table("t", data=pa.table({"replicate": [1, 2, 3]}), mode="overwrite")

    txn = CurationTransaction(
        table_name="t",
        changes=[CastColumn(column="replicate", data_type="string", tool="finalize")],
    )

    applicator = CurationApplicator(db_uri, audit_db_path=audit_path)
    try:
        result = applicator.apply(txn)
    finally:
        applicator.close()

    assert result.status == TransactionStatus.APPLIED
    table = db.open_table("t")
    assert pa.types.is_string(table.schema.field("replicate").type)
    assert table.to_arrow()["replicate"].to_pylist() == ["1", "2", "3"]


def test_set_column_with_sql_expression(atlas_dirs):
    db_uri, audit_path = atlas_dirs
    db = lancedb.connect(db_uri)
    db.create_table("t", data=pa.table({"a": [1, 2], "b": [10, 20]}), mode="overwrite")

    txn = CurationTransaction(
        table_name="t",
        changes=[SetColumn(column="a", value_sql="b + 1", tool="compute")],
    )

    applicator = CurationApplicator(db_uri, audit_db_path=audit_path)
    try:
        result = applicator.apply(txn)
    finally:
        applicator.close()

    assert result.status == TransactionStatus.APPLIED
    rows = db.open_table("t").to_arrow().to_pydict()
    assert rows["a"] == [11, 21]


def test_revert_after_schema_ops(atlas_dirs):
    db_uri, audit_path = atlas_dirs
    table = _make_gene_table(db_uri)
    version_before = table.version

    txn = CurationTransaction(
        table_name="gene_expression",
        changes=[
            AddColumn(column="organism", value="human", tool="t"),
            DropColumn(column="ensembl_id", tool="t"),
        ],
    )

    applicator = CurationApplicator(db_uri, audit_db_path=audit_path)
    try:
        result = applicator.apply(txn)
        revert_version = applicator.get_revert_version(result.transaction_id)
    finally:
        applicator.close()

    assert result.status == TransactionStatus.APPLIED
    assert revert_version == version_before

    # Version-based revert restores the pre-transaction schema.
    db = lancedb.connect(db_uri)
    table = db.open_table("gene_expression")
    table.checkout(version_before)
    table.restore()
    names = table.schema.names
    assert "organism" not in names
    assert "ensembl_id" in names
