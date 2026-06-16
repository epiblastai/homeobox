import json

import lancedb

from polycomb import cli
from polycomb.metadata_table import REFERENCE_TABLE_SCHEMAS, initialize_reference_db


def test_initialize_reference_db_creates_empty_schema_tables(tmp_path) -> None:
    db_path = str(tmp_path / "reference_db")

    statuses = initialize_reference_db(db_path)

    assert set(statuses) == set(REFERENCE_TABLE_SCHEMAS)
    assert set(statuses.values()) == {"created"}

    db = lancedb.connect(db_path)
    assert set(db.list_tables().tables) == set(REFERENCE_TABLE_SCHEMAS)
    for table_name in REFERENCE_TABLE_SCHEMAS:
        assert db.open_table(table_name).count_rows() == 0


def test_cli_setup_writes_config_and_initializes_tables(tmp_path, monkeypatch, capsys) -> None:
    db_path = str(tmp_path / "reference_db")
    config_path = str(tmp_path / ".polycomb" / "config.json")

    import polycomb.metadata_table as metadata_table

    monkeypatch.setattr(metadata_table, "CONFIG_PATH", config_path)

    rc = cli.main(
        [
            "setup",
            "--db-path",
            db_path,
            "--storage-options-json",
            '{"region": "auto"}',
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert "Reference DB ready" in captured.out

    with open(config_path) as f:
        payload = json.load(f)
    assert payload == {
        "reference_db": {
            "path": db_path,
            "storage_options": {"region": "auto"},
        }
    }

    db = lancedb.connect(db_path)
    assert set(db.list_tables().tables) == set(REFERENCE_TABLE_SCHEMAS)


def test_cli_optimize_cache_skips_empty_tables(tmp_path, capsys) -> None:
    db_path = str(tmp_path / "reference_db")
    initialize_reference_db(db_path)

    rc = cli.main(["optimize-cache", "--db-path", db_path, "--dry-run"])

    assert rc == 0
    captured = capsys.readouterr()
    assert "table is empty" in captured.out
    assert "DRY-RUN optimize" in captured.out
    assert "failed" in captured.out
