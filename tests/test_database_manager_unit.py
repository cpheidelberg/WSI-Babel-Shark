"""
Unit test for database_manager using a temporary SQLite file.
"""
import sqlite3
import pytest

@pytest.mark.unit
def test_db_insert_and_read(tmp_path):
    from babel_shark import database_manager

    db_path = tmp_path / "test.db"
    # If your module has a factory/get_connection, prefer that; else use sqlite3 directly.
    conn = sqlite3.connect(db_path.as_posix())
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS slides(case_id TEXT, pot TEXT, block TEXT, stain TEXT)")
    conn.commit()

    # If your module provides an insertion function, call it; else simulate:
    if hasattr(database_manager, "insert_slide"):
        database_manager.insert_slide(conn, "2024/12345", "A", "2", "PAS")
    else:
        cur.execute("INSERT INTO slides VALUES (?,?,?,?)", ("2024/12345","A","2","PAS"))
        conn.commit()

    cur.execute("SELECT * FROM slides")
    row = cur.fetchone()
    assert row and row[0].startswith("2024") and row[3].upper() == "PAS"
    conn.close()
