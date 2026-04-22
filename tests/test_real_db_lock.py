"""Test that Database.inject_lock() creates a REAL SQLite lock."""
import sys, os, tempfile, sqlite3
sys.path.insert(0, "D:/Meta")

from cloud_sre_v2.infra.database import Database

# Use a temp file so we don't mess with any existing DB
with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
    db_path = f.name

try:
    db = Database(db_path)
    db.initialize()

    # 1. Normal operation works
    db.execute("INSERT INTO payments (amount, user_id, status) VALUES (100.0, 'u1', 'completed')")
    rows = db.query("SELECT * FROM payments")
    print(f"1. Normal write OK: {len(rows)} rows")
    assert len(rows) == 1

    # 2. Inject REAL lock
    db.inject_lock()
    print(f"2. Lock injected (lock_conn is {'set' if db._lock_conn else 'None'})")
    assert db._lock_conn is not None, "Lock connection should exist!"
    assert db._is_fault_locked is True

    # 3. Try to write — should REALLY fail with SQLite error
    try:
        db.execute("INSERT INTO payments (amount, user_id, status) VALUES (200.0, 'u2', 'pending')")
        print("3. FAIL: Write should have been blocked!")
        assert False, "Write should have failed!"
    except sqlite3.OperationalError as e:
        print(f"3. Write blocked with REAL SQLite error: {e}")
        assert "locked" in str(e).lower() or "busy" in str(e).lower(), f"Unexpected error: {e}"

    # 4. Release lock
    db.release_lock()
    print(f"4. Lock released (lock_conn is {'set' if db._lock_conn else 'None'})")
    assert db._lock_conn is None
    assert db._is_fault_locked is False

    # 5. Write works again
    db.execute("INSERT INTO payments (amount, user_id, status) VALUES (300.0, 'u3', 'completed')")
    rows = db.query("SELECT * FROM payments")
    print(f"5. Write after unlock OK: {len(rows)} rows")
    assert len(rows) == 2

    # 6. Reset releases locks
    db.inject_lock()
    assert db._is_fault_locked is True
    db.reset()
    assert db._is_fault_locked is False
    assert db._lock_conn is None
    db.execute("INSERT INTO payments (amount, user_id, status) VALUES (50.0, 'u4', 'ok')")
    print("6. Reset + write after reset OK")

    print("\n=== ALL REAL DB LOCK TESTS PASSED ===")

finally:
    # Cleanup
    for suffix in ["", "-wal", "-shm"]:
        try:
            os.remove(db_path + suffix)
        except:
            pass
