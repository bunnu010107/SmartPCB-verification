"""Plain-assert tests for quality_control/database.py. Run: python tests/test_database.py"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quality_control.database import next_inspection_id, open_db, recent, save_report  # noqa: E402
from quality_control.rules import build_report, load_config  # noqa: E402

CFG = load_config(ROOT / "quality_control" / "config_v1_1024.json")
BOX = [[100, 100], [140, 100], [140, 120], [100, 120]]


def make_report(conn, dets):
    iid = next_inspection_id(conn)
    return build_report(dets, CFG, iid, "x.jpg", [2048, 1500])


def test_ids_increase_and_rows_are_stored():
    with tempfile.TemporaryDirectory() as tmp:
        conn = open_db(Path(tmp) / "t.db")
        assert next_inspection_id(conn) == "PCB-000001"
        r1 = make_report(conn, [{"class_id": 3, "confidence": 0.9, "polygon": BOX}])
        save_report(conn, r1, "a.jpg")
        r2 = make_report(conn, [])
        assert r2["inspection_id"] == "PCB-000002"
        save_report(conn, r2, "b.jpg")
        assert [row[0] for row in recent(conn)] == ["PCB-000002", "PCB-000001"]
        n_def = conn.execute("SELECT COUNT(*) FROM defects").fetchone()[0]
        assert n_def == 1
        conn.close()


def test_report_json_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        conn = open_db(Path(tmp) / "t.db")
        r = make_report(conn, [{"class_id": 1, "confidence": 0.45, "polygon": BOX}])
        save_report(conn, r, "a.jpg")
        stored = conn.execute("SELECT report_json FROM inspections").fetchone()[0]
        assert json.loads(stored) == r
        conn.close()


def test_duplicate_id_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        conn = open_db(Path(tmp) / "t.db")
        r = make_report(conn, [])
        save_report(conn, r, "a.jpg")
        try:
            save_report(conn, r, "a.jpg")
        except Exception:
            conn.close()
            return
        raise AssertionError("expected the duplicate id to be rejected")


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"ALL {len(tests)} TESTS PASSED")
