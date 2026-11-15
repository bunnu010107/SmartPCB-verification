"""SQLite storage for SmartPCB inspections (standard library only)."""
import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    inspection_id  TEXT PRIMARY KEY,
    timestamp_utc  TEXT NOT NULL,
    image_path     TEXT NOT NULL,
    model_name     TEXT NOT NULL,
    weights_sha256 TEXT NOT NULL,
    board_status   TEXT NOT NULL,
    reason         TEXT NOT NULL,
    n_defects      INTEGER NOT NULL,
    report_json    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS defects (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id TEXT NOT NULL REFERENCES inspections(inspection_id),
    class_name    TEXT NOT NULL,
    confidence    REAL NOT NULL,
    tier          TEXT NOT NULL,
    center_x      REAL NOT NULL,
    center_y      REAL NOT NULL,
    long_px       REAL NOT NULL,
    short_px      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_defects_inspection ON defects(inspection_id);
"""


def open_db(path):
    """Open (and create if needed) the inspection database."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def next_inspection_id(conn):
    """PCB-000001, PCB-000002, ... based on the highest id stored so far."""
    (last,) = conn.execute(
        "SELECT MAX(CAST(SUBSTR(inspection_id, 5) AS INTEGER)) FROM inspections"
    ).fetchone()
    return f"PCB-{(last or 0) + 1:06d}"


def save_report(conn, report, image_path):
    """Store one report (the board row and its defect rows) in a single transaction."""
    with conn:
        conn.execute(
            "INSERT INTO inspections (inspection_id, timestamp_utc, image_path, model_name, "
            "weights_sha256, board_status, reason, n_defects, report_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (report["inspection_id"], report["timestamp_utc"], str(image_path),
             report["model"]["name"], report["model"]["weights_sha256"],
             report["board_status"], report["reason"], len(report["defects"]),
             json.dumps(report)),
        )
        conn.executemany(
            "INSERT INTO defects (inspection_id, class_name, confidence, tier, "
            "center_x, center_y, long_px, short_px) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(report["inspection_id"], d["class"], d["confidence"], d["tier"],
              d["center_px"][0], d["center_px"][1], d["size_px"][0], d["size_px"][1])
             for d in report["defects"]],
        )


def recent(conn, limit=10):
    """Most recent inspections, newest first."""
    return conn.execute(
        "SELECT inspection_id, timestamp_utc, image_path, board_status, n_defects "
        "FROM inspections ORDER BY inspection_id DESC LIMIT ?", (limit,)
    ).fetchall()
