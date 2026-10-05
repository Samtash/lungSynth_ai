"""Tiny SQLite persistence layer for session history (mirrors the shape
of HistoryEntry in the frontend's src/lib/lungsynth.ts)."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from app.config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    session_code TEXT NOT NULL,
    user_email TEXT,
    created_at TEXT NOT NULL,
    phases INTEGER NOT NULL,
    processing_seconds INTEGER NOT NULL,
    status TEXT NOT NULL
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute(_SCHEMA)


def add_session_record(
    record_id: str,
    session_code: str,
    user_email: str | None,
    created_at: str,
    phases: int,
    processing_seconds: int,
    status: str = "Completed",
):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (id, session_code, user_email, created_at, phases, processing_seconds, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (record_id, session_code, user_email, created_at, phases, processing_seconds, status),
        )


def list_sessions(user_email: str | None = None) -> list[dict]:
    with get_conn() as conn:
        if user_email:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE user_email = ? ORDER BY created_at DESC", (user_email,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM sessions ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def clear_sessions(user_email: str | None = None):
    with get_conn() as conn:
        if user_email:
            conn.execute("DELETE FROM sessions WHERE user_email = ?", (user_email,))
        else:
            conn.execute("DELETE FROM sessions")
