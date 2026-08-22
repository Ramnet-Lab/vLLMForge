"""SQLite persistence.

The dashboard owns very little state: server definitions, job records, chat
transcripts and a handful of settings. sqlite3 from the stdlib is enough, and
keeps the install free of a database dependency. All calls are synchronous and
are pushed onto a worker thread by the callers in the router layer.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS servers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL UNIQUE,
    model           TEXT NOT NULL,
    served_name     TEXT,
    port            INTEGER NOT NULL,
    image           TEXT NOT NULL,
    args            TEXT NOT NULL DEFAULT '{}',
    env             TEXT NOT NULL DEFAULT '{}',
    notes           TEXT NOT NULL DEFAULT '',
    autostart       INTEGER NOT NULL DEFAULT 0,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    last_started_at REAL
);

CREATE TABLE IF NOT EXISTS jobs (
    id             TEXT PRIMARY KEY,
    kind           TEXT NOT NULL,
    title          TEXT NOT NULL,
    status         TEXT NOT NULL,
    spec           TEXT NOT NULL DEFAULT '{}',
    result         TEXT NOT NULL DEFAULT '{}',
    progress       TEXT NOT NULL DEFAULT '{}',
    container_name TEXT,
    exit_code      INTEGER,
    error          TEXT,
    created_at     REAL NOT NULL,
    started_at     REAL,
    finished_at    REAL
);
CREATE INDEX IF NOT EXISTS jobs_kind_created ON jobs (kind, created_at DESC);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs (status);

CREATE TABLE IF NOT EXISTS chats (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT NOT NULL,
    endpoint   TEXT NOT NULL,
    model      TEXT NOT NULL,
    params     TEXT NOT NULL DEFAULT '{}',
    messages   TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS presets (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT NOT NULL,
    name       TEXT NOT NULL,
    data       TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    UNIQUE (kind, name)
);

CREATE TABLE IF NOT EXISTS datasets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    source      TEXT NOT NULL,          -- 'hub' | 'upload'
    reference   TEXT NOT NULL,          -- hub id, or host path for uploads
    format      TEXT NOT NULL DEFAULT '',
    rows        INTEGER,
    size_bytes  INTEGER,
    preview     TEXT NOT NULL DEFAULT '[]',
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS recipes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT NOT NULL,           -- 'finetune' | 'heretic'
    name       TEXT NOT NULL,
    data       TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE (kind, name)
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_local = threading.local()


def connect() -> sqlite3.Connection:
    """One connection per thread; WAL so readers never block the writer."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        settings.state_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(settings.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        _local.conn = conn
    return conn


# CREATE TABLE IF NOT EXISTS cannot add a column to a table that already
# exists, so anything added after the first release needs an explicit step.
MIGRATIONS = [
    ("servers", "node", "ALTER TABLE servers ADD COLUMN node TEXT NOT NULL DEFAULT 'local'"),
    ("servers", "pool_nodes",
     "ALTER TABLE servers ADD COLUMN pool_nodes TEXT NOT NULL DEFAULT '[]'"),
]


def init_db() -> None:
    conn = connect()
    conn.executescript(SCHEMA)
    for table, column, statement in MIGRATIONS:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(statement)
    conn.commit()


def query(sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    return [dict(row) for row in connect().execute(sql, tuple(params)).fetchall()]


def query_one(sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
    row = connect().execute(sql, tuple(params)).fetchone()
    return dict(row) if row else None


def execute(sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
    conn = connect()
    cursor = conn.execute(sql, tuple(params))
    conn.commit()
    return cursor


def now() -> float:
    return time.time()


# --- JSON column helpers -------------------------------------------------
# Every JSON-typed column is stored as TEXT; these keep the (de)serialisation
# in one place so a malformed row can never crash a request handler.

def dumps(value: Any) -> str:
    return json.dumps(value, default=str)


def loads(value: Any, fallback: Any = None) -> Any:
    if value in (None, ""):
        return fallback
    if not isinstance(value, (str, bytes)):
        return value
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return fallback


def hydrate(row: dict[str, Any] | None, json_fields: Iterable[str]) -> dict[str, Any] | None:
    if row is None:
        return None
    out = dict(row)
    for name in json_fields:
        if name in out:
            out[name] = loads(out[name], {} if name != "messages" else [])
    return out


# --- settings key/value --------------------------------------------------

def get_setting(key: str, fallback: Any = None) -> Any:
    row = query_one("SELECT value FROM settings WHERE key = ?", (key,))
    return loads(row["value"], fallback) if row else fallback


def set_setting(key: str, value: Any) -> None:
    execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, dumps(value)),
    )


def log_path(job_id: str) -> Path:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    return settings.log_dir / f"{job_id}.log"
