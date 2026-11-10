from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "as_tuple"):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


class AuditLog:
    """Append-only local event log for the v1 prototype."""

    def __init__(self, database: str | Path = ":memory:") -> None:
        self.connection = sqlite3.connect(str(database))
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS audit_events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, "
            "event_type TEXT NOT NULL, actor TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        self.connection.execute(
            "CREATE TRIGGER IF NOT EXISTS audit_events_no_update "
            "BEFORE UPDATE ON audit_events BEGIN SELECT RAISE(ABORT, 'audit events are immutable'); END"
        )
        self.connection.execute(
            "CREATE TRIGGER IF NOT EXISTS audit_events_no_delete "
            "BEFORE DELETE ON audit_events BEGIN SELECT RAISE(ABORT, 'audit events are immutable'); END"
        )
        self.connection.commit()

    def append(self, timestamp: datetime, event_type: str, actor: str, payload: Any) -> int:
        serialized = json.dumps(payload, default=_json_default, sort_keys=True)
        cursor = self.connection.execute(
            "INSERT INTO audit_events(timestamp, event_type, actor, payload) VALUES (?, ?, ?, ?)",
            (timestamp.astimezone(timezone.utc).isoformat(), event_type, actor, serialized),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def events(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT id, timestamp, event_type, actor, payload FROM audit_events ORDER BY id"
        ).fetchall()
        return [
            {"id": row[0], "timestamp": row[1], "event_type": row[2], "actor": row[3], "payload": json.loads(row[4])}
            for row in rows
        ]

    def close(self) -> None:
        self.connection.close()
