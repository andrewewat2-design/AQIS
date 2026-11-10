"""Decision Ledger — persistent, append-only decision and outcome recording.

The ledger records every decision AQIS makes, the information available at
decision time, risk evaluations, executions, and subsequent outcomes. It is
read-only after write: immutability is enforced by SQLite triggers. Ledger
writes never influence trading, risk, or execution decisions.

All identifiers are deterministic SHA-256 hashes derived from replay identity
and decision identity. Identical replays produce identical ledger records.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LEDGER_SCHEMA_VERSION = 2


def generate_replay_id(
    strategy_set_version: str,
    first_ts: str,
    last_ts: str,
    candle_count: int,
) -> str:
    """Deterministic replay ID from replay identity."""
    material = f"{strategy_set_version}:{first_ts}:{last_ts}:{candle_count}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def generate_decision_id(replay_id: str, candle_timestamp: str, candle_index: int) -> str:
    """Deterministic decision ID from replay identity and candle identity."""
    material = f"{replay_id}:{candle_timestamp}:{candle_index}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def generate_outcome_id(outcome_type: str, correlation_id: str, horizon: str) -> str:
    """Deterministic outcome ID including outcome type.

    Market and portfolio rows for the same decision horizon must produce
    distinct IDs so both can exist in the insert-only ledger.
    """
    material = f"{outcome_type}:{correlation_id}:{horizon}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def generate_status_id(
    correlation_id: str, horizon: str, status: str, seq: int
) -> str:
    """Deterministic status event ID unique per transition.

    The ledger is insert-only and immutable, so successive status events for
    the same (correlation_id, horizon) need distinct IDs; seq disambiguates
    them.
    """
    material = f"{correlation_id}:{horizon}:{status}:{seq}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class LedgerHealth:
    health_id: str
    last_successful_write: str | None
    last_failed_write: str | None
    failed_write_count: int
    pending_outcome_count: int
    ledger_complete: bool


class LedgerDatabase:
    """SQLite-backed persistent decision ledger.

    Shares the existing AQIS database file (data/aqis.sqlite3) with
    AuditLog, LifecycleManager, and PersistentPolicyRegistry. All ledger
    tables are append-only; immutability is enforced by SQLite triggers.
    """

    def __init__(self, database: str = "data/aqis.sqlite3") -> None:
        if database != ":memory:":
            Path(database).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self._create_schema()
        self._init_health()
        self.connection.commit()

    # ------------------------------------------------------------------
    # Schema creation
    # ------------------------------------------------------------------

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS ledger_schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ledger_replays (
                replay_id TEXT PRIMARY KEY,
                strategy_set_version TEXT NOT NULL,
                market TEXT NOT NULL,
                venue TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                first_candle_timestamp TEXT NOT NULL,
                last_candle_timestamp TEXT NOT NULL,
                candle_count INTEGER NOT NULL,
                policy_id TEXT NOT NULL,
                policy_version INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ledger_decisions (
                decision_id TEXT PRIMARY KEY,
                replay_id TEXT NOT NULL,
                correlation_id TEXT NOT NULL UNIQUE,
                candle_timestamp TEXT NOT NULL,
                candle_index INTEGER NOT NULL,
                market_data_snapshot TEXT NOT NULL,
                regime TEXT NOT NULL,
                strategy_signals TEXT NOT NULL,
                selector_decision TEXT NOT NULL,
                effective_threshold REAL NOT NULL,
                opportunity_evidence_id TEXT,
                risk_policy_id TEXT NOT NULL,
                risk_policy_version INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(replay_id) REFERENCES ledger_replays(replay_id)
            );

            CREATE TABLE IF NOT EXISTS ledger_risk_records (
                risk_record_id TEXT PRIMARY KEY,
                correlation_id TEXT NOT NULL UNIQUE,
                order_intent TEXT NOT NULL,
                risk_decision TEXT NOT NULL,
                risk_budget_usdt TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(correlation_id) REFERENCES ledger_decisions(correlation_id)
            );

            CREATE TABLE IF NOT EXISTS ledger_executions (
                execution_id TEXT PRIMARY KEY,
                correlation_id TEXT NOT NULL UNIQUE,
                order_attempt_timestamp TEXT NOT NULL,
                fill TEXT,
                rejection_code TEXT,
                execution_timestamp TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(correlation_id) REFERENCES ledger_decisions(correlation_id)
            );

            CREATE TABLE IF NOT EXISTS ledger_outcomes (
                outcome_id TEXT PRIMARY KEY,
                correlation_id TEXT NOT NULL,
                horizon TEXT NOT NULL,
                horizon_timestamp TEXT NOT NULL,
                outcome_type TEXT NOT NULL,
                anchor TEXT NOT NULL DEFAULT 'decision',
                direction INTEGER,
                baseline_price TEXT,
                baseline_equity TEXT,
                horizon_return_bps TEXT,
                price_at_horizon TEXT,
                portfolio_equity_at_horizon TEXT,
                realized_pnl_at_horizon TEXT,
                unrealized_pnl_at_horizon TEXT,
                max_adverse_excursion TEXT,
                max_favourable_excursion TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(correlation_id) REFERENCES ledger_decisions(correlation_id)
            );

            CREATE TABLE IF NOT EXISTS ledger_outcome_status (
                status_id TEXT PRIMARY KEY,
                correlation_id TEXT NOT NULL,
                horizon TEXT NOT NULL,
                status TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                FOREIGN KEY(correlation_id) REFERENCES ledger_decisions(correlation_id)
            );

            CREATE TABLE IF NOT EXISTS ledger_health (
                health_id TEXT PRIMARY KEY,
                last_successful_write TEXT,
                last_failed_write TEXT,
                failed_write_count INTEGER NOT NULL DEFAULT 0,
                pending_outcome_count INTEGER NOT NULL DEFAULT 0,
                ledger_complete INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );
            """
        )

        # Indexes
        self.connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_decisions_replay ON ledger_decisions(replay_id);
            CREATE INDEX IF NOT EXISTS idx_decisions_correlation ON ledger_decisions(correlation_id);
            CREATE INDEX IF NOT EXISTS idx_outcomes_correlation ON ledger_outcomes(correlation_id, horizon);
            CREATE INDEX IF NOT EXISTS idx_outcomes_pending ON ledger_outcome_status(correlation_id, horizon);
            CREATE INDEX IF NOT EXISTS idx_risk_correlation ON ledger_risk_records(correlation_id);
            CREATE INDEX IF NOT EXISTS idx_executions_correlation ON ledger_executions(correlation_id);
            """
        )

        # Immutability triggers (same pattern as audit_events, risk_policy_versions)
        # ledger_health is excluded: it tracks write status and must be updatable.
        for table in (
            "ledger_replays",
            "ledger_decisions",
            "ledger_risk_records",
            "ledger_executions",
            "ledger_outcomes",
            "ledger_outcome_status",
        ):
            self.connection.execute(
                f"CREATE TRIGGER IF NOT EXISTS {table}_no_update "
                f"BEFORE UPDATE ON {table} BEGIN SELECT RAISE(ABORT, '{table} rows are immutable'); END"
            )
            self.connection.execute(
                f"CREATE TRIGGER IF NOT EXISTS {table}_no_delete "
                f"BEFORE DELETE ON {table} BEGIN SELECT RAISE(ABORT, '{table} rows are immutable'); END"
            )

        # Schema version
        row = self.connection.execute(
            "SELECT version FROM ledger_schema_version ORDER BY version DESC LIMIT 1"
        ).fetchone()
        if row is None:
            self.connection.execute(
                "INSERT INTO ledger_schema_version(version, applied_at) VALUES (?, ?)",
                (LEDGER_SCHEMA_VERSION, datetime.now(timezone.utc).isoformat()),
            )
        elif row[0] < LEDGER_SCHEMA_VERSION:
            self._migrate_schema(row[0])

    def _migrate_schema(self, from_version: int) -> None:
        applied = LEDGER_SCHEMA_VERSION
        now = datetime.now(timezone.utc).isoformat()
        if from_version < 2:
            existing = self.connection.execute(
                "PRAGMA table_info(ledger_outcomes)"
            ).fetchall()
            columns = {col[1] for col in existing}
            migrations = {
                "anchor": "ALTER TABLE ledger_outcomes ADD COLUMN anchor TEXT NOT NULL DEFAULT 'decision'",
                "direction": "ALTER TABLE ledger_outcomes ADD COLUMN direction INTEGER",
                "baseline_price": "ALTER TABLE ledger_outcomes ADD COLUMN baseline_price TEXT",
                "baseline_equity": "ALTER TABLE ledger_outcomes ADD COLUMN baseline_equity TEXT",
                "horizon_return_bps": "ALTER TABLE ledger_outcomes ADD COLUMN horizon_return_bps TEXT",
            }
            for name, statement in migrations.items():
                if name not in columns:
                    self.connection.execute(statement)
            self.connection.execute(
                "INSERT INTO ledger_schema_version(version, applied_at) VALUES (?, ?)",
                (2, now),
            )
            applied = 2

    def _init_health(self) -> None:
        row = self.connection.execute(
            "SELECT health_id FROM ledger_health WHERE health_id='ledger'"
        ).fetchone()
        if row is None:
            self.connection.execute(
                "INSERT INTO ledger_health(health_id, updated_at) VALUES (?, ?)",
                ("ledger", datetime.now(timezone.utc).isoformat()),
            )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def record_successful_write(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.connection.execute(
            "UPDATE ledger_health SET last_successful_write=?, updated_at=? WHERE health_id='ledger'",
            (now, now),
        )
        self.connection.commit()

    def record_failed_write(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.connection.execute(
            """
            UPDATE ledger_health
            SET last_failed_write=?, updated_at=?,
                failed_write_count=failed_write_count + 1
            WHERE health_id='ledger'
            """,
            (now, now),
        )
        self.connection.commit()

    def get_health(self) -> LedgerHealth:
        row = self.connection.execute(
            "SELECT health_id, last_successful_write, last_failed_write, "
            "failed_write_count, pending_outcome_count, ledger_complete "
            "FROM ledger_health WHERE health_id='ledger'"
        ).fetchone()
        if row is None:
            return LedgerHealth(
                health_id="ledger",
                last_successful_write=None,
                last_failed_write=None,
                failed_write_count=0,
                pending_outcome_count=0,
                ledger_complete=False,
            )
        return LedgerHealth(
            health_id=row[0],
            last_successful_write=row[1],
            last_failed_write=row[2],
            failed_write_count=row[3],
            pending_outcome_count=row[4],
            ledger_complete=bool(row[5]),
        )

    # ------------------------------------------------------------------
    # Replay registration
    # ------------------------------------------------------------------

    def register_replay(
        self,
        replay_id: str,
        strategy_set_version: str,
        market: str,
        venue: str,
        timeframe: str,
        first_ts: str,
        last_ts: str,
        candle_count: int,
        policy_id: str,
        policy_version: int,
    ) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO ledger_replays(replay_id, strategy_set_version, market, venue, "
            "timeframe, first_candle_timestamp, last_candle_timestamp, candle_count, "
            "policy_id, policy_version, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                replay_id,
                strategy_set_version,
                market,
                venue,
                timeframe,
                first_ts,
                last_ts,
                candle_count,
                policy_id,
                policy_version,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.connection.commit()

    # ------------------------------------------------------------------
    # Decision recording (decision-time data only, no future data)
    # ------------------------------------------------------------------

    def record_decision(
        self,
        decision_id: str,
        replay_id: str,
        correlation_id: str,
        candle_timestamp: str,
        candle_index: int,
        market_data_snapshot: dict[str, Any],
        regime: str,
        strategy_signals: list[dict[str, Any]],
        selector_decision: dict[str, Any],
        effective_threshold: float,
        opportunity_evidence_id: str | None,
        risk_policy_id: str,
        risk_policy_version: int,
    ) -> None:
        self.connection.execute(
            "INSERT INTO ledger_decisions(decision_id, replay_id, correlation_id, "
            "candle_timestamp, candle_index, market_data_snapshot, regime, "
            "strategy_signals, selector_decision, effective_threshold, "
            "opportunity_evidence_id, risk_policy_id, risk_policy_version, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                decision_id,
                replay_id,
                correlation_id,
                candle_timestamp,
                candle_index,
                json.dumps(market_data_snapshot, sort_keys=True),
                regime,
                json.dumps(strategy_signals, sort_keys=True),
                json.dumps(selector_decision, sort_keys=True),
                effective_threshold,
                opportunity_evidence_id,
                risk_policy_id,
                risk_policy_version,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.connection.commit()

    # ------------------------------------------------------------------
    # Risk recording
    # ------------------------------------------------------------------

    def record_risk(
        self,
        correlation_id: str,
        order_intent: dict[str, Any],
        risk_decision: dict[str, Any],
        risk_budget_usdt: str | None,
    ) -> None:
        risk_record_id = hashlib.sha256(
            f"{correlation_id}:risk".encode("utf-8")
        ).hexdigest()[:16]
        self.connection.execute(
            "INSERT INTO ledger_risk_records(risk_record_id, correlation_id, "
            "order_intent, risk_decision, risk_budget_usdt, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                risk_record_id,
                correlation_id,
                json.dumps(order_intent, sort_keys=True),
                json.dumps(risk_decision, sort_keys=True),
                risk_budget_usdt,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.connection.commit()

    # ------------------------------------------------------------------
    # Execution recording
    # ------------------------------------------------------------------

    def record_execution(
        self,
        correlation_id: str,
        order_attempt_timestamp: str,
        fill: dict[str, Any] | None,
        rejection_code: str | None,
        execution_timestamp: str,
    ) -> None:
        execution_id = hashlib.sha256(
            f"{correlation_id}:execution".encode("utf-8")
        ).hexdigest()[:16]
        self.connection.execute(
            "INSERT INTO ledger_executions(execution_id, correlation_id, "
            "order_attempt_timestamp, fill, rejection_code, execution_timestamp, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                execution_id,
                correlation_id,
                order_attempt_timestamp,
                json.dumps(fill, sort_keys=True) if fill is not None else None,
                rejection_code,
                execution_timestamp,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.connection.commit()

    # ------------------------------------------------------------------
    # Outcome recording (insert-only, no updates)
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        correlation_id: str,
        horizon: str,
        horizon_timestamp: str,
        outcome_type: str,
        anchor: str,
        direction: int | None,
        baseline_price: str | None,
        baseline_equity: str | None,
        horizon_return_bps: str | None,
        price_at_horizon: str | None,
        portfolio_equity_at_horizon: str | None,
        realized_pnl_at_horizon: str | None,
        unrealized_pnl_at_horizon: str | None,
        max_adverse_excursion: str | None,
        max_favourable_excursion: str | None,
    ) -> None:
        outcome_id = generate_outcome_id(outcome_type, correlation_id, horizon)
        self.connection.execute(
            "INSERT INTO ledger_outcomes(outcome_id, correlation_id, horizon, "
            "horizon_timestamp, outcome_type, anchor, direction, baseline_price, "
            "baseline_equity, horizon_return_bps, price_at_horizon, "
            "portfolio_equity_at_horizon, realized_pnl_at_horizon, "
            "unrealized_pnl_at_horizon, max_adverse_excursion, "
            "max_favourable_excursion, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                outcome_id,
                correlation_id,
                horizon,
                horizon_timestamp,
                outcome_type,
                anchor,
                direction,
                baseline_price,
                baseline_equity,
                horizon_return_bps,
                price_at_horizon,
                portfolio_equity_at_horizon,
                realized_pnl_at_horizon,
                unrealized_pnl_at_horizon,
                max_adverse_excursion,
                max_favourable_excursion,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.connection.commit()

    def record_outcome_status(
        self,
        correlation_id: str,
        horizon: str,
        status: str,
    ) -> None:
        seq = self.connection.execute(
            "SELECT COUNT(*) FROM ledger_outcome_status WHERE correlation_id=? AND horizon=?",
            (correlation_id, horizon),
        ).fetchone()[0]
        status_id = generate_status_id(correlation_id, horizon, status, seq)
        self.connection.execute(
            "INSERT INTO ledger_outcome_status(status_id, correlation_id, horizon, "
            "status, recorded_at) VALUES (?, ?, ?, ?, ?)",
            (
                status_id,
                correlation_id,
                horizon,
                status,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.connection.commit()

    # ------------------------------------------------------------------
    # Read-only queries
    # ------------------------------------------------------------------

    def get_decision_chain(self, correlation_id: str) -> dict[str, Any]:
        """Read-only traceability: all records linked to a correlation ID."""
        decision = self.connection.execute(
            "SELECT * FROM ledger_decisions WHERE correlation_id=?", (correlation_id,)
        ).fetchone()
        risk = self.connection.execute(
            "SELECT * FROM ledger_risk_records WHERE correlation_id=?", (correlation_id,)
        ).fetchone()
        execution = self.connection.execute(
            "SELECT * FROM ledger_executions WHERE correlation_id=?", (correlation_id,)
        ).fetchone()
        outcomes = self.connection.execute(
            "SELECT * FROM ledger_outcomes WHERE correlation_id=? ORDER BY horizon",
            (correlation_id,),
        ).fetchall()
        statuses = self.connection.execute(
            "SELECT * FROM ledger_outcome_status WHERE correlation_id=? ORDER BY horizon",
            (correlation_id,),
        ).fetchall()
        return {
            "decision": dict(decision) if decision else None,
            "risk": dict(risk) if risk else None,
            "execution": dict(execution) if execution else None,
            "outcomes": [dict(o) for o in outcomes],
            "statuses": [dict(s) for s in statuses],
        }

    def get_pending_outcomes(self) -> list[dict[str, Any]]:
        """Decisions with horizons that have no completed outcome measurement."""
        rows = self.connection.execute(
            "SELECT d.correlation_id, d.candle_timestamp, s.horizon, s.status "
            "FROM ledger_outcome_status s "
            "JOIN ledger_decisions d ON d.correlation_id = s.correlation_id "
            "WHERE s.status IN ('pending', 'data_insufficient', 'horizon_not_reached') "
            "ORDER BY d.candle_timestamp, s.horizon"
        ).fetchall()
        return [
            {
                "correlation_id": row[0],
                "candle_timestamp": row[1],
                "horizon": row[2],
                "status": row[3],
            }
            for row in rows
        ]

    def get_outcomes_by_horizon(self, horizon: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM ledger_outcomes WHERE horizon=? ORDER BY horizon_timestamp",
            (horizon,),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> LedgerDatabase:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
