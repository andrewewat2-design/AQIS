from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import sqlite3
from typing import Iterable
from pathlib import Path

from aqis.models import AuditEvent, LifecycleStage, is_agent_actor


_TRANSITIONS: dict[LifecycleStage, set[LifecycleStage]] = {
    LifecycleStage.DISCOVERY: {LifecycleStage.RESEARCH, LifecycleStage.REJECTED},
    LifecycleStage.RESEARCH: {LifecycleStage.BACKTEST, LifecycleStage.REJECTED},
    LifecycleStage.BACKTEST: {LifecycleStage.VALIDATION, LifecycleStage.RESEARCH, LifecycleStage.REJECTED},
    LifecycleStage.VALIDATION: {LifecycleStage.PAPER_TRADING, LifecycleStage.RESEARCH, LifecycleStage.REJECTED},
    LifecycleStage.PAPER_TRADING: {LifecycleStage.MONITORING, LifecycleStage.RETIRED},
    LifecycleStage.MONITORING: {LifecycleStage.RETIRED, LifecycleStage.RESTRICTED_DEPLOYMENT},
    LifecycleStage.RESTRICTED_DEPLOYMENT: {LifecycleStage.MONITORING, LifecycleStage.RETIRED},
    LifecycleStage.REJECTED: set(),
    LifecycleStage.RETIRED: set(),
}


@dataclass(frozen=True, slots=True)
class StrategyVersion:
    candidate_id: str
    version: int
    strategy_set_version: str
    parameters: tuple[tuple[str, str], ...]
    confidence_scales: tuple[tuple[str, str], ...]
    source_hash: str
    risk_policy_version: int


@dataclass(slots=True)
class Candidate:
    candidate_id: str
    version: StrategyVersion
    stage: LifecycleStage = LifecycleStage.DISCOVERY
    human_approved_for_paper: bool = False
    validation_passed: bool = False


class LifecycleManager:
    def __init__(self) -> None:
        self.candidates: dict[str, Candidate] = {}
        self.events: list[AuditEvent] = []

    def create_candidate(self, version: StrategyVersion, actor: str = "research_agent") -> Candidate:
        if version.candidate_id in self.candidates:
            raise ValueError("candidate id already exists; create a new candidate id")
        candidate = Candidate(candidate_id=version.candidate_id, version=version)
        self.candidates[version.candidate_id] = candidate
        self._event("candidate_created", actor, candidate, {"version": version.version})
        return candidate

    def transition(
        self,
        candidate_id: str,
        to_stage: LifecycleStage,
        actor: str,
        *,
        validation_passed: bool | None = None,
        human_approved: bool = False,
        reason: str = "",
    ) -> Candidate:
        candidate = self.candidates[candidate_id]
        if to_stage not in _TRANSITIONS[candidate.stage]:
            raise ValueError(f"invalid lifecycle transition: {candidate.stage.value} -> {to_stage.value}")
        if candidate.stage is LifecycleStage.VALIDATION and to_stage is LifecycleStage.PAPER_TRADING:
            if is_agent_actor(actor):
                raise PermissionError("A human must explicitly approve paper deployment")
            if not candidate.validation_passed:
                raise ValueError("validation must pass before paper trading")
            if not candidate.human_approved_for_paper and not human_approved:
                raise PermissionError("candidate has no explicit human paper approval")
        if to_stage is LifecycleStage.RESTRICTED_DEPLOYMENT:
            raise PermissionError("Restricted deployment is disabled in the AQIS v1 prototype")
        if validation_passed is not None:
            candidate.validation_passed = validation_passed
        if human_approved:
            candidate.human_approved_for_paper = True
        previous = candidate.stage
        candidate.stage = to_stage
        self._event(
            "lifecycle_transition",
            actor,
            candidate,
            {"from": previous.value, "to": to_stage.value, "reason": reason},
        )
        return candidate

    def record_validation(
        self, candidate_id: str, passed: bool, report_id: str, actor: str = "validation_engine"
    ) -> Candidate:
        candidate = self.candidates[candidate_id]
        if candidate.stage is not LifecycleStage.VALIDATION:
            raise ValueError("validation results can only be recorded at the validation stage")
        candidate.validation_passed = passed
        self._event(
            "validation_recorded",
            actor,
            candidate,
            {"report_id": report_id, "passed": passed},
        )
        return candidate

    def approve_for_paper(self, candidate_id: str, actor: str, report_id: str) -> Candidate:
        candidate = self.candidates[candidate_id]
        if candidate.stage is not LifecycleStage.VALIDATION:
            raise ValueError("paper approval is only available after validation")
        if is_agent_actor(actor):
            raise PermissionError("only a human can approve paper deployment")
        if not candidate.validation_passed:
            raise ValueError("candidate has not passed validation")
        candidate.human_approved_for_paper = True
        self._event("paper_deployment_approved", actor, candidate, {"report_id": report_id})
        return candidate

    def new_version(self, prior_candidate_id: str, new_version: StrategyVersion) -> Candidate:
        prior = self.candidates[prior_candidate_id]
        if new_version.candidate_id == prior_candidate_id or new_version.version <= prior.version.version:
            raise ValueError("modifications require a new candidate id and higher version")
        if new_version.risk_policy_version != prior.version.risk_policy_version:
            # A strategy and policy may be versioned together, but both need explicit review.
            pass
        candidate = self.create_candidate(new_version, actor="human_or_research_agent")
        # A revision prompted by poor performance is a new immutable candidate
        # that re-enters at Research; it does not inherit validation or approval.
        return self.transition(
            candidate.candidate_id,
            LifecycleStage.RESEARCH,
            actor="supervisor_agent",
            reason=f"new version of {prior_candidate_id}; research and validation required",
        )

    def _event(
        self, event_type: str, actor: str, candidate: Candidate, details: dict[str, object]
    ) -> None:
        self.events.append(
            AuditEvent(
                timestamp=datetime.now(timezone.utc),
                event_type=event_type,
                actor=actor,
                details={"candidate_id": candidate.candidate_id, **details},
            )
        )


class PersistentLifecycleManager(LifecycleManager):
    """SQLite-backed lifecycle state and append-only decision history."""

    def __init__(self, database: str = "data/aqis.sqlite3") -> None:
        super().__init__()
        if database != ":memory:":
            Path(database).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS strategy_candidates ("
            "candidate_id TEXT PRIMARY KEY, version INTEGER NOT NULL, strategy_set_version TEXT NOT NULL, "
            "parameters TEXT NOT NULL, confidence_scales TEXT NOT NULL, source_hash TEXT NOT NULL, "
            "risk_policy_version INTEGER NOT NULL, stage TEXT NOT NULL, human_approved INTEGER NOT NULL, "
            "validation_passed INTEGER NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS lifecycle_events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, event_type TEXT NOT NULL, "
            "actor TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        self.connection.execute(
            "CREATE TRIGGER IF NOT EXISTS lifecycle_events_no_update "
            "BEFORE UPDATE ON lifecycle_events BEGIN SELECT RAISE(ABORT, 'lifecycle events are immutable'); END"
        )
        self.connection.execute(
            "CREATE TRIGGER IF NOT EXISTS lifecycle_events_no_delete "
            "BEFORE DELETE ON lifecycle_events BEGIN SELECT RAISE(ABORT, 'lifecycle events are immutable'); END"
        )
        self.connection.execute(
            "CREATE TRIGGER IF NOT EXISTS strategy_candidate_version_immutable "
            "BEFORE UPDATE OF version, strategy_set_version, parameters, confidence_scales, source_hash, risk_policy_version "
            "ON strategy_candidates BEGIN SELECT RAISE(ABORT, 'strategy versions are immutable'); END"
        )
        self.connection.execute(
            "CREATE TRIGGER IF NOT EXISTS strategy_candidate_no_delete "
            "BEFORE DELETE ON strategy_candidates BEGIN SELECT RAISE(ABORT, 'strategy candidates are immutable'); END"
        )
        self.connection.commit()
        self._load_candidates()

    def _load_candidates(self) -> None:
        rows = self.connection.execute(
            "SELECT candidate_id, version, strategy_set_version, parameters, confidence_scales, "
            "source_hash, risk_policy_version, stage, human_approved, validation_passed "
            "FROM strategy_candidates"
        ).fetchall()
        for row in rows:
            version = StrategyVersion(
                candidate_id=row[0],
                version=row[1],
                strategy_set_version=row[2],
                parameters=tuple(tuple(item) for item in json.loads(row[3])),
                confidence_scales=tuple(tuple(item) for item in json.loads(row[4])),
                source_hash=row[5],
                risk_policy_version=row[6],
            )
            self.candidates[row[0]] = Candidate(
                candidate_id=row[0],
                version=version,
                stage=LifecycleStage(row[7]),
                human_approved_for_paper=bool(row[8]),
                validation_passed=bool(row[9]),
            )

    def _event(
        self, event_type: str, actor: str, candidate: Candidate, details: dict[str, object]
    ) -> None:
        super()._event(event_type, actor, candidate, details)
        version = candidate.version
        self.connection.execute(
            "INSERT INTO strategy_candidates(candidate_id, version, strategy_set_version, parameters, "
            "confidence_scales, source_hash, risk_policy_version, stage, human_approved, validation_passed) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(candidate_id) DO UPDATE SET stage=excluded.stage, "
            "human_approved=excluded.human_approved, validation_passed=excluded.validation_passed",
            (
                candidate.candidate_id,
                version.version,
                version.strategy_set_version,
                json.dumps(version.parameters),
                json.dumps(version.confidence_scales),
                version.source_hash,
                version.risk_policy_version,
                candidate.stage.value,
                int(candidate.human_approved_for_paper),
                int(candidate.validation_passed),
            ),
        )
        event = self.events[-1]
        self.connection.execute(
            "INSERT INTO lifecycle_events(timestamp, event_type, actor, payload) VALUES (?, ?, ?, ?)",
            (
                event.timestamp.isoformat(),
                event.event_type,
                event.actor,
                json.dumps(event.details, sort_keys=True),
            ),
        )
        self.connection.commit()

    def event_history(self) -> list[dict[str, object]]:
        rows = self.connection.execute(
            "SELECT id, timestamp, event_type, actor, payload FROM lifecycle_events ORDER BY id"
        ).fetchall()
        return [
            {
                "id": row[0],
                "timestamp": row[1],
                "event_type": row[2],
                "actor": row[3],
                "payload": json.loads(row[4]),
            }
            for row in rows
        ]

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> PersistentLifecycleManager:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
