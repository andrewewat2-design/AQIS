import unittest
import tempfile
from pathlib import Path
import sqlite3

from aqis.lifecycle import LifecycleManager, PersistentLifecycleManager, StrategyVersion
from aqis.models import LifecycleStage


def version(candidate: str = "candidate-a", number: int = 1) -> StrategyVersion:
    return StrategyVersion(
        candidate_id=candidate,
        version=number,
        strategy_set_version=f"set-{number}",
        parameters=(("fast", "20"),),
        confidence_scales=(("ema", "1 ATR"),),
        source_hash=f"hash-{number}",
        risk_policy_version=1,
    )


class LifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = LifecycleManager()
        self.manager.create_candidate(version())

    def test_new_version_is_a_new_candidate_and_inherits_no_approval(self) -> None:
        first = self.manager.candidates["candidate-a"]
        first.stage = LifecycleStage.RETIRED
        second = self.manager.new_version("candidate-a", version("candidate-b", 2))
        self.assertEqual(second.stage, LifecycleStage.RESEARCH)
        self.assertFalse(second.human_approved_for_paper)
        self.assertFalse(second.validation_passed)

    def test_full_path_requires_validation_and_human_paper_approval(self) -> None:
        for stage in (LifecycleStage.RESEARCH, LifecycleStage.BACKTEST, LifecycleStage.VALIDATION):
            self.manager.transition("candidate-a", stage, "agent")
        self.manager.record_validation("candidate-a", True, "report-1")
        with self.assertRaises(PermissionError):
            self.manager.approve_for_paper("candidate-a", "supervisor", "report-1")
        self.manager.approve_for_paper("candidate-a", "operator", "report-1")
        self.manager.transition("candidate-a", LifecycleStage.PAPER_TRADING, "operator")
        self.assertEqual(self.manager.candidates["candidate-a"].stage, LifecycleStage.PAPER_TRADING)

    def test_failed_validation_cannot_enter_paper(self) -> None:
        for stage in (LifecycleStage.RESEARCH, LifecycleStage.BACKTEST, LifecycleStage.VALIDATION):
            self.manager.transition("candidate-a", stage, "agent")
        self.manager.record_validation("candidate-a", False, "report-fail")
        with self.assertRaises(ValueError):
            self.manager.approve_for_paper("candidate-a", "operator", "report-fail")

    def test_strategy_cannot_be_edited_in_place_by_transition(self) -> None:
        with self.assertRaises(ValueError):
            self.manager.transition("candidate-a", LifecycleStage.BACKTEST, "agent")

    def test_sqlite_lifecycle_persists_candidate_and_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "aqis.sqlite3")
            with PersistentLifecycleManager(path) as manager:
                manager.create_candidate(version("persisted"), actor="research_agent")
                manager.transition("persisted", LifecycleStage.RESEARCH, "strategy_agent")
            with PersistentLifecycleManager(path) as reopened:
                candidate = reopened.candidates["persisted"]
                self.assertEqual(candidate.stage, LifecycleStage.RESEARCH)
                self.assertEqual(len(reopened.event_history()), 2)
                with self.assertRaises(sqlite3.IntegrityError):
                    reopened.connection.execute(
                        "UPDATE strategy_candidates SET parameters='[]' WHERE candidate_id='persisted'"
                    )
