import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from aqis.models import Candle, OpportunityEvidence, RiskPolicy, SignalAction, StrategySignal, TargetPosition
from aqis.nemotron import (
    FindingKind,
    MockNemotronProvider,
    NemotronResearchAgent,
    ResearchReport,
)
from aqis.nemotron_integration import NemotronObserver, attach_nemotron_research, ResearchSnapshot
from aqis.paper import PaperReplayEngine, ReplayResult
from tests.helpers import flat_candles


class ScheduledAgent:
    """Minimal stand-in strategy for integration tests."""

    def __init__(self, strategy_id: str, action: SignalAction) -> None:
        self.strategy_id = strategy_id
        self.action = action
        self.version = "test"

    def reset(self) -> None:
        pass

    def evaluate(self, candles):
        from aqis.models import StrategySignal
        return StrategySignal(
            self.strategy_id, self.version, candles[-1].timestamp,
            self.action, 100, "test",
        )


def make_candles(count: int = 6) -> list[Candle]:
    timestamp = datetime(2024, 1, 1, 1, tzinfo=timezone.utc)
    return [
        Candle(timestamp + timedelta(hours=i), 100.0, 100.5, 99.5, 100.0, 10.0)
        for i in range(count)
    ]


class IntegrationTests(unittest.TestCase):
    def test_nemotron_cannot_change_replay_results(self) -> None:
        """Replay results must be identical with or without Nemotron observer."""
        candles = make_candles(8)
        agents = [
            ScheduledAgent("ema", SignalAction.BUY),
            ScheduledAgent("rsi", SignalAction.BUY),
            ScheduledAgent("mean", SignalAction.BUY),
            ScheduledAgent("breakout", SignalAction.ABSTAIN),
        ]

        # Replay without Nemotron.
        result_without = PaperReplayEngine(
            RiskPolicy(), {"default": OpportunityEvidence(
                "e", "aqis-four-agent-v1", "default", Decimal("0.01"), Decimal("0.10"), 50, "h"
            )}, agents
        ).run(candles)

        # Replay with Nemotron observer.
        result_with = PaperReplayEngine(
            RiskPolicy(), {"default": OpportunityEvidence(
                "e", "aqis-four-agent-v1", "default", Decimal("0.01"), Decimal("0.10"), 50, "h"
            )}, agents
        ).run(candles)
        attach_nemotron_research(result_with)

        # All trading-relevant fields must be identical.
        self.assertEqual(result_without.starting_equity_usdt, result_with.starting_equity_usdt)
        self.assertEqual(result_without.ending_equity_usdt, result_with.ending_equity_usdt)
        self.assertEqual(result_without.maximum_drawdown_pct, result_with.maximum_drawdown_pct)
        self.assertEqual(result_without.halted, result_with.halted)
        self.assertEqual(result_without.halt_reason, result_with.halt_reason)
        self.assertEqual(len(result_without.fills), len(result_with.fills))
        self.assertEqual(len(result_without.rejected_orders), len(result_with.rejected_orders))
        self.assertEqual(len(result_without.audit_events), len(result_with.audit_events))
        self.assertEqual(len(result_without.decisions), len(result_with.decisions))
        self.assertEqual(len(result_without.risk_interventions), len(result_with.risk_interventions))
        self.assertEqual(len(result_without.pause_resume_history), len(result_with.pause_resume_history))
        self.assertEqual(len(result_without.portfolio_snapshots), len(result_with.portfolio_snapshots))

    def test_research_reports_stored_separately_from_execution_events(self) -> None:
        """Nemotron reports go to nemotron_reports, not audit_events."""
        candles = make_candles(6)
        agents = [
            ScheduledAgent("ema", SignalAction.ABSTAIN),
            ScheduledAgent("rsi", SignalAction.ABSTAIN),
            ScheduledAgent("mean", SignalAction.ABSTAIN),
            ScheduledAgent("breakout", SignalAction.ABSTAIN),
        ]
        result = PaperReplayEngine(
            RiskPolicy(), strategies=agents
        ).run(candles)

        audit_count_before = len(result.audit_events)
        reports = attach_nemotron_research(result)

        # Reports produced.
        self.assertGreater(len(reports), 0)
        # Reports stored on result.
        self.assertEqual(len(result.nemotron_reports), len(reports))
        # Audit events unchanged (reports not mixed with trading events).
        self.assertEqual(len(result.audit_events), audit_count_before)
        # Each report is a ResearchReport.
        for report in reports:
            self.assertIsInstance(report, ResearchReport)
            self.assertIsInstance(report.analysis_id, str)
            self.assertGreater(len(report.analysis_id), 0)

    def test_mock_provider_works_offline(self) -> None:
        """No env vars needed; MockNemotronProvider is the default."""
        import os
        for key in ("NEMOTRON_API_URL", "NEMOTRON_API_KEY", "NEMOTRON_MODEL"):
            os.environ.pop(key, None)

        candles = make_candles(6)
        result = PaperReplayEngine(RiskPolicy(), strategies=[]).run(candles)
        observer = NemotronObserver()  # defaults to MockNemotronProvider
        reports = observer.observe(result)

        self.assertGreater(len(reports), 0)
        for report in reports:
            self.assertIsInstance(report, ResearchReport)

    def test_provider_failure_does_not_interrupt_replay(self) -> None:
        """A failing provider produces fallback reports; replay completes."""
        class FailingProvider:
            def model_info(self):
                return {"model_name": "fail", "model_version": "1", "base_url": "x"}
            def complete_json(self, *, system, user):
                raise RuntimeError("provider down")

        candles = make_candles(6)
        result = PaperReplayEngine(RiskPolicy(), strategies=[]).run(candles)
        agent = NemotronResearchAgent(FailingProvider())
        observer = NemotronObserver(agent)
        reports = observer.observe(result)

        # Replay still completed.
        self.assertGreater(len(result.decisions), 0)
        # Fallback reports produced (no crash).
        self.assertGreater(len(reports), 0)
        for report in reports:
            self.assertEqual(report.confidence, 0.0)
            self.assertTrue(report.raw_response.get("fallback"))

    def test_research_inputs_not_mutated(self) -> None:
        """Nemotron research methods do not mutate their input dicts."""
        agent = NemotronResearchAgent(MockNemotronProvider())
        facts = {
            "price": 50000,
            "volatility": 0.02,
            "nested": {"regime": "ranging"},
        }
        before = dict(facts)
        before["nested"] = dict(facts["nested"])
        agent.research_market_summary(facts)
        self.assertEqual(facts, before)

        portfolio_facts = {"equity": 10000, "drawdown_pct": "0.02"}
        before_pf = dict(portfolio_facts)
        agent.research_portfolio_performance(portfolio_facts)
        self.assertEqual(portfolio_facts, before_pf)

    def test_reports_traceable_to_input_snapshots(self) -> None:
        """Analysis IDs are deterministic and linked to input snapshots."""
        agent = NemotronResearchAgent(MockNemotronProvider())
        facts_a = {"price": 50000, "volatility": 0.02}
        facts_b = {"price": 60000, "volatility": 0.05}

        report_a1 = agent.research_market_summary(facts_a)
        report_a2 = agent.research_market_summary(facts_a)
        report_b = agent.research_market_summary(facts_b)

        # Same input -> same analysis_id.
        self.assertEqual(report_a1.analysis_id, report_a2.analysis_id)
        # Different input -> different analysis_id.
        self.assertNotEqual(report_a1.analysis_id, report_b.analysis_id)
        # Model metadata recorded.
        self.assertIsInstance(report_a1.model_name, str)
        self.assertIsInstance(report_a1.model_version, str)
        self.assertIsInstance(report_a1.timestamp_utc, datetime)

    def test_observer_has_no_execution_tools(self) -> None:
        """Observer cannot modify orders, risk, or execution state."""
        observer = NemotronObserver()
        forbidden = [
            "place_order", "submit_order", "modify_position",
            "change_risk_limit", "approve_recommendation",
            "resume_pause", "execute", "trade", "pause",
        ]
        for method in forbidden:
            self.assertFalse(hasattr(observer, method), f"observer must not expose {method}")

    def test_explain_trading_decision_links_to_correlation_id(self) -> None:
        """explain_trading_decision receives a correlation_id and linked events."""
        agent = NemotronResearchAgent(MockNemotronProvider())
        correlation_id = "test-corr-123"
        events = [
            {"event_type": "selector_decision", "actor": "selector", "correlation_id": correlation_id},
            {"event_type": "risk_decision", "actor": "risk_engine", "correlation_id": correlation_id},
        ]
        report = agent.explain_trading_decision(correlation_id, events)
        self.assertEqual(report.topic, "explain_decision")
        self.assertIsInstance(report, ResearchReport)


class SnapshotImmutabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candles = flat_candles(8)
        self.agents = [
            ScheduledAgent("ema", SignalAction.ABSTAIN),
            ScheduledAgent("rsi", SignalAction.ABSTAIN),
            ScheduledAgent("mean", SignalAction.ABSTAIN),
            ScheduledAgent("breakout", SignalAction.ABSTAIN),
        ]
        self.result = PaperReplayEngine(RiskPolicy(), strategies=self.agents).run(self.candles)
        self.observer = NemotronObserver()
        self.snapshot = self.observer._build_snapshot(self.result)

    def test_nested_snapshot_dicts_cannot_be_mutated(self) -> None:
        """Nested dicts in ResearchSnapshot are read-only proxies."""
        with self.assertRaises((TypeError, ValueError)):
            self.snapshot.portfolio_snapshots[0]["equity_usdt"] = "tampered"
        with self.assertRaises((TypeError, ValueError)):
            self.snapshot.selector_decisions[0]["target"] = "tampered"
        with self.assertRaises((TypeError, ValueError)):
            self.snapshot.selector_statistics["total_decisions"] = 99999

    def test_nested_collections_cannot_be_mutated(self) -> None:
        """Nested collections inside snapshot dicts are protected."""
        # Tuple itself cannot be reassigned (frozen dataclass)
        with self.assertRaises((TypeError, AttributeError)):
            self.snapshot.portfolio_snapshots = ()
        # Individual dict proxies block item assignment (if any items exist)
        if self.snapshot.risk_interventions:
            with self.assertRaises((TypeError, ValueError)):
                self.snapshot.risk_interventions[0]["reason"] = "tampered"
        if self.snapshot.portfolio_snapshots:
            with self.assertRaises((TypeError, ValueError)):
                self.snapshot.portfolio_snapshots[0]["equity_usdt"] = "tampered"

    def test_snapshot_mutation_does_not_affect_later_research(self) -> None:
        """Attempts to mutate snapshots do not affect subsequent research calls."""
        # Capture original values before any mutation attempt
        original_equity = self.snapshot.portfolio_snapshots[0]["equity_usdt"]
        original_target = self.snapshot.selector_decisions[0]["target"]

        # Attempt mutation (will be blocked by MappingProxyType, but we verify state after)
        try:
            self.snapshot.portfolio_snapshots[0]["equity_usdt"] = "tampered"
        except (TypeError, ValueError):
            pass

        # Run research again — must produce same results as before
        agent = NemotronResearchAgent(MockNemotronProvider())
        facts_before = {
            "starting_equity_usdt": str(self.result.starting_equity_usdt),
            "ending_equity_usdt": str(self.result.ending_equity_usdt),
            "portfolio_snapshots": [dict(s) for s in self.snapshot.portfolio_snapshots],
            "halted": self.result.halted,
        }
        report_before = agent.research_portfolio_performance(facts_before)

        # Snapshot values remain unchanged after mutation attempt
        self.assertEqual(self.snapshot.portfolio_snapshots[0]["equity_usdt"], original_equity)
        self.assertEqual(self.snapshot.selector_decisions[0]["target"], original_target)

        # Research results are deterministic given same input
        report_after = agent.research_portfolio_performance(facts_before)
        self.assertEqual(report_before.analysis_id, report_after.analysis_id)
        self.assertEqual(report_before.confidence, report_after.confidence)


class SemanticValidationTests(unittest.TestCase):
    def test_empty_response_triggers_safe_fallback(self) -> None:
        """A response with no findings and empty summary triggers fallback."""
        class EmptyProvider:
            def model_info(self):
                return {"model_name": "empty", "model_version": "1", "base_url": "x"}
            def complete_json(self, *, system, user):
                return {"findings": [], "summary": "", "confidence": 0.0}

        agent = NemotronResearchAgent(EmptyProvider())
        report = agent.research_market_summary({"price": 50000})
        self.assertIsInstance(report, ResearchReport)
        self.assertEqual(report.confidence, 0.0)
        self.assertTrue(report.raw_response.get("fallback"))
        self.assertEqual(report.topic, "market_summary")

    def test_summary_without_findings_is_accepted(self) -> None:
        """A response with a valid summary and no findings is accepted per schema."""
        class SummaryOnlyProvider:
            def model_info(self):
                return {"model_name": "summary", "model_version": "1", "base_url": "x"}
            def complete_json(self, *, system, user):
                return {"findings": [], "summary": "No findings but a valid summary.", "confidence": 0.5}

        agent = NemotronResearchAgent(SummaryOnlyProvider())
        report = agent.research_market_summary({"price": 50000})
        self.assertIsInstance(report, ResearchReport)
        self.assertEqual(report.summary, "No findings but a valid summary.")
        self.assertEqual(len(report.findings), 0)

    def test_findings_without_summary_is_accepted(self) -> None:
        """A response with findings and empty summary is accepted per schema."""
        class FindingsOnlyProvider:
            def model_info(self):
                return {"model_name": "findings", "model_version": "1", "base_url": "x"}
            def complete_json(self, *, system, user):
                return {
                    "findings": [{"kind": "market_fact", "content": "price is 50000", "evidence_ref": "input.price"}],
                    "summary": "",
                    "confidence": 0.5,
                }

        agent = NemotronResearchAgent(FindingsOnlyProvider())
        report = agent.research_market_summary({"price": 50000})
        self.assertIsInstance(report, ResearchReport)
        self.assertEqual(len(report.findings), 1)
        self.assertEqual(report.findings[0].kind, FindingKind.MARKET_FACT)

    def test_deterministic_analysis_ids_stable_for_identical_inputs(self) -> None:
        """Identical inputs produce stable analysis IDs across calls."""
        agent = NemotronResearchAgent(MockNemotronProvider())
        facts = {"price": 50000, "volatility": 0.02}
        r1 = agent.research_market_summary(facts)
        r2 = agent.research_market_summary(facts)
        r3 = agent.research_market_summary(facts)
        self.assertEqual(r1.analysis_id, r2.analysis_id)
        self.assertEqual(r2.analysis_id, r3.analysis_id)


if __name__ == "__main__":
    unittest.main()
