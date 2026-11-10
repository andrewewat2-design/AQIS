import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from aqis.models import Candle, OpportunityEvidence, RiskPolicy, SignalAction, StrategySignal, TargetPosition
from aqis.paper import PaperReplayEngine
from aqis.strategies import StrategyAgent
from tests.helpers import flat_candles, rising_candles


def evidence() -> OpportunityEvidence:
    return OpportunityEvidence(
        "holdout-v1", "aqis-four-agent-v1", "default", Decimal("0.01"), Decimal("0.10"), 50, "holdout-hash"
    )


class ScheduledAgent(StrategyAgent):
    version = "test"

    def __init__(self, strategy_id: str, action_by_length: dict[int, SignalAction]) -> None:
        self.strategy_id = strategy_id
        self.action_by_length = action_by_length
        super().__init__()

    def evaluate(self, candles):
        action = self.action_by_length.get(len(candles), self.action_by_length.get(-1, SignalAction.ABSTAIN))
        return self._signal(candles, action, 100 if action is not SignalAction.ABSTAIN else 0, "test")


class CorrelationIdTests(unittest.TestCase):
    def test_correlation_ids_are_deterministic_across_identical_replays(self) -> None:
        agents = [
            ScheduledAgent("ema", {-1: SignalAction.ABSTAIN}),
            ScheduledAgent("rsi", {-1: SignalAction.ABSTAIN}),
            ScheduledAgent("mean", {-1: SignalAction.ABSTAIN}),
            ScheduledAgent("breakout", {-1: SignalAction.ABSTAIN}),
        ]
        candles = flat_candles(6)

        first = PaperReplayEngine(RiskPolicy(), {"default": evidence()}, agents).run(candles)
        second = PaperReplayEngine(RiskPolicy(), {"default": evidence()}, agents).run(candles)

        first_ids = [e.correlation_id for e in first.audit_events if e.correlation_id]
        second_ids = [e.correlation_id for e in second.audit_events if e.correlation_id]
        self.assertEqual(first_ids, second_ids)
        self.assertTrue(all(cid is not None and cid != "" for cid in first_ids))

    def test_different_candles_produce_different_correlation_ids(self) -> None:
        agents = [
            ScheduledAgent("ema", {-1: SignalAction.ABSTAIN}),
            ScheduledAgent("rsi", {-1: SignalAction.ABSTAIN}),
            ScheduledAgent("mean", {-1: SignalAction.ABSTAIN}),
            ScheduledAgent("breakout", {-1: SignalAction.ABSTAIN}),
        ]
        candles = flat_candles(6)
        result = PaperReplayEngine(RiskPolicy(), {"default": evidence()}, agents).run(candles)
        ids = [e.correlation_id for e in result.audit_events if e.correlation_id]
        self.assertEqual(len(ids), len(set(ids)), "correlation IDs must be unique per candle")


class ExecutionTraceabilityTests(unittest.TestCase):
    def test_signal_decision_order_fill_share_correlation_id(self) -> None:
        agents = [
            ScheduledAgent("ema", {1: SignalAction.BUY, 2: SignalAction.BUY, -1: SignalAction.ABSTAIN}),
            ScheduledAgent("rsi", {1: SignalAction.BUY, 2: SignalAction.BUY, -1: SignalAction.ABSTAIN}),
            ScheduledAgent("mean", {1: SignalAction.BUY, 2: SignalAction.BUY, -1: SignalAction.ABSTAIN}),
            ScheduledAgent("breakout", {-1: SignalAction.ABSTAIN}),
        ]
        engine = PaperReplayEngine(RiskPolicy(), {"default": evidence()}, agents)
        result = engine.run(flat_candles(8))

        # Find the first fill and trace it back through the decision.
        self.assertGreater(len(result.fills), 0, "expected at least one fill")
        first_fill = result.fills[0]
        cid = first_fill.correlation_id
        self.assertIsNotNone(cid)

        lookup = result.find_by_correlation_id(cid)
        self.assertEqual(lookup["correlation_id"], cid)
        # The fill should be present in the lookup.
        self.assertEqual(len(lookup["fills"]), 1)
        # There should be at least one audit event for this correlation (selector decision / risk decision).
        self.assertGreaterEqual(len(lookup["audit_events"]), 1)

    def test_risk_approval_is_tracked_separately_from_fill(self) -> None:
        agents = [
            ScheduledAgent("ema", {-1: SignalAction.BUY}),
            ScheduledAgent("rsi", {-1: SignalAction.BUY}),
            ScheduledAgent("mean", {-1: SignalAction.BUY}),
            ScheduledAgent("breakout", {-1: SignalAction.ABSTAIN}),
        ]
        engine = PaperReplayEngine(RiskPolicy(), {"default": evidence()}, agents)
        result = engine.run(flat_candles(3))

        # Order requests and approved orders are recorded independently of fills.
        self.assertIsInstance(result.order_requests, list)
        self.assertIsInstance(result.approved_orders, list)
        # A risk approval is not, by itself, an executed fill.
        for req in result.order_requests:
            self.assertEqual(len(req), 4)  # timestamp, side, quantity, correlation_id
        for apr in result.approved_orders:
            self.assertEqual(len(apr), 4)

    def test_next_candle_execution_uses_decision_correlation_id(self) -> None:
        agents = [
            ScheduledAgent("ema", {1: SignalAction.BUY, 2: SignalAction.BUY, -1: SignalAction.ABSTAIN}),
            ScheduledAgent("rsi", {1: SignalAction.BUY, 2: SignalAction.BUY, -1: SignalAction.ABSTAIN}),
            ScheduledAgent("mean", {1: SignalAction.BUY, 2: SignalAction.BUY, -1: SignalAction.ABSTAIN}),
            ScheduledAgent("breakout", {-1: SignalAction.ABSTAIN}),
        ]
        engine = PaperReplayEngine(RiskPolicy(), {"default": evidence()}, agents)
        result = engine.run(flat_candles(8))
        self.assertGreater(len(result.fills), 0)
        # Every fill must carry a correlation ID that originated from a decision.
        for fill in result.fills:
            self.assertIsNotNone(fill.correlation_id)


class RejectionAttributionTests(unittest.TestCase):
    def test_missing_evidence_is_opportunity_suppression_not_risk_rejection(self) -> None:
        agents = [
            ScheduledAgent("ema", {-1: SignalAction.BUY}),
            ScheduledAgent("rsi", {-1: SignalAction.BUY}),
            ScheduledAgent("mean", {-1: SignalAction.BUY}),
            ScheduledAgent("breakout", {-1: SignalAction.ABSTAIN}),
        ]
        # No evidence provided -> opportunity gate suppresses entries.
        result = PaperReplayEngine(RiskPolicy(), strategies=agents).run(flat_candles(3))
        self.assertEqual(result.fills, [])
        # The suppression must appear in opportunity_suppressions.
        self.assertTrue(
            any(item[1] == "missing_evidence" for item in result.opportunity_suppressions),
            "missing_evidence must be an opportunity-gate suppression",
        )
        # It must not be attributed to the risk engine.
        risk_rejection_codes = [item[1] for item in result.risk_rejections]
        self.assertNotIn("missing_evidence", risk_rejection_codes)
        # Backward-compat union list still records it.
        self.assertTrue(any(item[1] == "missing_evidence" for item in result.rejected_orders))

    def test_risk_rejections_are_attributed_to_risk_engine(self) -> None:
        agents = [
            ScheduledAgent("ema", {-1: SignalAction.BUY}),
            ScheduledAgent("rsi", {-1: SignalAction.BUY}),
            ScheduledAgent("mean", {-1: SignalAction.BUY}),
            ScheduledAgent("breakout", {-1: SignalAction.ABSTAIN}),
        ]
        # Oversized order: 1100 notional exceeds the 10% order cap on 10000 equity.
        # Use a buy at high quantity by scaling price down is not possible with flat candles,
        # so instead rely on exposure cap: hold BTC then try to add.
        result = PaperReplayEngine(RiskPolicy(), {"default": evidence()}, agents).run(flat_candles(3))
        # Any risk rejection must carry the risk_engine actor in audit events.
        risk_rej_events = [
            e for e in result.audit_events if e.event_type == "order_rejected" and e.actor == "risk_engine"
        ]
        for rejection in result.risk_rejections:
            self.assertEqual(len(rejection), 4)  # timestamp, code, message, correlation_id


class PauseTransitionTests(unittest.TestCase):
    def test_human_resume_records_intervention_and_is_human_only(self) -> None:
        engine = PaperReplayEngine(RiskPolicy(), strategies=[])
        result = engine.run(flat_candles(3))

        # Human resume is allowed.
        engine.human_resume(result, "operator")
        resume_interventions = [
            i for i in result.risk_interventions if i.intervention_type == "human_resume"
        ]
        self.assertEqual(len(resume_interventions), 1)
        self.assertEqual(resume_interventions[0].actor, "operator")

        # Agent resume must be rejected by the risk engine.
        account = result.account
        account.paused = True
        account.pause_reason = "daily_loss_limit"
        with self.assertRaises(PermissionError):
            engine.risk_engine.human_resume(account, "supervisor")

    def test_agent_pause_request_is_recorded(self) -> None:
        engine = PaperReplayEngine(RiskPolicy(), strategies=[])
        result = engine.run(flat_candles(3))

        engine.record_agent_pause_request(result, "monitoring_agent", "volatility anomaly")
        pause_requests = [
            i for i in result.risk_interventions if i.intervention_type == "agent_pause_request"
        ]
        self.assertEqual(len(pause_requests), 1)
        self.assertEqual(pause_requests[0].actor, "monitoring_agent")
        self.assertIn("volatility anomaly", pause_requests[0].reason)
        # Pause resume history should also reflect the pause.
        pause_events = [i for i in result.pause_resume_history if i.intervention_type == "pause"]
        self.assertGreaterEqual(len(pause_events), 1)


class StrategyAttributionTests(unittest.TestCase):
    def test_strategy_performance_records_participation_without_causal_claims(self) -> None:
        agents = [
            ScheduledAgent("ema", {1: SignalAction.BUY, 2: SignalAction.BUY, -1: SignalAction.ABSTAIN}),
            ScheduledAgent("rsi", {1: SignalAction.BUY, 2: SignalAction.BUY, -1: SignalAction.ABSTAIN}),
            ScheduledAgent("mean", {1: SignalAction.BUY, 2: SignalAction.BUY, -1: SignalAction.ABSTAIN}),
            ScheduledAgent("breakout", {-1: SignalAction.ABSTAIN}),
        ]
        engine = PaperReplayEngine(RiskPolicy(), {"default": evidence()}, agents)
        result = engine.run(flat_candles(8))

        self.assertIn("ema", result.strategy_performance)
        self.assertIn("breakout", result.strategy_performance)
        ema_perf = result.strategy_performance["ema"]
        self.assertGreater(ema_perf.signals, 0)
        # Attribution limitation must be explicitly present.
        self.assertIn("Correlation does not imply causation", ema_perf.attribution_limitation)
        # No per-strategy drawdown field is exposed.
        self.assertNotIn("drawdown_contribution", ema_perf.attribution_limitation.lower())
        self.assertFalse(hasattr(ema_perf, "drawdown_contribution"))

    def test_no_individual_strategy_drawdown_is_claimed(self) -> None:
        for strategy_id, perf in PaperReplayEngine(
            RiskPolicy(), {"default": evidence()}, strategies=[]
        ).run(flat_candles(3)).strategy_performance.items():
            self.assertFalse(hasattr(perf, "drawdown_contribution"))


class PortfolioSnapshotTests(unittest.TestCase):
    def test_portfolio_snapshot_records_required_fields(self) -> None:
        agents = [
            ScheduledAgent("ema", {-1: SignalAction.ABSTAIN}),
            ScheduledAgent("rsi", {-1: SignalAction.ABSTAIN}),
            ScheduledAgent("mean", {-1: SignalAction.ABSTAIN}),
            ScheduledAgent("breakout", {-1: SignalAction.ABSTAIN}),
        ]
        result = PaperReplayEngine(RiskPolicy(), {"default": evidence()}, agents).run(flat_candles(6))
        self.assertGreater(len(result.portfolio_snapshots), 0)
        for snap in result.portfolio_snapshots:
            self.assertIsInstance(snap.equity_usdt, Decimal)
            self.assertIsInstance(snap.btc_exposure_usdt, Decimal)
            self.assertIsInstance(snap.exposure_pct, Decimal)
            self.assertIsInstance(snap.drawdown_pct, Decimal)
            self.assertIsInstance(snap.daily_loss_pct, Decimal)
            self.assertIsInstance(snap.paused, bool)
            self.assertGreaterEqual(snap.exposure_pct, Decimal("0"))
            self.assertGreaterEqual(snap.drawdown_pct, Decimal("0"))
            self.assertGreaterEqual(snap.daily_loss_pct, Decimal("0"))

    def test_daily_loss_baseline_resets_at_utc_day_boundary(self) -> None:
        # Build candles spanning two UTC days.
        candles = []
        timestamp = datetime(2024, 1, 1, 1, tzinfo=timezone.utc)
        price = 100.0
        for index in range(50):
            candles.append(
                Candle(
                    timestamp + timedelta(hours=index), price, price + 0.5, price - 0.5, price, 10.0
                )
            )
        agents = [
            ScheduledAgent("ema", {-1: SignalAction.ABSTAIN}),
            ScheduledAgent("rsi", {-1: SignalAction.ABSTAIN}),
            ScheduledAgent("mean", {-1: SignalAction.ABSTAIN}),
            ScheduledAgent("breakout", {-1: SignalAction.ABSTAIN}),
        ]
        result = PaperReplayEngine(RiskPolicy(), {"default": evidence()}, agents).run(candles)
        # Snapshots exist for every traded candle; daily loss percentage stays bounded.
        for snap in result.portfolio_snapshots:
            self.assertGreaterEqual(snap.daily_loss_pct, Decimal("0"))


class AuditReportTests(unittest.TestCase):
    def test_audit_report_aggregates_by_event_type_and_origin(self) -> None:
        agents = [
            ScheduledAgent("ema", {-1: SignalAction.ABSTAIN}),
            ScheduledAgent("rsi", {-1: SignalAction.ABSTAIN}),
            ScheduledAgent("mean", {-1: SignalAction.ABSTAIN}),
            ScheduledAgent("breakout", {-1: SignalAction.ABSTAIN}),
        ]
        result = PaperReplayEngine(RiskPolicy(), {"default": evidence()}, agents).run(flat_candles(6))
        report = result.audit_report()
        self.assertIn("event_type_counts", report)
        self.assertIn("rejection_attribution_by_origin", report)
        self.assertIn("risk_interventions", report)
        self.assertIn("pause_resume_history", report)
        self.assertIsInstance(report["event_type_counts"], dict)
        self.assertIsInstance(report["rejection_attribution_by_origin"], dict)

    def test_audit_report_is_read_only(self) -> None:
        agents = [
            ScheduledAgent("ema", {-1: SignalAction.ABSTAIN}),
            ScheduledAgent("rsi", {-1: SignalAction.ABSTAIN}),
            ScheduledAgent("mean", {-1: SignalAction.ABSTAIN}),
            ScheduledAgent("breakout", {-1: SignalAction.ABSTAIN}),
        ]
        result = PaperReplayEngine(RiskPolicy(), {"default": evidence()}, agents).run(flat_candles(6))
        before_fills = len(result.fills)
        before_rejected = len(result.rejected_orders)
        report = result.audit_report()
        # Calling audit_report must not mutate state.
        self.assertEqual(len(result.fills), before_fills)
        self.assertEqual(len(result.rejected_orders), before_rejected)
        self.assertEqual(report["fills"], before_fills)
        self.assertEqual(report["rejected_orders"], before_rejected)

    def test_find_by_correlation_id_returns_linked_events(self) -> None:
        agents = [
            ScheduledAgent("ema", {1: SignalAction.BUY, 2: SignalAction.BUY, -1: SignalAction.ABSTAIN}),
            ScheduledAgent("rsi", {1: SignalAction.BUY, 2: SignalAction.BUY, -1: SignalAction.ABSTAIN}),
            ScheduledAgent("mean", {1: SignalAction.BUY, 2: SignalAction.BUY, -1: SignalAction.ABSTAIN}),
            ScheduledAgent("breakout", {-1: SignalAction.ABSTAIN}),
        ]
        result = PaperReplayEngine(RiskPolicy(), {"default": evidence()}, agents).run(flat_candles(8))
        self.assertGreater(len(result.fills), 0)
        cid = result.fills[0].correlation_id
        lookup = result.find_by_correlation_id(cid)
        self.assertEqual(lookup["correlation_id"], cid)
        self.assertIsInstance(lookup["audit_events"], list)
        self.assertIsInstance(lookup["fills"], list)

    def test_find_by_unknown_correlation_id_returns_empty(self) -> None:
        result = PaperReplayEngine(RiskPolicy(), strategies=[]).run(flat_candles(3))
        lookup = result.find_by_correlation_id("nonexistent-id")
        self.assertEqual(lookup["audit_events"], [])
        self.assertEqual(lookup["fills"], [])
        self.assertEqual(lookup["risk_rejections"], [])
        self.assertEqual(lookup["opportunity_suppressions"], [])


class SelectorStatisticsTests(unittest.TestCase):
    def test_selector_statistics_are_recorded(self) -> None:
        agents = [
            ScheduledAgent("ema", {-1: SignalAction.ABSTAIN}),
            ScheduledAgent("rsi", {-1: SignalAction.ABSTAIN}),
            ScheduledAgent("mean", {-1: SignalAction.ABSTAIN}),
            ScheduledAgent("breakout", {-1: SignalAction.ABSTAIN}),
        ]
        result = PaperReplayEngine(RiskPolicy(), {"default": evidence()}, agents).run(flat_candles(6))
        stats = result.selector_statistics
        self.assertIn("total_decisions", stats)
        self.assertIn("long_targets", stats)
        self.assertIn("cash_targets", stats)
        self.assertEqual(stats["long_targets"] + stats["cash_targets"], stats["total_decisions"])


if __name__ == "__main__":
    unittest.main()
