import unittest
from datetime import datetime, timezone
from decimal import Decimal

from aqis.agents import (
    AQISAgentOrchestrator,
    BacktestingAgent,
    DeploymentAgent,
    MonitoringAgent,
    Recommendation,
    RiskEvaluationAgent,
    StrategyResearchAgent,
    SupervisorAgent,
)
from aqis.audit import AuditLog
from aqis.models import AuditEvent, RiskPolicy
from aqis.opportunity import OpportunityGate
from aqis.validation import build_validation_report


class FakeProvider:
    def __init__(self, payload):
        self.payload = payload

    def complete_json(self, *, system: str, user: str):
        return self.payload


class AgentValidationAuditTests(unittest.TestCase):
    def test_agents_are_advisory_and_schema_checked(self) -> None:
        review = RiskEvaluationAgent().review({"drawdown": "0.01"})
        self.assertEqual(review.recommendation, Recommendation.NO_CHANGE)
        proposal = StrategyResearchAgent().propose({"market": "BTC/USDT"})
        self.assertIsNone(proposal)
        fake = FakeProvider({
            "recommendation": "request_pause",
            "summary": "risk changed",
            "evidence": ["volatility increased"],
            "confidence": 0.8,
        })
        self.assertEqual(RiskEvaluationAgent(fake).review({}).recommendation, Recommendation.REQUEST_PAUSE)
        self.assertEqual(SupervisorAgent().review({}).recommendation, Recommendation.NO_CHANGE)
        self.assertEqual(
            MonitoringAgent().review({"hard_limit_breached": True, "triggers": ["drawdown"]}).recommendation,
            Recommendation.REQUEST_PAUSE,
        )
        self.assertEqual(
            DeploymentAgent().review({"validation_passed": True, "human_approved": False}).recommendation,
            Recommendation.REJECT,
        )
        self.assertEqual(
            BacktestingAgent().review({"validation_passed": True}).recommendation,
            Recommendation.APPROVE_FOR_HUMAN_REVIEW,
        )

    def test_orchestrator_returns_recommendations_without_mutating_policy(self) -> None:
        orchestrator = AQISAgentOrchestrator(
            RiskEvaluationAgent(), MonitoringAgent(), SupervisorAgent()
        )
        round_result = orchestrator.review_snapshot({"equity_usdt": "10000", "findings": []})
        self.assertEqual(round_result.risk_review.agent_id, "risk_evaluation_agent")
        self.assertEqual(round_result.monitoring_review.agent_id, "monitoring_agent")
        self.assertEqual(round_result.supervisor_review.agent_id, "supervisor_agent")

    def test_validation_evidence_is_frozen_and_gate_checks_sample(self) -> None:
        returns = [Decimal("0.01")] * 38 + [Decimal("-0.01")] * 2
        report = build_validation_report(
            report_id="v1-holdout",
            strategy_set_version="aqis-four-agent-v1",
            regime="default",
            net_trade_returns_pct=returns,
            data_fingerprint="sha256:bars",
        )
        self.assertEqual(report.evidence.validation_trade_count, 40)
        self.assertTrue(report.evidence.data_hash)
        self.assertIsInstance(report.passed, bool)
        decision = OpportunityGate(RiskPolicy()).assess(report.evidence)
        self.assertEqual(decision.eligible, report.evidence.net_edge_lower_bound_pct > 0)

    def test_audit_log_appends_json_events(self) -> None:
        log = AuditLog()
        log.append(datetime.now(timezone.utc), "policy_activated", "operator", {"version": 1, "cap": Decimal("0.2")})
        events = log.events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["payload"]["cap"], "0.2")
        import sqlite3

        with self.assertRaises(sqlite3.IntegrityError):
            log.connection.execute("DELETE FROM audit_events")
        log.close()
