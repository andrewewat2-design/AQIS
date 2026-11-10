import unittest
from dataclasses import FrozenInstanceError
from decimal import Decimal
from datetime import datetime, timezone
import sqlite3

from aqis.models import OrderIntent, OrderSide, PaperAccount, RiskPolicy
from aqis.risk import IntegrityStatus, PersistentPolicyRegistry, PolicyRegistry, RiskEngine


def buy_intent(notional: str = "500") -> OrderIntent:
    return OrderIntent(
        OrderSide.BUY,
        Decimal(notional) / Decimal("100"),
        Decimal("100"),
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        net_edge_lower_bound_pct=Decimal("0.001"),
        stressed_loss_fraction=Decimal("0.10"),
        evidence_id="validation-v1",
    )


class RiskEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = RiskPolicy()
        self.engine = RiskEngine(self.policy)
        self.account = PaperAccount(
            cash_usdt=Decimal("10000"),
            high_watermark_usdt=Decimal("10000"),
            utc_day_start_equity_usdt=Decimal("10000"),
        )

    def test_approved_order_with_edge_and_risk_evidence(self) -> None:
        result = self.engine.evaluate_order(buy_intent(), self.account)
        self.assertTrue(result.approved)

    def test_missing_edge_evidence_fails_closed(self) -> None:
        order = OrderIntent(OrderSide.BUY, Decimal("1"), Decimal("100"), datetime.now(timezone.utc))
        result = self.engine.evaluate_order(order, self.account)
        self.assertFalse(result.approved)
        self.assertEqual(result.reason_code, "missing_opportunity_evidence")

    def test_order_and_exposure_limits_are_independent(self) -> None:
        order_too_large = buy_intent("1100")
        self.assertEqual(self.engine.evaluate_order(order_too_large, self.account).reason_code, "max_order_exceeded")

        account = PaperAccount(
            cash_usdt=Decimal("8000"),
            btc=Decimal("20"),
            high_watermark_usdt=Decimal("10000"),
            utc_day_start_equity_usdt=Decimal("10000"),
        )
        self.assertEqual(self.engine.evaluate_order(buy_intent("100"), account).reason_code, "max_exposure_exceeded")

    def test_loss_stop_latches_pause_and_allows_reduce_only_close(self) -> None:
        account = PaperAccount(
            cash_usdt=Decimal("0"),
            btc=Decimal("90"),
            high_watermark_usdt=Decimal("10000"),
            utc_day_start_equity_usdt=Decimal("10000"),
        )
        reason = self.engine.pause_if_required(account, Decimal("100"))
        self.assertEqual(reason, "daily_loss_limit")
        buy_result = self.engine.evaluate_order(buy_intent("100"), account)
        self.assertEqual(buy_result.reason_code, "policy_paused")
        sell = OrderIntent(OrderSide.SELL, Decimal("5"), Decimal("100"), datetime.now(timezone.utc), reduce_only=True)
        self.assertTrue(self.engine.evaluate_order(sell, account).approved)

    def test_data_integrity_halt_blocks_even_reduce_only_order(self) -> None:
        account = PaperAccount(cash_usdt=Decimal("1000"), btc=Decimal("1"), paused=True, pause_reason="data_integrity:gap")
        sell = OrderIntent(OrderSide.SELL, Decimal("0.1"), Decimal("100"), datetime.now(timezone.utc), reduce_only=True)
        result = self.engine.evaluate_order(sell, account)
        self.assertFalse(result.approved)
        self.assertEqual(result.reason_code, "integrity_halt")

    def test_policy_is_immutable_and_requires_human_activation(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            self.policy.version = 2  # type: ignore[misc]
        registry = PolicyRegistry()
        with self.assertRaises(PermissionError):
            registry.activate(self.policy, "risk_agent")
        registry.activate(self.policy, "operator")
        with self.assertRaises(ValueError):
            registry.activate(self.policy, "operator")
        self.assertEqual(registry.active(self.policy.policy_id), self.policy)

    def test_persistent_policy_registry_keeps_approved_versions(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            registry = PersistentPolicyRegistry(str(Path(directory) / "policy.sqlite3"))
            with self.assertRaises(PermissionError):
                registry.activate(self.policy, "supervisor")
            registry.activate(self.policy, "operator")
            self.assertEqual(registry.active(self.policy.policy_id), self.policy)
            updated = RiskPolicy(version=2, max_order_pct=Decimal("0.08"))
            registry.activate(updated, "operator")
            self.assertEqual(len(registry.versions(self.policy.policy_id)), 2)
            self.assertEqual(registry.active(self.policy.policy_id).version, 2)
            with self.assertRaises(sqlite3.IntegrityError):
                registry.connection.execute(
                    "UPDATE risk_policy_versions SET payload='{}' WHERE version=1"
                )
            registry.close()

    def test_human_only_resume(self) -> None:
        self.account.paused = True
        self.account.pause_reason = "daily_loss_limit"
        with self.assertRaises(PermissionError):
            self.engine.human_resume(self.account, "supervisor")
        self.engine.human_resume(self.account, "operator")
        self.assertFalse(self.account.paused)

    def test_monitor_agent_can_only_latch_pause_not_resume(self) -> None:
        self.engine.apply_agent_pause_request(self.account, "monitoring_agent", "slippage anomaly")
        self.assertTrue(self.account.paused)
        self.assertTrue(self.account.pause_reason.startswith("agent_requested_pause"))
        with self.assertRaises(PermissionError):
            self.engine.human_resume(self.account, "monitoring_agent")
