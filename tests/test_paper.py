import unittest
from datetime import timedelta
from decimal import Decimal

from aqis.models import OpportunityEvidence, RiskPolicy, SignalAction, StrategySignal, TargetPosition
from aqis.paper import PaperReplayEngine
from aqis.strategies import StrategyAgent
from tests.helpers import flat_candles


class ScheduledAgent(StrategyAgent):
    version = "test"

    def __init__(self, strategy_id: str, action_by_length: dict[int, SignalAction]) -> None:
        self.strategy_id = strategy_id
        self.action_by_length = action_by_length
        super().__init__()

    def evaluate(self, candles):
        action = self.action_by_length.get(len(candles), self.action_by_length.get(-1, SignalAction.ABSTAIN))
        return self._signal(candles, action, 100 if action is not SignalAction.ABSTAIN else 0, "test")


def evidence() -> OpportunityEvidence:
    return OpportunityEvidence(
        "holdout-v1", "aqis-four-agent-v1", "default", Decimal("0.01"), Decimal("0.10"), 50, "holdout-hash"
    )


class PaperReplayTests(unittest.TestCase):
    def test_three_buy_votes_open_and_no_majority_closes_to_cash(self) -> None:
        agents = [
            ScheduledAgent("ema", {1: SignalAction.BUY, 2: SignalAction.BUY, -1: SignalAction.ABSTAIN}),
            ScheduledAgent("rsi", {1: SignalAction.BUY, 2: SignalAction.BUY, -1: SignalAction.ABSTAIN}),
            ScheduledAgent("mean", {1: SignalAction.BUY, 2: SignalAction.BUY, -1: SignalAction.ABSTAIN}),
            ScheduledAgent("breakout", {-1: SignalAction.ABSTAIN}),
        ]
        engine = PaperReplayEngine(RiskPolicy(), {"default": evidence()}, agents)
        result = engine.run(flat_candles(8))
        self.assertEqual([fill.side.value for fill in result.fills[:2]], ["buy", "buy"])
        self.assertTrue(all(fill.side.value == "sell" for fill in result.fills[2:]))
        self.assertEqual(result.account.btc, Decimal("0"))
        self.assertGreater(result.account.fees_usdt, Decimal("0"))
        self.assertGreater(result.account.slippage_usdt, Decimal("0"))
        self.assertLess(result.ending_equity_usdt, result.starting_equity_usdt)
        self.assertEqual(len(result.completed_trade_returns_pct), 1)
        self.assertLess(result.completed_trade_returns_pct[0], Decimal("0"))

    def test_missing_validation_evidence_fails_closed_for_entries(self) -> None:
        agents = [
            ScheduledAgent("ema", {-1: SignalAction.BUY}),
            ScheduledAgent("rsi", {-1: SignalAction.BUY}),
            ScheduledAgent("mean", {-1: SignalAction.BUY}),
            ScheduledAgent("breakout", {-1: SignalAction.ABSTAIN}),
        ]
        result = PaperReplayEngine(RiskPolicy(), strategies=agents).run(flat_candles(3))
        self.assertEqual(result.fills, [])
        self.assertTrue(any(item[1] == "missing_evidence" for item in result.rejected_orders))

    def test_validation_mode_runs_hard_limits_without_promotable_evidence(self) -> None:
        agents = [
            ScheduledAgent("ema", {-1: SignalAction.BUY}),
            ScheduledAgent("rsi", {-1: SignalAction.BUY}),
            ScheduledAgent("mean", {-1: SignalAction.BUY}),
            ScheduledAgent("breakout", {-1: SignalAction.ABSTAIN}),
        ]
        result = PaperReplayEngine(RiskPolicy(), strategies=agents, validation_mode=True).run(flat_candles(3))
        self.assertGreater(len(result.fills), 0)

    def test_gap_fails_before_any_order_can_be_simulated(self) -> None:
        candles = flat_candles(4)
        candles[2] = type(candles[2])(
            candles[2].timestamp + timedelta(hours=1),
            candles[2].open,
            candles[2].high,
            candles[2].low,
            candles[2].close,
            candles[2].volume,
        )
        with self.assertRaisesRegex(ValueError, "missing or non-hourly"):
            PaperReplayEngine(RiskPolicy()).run(candles)
