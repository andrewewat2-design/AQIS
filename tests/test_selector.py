import unittest
from datetime import datetime, timezone

from aqis.models import SignalAction, StrategySignal, TargetPosition
from aqis.selector import ConsensusSelector


def vote(name: str, action: SignalAction, confidence: float = 80.0) -> StrategySignal:
    return StrategySignal(name, "1.0.0", datetime(2024, 1, 1, tzinfo=timezone.utc), action, confidence, "test")


class SelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.selector = ConsensusSelector()

    def test_three_valid_buys_target_long(self) -> None:
        result = self.selector.decide([
            vote("ema", SignalAction.BUY),
            vote("rsi", SignalAction.BUY),
            vote("mean", SignalAction.BUY),
            vote("breakout", SignalAction.SELL),
        ])
        self.assertEqual(result.target, TargetPosition.LONG)
        self.assertEqual(len(result.buy_votes), 3)

    def test_three_valid_sells_target_cash(self) -> None:
        result = self.selector.decide([
            vote("ema", SignalAction.SELL),
            vote("rsi", SignalAction.SELL),
            vote("mean", SignalAction.SELL),
            vote("breakout", SignalAction.BUY),
        ])
        self.assertEqual(result.target, TargetPosition.CASH)
        self.assertEqual(result.reason, "sell_quorum")

    def test_tie_and_abstention_target_cash(self) -> None:
        result = self.selector.decide([
            vote("ema", SignalAction.BUY),
            vote("rsi", SignalAction.BUY),
            vote("mean", SignalAction.SELL),
            vote("breakout", SignalAction.SELL),
        ])
        self.assertEqual(result.target, TargetPosition.CASH)
        self.assertEqual(result.reason, "no_clear_majority_cash_target")

    def test_confidence_below_sixty_abstains(self) -> None:
        result = self.selector.decide([
            vote("ema", SignalAction.BUY, 60),
            vote("rsi", SignalAction.BUY, 60),
            vote("mean", SignalAction.BUY, 59.99),
            vote("breakout", SignalAction.ABSTAIN, 100),
        ])
        self.assertEqual(result.target, TargetPosition.CASH)
        self.assertEqual(result.buy_votes, ("ema", "rsi"))
        self.assertEqual(set(result.abstentions), {"mean", "breakout"})

    def test_duplicate_or_misaligned_votes_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.selector.decide([vote("ema", SignalAction.BUY), vote("ema", SignalAction.SELL)])
        other = StrategySignal("rsi", "1.0.0", datetime(2024, 1, 2, tzinfo=timezone.utc), SignalAction.BUY, 90, "test")
        with self.assertRaises(ValueError):
            self.selector.decide([vote("ema", SignalAction.BUY), other])
