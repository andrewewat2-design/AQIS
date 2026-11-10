from __future__ import annotations

from collections.abc import Iterable

from aqis.models import SelectorDecision, SignalAction, StrategySignal, TargetPosition
from aqis.regime import RegimeAssessment


class ConsensusSelector:
    """Equal votes, regime-aware confidence threshold, 3-of-4 quorum; no majority targets cash.

    The risk engine remains the authoritative hard gate; regime adjustments only
    modulate the selector's vote threshold, never override risk-engine rejections.
    """

    REGIME_THRESHOLDS: dict[str, float] = {
        "trending": 60.0,
        "ranging": 65.0,
        "high_volatility": 60.0,
        "warming_up": 65.0,
    }

    def __init__(
        self,
        confidence_threshold: float = 60.0,
        quorum: int = 3,
        *,
        regime: RegimeAssessment | None = None,
    ) -> None:
        if not 0 <= confidence_threshold <= 100:
            raise ValueError("confidence threshold must be in [0, 100]")
        if quorum < 1:
            raise ValueError("quorum must be positive")
        self.confidence_threshold = confidence_threshold
        self.quorum = quorum
        self.regime = regime

    def _effective_threshold(self) -> float:
        if self.regime is None:
            return self.confidence_threshold
        regime_name = self.regime.regime
        return self.REGIME_THRESHOLDS.get(regime_name, self.confidence_threshold)

    def decide(self, signals: Iterable[StrategySignal]) -> SelectorDecision:
        items = tuple(signals)
        if not items:
            raise ValueError("selector requires strategy signals")
        timestamps = {signal.timestamp for signal in items}
        if len(timestamps) != 1:
            raise ValueError("all votes must refer to the same completed candle")
        ids = [signal.strategy_id for signal in items]
        if len(set(ids)) != len(ids):
            raise ValueError("each strategy may cast at most one vote per candle")

        effective_thresh = self._effective_threshold()

        buys = tuple(
            signal.strategy_id
            for signal in items
            if signal.action is SignalAction.BUY
            and signal.confidence >= effective_thresh
        )
        sells = tuple(
            signal.strategy_id
            for signal in items
            if signal.action is SignalAction.SELL
            and signal.confidence >= effective_thresh
        )
        abstentions = tuple(
            signal.strategy_id
            for signal in items
            if signal.strategy_id not in buys and signal.strategy_id not in sells
        )

        if len(buys) >= self.quorum:
            target = TargetPosition.LONG
            reason = "buy_quorum"
        elif len(sells) >= self.quorum:
            target = TargetPosition.CASH
            reason = "sell_quorum"
        else:
            target = TargetPosition.CASH
            reason = "no_clear_majority_cash_target"

        return SelectorDecision(
            timestamp=items[0].timestamp,
            target=target,
            buy_votes=buys,
            sell_votes=sells,
            abstentions=abstentions,
            reason=reason,
        )
