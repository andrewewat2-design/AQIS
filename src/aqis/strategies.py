from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from aqis.indicators import atr, bollinger, donchian, ema, rsi
from aqis.models import Candle, SignalAction, StrategySignal


CONFIDENCE_THRESHOLD = 60.0


def _confidence(margin: float, scale: float) -> float:
    if margin <= 0 or scale <= 0:
        return 0.0
    return min(100.0, max(0.0, 100.0 * margin / scale))


class StrategyAgent(ABC):
    """Independent deterministic runtime strategy; research agents propose versions."""

    strategy_id: str
    version: str

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._active_confidence = 0.0

    @abstractmethod
    def evaluate(self, candles: Sequence[Candle]) -> StrategySignal:
        raise NotImplementedError

    def _signal(
        self,
        candles: Sequence[Candle],
        action: SignalAction,
        confidence: float,
        reason: str,
        strength: float | None = None,
    ) -> StrategySignal:
        timestamp = candles[-1].timestamp
        if confidence < CONFIDENCE_THRESHOLD:
            action = SignalAction.ABSTAIN
        return StrategySignal(
            strategy_id=self.strategy_id,
            strategy_version=self.version,
            timestamp=timestamp,
            action=action,
            confidence=confidence,
            reason=reason,
            strength=strength,
        )


class EmaCrossoverAgent(StrategyAgent):
    strategy_id = "ema_crossover"
    version = "1.0.0"
    fast_period = 20
    slow_period = 50
    atr_period = 14
    confidence_scale_atr = 1.0

    def evaluate(self, candles: Sequence[Candle]) -> StrategySignal:
        closes = [candle.close for candle in candles]
        fast = ema(closes, self.fast_period)
        slow = ema(closes, self.slow_period)
        volatility = atr(candles, self.atr_period)
        if fast is None or slow is None or volatility is None or volatility <= 0:
            return self._signal(candles, SignalAction.ABSTAIN, 0, "indicator_warmup")
        spread = (fast - slow) / volatility
        confidence = _confidence(abs(spread), self.confidence_scale_atr)
        action = SignalAction.BUY if spread > 0 else SignalAction.SELL
        return self._signal(
            candles,
            action,
            confidence,
            "ema20_above_ema50" if spread > 0 else "ema20_at_or_below_ema50",
            spread,
        )


class RsiMomentumAgent(StrategyAgent):
    strategy_id = "rsi_momentum"
    version = "1.0.0"
    period = 14
    buy_threshold = 55.0
    sell_threshold = 45.0
    confidence_scale_points = 15.0

    def evaluate(self, candles: Sequence[Candle]) -> StrategySignal:
        closes = [candle.close for candle in candles]
        current = rsi(closes, self.period)
        previous = rsi(closes[:-1], self.period) if len(closes) > self.period + 1 else None
        if current is None or previous is None:
            return self._signal(candles, SignalAction.ABSTAIN, 0, "indicator_warmup")
        if current >= self.buy_threshold and current > previous:
            margin = current - self.buy_threshold
            return self._signal(
                candles,
                SignalAction.BUY,
                _confidence(margin, self.confidence_scale_points),
                "rsi_above_55_and_rising",
                margin,
            )
        if current <= self.sell_threshold and current < previous:
            margin = self.sell_threshold - current
            return self._signal(
                candles,
                SignalAction.SELL,
                _confidence(margin, self.confidence_scale_points),
                "rsi_below_45_and_falling",
                margin,
            )
        return self._signal(candles, SignalAction.ABSTAIN, 0, "rsi_momentum_neutral", current)


class MeanReversionAgent(StrategyAgent):
    strategy_id = "mean_reversion"
    version = "1.0.0"
    band_period = 20
    band_deviations = 2.0
    atr_period = 14
    confidence_scale_atr = 1.0

    def reset(self) -> None:
        super().reset()
        self._entry_confidence = 0.0

    def evaluate(self, candles: Sequence[Candle]) -> StrategySignal:
        closes = [candle.close for candle in candles]
        if len(candles) < self.band_period + 1:
            return self._signal(candles, SignalAction.ABSTAIN, 0, "indicator_warmup")
        previous_bands = bollinger(closes[:-1], self.band_period, self.band_deviations)
        current_bands = bollinger(closes, self.band_period, self.band_deviations)
        volatility = atr(candles, self.atr_period)
        if previous_bands is None or current_bands is None or volatility is None or volatility <= 0:
            return self._signal(candles, SignalAction.ABSTAIN, 0, "indicator_warmup")
        previous_lower, _, _ = previous_bands
        current_lower, middle, _ = current_bands
        previous_close = closes[-2]
        current_close = closes[-1]

        if self._active:
            if current_close >= middle or current_close < current_lower:
                self._active = False
                return self._signal(candles, SignalAction.SELL, 100, "mean_reversion_exit")
            return self._signal(
                candles,
                SignalAction.BUY,
                self._entry_confidence,
                "mean_reversion_position_active",
            )

        if previous_close < previous_lower and current_close > current_lower:
            penetration = max(0.0, previous_lower - previous_close)
            reentry = max(0.0, current_close - current_lower)
            margin_atr = min(penetration, reentry) / volatility
            confidence = _confidence(margin_atr, self.confidence_scale_atr)
            if confidence >= CONFIDENCE_THRESHOLD:
                self._active = True
                self._entry_confidence = confidence
            return self._signal(
                candles,
                SignalAction.BUY,
                confidence,
                "lower_band_reentry_after_oversold_close",
                margin_atr,
            )
        return self._signal(candles, SignalAction.ABSTAIN, 0, "no_mean_reversion_setup")


class BreakoutAgent(StrategyAgent):
    strategy_id = "breakout"
    entry_period = 20
    exit_period = 10
    atr_period = 14
    confidence_scale_atr = 1.0
    version = "1.0.0"

    def reset(self) -> None:
        super().reset()
        self._entry_confidence = 0.0

    def evaluate(self, candles: Sequence[Candle]) -> StrategySignal:
        entry_channel = donchian(candles, self.entry_period)
        exit_channel = donchian(candles, self.exit_period)
        volatility = atr(candles, self.atr_period)
        if entry_channel is None or exit_channel is None or volatility is None or volatility <= 0:
            return self._signal(candles, SignalAction.ABSTAIN, 0, "indicator_warmup")
        previous_high, _ = entry_channel
        _, previous_low = exit_channel
        close = candles[-1].close

        if self._active:
            if close < previous_low:
                self._active = False
                return self._signal(candles, SignalAction.SELL, 100, "donchian_10_low_exit")
            return self._signal(
                candles,
                SignalAction.BUY,
                self._entry_confidence,
                "breakout_position_active",
            )

        if close > previous_high:
            margin_atr = (close - previous_high) / volatility
            confidence = _confidence(margin_atr, self.confidence_scale_atr)
            if confidence >= CONFIDENCE_THRESHOLD:
                self._active = True
                self._entry_confidence = confidence
            return self._signal(
                candles,
                SignalAction.BUY,
                confidence,
                "close_above_prior_20_hour_high",
                margin_atr,
            )
        return self._signal(candles, SignalAction.ABSTAIN, 0, "no_breakout")


def default_strategy_agents() -> list[StrategyAgent]:
    return [EmaCrossoverAgent(), RsiMomentumAgent(), MeanReversionAgent(), BreakoutAgent()]
