from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from aqis.indicators import atr, directional_index
from aqis.models import Candle


@dataclass(frozen=True, slots=True)
class RegimeAssessment:
    regime: str
    trend_strength: float | None
    hourly_atr_fraction: float | None
    selector_effect: str = "shadow_only"


class ShadowRegimeClassifier:
    """Regime telemetry only; never changes the v1 active selector."""

    def __init__(self, trend_threshold: float = 25.0, high_volatility_fraction: float = 0.03) -> None:
        self.trend_threshold = trend_threshold
        self.high_volatility_fraction = high_volatility_fraction

    def classify(self, candles: Sequence[Candle]) -> RegimeAssessment:
        adx = directional_index(candles, 14)
        current_atr = atr(candles, 14)
        atr_fraction = None if current_atr is None else current_atr / candles[-1].close
        if adx is None or atr_fraction is None:
            return RegimeAssessment("warming_up", adx, atr_fraction)
        if atr_fraction >= self.high_volatility_fraction:
            regime = "high_volatility"
        elif adx >= self.trend_threshold:
            regime = "trending"
        else:
            regime = "ranging"
        return RegimeAssessment(regime, adx, atr_fraction)
