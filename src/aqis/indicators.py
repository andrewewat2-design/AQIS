from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

from aqis.models import Candle


def ema(values: Sequence[float], period: int) -> float | None:
    if period <= 0 or len(values) < period:
        return None
    alpha = 2 / (period + 1)
    current = sum(values[:period]) / period
    for value in values[period:]:
        current = alpha * value + (1 - alpha) * current
    return current


def atr(candles: Sequence[Candle], period: int = 14) -> float | None:
    if period <= 0 or len(candles) < period + 1:
        return None
    true_ranges: list[float] = []
    for index in range(1, len(candles)):
        candle = candles[index]
        previous_close = candles[index - 1].close
        true_ranges.append(
            max(
                candle.high - candle.low,
                abs(candle.high - previous_close),
                abs(candle.low - previous_close),
            )
        )
    value = sum(true_ranges[:period]) / period
    for item in true_ranges[period:]:
        value = ((period - 1) * value + item) / period
    return value


def rsi(values: Sequence[float], period: int = 14) -> float | None:
    if period <= 0 or len(values) < period + 1:
        return None
    changes = [values[i] - values[i - 1] for i in range(1, len(values))]
    gains = [max(change, 0.0) for change in changes]
    losses = [max(-change, 0.0) for change in changes]
    average_gain = sum(gains[:period]) / period
    average_loss = sum(losses[:period]) / period
    for gain, loss in zip(gains[period:], losses[period:]):
        average_gain = ((period - 1) * average_gain + gain) / period
        average_loss = ((period - 1) * average_loss + loss) / period
    if average_loss == 0:
        return 100.0 if average_gain > 0 else 50.0
    relative_strength = average_gain / average_loss
    return 100 - 100 / (1 + relative_strength)


def bollinger(
    values: Sequence[float], period: int = 20, deviations: float = 2.0
) -> tuple[float, float, float] | None:
    if period <= 0 or len(values) < period:
        return None
    window = values[-period:]
    middle = sum(window) / period
    variance = sum((value - middle) ** 2 for value in window) / period
    width = deviations * sqrt(variance)
    return middle - width, middle, middle + width


def donchian(
    candles: Sequence[Candle], period: int
) -> tuple[float, float] | None:
    """Return the prior-channel high/low, excluding the current candle."""
    if period <= 0 or len(candles) < period + 1:
        return None
    window = candles[-period - 1 : -1]
    return max(candle.high for candle in window), min(candle.low for candle in window)


def directional_index(candles: Sequence[Candle], period: int = 14) -> float | None:
    """Compact ADX-like regime statistic used only for shadow analysis."""
    if len(candles) < period + 1:
        return None
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    tr: list[float] = []
    for previous, current in zip(candles[-period - 1 : -1], candles[-period:]):
        up = current.high - previous.high
        down = previous.low - current.low
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
        tr.append(
            max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )
        )
    tr_sum = sum(tr)
    if tr_sum == 0:
        return 0.0
    plus_di = 100 * sum(plus_dm) / tr_sum
    minus_di = 100 * sum(minus_dm) / tr_sum
    denominator = plus_di + minus_di
    return 0.0 if denominator == 0 else 100 * abs(plus_di - minus_di) / denominator
