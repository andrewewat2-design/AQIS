import unittest

from aqis.indicators import donchian, ema, rsi
from aqis.models import SignalAction
from aqis.strategies import (
    BreakoutAgent,
    EmaCrossoverAgent,
    MeanReversionAgent,
    RsiMomentumAgent,
)
from tests.helpers import flat_candles, rising_candles


class StrategyTests(unittest.TestCase):
    def test_ema_crossover_emits_buy_with_frozen_atr_scale(self) -> None:
        agent = EmaCrossoverAgent()
        signal = agent.evaluate(rising_candles(100, step=1.0))
        self.assertEqual(signal.action, SignalAction.BUY)
        self.assertGreaterEqual(signal.confidence, 60)

    def test_rsi_momentum_emits_buy_on_strong_rising_rsi(self) -> None:
        from datetime import datetime, timedelta, timezone
        from aqis.models import Candle

        closes = [100.0 + (0.5 if index % 2 == 0 else -0.5) for index in range(50)]
        closes.extend(100.0 + index for index in range(1, 6))
        timestamp = datetime(2024, 1, 1, 1, tzinfo=timezone.utc)
        candles = []
        previous = closes[0]
        for index, close in enumerate(closes):
            open_price = previous
            candles.append(
                Candle(timestamp + timedelta(hours=index), open_price, max(open_price, close) + 0.1,
                       min(open_price, close) - 0.1, close, 10)
            )
            previous = close
        signal = RsiMomentumAgent().evaluate(candles)
        self.assertEqual(signal.action, SignalAction.BUY)
        self.assertGreaterEqual(signal.confidence, 60)

    def test_mean_reversion_detects_lower_band_reentry(self) -> None:
        candles = flat_candles(25)
        from aqis.models import Candle

        candles[-2] = Candle(candles[-2].timestamp, 95, 95.2, 94.8, 95, 20)
        candles[-1] = Candle(candles[-1].timestamp, 95, 101.2, 94.8, 101, 20)
        signal = MeanReversionAgent().evaluate(candles)
        self.assertEqual(signal.action, SignalAction.BUY)
        self.assertGreaterEqual(signal.confidence, 60)

    def test_breakout_uses_prior_channel_and_emits_buy(self) -> None:
        signal = BreakoutAgent().evaluate(rising_candles(80, step=2.0))
        self.assertEqual(signal.action, SignalAction.BUY)
        self.assertGreaterEqual(signal.confidence, 60)

    def test_donchian_excludes_current_candle(self) -> None:
        candles = flat_candles(22)
        from aqis.models import Candle

        current = candles[-1]
        candles[-1] = Candle(current.timestamp, 100, 999, 99, 100, 1)
        high, low = donchian(candles, 20)  # type: ignore[misc]
        self.assertEqual(high, 100.5)
        self.assertEqual(low, 99.5)

    def test_indicators_warm_up_without_lookahead(self) -> None:
        closes = [float(value) for value in range(1, 30)]
        self.assertIsNone(ema(closes[:10], 20))
        self.assertIsNone(rsi(closes[:10], 14))
