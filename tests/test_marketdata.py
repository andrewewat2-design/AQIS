import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from aqis.marketdata import DataIntegrityError, load_csv, validate_candles
from tests.helpers import flat_candles


class MarketDataTests(unittest.TestCase):
    def test_completed_candle_stream_is_contiguous(self) -> None:
        report = validate_candles(flat_candles(4))
        self.assertTrue(report.valid)

    def test_gap_is_reported_and_csv_loader_fails_closed(self) -> None:
        candles = flat_candles(4)
        candles[2] = type(candles[2])(
            candles[2].timestamp + timedelta(hours=1),
            candles[2].open,
            candles[2].high,
            candles[2].low,
            candles[2].close,
            candles[2].volume,
        )
        self.assertEqual(len(validate_candles(candles).missing_intervals), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candles.csv"
            path.write_text(
                "timestamp,open,high,low,close,volume\n"
                + "\n".join(
                    f"{c.utc_timestamp.isoformat()},{c.open},{c.high},{c.low},{c.close},{c.volume}"
                    for c in candles
                ),
                encoding="utf-8",
            )
            with self.assertRaises(DataIntegrityError):
                load_csv(path)

    def test_naive_timestamp_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candles.csv"
            path.write_text("timestamp,open,high,low,close,volume\n2024-01-01T01:00:00,1,2,0.5,1.5,1\n")
            with self.assertRaises(DataIntegrityError):
                load_csv(path)

    def test_unfinished_candle_rejected(self) -> None:
        candle = flat_candles(1)[0]
        with self.assertRaises(ValueError):
            type(candle)(candle.timestamp, candle.open, candle.high, candle.low, candle.close, candle.volume, False)
