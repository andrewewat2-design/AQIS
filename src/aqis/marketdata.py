from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from aqis.models import Candle


class DataIntegrityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DataValidation:
    candle_count: int
    missing_intervals: tuple[datetime, ...]
    duplicate_timestamps: tuple[datetime, ...]
    invalid_rows: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.missing_intervals and not self.duplicate_timestamps and not self.invalid_rows


def parse_timestamp(raw: str) -> datetime:
    value = raw.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone (use UTC/Z)")
    return parsed.astimezone(timezone.utc)


def validate_candles(
    candles: list[Candle], interval: timedelta = timedelta(hours=1)
) -> DataValidation:
    invalid: list[str] = []
    duplicates: list[datetime] = []
    missing: list[datetime] = []
    seen: set[datetime] = set()
    previous: Candle | None = None
    for candle in candles:
        timestamp = candle.utc_timestamp
        if timestamp.minute or timestamp.second or timestamp.microsecond:
            invalid.append(f"hourly close timestamp is not on the UTC hour: {timestamp.isoformat()}")
        if timestamp in seen:
            duplicates.append(timestamp)
            continue
        seen.add(timestamp)
        if previous is not None:
            delta = timestamp - previous.utc_timestamp
            if delta <= timedelta(0):
                invalid.append(f"timestamps are not strictly increasing at {timestamp.isoformat()}")
            elif delta > interval:
                whole_intervals, remainder = divmod(delta, interval)
                missing.extend(
                    previous.utc_timestamp + interval * step
                    for step in range(1, whole_intervals)
                )
                if remainder:
                    invalid.append(f"non-hourly timestamp gap before {timestamp.isoformat()}")
            elif delta < interval:
                invalid.append(f"non-hourly timestamp gap before {timestamp.isoformat()}")
        previous = candle
    return DataValidation(len(candles), tuple(missing), tuple(duplicates), tuple(invalid))


def load_csv(path: str | Path, require_contiguous: bool = True) -> list[Candle]:
    candles: list[Candle] = []
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"open", "high", "low", "close", "volume"}
        if reader.fieldnames is None:
            raise DataIntegrityError("CSV is missing a header row")
        timestamp_field = "timestamp" if "timestamp" in reader.fieldnames else "close_time"
        if timestamp_field not in reader.fieldnames or not required.issubset(reader.fieldnames):
            raise DataIntegrityError(
                "CSV requires timestamp (or close_time), open, high, low, close, volume columns"
            )
        for row_number, row in enumerate(reader, start=2):
            try:
                candles.append(
                    Candle(
                        timestamp=parse_timestamp(row[timestamp_field]),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                    )
                )
            except (TypeError, ValueError) as error:
                raise DataIntegrityError(f"invalid row {row_number}: {error}") from error
    report = validate_candles(candles)
    if report.duplicate_timestamps or report.invalid_rows:
        raise DataIntegrityError("duplicate or out-of-order hourly candle timestamps")
    if require_contiguous and report.missing_intervals:
        raise DataIntegrityError(
            f"missing {len(report.missing_intervals)} hourly candles; first gap starts "
            f"at {report.missing_intervals[0].isoformat()}"
        )
    return candles


def download_binance_spot_klines(
    start: datetime,
    end: datetime,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    timeout: int = 20,
) -> list[Candle]:
    """Download public Binance spot klines; no account credentials are used."""
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("start/end must be timezone-aware and start before end")
    if interval != "1h":
        raise ValueError("AQIS v1 supports only finalized 1h Binance Spot candles")
    start_ms = int(start.astimezone(timezone.utc).timestamp() * 1000)
    end_ms = int(end.astimezone(timezone.utc).timestamp() * 1000)
    rows: list[list[object]] = []
    cursor = start_ms
    while cursor < end_ms:
        params = urlencode(
            {"symbol": symbol, "interval": interval, "startTime": cursor, "endTime": end_ms, "limit": 1000}
        )
        request_url = f"https://api.binance.com/api/v3/klines?{params}"
        with urlopen(request_url, timeout=timeout) as response:  # noqa: S310 - fixed Binance public HTTPS endpoint
            page = json.loads(response.read().decode("utf-8"))
        if not page:
            break
        rows.extend(page)
        last_open_ms = int(page[-1][0])
        next_cursor = last_open_ms + 3_600_000
        if next_cursor <= cursor:
            raise DataIntegrityError("Binance kline pagination did not advance")
        cursor = next_cursor

    candles: list[Candle] = []
    for row in rows:
        open_time = datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc)
        close_time = open_time + timedelta(hours=1)
        # Exclude the still-forming last candle even if the endpoint returns it.
        if close_time > datetime.now(timezone.utc) or close_time <= start or close_time > end:
            continue
        candles.append(
            Candle(
                timestamp=close_time,
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
        )
    report = validate_candles(candles)
    if not report.valid:
        raise DataIntegrityError("downloaded Binance data failed hourly continuity validation")
    return candles
