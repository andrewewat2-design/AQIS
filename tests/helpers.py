from datetime import datetime, timedelta, timezone

from aqis.models import Candle


def flat_candles(count: int, price: float = 100.0) -> list[Candle]:
    timestamp = datetime(2024, 1, 1, 1, tzinfo=timezone.utc)
    return [
        Candle(timestamp + timedelta(hours=index), price, price + 0.5, price - 0.5, price, 10.0)
        for index in range(count)
    ]


def rising_candles(count: int, start: float = 100.0, step: float = 0.2) -> list[Candle]:
    timestamp = datetime(2024, 1, 1, 1, tzinfo=timezone.utc)
    result = []
    price = start
    for index in range(count):
        close = price + step
        result.append(Candle(timestamp + timedelta(hours=index), price, close + 0.2, price - 0.2, close, 10.0))
        price = close
    return result
