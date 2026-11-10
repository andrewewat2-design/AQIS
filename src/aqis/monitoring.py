from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from math import log, sqrt

from aqis.models import Candle, PaperAccount, RiskPolicy


@dataclass(frozen=True, slots=True)
class PortfolioRiskSnapshot:
    timestamp: datetime
    equity_usdt: Decimal
    btc_exposure_usdt: Decimal
    exposure_pct: Decimal
    unrealized_pnl_usdt: Decimal
    realized_pnl_usdt: Decimal
    fees_usdt: Decimal
    slippage_usdt: Decimal
    daily_loss_pct: Decimal
    drawdown_pct: Decimal
    current_hourly_volatility: float | None
    volatility_baseline: float | None
    volatility_ratio: float | None
    strategy_correlations: dict[str, float | None]
    execution_errors: int
    expected_slippage_bps: Decimal | None
    actual_slippage_bps: Decimal | None
    findings: tuple[str, ...]

    def agent_facts(self) -> dict[str, object]:
        values = asdict(self)
        for key in (
            "equity_usdt",
            "btc_exposure_usdt",
            "exposure_pct",
            "unrealized_pnl_usdt",
            "realized_pnl_usdt",
            "fees_usdt",
            "slippage_usdt",
            "daily_loss_pct",
            "drawdown_pct",
            "expected_slippage_bps",
            "actual_slippage_bps",
        ):
            if values[key] is not None:
                values[key] = str(values[key])
        values["timestamp"] = self.timestamp.isoformat()
        values["findings"] = list(self.findings)
        return values


def rolling_hourly_volatility(candles: Sequence[Candle], window: int = 24) -> float | None:
    if window <= 1 or len(candles) < window + 1:
        return None
    prices = [candle.close for candle in candles[-window - 1 :]]
    returns = [log(prices[index] / prices[index - 1]) for index in range(1, len(prices))]
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
    return sqrt(max(0.0, variance))


def pairwise_correlations(
    strategy_returns: Mapping[str, Sequence[float]], minimum_samples: int = 30
) -> dict[str, float | None]:
    names = sorted(strategy_returns)
    results: dict[str, float | None] = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            key = f"{left}:{right}"
            left_values = list(strategy_returns[left])
            right_values = list(strategy_returns[right])
            count = min(len(left_values), len(right_values))
            if count < minimum_samples:
                results[key] = None
                continue
            x = left_values[-count:]
            y = right_values[-count:]
            x_mean = sum(x) / count
            y_mean = sum(y) / count
            x_centered = [value - x_mean for value in x]
            y_centered = [value - y_mean for value in y]
            x_var = sum(value * value for value in x_centered)
            y_var = sum(value * value for value in y_centered)
            if x_var == 0 or y_var == 0:
                results[key] = None
                continue
            covariance = sum(a * b for a, b in zip(x_centered, y_centered))
            results[key] = max(-1.0, min(1.0, covariance / sqrt(x_var * y_var)))
    return results


class PortfolioMonitor:
    """Computes observable portfolio/operations facts for the Risk Agent."""

    def __init__(
        self,
        policy: RiskPolicy,
        *,
        high_volatility_ratio: float | None = None,
        high_correlation_threshold: float | None = None,
        maximum_slippage_deviation_bps: Decimal | None = None,
    ) -> None:
        self.policy = policy
        self.high_volatility_ratio = float(
            policy.volatility_review_multiple if high_volatility_ratio is None else high_volatility_ratio
        )
        self.high_correlation_threshold = float(
            policy.strategy_correlation_review
            if high_correlation_threshold is None
            else high_correlation_threshold
        )
        self.maximum_slippage_deviation_bps = (
            policy.slippage_deviation_review_bps
            if maximum_slippage_deviation_bps is None
            else maximum_slippage_deviation_bps
        )

    def snapshot(
        self,
        account: PaperAccount,
        mark_price: Decimal,
        timestamp: datetime,
        candles: Sequence[Candle],
        strategy_returns: Mapping[str, Sequence[float]],
        *,
        execution_errors: int = 0,
        volatility_baseline: float | None = None,
        expected_slippage_bps: Decimal | None = None,
        actual_slippage_bps: Decimal | None = None,
    ) -> PortfolioRiskSnapshot:
        equity = account.equity(mark_price)
        exposure = account.btc * mark_price
        exposure_pct = Decimal("0") if equity <= 0 else exposure / equity
        unrealized = exposure - account.position_cost_basis_usdt
        day_start = account.utc_day_start_equity_usdt or equity
        high_water = account.high_watermark_usdt or equity
        daily_loss = max(Decimal("0"), day_start - equity)
        drawdown = max(Decimal("0"), high_water - equity)
        daily_loss_pct = Decimal("0") if day_start <= 0 else daily_loss / day_start
        drawdown_pct = Decimal("0") if high_water <= 0 else drawdown / high_water
        current_vol = rolling_hourly_volatility(candles)
        vol_ratio = None
        if current_vol is not None and volatility_baseline is not None and volatility_baseline > 0:
            vol_ratio = current_vol / volatility_baseline
        correlations = pairwise_correlations(
            strategy_returns, minimum_samples=self.policy.minimum_correlation_samples
        )
        findings: list[str] = []
        if exposure_pct > self.policy.max_position_pct:
            findings.append("portfolio_exposure_above_policy")
        if daily_loss_pct >= self.policy.daily_loss_stop_pct:
            findings.append("daily_loss_stop_breached")
        if drawdown_pct >= self.policy.max_drawdown_stop_pct:
            findings.append("maximum_drawdown_breached")
        if vol_ratio is not None and vol_ratio >= self.high_volatility_ratio:
            findings.append("hourly_volatility_abnormal_vs_baseline")
        if any(value is not None and value >= self.high_correlation_threshold for value in correlations.values()):
            findings.append("strategy_returns_highly_correlated")
        if execution_errors:
            findings.append("execution_errors_observed")
        if (
            expected_slippage_bps is not None
            and actual_slippage_bps is not None
            and actual_slippage_bps - expected_slippage_bps >= self.maximum_slippage_deviation_bps
        ):
            findings.append("execution_slippage_above_expectation")

        return PortfolioRiskSnapshot(
            timestamp=timestamp,
            equity_usdt=equity,
            btc_exposure_usdt=exposure,
            exposure_pct=exposure_pct,
            unrealized_pnl_usdt=unrealized,
            realized_pnl_usdt=account.realized_pnl_usdt,
            fees_usdt=account.fees_usdt,
            slippage_usdt=account.slippage_usdt,
            daily_loss_pct=daily_loss_pct,
            drawdown_pct=drawdown_pct,
            current_hourly_volatility=current_vol,
            volatility_baseline=volatility_baseline,
            volatility_ratio=vol_ratio,
            strategy_correlations=correlations,
            execution_errors=execution_errors,
            expected_slippage_bps=expected_slippage_bps,
            actual_slippage_bps=actual_slippage_bps,
            findings=tuple(findings),
        )
