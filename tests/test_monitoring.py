import unittest
from decimal import Decimal

from aqis.models import PaperAccount, RiskPolicy
from aqis.monitoring import PortfolioMonitor, pairwise_correlations, rolling_hourly_volatility
from tests.helpers import rising_candles


class MonitoringTests(unittest.TestCase):
    def test_portfolio_snapshot_includes_exposure_unrealized_loss_and_findings(self) -> None:
        account = PaperAccount(
            cash_usdt=Decimal("900"),
            btc=Decimal("1"),
            position_cost_basis_usdt=Decimal("100"),
            realized_pnl_usdt=Decimal("-5"),
            fees_usdt=Decimal("2"),
            slippage_usdt=Decimal("1"),
            high_watermark_usdt=Decimal("1010"),
            utc_day_start_equity_usdt=Decimal("1010"),
        )
        snapshot = PortfolioMonitor(RiskPolicy()).snapshot(
            account,
            Decimal("90"),
            rising_candles(40)[-1].timestamp,
            rising_candles(40),
            {"ema": [0.1 * index for index in range(35)], "rsi": [0.2 * index for index in range(35)]},
            execution_errors=1,
            volatility_baseline=0.000000001,
            expected_slippage_bps=Decimal("10"),
            actual_slippage_bps=Decimal("40"),
        )
        self.assertEqual(snapshot.unrealized_pnl_usdt, Decimal("-10"))
        self.assertGreater(snapshot.exposure_pct, 0)
        self.assertIn("hourly_volatility_abnormal_vs_baseline", snapshot.findings)
        self.assertIn("strategy_returns_highly_correlated", snapshot.findings)
        self.assertIn("execution_errors_observed", snapshot.findings)
        self.assertIn("execution_slippage_above_expectation", snapshot.findings)

    def test_correlation_requires_sufficient_variation_and_sample(self) -> None:
        correlations = pairwise_correlations({"a": [float(i) for i in range(40)], "b": [float(i) for i in range(40)]})
        self.assertAlmostEqual(correlations["a:b"], 1.0)
        short = pairwise_correlations({"a": [1.0, 2.0], "b": [1.0, 2.0]})
        self.assertIsNone(short["a:b"])

    def test_hourly_volatility_needs_a_window(self) -> None:
        self.assertIsNotNone(rolling_hourly_volatility(rising_candles(30), window=24))
