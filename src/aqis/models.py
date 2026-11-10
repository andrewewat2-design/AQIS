from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any


ZERO = Decimal("0")
ONE = Decimal("1")


def is_agent_actor(actor: str) -> bool:
    normalized = actor.strip().casefold()
    return normalized in {"agent", "system", "supervisor", "validation_engine"} or normalized.endswith("_agent")


class SignalAction(str, Enum):
    BUY = "buy"
    SELL = "sell"  # Spot exit / target cash; never a short.
    ABSTAIN = "abstain"


class TargetPosition(str, Enum):
    LONG = "long"
    CASH = "cash"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class LifecycleStage(str, Enum):
    DISCOVERY = "discovery"
    RESEARCH = "research"
    BACKTEST = "backtest"
    VALIDATION = "validation"
    PAPER_TRADING = "paper_trading"
    MONITORING = "monitoring"
    RESTRICTED_DEPLOYMENT = "restricted_deployment"
    RETIRED = "retired"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class Candle:
    """Finalized OHLCV candle. timestamp is the candle close time in UTC."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    finalized: bool = True

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("candle timestamp must be timezone-aware")
        if not self.finalized:
            raise ValueError("AQIS strategy inputs must be completed candles")
        values = (self.open, self.high, self.low, self.close, self.volume)
        if any(not isinstance(value, (int, float)) for value in values):
            raise TypeError("OHLCV values must be numeric")
        if any(value != value for value in values):
            raise ValueError("OHLCV values cannot be NaN")
        if min(self.open, self.high, self.low, self.close) <= 0:
            raise ValueError("OHLC prices must be positive")
        if self.volume < 0:
            raise ValueError("volume cannot be negative")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high is inconsistent with OHLC values")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low is inconsistent with OHLC values")

    @property
    def utc_timestamp(self) -> datetime:
        return self.timestamp.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class StrategySignal:
    strategy_id: str
    strategy_version: str
    timestamp: datetime
    action: SignalAction
    confidence: float
    reason: str
    strength: float | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 100:
            raise ValueError("confidence must be between 0 and 100")
        if self.timestamp.tzinfo is None:
            raise ValueError("signal timestamp must be timezone-aware")


@dataclass(frozen=True, slots=True)
class SelectorDecision:
    timestamp: datetime
    target: TargetPosition
    buy_votes: tuple[str, ...]
    sell_votes: tuple[str, ...]
    abstentions: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class OpportunityEvidence:
    """Frozen validation evidence for an ensemble/regime, not an AI estimate."""

    evidence_id: str
    strategy_set_version: str
    regime: str
    net_edge_lower_bound_pct: Decimal
    stressed_loss_fraction: Decimal
    validation_trade_count: int
    data_hash: str
    risk_policy_id: str = "aqis-paper-btcusdt"
    risk_policy_version: int = 1

    def __post_init__(self) -> None:
        if self.validation_trade_count < 0:
            raise ValueError("validation_trade_count cannot be negative")
        if self.stressed_loss_fraction < ZERO:
            raise ValueError("stressed_loss_fraction cannot be negative")


@dataclass(frozen=True, slots=True)
class RiskPolicy:
    """Immutable after activation; changes require a new approved version."""

    policy_id: str = "aqis-paper-btcusdt"
    version: int = 1
    starting_equity_usdt: Decimal = Decimal("10000")
    max_position_pct: Decimal = Decimal("0.20")
    max_order_pct: Decimal = Decimal("0.10")
    daily_loss_stop_pct: Decimal = Decimal("0.02")
    max_drawdown_stop_pct: Decimal = Decimal("0.05")
    fee_rate: Decimal = Decimal("0.001")
    slippage_bps: Decimal = Decimal("10")
    min_net_edge_pct: Decimal = ZERO
    max_candle_lag_seconds: int = 300
    max_consecutive_execution_errors: int = 3
    volatility_review_multiple: Decimal = Decimal("2")
    strategy_correlation_review: Decimal = Decimal("0.85")
    slippage_deviation_review_bps: Decimal = Decimal("20")
    minimum_correlation_samples: int = 30
    market: str = "BTC/USDT"
    venue: str = "binance"
    timeframe: str = "1h"
    live_trading_enabled: bool = False

    def __post_init__(self) -> None:
        if self.version <= 0:
            raise ValueError("policy version must be positive")
        for field_name in (
            "max_position_pct",
            "max_order_pct",
            "daily_loss_stop_pct",
            "max_drawdown_stop_pct",
            "fee_rate",
        ):
            value = getattr(self, field_name)
            if not ZERO <= value <= ONE:
                raise ValueError(f"{field_name} must be between 0 and 1")
        if self.starting_equity_usdt <= ZERO:
            raise ValueError("starting equity must be positive")
        if self.max_position_pct <= ZERO or self.max_order_pct <= ZERO:
            raise ValueError("position and order caps must be positive")
        if self.slippage_bps < ZERO or self.min_net_edge_pct < ZERO:
            raise ValueError("slippage and expected-edge thresholds cannot be negative")
        if self.max_candle_lag_seconds <= 0:
            raise ValueError("max candle lag must be positive")
        if self.max_consecutive_execution_errors <= 0:
            raise ValueError("execution error threshold must be positive")
        if self.volatility_review_multiple <= ZERO:
            raise ValueError("volatility review multiple must be positive")
        if not ZERO <= self.strategy_correlation_review <= ONE:
            raise ValueError("strategy correlation threshold must be between 0 and 1")
        if self.slippage_deviation_review_bps < ZERO:
            raise ValueError("slippage deviation threshold cannot be negative")
        if self.minimum_correlation_samples < 3:
            raise ValueError("correlation sample threshold must be at least 3")
        if self.market != "BTC/USDT" or self.venue != "binance" or self.timeframe != "1h":
            raise ValueError("AQIS v1 paper policy is restricted to Binance Spot BTC/USDT 1h")
        if self.live_trading_enabled:
            raise ValueError("live trading is outside the AQIS v1 prototype")


@dataclass(slots=True)
class PaperAccount:
    cash_usdt: Decimal
    btc: Decimal = ZERO
    realized_pnl_usdt: Decimal = ZERO
    fees_usdt: Decimal = ZERO
    slippage_usdt: Decimal = ZERO
    position_cost_basis_usdt: Decimal = ZERO
    position_entry_cost_accum_usdt: Decimal = ZERO
    position_realized_pnl_usdt: Decimal = ZERO
    high_watermark_usdt: Decimal | None = None
    utc_day_start_equity_usdt: Decimal | None = None
    paused: bool = False
    pause_reason: str | None = None
    consecutive_execution_errors: int = 0

    def equity(self, mark_price: Decimal) -> Decimal:
        value = self.cash_usdt + self.btc * mark_price
        if self.high_watermark_usdt is None or value > self.high_watermark_usdt:
            self.high_watermark_usdt = value
        return value


@dataclass(frozen=True, slots=True)
class OrderIntent:
    side: OrderSide
    quantity_btc: Decimal
    reference_price: Decimal
    timestamp: datetime
    reduce_only: bool = False
    net_edge_lower_bound_pct: Decimal | None = None
    stressed_loss_fraction: Decimal | None = None
    evidence_id: str | None = None
    correlation_id: str | None = None
    reason: str = ""

    @property
    def reference_notional(self) -> Decimal:
        return abs(self.quantity_btc * self.reference_price)


@dataclass(frozen=True, slots=True)
class RiskDecision:
    approved: bool
    reason_code: str
    message: str
    risk_budget_usdt: Decimal | None = None
    correlation_id: str | None = None


@dataclass(frozen=True, slots=True)
class PaperFill:
    timestamp: datetime
    side: OrderSide
    quantity_btc: Decimal
    reference_price: Decimal
    fill_price: Decimal
    fee_usdt: Decimal
    slippage_usdt: Decimal
    reason: str
    correlation_id: str | None = None
    post_fill_equity_usdt: Decimal | None = None


@dataclass(frozen=True, slots=True)
class AuditEvent:
    timestamp: datetime
    event_type: str
    actor: str
    details: dict[str, Any] = field(default_factory=dict)
    correlation_id: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    report_id: str
    strategy_set_version: str
    evidence: OpportunityEvidence
    passed: bool
    net_expectancy_pct: Decimal
    expected_shortfall_pct: Decimal
    maximum_drawdown_pct: Decimal
    notes: tuple[str, ...] = ()
    risk_policy_id: str = "aqis-paper-btcusdt"
    risk_policy_version: int = 1
