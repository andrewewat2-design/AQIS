from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import sqlite3

from aqis.models import (
    ONE,
    ZERO,
    OrderIntent,
    OrderSide,
    PaperAccount,
    RiskDecision,
    RiskPolicy,
    is_agent_actor,
)


@dataclass(frozen=True, slots=True)
class IntegrityStatus:
    market_data_fresh: bool = True
    positions_reconciled: bool = True
    pending_order_state_known: bool = True
    reason: str = "ok"


class PolicyRegistry:
    """Human-approved immutable policy versions; active policy objects are frozen."""

    def __init__(self) -> None:
        self._versions: dict[tuple[str, int], RiskPolicy] = {}
        self._active: dict[str, int] = {}

    def activate(self, policy: RiskPolicy, approved_by: str) -> None:
        if not approved_by or is_agent_actor(approved_by):
            raise PermissionError("only a human may approve a risk policy version")
        key = (policy.policy_id, policy.version)
        if key in self._versions:
            raise ValueError("policy versions are immutable and cannot be re-activated with changes")
        current = self._active.get(policy.policy_id)
        if current is not None and policy.version <= current:
            raise ValueError("new policy versions must increase monotonically")
        self._versions[key] = policy
        self._active[policy.policy_id] = policy.version

    def active(self, policy_id: str) -> RiskPolicy:
        version = self._active[policy_id]
        return self._versions[(policy_id, version)]

    def versions(self, policy_id: str) -> tuple[RiskPolicy, ...]:
        return tuple(
            self._versions[key]
            for key in sorted(self._versions)
            if key[0] == policy_id
        )


class PersistentPolicyRegistry:
    """Durable human-approved risk policy history and active-version pointer."""

    def __init__(self, database: str = "data/aqis.sqlite3") -> None:
        if database != ":memory:":
            Path(database).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS risk_policy_versions ("
            "policy_id TEXT NOT NULL, version INTEGER NOT NULL, payload TEXT NOT NULL, "
            "approved_by TEXT NOT NULL, activated_at TEXT NOT NULL, active INTEGER NOT NULL, "
            "PRIMARY KEY(policy_id, version))"
        )
        self.connection.execute(
            "CREATE TRIGGER IF NOT EXISTS risk_policy_payload_immutable "
            "BEFORE UPDATE OF payload, approved_by, activated_at, policy_id, version "
            "ON risk_policy_versions BEGIN SELECT RAISE(ABORT, 'risk policy versions are immutable'); END"
        )
        self.connection.execute(
            "CREATE TRIGGER IF NOT EXISTS risk_policy_no_delete "
            "BEFORE DELETE ON risk_policy_versions BEGIN SELECT RAISE(ABORT, 'risk policy versions are immutable'); END"
        )
        self.connection.commit()

    def activate(self, policy: RiskPolicy, approved_by: str) -> None:
        if not approved_by or is_agent_actor(approved_by):
            raise PermissionError("only a human may approve a risk policy version")
        row = self.connection.execute(
            "SELECT MAX(version) FROM risk_policy_versions WHERE policy_id=?", (policy.policy_id,)
        ).fetchone()
        current_version = row[0]
        if current_version is not None and policy.version <= int(current_version):
            raise ValueError("risk policy versions must increase monotonically")
        if policy.live_trading_enabled:
            raise ValueError("live execution is disabled in AQIS v1")
        payload = {
            "policy_id": policy.policy_id,
            "version": policy.version,
            "starting_equity_usdt": str(policy.starting_equity_usdt),
            "max_position_pct": str(policy.max_position_pct),
            "max_order_pct": str(policy.max_order_pct),
            "daily_loss_stop_pct": str(policy.daily_loss_stop_pct),
            "max_drawdown_stop_pct": str(policy.max_drawdown_stop_pct),
            "fee_rate": str(policy.fee_rate),
            "slippage_bps": str(policy.slippage_bps),
            "min_net_edge_pct": str(policy.min_net_edge_pct),
            "max_candle_lag_seconds": policy.max_candle_lag_seconds,
            "max_consecutive_execution_errors": policy.max_consecutive_execution_errors,
            "volatility_review_multiple": str(policy.volatility_review_multiple),
            "strategy_correlation_review": str(policy.strategy_correlation_review),
            "slippage_deviation_review_bps": str(policy.slippage_deviation_review_bps),
            "minimum_correlation_samples": policy.minimum_correlation_samples,
            "market": policy.market,
            "venue": policy.venue,
            "timeframe": policy.timeframe,
            "live_trading_enabled": False,
        }
        with self.connection:
            self.connection.execute(
                "UPDATE risk_policy_versions SET active=0 WHERE policy_id=?", (policy.policy_id,)
            )
            self.connection.execute(
                "INSERT INTO risk_policy_versions(policy_id, version, payload, approved_by, activated_at, active) "
                "VALUES (?, ?, ?, ?, ?, 1)",
                (
                    policy.policy_id,
                    policy.version,
                    json.dumps(payload, sort_keys=True),
                    approved_by,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def active(self, policy_id: str) -> RiskPolicy:
        row = self.connection.execute(
            "SELECT payload FROM risk_policy_versions WHERE policy_id=? AND active=1", (policy_id,)
        ).fetchone()
        if row is None:
            raise KeyError(policy_id)
        return self._decode(json.loads(row[0]))

    def versions(self, policy_id: str) -> tuple[RiskPolicy, ...]:
        rows = self.connection.execute(
            "SELECT payload FROM risk_policy_versions WHERE policy_id=? ORDER BY version", (policy_id,)
        ).fetchall()
        return tuple(self._decode(json.loads(row[0])) for row in rows)

    @staticmethod
    def _decode(payload: dict[str, object]) -> RiskPolicy:
        decimal_fields = (
            "starting_equity_usdt",
            "max_position_pct",
            "max_order_pct",
            "daily_loss_stop_pct",
            "max_drawdown_stop_pct",
            "fee_rate",
            "slippage_bps",
            "min_net_edge_pct",
            "volatility_review_multiple",
            "strategy_correlation_review",
            "slippage_deviation_review_bps",
        )
        for key in decimal_fields:
            payload[key] = Decimal(str(payload[key]))
        return RiskPolicy(**payload)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> PersistentPolicyRegistry:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class RiskEngine:
    """Deterministic hard gate. It has no API for modifying the active policy."""

    def __init__(self, policy: RiskPolicy) -> None:
        self.policy = policy

    def account_loss_trigger(
        self, account: PaperAccount, mark_price: Decimal
    ) -> tuple[str, Decimal] | None:
        equity = account.equity(mark_price)
        if account.utc_day_start_equity_usdt is None:
            account.utc_day_start_equity_usdt = equity
        if account.high_watermark_usdt is None:
            account.high_watermark_usdt = equity

        daily_loss = max(ZERO, account.utc_day_start_equity_usdt - equity)
        daily_limit = account.utc_day_start_equity_usdt * self.policy.daily_loss_stop_pct
        if daily_limit > ZERO and daily_loss >= daily_limit:
            return "daily_loss_limit", daily_loss

        drawdown = max(ZERO, account.high_watermark_usdt - equity)
        drawdown_limit = account.high_watermark_usdt * self.policy.max_drawdown_stop_pct
        if drawdown_limit > ZERO and drawdown >= drawdown_limit:
            return "maximum_drawdown_limit", drawdown
        return None

    def evaluate_order(
        self,
        order: OrderIntent,
        account: PaperAccount,
        integrity: IntegrityStatus = IntegrityStatus(),
        *,
        require_opportunity_evidence: bool = True,
    ) -> RiskDecision:
        if order.quantity_btc <= ZERO or order.reference_price <= ZERO:
            return RiskDecision(False, "invalid_order", "Quantity and reference price must be positive")
        if not integrity.market_data_fresh:
            return RiskDecision(False, "stale_market_data", integrity.reason or "Market data is stale")
        if not integrity.positions_reconciled:
            return RiskDecision(False, "position_mismatch", integrity.reason or "Positions are unreconciled")
        if not integrity.pending_order_state_known:
            return RiskDecision(False, "order_state_unknown", integrity.reason or "Pending order state is unknown")

        # Check realized + unrealized account loss before accepting any new exposure.
        self.pause_if_required(account, order.reference_price)
        if account.paused and (account.pause_reason or "").startswith("data_integrity:"):
            return RiskDecision(False, "integrity_halt", account.pause_reason or "Data-integrity halt")
        if account.paused:
            # Only a risk-reducing close may pass a loss-limit pause. Integrity halts
            # are represented by failed IntegrityStatus and block every order.
            if not (order.side is OrderSide.SELL and order.reduce_only):
                return RiskDecision(False, "policy_paused", account.pause_reason or "Trading is paused")

        equity = account.equity(order.reference_price)
        notional = order.reference_notional
        max_order = equity * self.policy.max_order_pct
        if notional > max_order:
            return RiskDecision(False, "max_order_exceeded", "Order exceeds the policy order cap")

        if order.side is OrderSide.SELL:
            if order.quantity_btc > account.btc:
                return RiskDecision(False, "would_short_spot", "Spot sell exceeds the held BTC position")
            if order.reduce_only is False and account.paused:
                return RiskDecision(False, "reduce_only_required", "Paused exits must be reduce-only")
            return RiskDecision(True, "risk_reducing_order", "Spot sell reduces the current BTC position")

        if account.paused:
            return RiskDecision(False, "policy_paused", account.pause_reason or "Trading is paused")
        if require_opportunity_evidence:
            if order.net_edge_lower_bound_pct is None:
                return RiskDecision(False, "missing_opportunity_evidence", "Buy requires frozen validation evidence")
            if order.net_edge_lower_bound_pct <= self.policy.min_net_edge_pct:
                return RiskDecision(False, "net_edge_below_hurdle", "Expected net opportunity is below policy hurdle")
            if order.stressed_loss_fraction is None or order.stressed_loss_fraction <= ZERO:
                return RiskDecision(False, "missing_stress_estimate", "Buy requires a positive stressed-loss estimate")
            if not order.evidence_id:
                return RiskDecision(False, "missing_evidence_id", "Buy must reference its frozen validation evidence")

        current_exposure = account.btc * order.reference_price
        estimated_fill_exposure = notional * (ONE + self.policy.slippage_bps / Decimal("10000"))
        projected_exposure = current_exposure + estimated_fill_exposure
        max_exposure = equity * self.policy.max_position_pct
        if projected_exposure > max_exposure:
            return RiskDecision(False, "max_exposure_exceeded", "Order exceeds aggregate BTC exposure cap")

        loss_headroom = self._loss_headroom(account, order.reference_price)
        if require_opportunity_evidence:
            stressed_loss = notional * order.stressed_loss_fraction
            if stressed_loss > loss_headroom:
                return RiskDecision(
                    False,
                    "risk_budget_exceeded",
                    "Stressed order loss exceeds remaining daily/drawdown risk budget",
                    risk_budget_usdt=loss_headroom,
                )
        if notional > account.cash_usdt:
            return RiskDecision(False, "insufficient_cash", "Paper account has insufficient USDT")
        return RiskDecision(True, "approved", "Order passes the independent risk policy", loss_headroom)

    def _loss_headroom(self, account: PaperAccount, mark_price: Decimal) -> Decimal:
        equity = account.equity(mark_price)
        day_start = account.utc_day_start_equity_usdt or equity
        high_water = account.high_watermark_usdt or equity
        daily_headroom = max(ZERO, day_start * self.policy.daily_loss_stop_pct - max(ZERO, day_start - equity))
        drawdown_headroom = max(
            ZERO,
            high_water * self.policy.max_drawdown_stop_pct - max(ZERO, high_water - equity),
        )
        return min(daily_headroom, drawdown_headroom)

    def pause_if_required(self, account: PaperAccount, mark_price: Decimal) -> str | None:
        if account.paused:
            return account.pause_reason
        trigger = self.account_loss_trigger(account, mark_price)
        if trigger is None:
            return None
        reason, _ = trigger
        account.paused = True
        account.pause_reason = reason
        return reason

    def data_integrity_halt(self, account: PaperAccount, reason: str) -> None:
        account.paused = True
        account.pause_reason = f"data_integrity:{reason}"

    def record_execution_result(self, account: PaperAccount, success: bool) -> bool:
        if success:
            account.consecutive_execution_errors = 0
            return False
        account.consecutive_execution_errors += 1
        if account.consecutive_execution_errors >= self.policy.max_consecutive_execution_errors:
            account.paused = True
            account.pause_reason = "repeated_execution_errors"
            return True
        return False

    def human_resume(self, account: PaperAccount, actor: str) -> None:
        if not actor or is_agent_actor(actor):
            raise PermissionError("Only a human operator can resume a paused account")
        account.paused = False
        account.pause_reason = None
        account.consecutive_execution_errors = 0

    def apply_agent_pause_request(self, account: PaperAccount, actor: str, reason: str) -> None:
        allowed = {
            "risk_evaluation_agent",
            "monitoring_agent",
            "supervisor_agent",
        }
        if actor not in allowed:
            raise PermissionError("only the Risk, Monitoring, or Supervisor Agent may request a pause")
        account.paused = True
        account.pause_reason = f"agent_requested_pause:{actor}:{reason[:160]}"


def is_stale(
    candle_timestamp: datetime, now: datetime, policy: RiskPolicy
) -> bool:
    if candle_timestamp.tzinfo is None or now.tzinfo is None:
        return True
    return (now.astimezone(timezone.utc) - candle_timestamp.astimezone(timezone.utc)).total_seconds() > policy.max_candle_lag_seconds
