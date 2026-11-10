from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Sequence

import hashlib

from aqis.monitoring import PortfolioMonitor
from aqis.nemotron import ResearchReport
from aqis.ledger import LedgerDatabase, generate_replay_id, generate_decision_id
from aqis.models import (
    AuditEvent,
    Candle,
    OpportunityEvidence,
    OrderIntent,
    OrderSide,
    PaperAccount,
    PaperFill,
    RiskDecision,
    RiskPolicy,
    SelectorDecision,
    SignalAction,
    StrategySignal,
    TargetPosition,
)
from aqis.opportunity import OpportunityGate
from aqis.regime import RegimeAssessment, ShadowRegimeClassifier
from aqis.risk import IntegrityStatus, RiskEngine
from aqis.selector import ConsensusSelector
from aqis.strategies import StrategyAgent, default_strategy_agents


@dataclass(frozen=True, slots=True)
class CandleDecision:
    candle: Candle
    signals: tuple[StrategySignal, ...]
    selection: SelectorDecision
    regime: RegimeAssessment


@dataclass(frozen=True, slots=True)
class StrategyPerformance:
    """Per-strategy participation and vote attribution.

    Attribution limitation: a strategy's vote agreeing with the selector is a
    correlation, not evidence that the strategy caused the fill outcome.
    Individual drawdown contribution is intentionally not calculated because no
    defensible counterfactual methodology exists.
    """

    strategy_id: str
    signals: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    abstentions: int = 0
    valid_votes: int = 0
    contributing_votes: int = 0
    fills_while_voting_with_selector: int = 0
    attribution_limitation: str = (
        "Correlation does not imply causation: a strategy vote agreeing with the "
        "selector does not mean the strategy caused the fill outcome. Individual "
        "drawdown contribution is not calculated without a defensible "
        "counterfactual methodology."
    )


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    timestamp: datetime
    equity_usdt: Decimal
    btc_exposure_usdt: Decimal
    exposure_pct: Decimal
    realized_pnl_usdt: Decimal
    unrealized_pnl_usdt: Decimal
    drawdown_pct: Decimal
    daily_loss_pct: Decimal
    paused: bool


@dataclass(frozen=True, slots=True)
class RiskIntervention:
    timestamp: datetime
    intervention_type: str  # "pause_transition", "agent_pause_request", "human_resume"
    actor: str
    reason: str
    correlation_id: str | None = None


@dataclass(slots=True)
class ReplayResult:
    starting_equity_usdt: Decimal
    ending_equity_usdt: Decimal
    account: PaperAccount
    fills: list[PaperFill] = field(default_factory=list)
    decisions: list[CandleDecision] = field(default_factory=list)
    audit_events: list[AuditEvent] = field(default_factory=list)
    rejected_orders: list[tuple[datetime, str, str]] = field(default_factory=list)
    order_requests: list[tuple[datetime, str, Decimal, str | None]] = field(
        default_factory=list
    )
    approved_orders: list[tuple[datetime, str, Decimal, str | None]] = field(
        default_factory=list
    )
    risk_rejections: list[tuple[datetime, str, str, str | None]] = field(
        default_factory=list
    )
    opportunity_suppressions: list[tuple[datetime, str, str, str | None]] = field(
        default_factory=list
    )
    risk_interventions: list[RiskIntervention] = field(default_factory=list)
    pause_resume_history: list[RiskIntervention] = field(default_factory=list)
    portfolio_snapshots: list[PortfolioSnapshot] = field(default_factory=list)
    strategy_performance: dict[str, StrategyPerformance] = field(default_factory=dict)
    selector_statistics: dict[str, object] = field(default_factory=dict)
    nemotron_reports: list[ResearchReport] = field(default_factory=list)
    completed_trade_returns_pct: list[Decimal] = field(default_factory=list)
    halted: bool = False
    halt_reason: str | None = None
    maximum_drawdown_pct: Decimal = Decimal("0")

    def audit_report(self) -> dict[str, object]:
        """Read-only aggregation of replay observability data.

        Does not modify trading or risk state.
        """
        event_type_counts: dict[str, int] = {}
        rejection_origin: dict[str, int] = {}
        for event in self.audit_events:
            event_type_counts[event.event_type] = (
                event_type_counts.get(event.event_type, 0) + 1
            )
            actor = event.actor
            if event.event_type == "order_rejected":
                rejection_origin[actor] = rejection_origin.get(actor, 0) + 1

        intervention_summary: list[dict[str, object]] = []
        for intervention in self.risk_interventions:
            intervention_summary.append(
                {
                    "timestamp": intervention.timestamp.isoformat(),
                    "type": intervention.intervention_type,
                    "actor": intervention.actor,
                    "reason": intervention.reason,
                    "correlation_id": intervention.correlation_id,
                }
            )

        pause_resume_summary: list[dict[str, object]] = []
        for event in self.pause_resume_history:
            pause_resume_summary.append(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "type": event.intervention_type,
                    "actor": event.actor,
                    "reason": event.reason,
                    "correlation_id": event.correlation_id,
                }
            )

        return {
            "event_type_counts": event_type_counts,
            "rejection_attribution_by_origin": rejection_origin,
            "risk_interventions": intervention_summary,
            "pause_resume_history": pause_resume_summary,
            "order_requests": len(self.order_requests),
            "approved_orders": len(self.approved_orders),
            "risk_rejections": len(self.risk_rejections),
            "opportunity_suppressions": len(self.opportunity_suppressions),
            "fills": len(self.fills),
            "rejected_orders": len(self.rejected_orders),
            "strategy_performance": {
                sid: {
                    "signals": perf.signals,
                    "buy_signals": perf.buy_signals,
                    "sell_signals": perf.sell_signals,
                    "abstentions": perf.abstentions,
                    "valid_votes": perf.valid_votes,
                    "contributing_votes": perf.contributing_votes,
                    "fills_while_voting_with_selector": perf.fills_while_voting_with_selector,
                    "attribution_limitation": perf.attribution_limitation,
                }
                for sid, perf in self.strategy_performance.items()
            },
            "selector_statistics": dict(self.selector_statistics),
            "portfolio_snapshots": [
                {
                    "timestamp": snap.timestamp.isoformat(),
                    "equity_usdt": str(snap.equity_usdt),
                    "btc_exposure_usdt": str(snap.btc_exposure_usdt),
                    "exposure_pct": str(snap.exposure_pct),
                    "drawdown_pct": str(snap.drawdown_pct),
                    "daily_loss_pct": str(snap.daily_loss_pct),
                    "paused": snap.paused,
                }
                for snap in self.portfolio_snapshots
            ],
        }

    def find_by_correlation_id(self, correlation_id: str) -> dict[str, object]:
        """Read-only lookup of all events linked to a correlation ID.

        Does not modify trading or risk state.
        """
        matched_events = [
            {
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "actor": event.actor,
                "details": event.details,
            }
            for event in self.audit_events
            if event.correlation_id == correlation_id
        ]
        matched_fills = [
            {
                "timestamp": fill.timestamp.isoformat(),
                "side": fill.side.value,
                "quantity_btc": str(fill.quantity_btc),
                "fill_price": str(fill.fill_price),
            }
            for fill in self.fills
            if fill.correlation_id == correlation_id
        ]
        matched_rejections = [
            {
                "timestamp": ts.isoformat(),
                "reason_code": code,
                "message": message,
            }
            for ts, code, message, cid in self.risk_rejections
            if cid == correlation_id
        ]
        matched_suppressions = [
            {
                "timestamp": ts.isoformat(),
                "reason_code": code,
                "message": message,
            }
            for ts, code, message, cid in self.opportunity_suppressions
            if cid == correlation_id
        ]
        return {
            "correlation_id": correlation_id,
            "audit_events": matched_events,
            "fills": matched_fills,
            "risk_rejections": matched_rejections,
            "opportunity_suppressions": matched_suppressions,
        }


class PaperReplayEngine:
    """Historical replay with next-hour-open fills, fees, and adverse slippage."""

    def __init__(
        self,
        policy: RiskPolicy,
        opportunity_evidence: dict[str, OpportunityEvidence] | None = None,
        strategies: Sequence[StrategyAgent] | None = None,
        selector: ConsensusSelector | None = None,
        regime_classifier: ShadowRegimeClassifier | None = None,
        *,
        validation_mode: bool = False,
        strategy_set_version: str = "aqis-four-agent-v1",
        startup_candles: int = 250,
        ledger: LedgerDatabase | None = None,
    ) -> None:
        if policy.market != "BTC/USDT" or policy.venue != "binance" or policy.timeframe != "1h":
            raise ValueError("paper replay v1 is restricted to Binance Spot BTC/USDT 1h")
        self.policy = policy
        self.opportunity_evidence = opportunity_evidence or {}
        self.strategies = list(strategies or default_strategy_agents())
        self.selector = selector or ConsensusSelector()
        self.regime_classifier = regime_classifier or ShadowRegimeClassifier()
        self.validation_mode = validation_mode
        self.strategy_set_version = strategy_set_version
        if startup_candles < 50:
            raise ValueError("startup history must cover at least the 50-hour EMA")
        self.startup_candles = startup_candles
        self.risk_engine = RiskEngine(policy)
        self.opportunity_gate = OpportunityGate(policy, strategy_set_version=strategy_set_version)
        self.portfolio_monitor = PortfolioMonitor(policy)
        self.ledger = ledger or LedgerDatabase()
        self._replay_id: str | None = None

    @staticmethod
    def _correlation_id(strategy_set_version: str, candle_timestamp: datetime, candle_index: int) -> str:
        material = f"{strategy_set_version}:{candle_timestamp.isoformat()}:{candle_index}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    def run(self, candles: Sequence[Candle], trade_start_index: int = 0) -> ReplayResult:
        self._validate_stream(candles)
        if not 0 <= trade_start_index < len(candles):
            raise ValueError("trade_start_index must select a candle in the stream")
        for strategy in self.strategies:
            strategy.reset()
        account = PaperAccount(cash_usdt=self.policy.starting_equity_usdt)
        initial = self.policy.starting_equity_usdt
        account.high_watermark_usdt = initial
        account.utc_day_start_equity_usdt = initial
        result = ReplayResult(initial, initial, account)
        self._replay_id = generate_replay_id(
            self.strategy_set_version,
            candles[0].utc_timestamp.isoformat(),
            candles[-1].utc_timestamp.isoformat(),
            len(candles),
        )
        try:
            self.ledger.register_replay(
                replay_id=self._replay_id,
                strategy_set_version=self.strategy_set_version,
                market=self.policy.market,
                venue=self.policy.venue,
                timeframe=self.policy.timeframe,
                first_ts=candles[0].utc_timestamp.isoformat(),
                last_ts=candles[-1].utc_timestamp.isoformat(),
                candle_count=len(candles),
                policy_id=self.policy.policy_id,
                policy_version=self.policy.version,
            )
            self.ledger.record_successful_write()
        except Exception:
            self.ledger.record_failed_write()
        pending_target: TargetPosition | None = None
        pending_reason = ""
        pending_correlation_id: str | None = None
        pending_decision_timestamp: datetime | None = None
        pending_contributing_ids: frozenset[str] = frozenset()
        previous_day = None
        equity_peak = initial
        max_drawdown = Decimal("0")
        was_paused = account.paused

        # Initialise selector statistics.
        result.selector_statistics = {
            "total_decisions": 0,
            "long_targets": 0,
            "cash_targets": 0,
            "buy_quorum": 0,
            "sell_quorum": 0,
            "no_clear_majority": 0,
            "regime_counts": {},
        }

        # Initialise strategy performance registries.
        for strategy in self.strategies:
            result.strategy_performance[strategy.strategy_id] = StrategyPerformance(
                strategy_id=strategy.strategy_id
            )

        for index, candle in enumerate(candles):
            open_time = candle.utc_timestamp - timedelta(hours=1)
            current_day = open_time.date()
            if previous_day is not None and current_day != previous_day:
                account.utc_day_start_equity_usdt = account.equity(Decimal(str(candle.open)))
            previous_day = current_day

            correlation_id = self._correlation_id(
                self.strategy_set_version, candle.utc_timestamp, index
            )

            if pending_target is not None and index >= trade_start_index:
                self._execute_target(
                    result,
                    pending_target,
                    pending_reason,
                    candle,
                    open_time,
                    correlation_id=pending_correlation_id,
                    decision_timestamp=pending_decision_timestamp,
                    contributing_ids=pending_contributing_ids,
                )

            close_price = Decimal(str(candle.close))
            equity = account.equity(close_price)
            equity_peak = max(equity_peak, equity)
            drawdown = ZERO if equity_peak <= ZERO else (equity_peak - equity) / equity_peak
            max_drawdown = max(max_drawdown, drawdown)

            # Portfolio snapshot for this candle (end-of-candle state).
            if index >= trade_start_index:
                window = candles[max(0, index - self.policy.minimum_correlation_samples + 1) : index + 1]
                monitor_snapshot = self.portfolio_monitor.snapshot(
                    account,
                    close_price,
                    candle.utc_timestamp,
                    window,
                    {},
                    execution_errors=account.consecutive_execution_errors,
                )
                result.portfolio_snapshots.append(
                    PortfolioSnapshot(
                        timestamp=candle.utc_timestamp,
                        equity_usdt=monitor_snapshot.equity_usdt,
                        btc_exposure_usdt=monitor_snapshot.btc_exposure_usdt,
                        exposure_pct=monitor_snapshot.exposure_pct,
                        realized_pnl_usdt=monitor_snapshot.realized_pnl_usdt,
                        unrealized_pnl_usdt=monitor_snapshot.unrealized_pnl_usdt,
                        drawdown_pct=monitor_snapshot.drawdown_pct,
                        daily_loss_pct=monitor_snapshot.daily_loss_pct,
                        paused=account.paused,
                    )
                )

            stop_reason = self.risk_engine.pause_if_required(account, close_price)
            newly_paused = stop_reason is not None and not was_paused
            if stop_reason and index >= trade_start_index:
                self._audit(
                    result,
                    candle.utc_timestamp,
                    "risk_pause",
                    "risk_engine",
                    {"reason": stop_reason, "equity_usdt": str(equity), "correlation_id": correlation_id},
                    correlation_id=correlation_id,
                )
                if newly_paused:
                    result.risk_interventions.append(
                        RiskIntervention(
                            timestamp=candle.utc_timestamp,
                            intervention_type="pause_transition",
                            actor="risk_engine",
                            reason=stop_reason,
                            correlation_id=correlation_id,
                        )
                    )
                    result.pause_resume_history.append(
                        RiskIntervention(
                            timestamp=candle.utc_timestamp,
                            intervention_type="pause",
                            actor="risk_engine",
                            reason=stop_reason,
                            correlation_id=correlation_id,
                        )
                    )
                pending_target = TargetPosition.CASH
                pending_reason = f"risk_stop:{stop_reason}"
                pending_correlation_id = correlation_id
                pending_decision_timestamp = candle.utc_timestamp
                pending_contributing_ids = frozenset()
                result.halted = True
                result.halt_reason = stop_reason
                try:
                    self.ledger.record_decision(
                        decision_id=generate_decision_id(
                            self._replay_id, candle.utc_timestamp.isoformat(), index
                        ),
                        replay_id=self._replay_id,
                        correlation_id=correlation_id,
                        candle_timestamp=candle.utc_timestamp.isoformat(),
                        candle_index=index,
                        market_data_snapshot={
                            "timestamp": candle.utc_timestamp.isoformat(),
                            "open": candle.open,
                            "high": candle.high,
                            "low": candle.low,
                            "close": candle.close,
                            "volume": candle.volume,
                            "equity_usdt": str(equity),
                        },
                        regime="risk_stop",
                        strategy_signals=[],
                        selector_decision={
                            "target": TargetPosition.CASH.value,
                            "reason": f"risk_stop:{stop_reason}",
                        },
                        effective_threshold=self.selector._effective_threshold(),
                        opportunity_evidence_id=None,
                        risk_policy_id=self.policy.policy_id,
                        risk_policy_version=self.policy.version,
                    )
                    self.ledger.record_successful_write()
                except Exception:
                    self.ledger.record_failed_write()
            else:
                history = candles[max(0, index - self.startup_candles + 1) : index + 1]
                signals = tuple(strategy.evaluate(history) for strategy in self.strategies)
                selection = self.selector.decide(signals)
                regime = self.regime_classifier.classify(history)
                if index >= trade_start_index:
                    result.decisions.append(CandleDecision(candle, signals, selection, regime))
                    self._audit(
                        result,
                        candle.utc_timestamp,
                        "selector_decision",
                        "selector",
                        {
                            "target": selection.target.value,
                            "buy_votes": list(selection.buy_votes),
                            "sell_votes": list(selection.sell_votes),
                            "abstentions": list(selection.abstentions),
                            "reason": selection.reason,
                            "regime_shadow": regime.regime,
                            "correlation_id": correlation_id,
                        },
                        correlation_id=correlation_id,
                    )
                    try:
                        evidence = self._evidence_for(candle)
                        self.ledger.record_decision(
                            decision_id=correlation_id,
                            replay_id=self._replay_id,
                            correlation_id=correlation_id,
                            candle_timestamp=candle.utc_timestamp.isoformat(),
                            candle_index=index,
                            market_data_snapshot={
                                "timestamp": candle.utc_timestamp.isoformat(),
                                "open": candle.open,
                                "high": candle.high,
                                "low": candle.low,
                                "close": candle.close,
                                "volume": candle.volume,
                                "history_candles": len(history),
                            },
                            regime=regime.regime,
                            strategy_signals=[
                                {
                                    "strategy_id": s.strategy_id,
                                    "strategy_version": s.strategy_version,
                                    "action": s.action.value,
                                    "confidence": s.confidence,
                                    "reason": s.reason,
                                    "strength": s.strength,
                                }
                                for s in signals
                            ],
                            selector_decision={
                                "target": selection.target.value,
                                "buy_votes": list(selection.buy_votes),
                                "sell_votes": list(selection.sell_votes),
                                "abstentions": list(selection.abstentions),
                                "reason": selection.reason,
                            },
                            effective_threshold=self.selector._effective_threshold(),
                            opportunity_evidence_id=(
                                None if evidence is None else evidence.evidence_id
                            ),
                            risk_policy_id=self.policy.policy_id,
                            risk_policy_version=self.policy.version,
                        )
                        self.ledger.record_successful_write()
                    except Exception:
                        self.ledger.record_failed_write()
                    pending_target = selection.target
                    pending_reason = selection.reason
                    pending_correlation_id = correlation_id
                    pending_decision_timestamp = candle.utc_timestamp

                    if selection.target is TargetPosition.LONG:
                        pending_contributing_ids = frozenset(selection.buy_votes)
                    elif selection.reason == "sell_quorum":
                        pending_contributing_ids = frozenset(selection.sell_votes)
                    else:
                        pending_contributing_ids = frozenset()

                    self._update_strategy_performance(result, signals, selection)
                    self._update_selector_statistics(result, selection, regime)
                else:
                    pending_target = None
                    pending_correlation_id = None
                    pending_decision_timestamp = None
                    pending_contributing_ids = frozenset()
                    if index + 1 == trade_start_index:
                        account.utc_day_start_equity_usdt = account.equity(close_price)

            was_paused = account.paused
            result.ending_equity_usdt = equity

        result.maximum_drawdown_pct = max_drawdown

        # Record outcomes for all decisions at defined horizons.
        try:
            self._record_outcomes(candles, result, trade_start_index)
            self.ledger.record_successful_write()
        except Exception:
            self.ledger.record_failed_write()

        return result

    def _update_strategy_performance(
        self, result: ReplayResult, signals: tuple[StrategySignal, ...], selection: SelectorDecision
    ) -> None:
        if selection.target is TargetPosition.LONG:
            contributing = set(selection.buy_votes)
        elif selection.reason == "sell_quorum":
            contributing = set(selection.sell_votes)
        else:
            contributing = set()

        for signal in signals:
            perf = result.strategy_performance.get(signal.strategy_id)
            if perf is None:
                continue
            # Dataclass is frozen; rebuild with updated counts.
            buy_signals = perf.buy_signals
            sell_signals = perf.sell_signals
            abstentions = perf.abstentions
            if signal.action is SignalAction.BUY:
                buy_signals += 1
            elif signal.action is SignalAction.SELL:
                sell_signals += 1
            else:
                abstentions += 1
            valid_votes = perf.valid_votes
            if signal.strategy_id in selection.buy_votes or signal.strategy_id in selection.sell_votes:
                valid_votes += 1
            contributing_votes = perf.contributing_votes
            if signal.strategy_id in contributing:
                contributing_votes += 1
            result.strategy_performance[signal.strategy_id] = StrategyPerformance(
                strategy_id=signal.strategy_id,
                signals=perf.signals + 1,
                buy_signals=buy_signals,
                sell_signals=sell_signals,
                abstentions=abstentions,
                valid_votes=valid_votes,
                contributing_votes=contributing_votes,
                fills_while_voting_with_selector=perf.fills_while_voting_with_selector,
            )

    def _update_selector_statistics(
        self, result: ReplayResult, selection: SelectorDecision, regime: RegimeAssessment
    ) -> None:
        stats = result.selector_statistics
        stats["total_decisions"] = int(stats.get("total_decisions", 0)) + 1
        if selection.target is TargetPosition.LONG:
            stats["long_targets"] = int(stats.get("long_targets", 0)) + 1
        else:
            stats["cash_targets"] = int(stats.get("cash_targets", 0)) + 1
        if selection.reason == "buy_quorum":
            stats["buy_quorum"] = int(stats.get("buy_quorum", 0)) + 1
        elif selection.reason == "sell_quorum":
            stats["sell_quorum"] = int(stats.get("sell_quorum", 0)) + 1
        else:
            stats["no_clear_majority"] = int(stats.get("no_clear_majority", 0)) + 1
        regime_counts = stats.setdefault("regime_counts", {})
        regime_counts[regime.regime] = int(regime_counts.get(regime.regime, 0)) + 1

    def human_resume(self, result: ReplayResult, actor: str) -> None:
        self.risk_engine.human_resume(result.account, actor)
        result.halted = False
        result.halt_reason = None
        resume_ts = datetime.now(timezone.utc)
        self._audit(
            result,
            resume_ts,
            "risk_pause_resumed",
            actor,
            {"policy_version": self.policy.version},
        )
        result.risk_interventions.append(
            RiskIntervention(
                timestamp=resume_ts,
                intervention_type="human_resume",
                actor=actor,
                reason="human resume",
            )
        )
        result.pause_resume_history.append(
            RiskIntervention(
                timestamp=resume_ts,
                intervention_type="resume",
                actor=actor,
                reason="human resume",
            )
        )

    def record_agent_pause_request(
        self, result: ReplayResult, actor: str, reason: str
    ) -> None:
        """Record an advisory agent pause request; does not override risk authority."""
        self.risk_engine.apply_agent_pause_request(result.account, actor, reason)
        request_ts = datetime.now(timezone.utc)
        result.risk_interventions.append(
            RiskIntervention(
                timestamp=request_ts,
                intervention_type="agent_pause_request",
                actor=actor,
                reason=reason,
            )
        )
        result.pause_resume_history.append(
            RiskIntervention(
                timestamp=request_ts,
                intervention_type="pause",
                actor=actor,
                reason=reason,
            )
        )

    def _execute_target(
        self,
        result: ReplayResult,
        target: TargetPosition,
        reason: str,
        candle: Candle,
        timestamp: datetime,
        *,
        correlation_id: str | None = None,
        decision_timestamp: datetime | None = None,
        contributing_ids: frozenset[str] = frozenset(),
    ) -> None:
        account = result.account
        reference_price = Decimal(str(candle.open))
        equity = account.equity(reference_price)
        max_order_notional = equity * self.policy.max_order_pct
        slippage_fraction = self.policy.slippage_bps / Decimal("10000")
        fee_fraction = self.policy.fee_rate

        if target is TargetPosition.LONG and not account.paused:
            current_exposure = account.btc * reference_price
            exposure_headroom = max(ZERO, equity * self.policy.max_position_pct - current_exposure) / (
                ONE + slippage_fraction
            )
            cash_headroom = max(ZERO, account.cash_usdt / (ONE + fee_fraction + slippage_fraction))
            requested_notional = min(max_order_notional, exposure_headroom, cash_headroom)
            if requested_notional <= ZERO:
                return
            evidence = self._evidence_for(candle)
            if not self.validation_mode:
                opportunity = self.opportunity_gate.assess(evidence)
                if not opportunity.eligible or evidence is None:
                    self._suppress_order(
                        result, timestamp, opportunity.reason_code, opportunity.message, correlation_id
                    )
                    return
                budget = self.risk_engine._loss_headroom(account, reference_price)
                if evidence is not None and evidence.stressed_loss_fraction <= ZERO:
                    self._suppress_order(
                        result,
                        timestamp,
                        "invalid_stress_loss",
                        "Stressed-loss estimate must be positive",
                        correlation_id,
                    )
                    return
                requested_notional = min(
                    requested_notional,
                    budget / evidence.stressed_loss_fraction,
                )
            if requested_notional <= ZERO:
                return
            order = OrderIntent(
                side=OrderSide.BUY,
                quantity_btc=requested_notional / reference_price,
                reference_price=reference_price,
                timestamp=timestamp,
                net_edge_lower_bound_pct=(
                    None if evidence is None else evidence.net_edge_lower_bound_pct
                ),
                stressed_loss_fraction=(
                    None if evidence is None else evidence.stressed_loss_fraction
                ),
                evidence_id=None if evidence is None else evidence.evidence_id,
                correlation_id=correlation_id,
                reason=reason,
            )
            result.order_requests.append(
                (timestamp, OrderSide.BUY.value, order.quantity_btc, correlation_id)
            )
            decision = self.risk_engine.evaluate_order(
                order, account, require_opportunity_evidence=not self.validation_mode
            )
            decision = RiskDecision(
                decision.approved,
                decision.reason_code,
                decision.message,
                decision.risk_budget_usdt,
                correlation_id=correlation_id,
            )
            self._audit_risk(result, timestamp, decision, order, correlation_id)
            try:
                self.ledger.record_risk(
                    correlation_id=correlation_id,
                    order_intent={
                        "side": order.side.value,
                        "quantity_btc": str(order.quantity_btc),
                        "reference_price": str(order.reference_price),
                        "timestamp": order.timestamp.isoformat(),
                        "reduce_only": order.reduce_only,
                        "evidence_id": order.evidence_id,
                        "reason": order.reason,
                    },
                    risk_decision={
                        "approved": decision.approved,
                        "reason_code": decision.reason_code,
                        "message": decision.message,
                        "risk_budget_usdt": str(decision.risk_budget_usdt) if decision.risk_budget_usdt is not None else None,
                    },
                    risk_budget_usdt=str(decision.risk_budget_usdt) if decision.risk_budget_usdt is not None else None,
                )
                self.ledger.record_successful_write()
            except Exception:
                self.ledger.record_failed_write()
            if not decision.approved:
                result.risk_rejections.append(
                    (timestamp, decision.reason_code, decision.message, correlation_id)
                )
                try:
                    self.ledger.record_execution(
                        correlation_id=correlation_id,
                        order_attempt_timestamp=timestamp.isoformat(),
                        fill=None,
                        rejection_code=decision.reason_code,
                        execution_timestamp=timestamp.isoformat(),
                    )
                    self.ledger.record_successful_write()
                except Exception:
                    self.ledger.record_failed_write()
                self._reject(result, timestamp, decision.reason_code, decision.message)
                return
            result.approved_orders.append(
                (timestamp, OrderSide.BUY.value, order.quantity_btc, correlation_id)
            )
            fill_price = reference_price * (ONE + slippage_fraction)
            notional = order.quantity_btc * fill_price
            fee = notional * fee_fraction
            slippage_cost = order.quantity_btc * (fill_price - reference_price)
            if notional + fee > account.cash_usdt:
                try:
                    self.ledger.record_execution(
                        correlation_id=correlation_id,
                        order_attempt_timestamp=timestamp.isoformat(),
                        fill=None,
                        rejection_code="insufficient_cash_after_costs",
                        execution_timestamp=timestamp.isoformat(),
                    )
                    self.ledger.record_successful_write()
                except Exception:
                    self.ledger.record_failed_write()
                self._reject(result, timestamp, "insufficient_cash_after_costs", "Cost-adjusted order exceeds cash")
                return
            account.cash_usdt -= notional + fee
            account.btc += order.quantity_btc
            entry_cost = notional + fee
            account.position_cost_basis_usdt += entry_cost
            account.position_entry_cost_accum_usdt += entry_cost
            account.fees_usdt += fee
            account.slippage_usdt += slippage_cost
            self._fill(
                result,
                PaperFill(
                    timestamp,
                    OrderSide.BUY,
                    order.quantity_btc,
                    reference_price,
                    fill_price,
                    fee,
                    slippage_cost,
                    reason,
                    correlation_id=correlation_id,
                    post_fill_equity_usdt=account.cash_usdt
                    + account.btc * reference_price,
                ),
                contributing_ids=contributing_ids,
            )
            try:
                self.ledger.record_execution(
                    correlation_id=correlation_id,
                    order_attempt_timestamp=timestamp.isoformat(),
                    fill={
                        "side": OrderSide.BUY.value,
                        "quantity_btc": str(order.quantity_btc),
                        "reference_price": str(reference_price),
                        "fill_price": str(fill_price),
                        "fee_usdt": str(fee),
                        "slippage_usdt": str(slippage_cost),
                        "reason": reason,
                    },
                    rejection_code=None,
                    execution_timestamp=timestamp.isoformat(),
                )
                self.ledger.record_successful_write()
            except Exception:
                self.ledger.record_failed_write()
            return

        # Any non-long target is a cash target. It may reduce exposure while paused.
        if target is TargetPosition.CASH and account.btc > ZERO:
            quantity = min(account.btc, max_order_notional / reference_price)
            if quantity <= ZERO:
                return
            order = OrderIntent(
                side=OrderSide.SELL,
                quantity_btc=quantity,
                reference_price=reference_price,
                timestamp=timestamp,
                reduce_only=True,
                correlation_id=correlation_id,
                reason=reason,
            )
            result.order_requests.append(
                (timestamp, OrderSide.SELL.value, quantity, correlation_id)
            )
            decision = self.risk_engine.evaluate_order(order, account)
            decision = RiskDecision(
                decision.approved,
                decision.reason_code,
                decision.message,
                decision.risk_budget_usdt,
                correlation_id=correlation_id,
            )
            self._audit_risk(result, timestamp, decision, order, correlation_id)
            try:
                self.ledger.record_risk(
                    correlation_id=correlation_id,
                    order_intent={
                        "side": order.side.value,
                        "quantity_btc": str(order.quantity_btc),
                        "reference_price": str(order.reference_price),
                        "timestamp": order.timestamp.isoformat(),
                        "reduce_only": order.reduce_only,
                        "evidence_id": order.evidence_id,
                        "reason": order.reason,
                    },
                    risk_decision={
                        "approved": decision.approved,
                        "reason_code": decision.reason_code,
                        "message": decision.message,
                        "risk_budget_usdt": str(decision.risk_budget_usdt) if decision.risk_budget_usdt is not None else None,
                    },
                    risk_budget_usdt=str(decision.risk_budget_usdt) if decision.risk_budget_usdt is not None else None,
                )
                self.ledger.record_successful_write()
            except Exception:
                self.ledger.record_failed_write()
            if not decision.approved:
                result.risk_rejections.append(
                    (timestamp, decision.reason_code, decision.message, correlation_id)
                )
                try:
                    self.ledger.record_execution(
                        correlation_id=correlation_id,
                        order_attempt_timestamp=timestamp.isoformat(),
                        fill=None,
                        rejection_code=decision.reason_code,
                        execution_timestamp=timestamp.isoformat(),
                    )
                    self.ledger.record_successful_write()
                except Exception:
                    self.ledger.record_failed_write()
                self._reject(result, timestamp, decision.reason_code, decision.message)
                return
            result.approved_orders.append(
                (timestamp, OrderSide.SELL.value, quantity, correlation_id)
            )
            fill_price = reference_price * (ONE - slippage_fraction)
            proceeds = quantity * fill_price
            fee = proceeds * fee_fraction
            slippage_cost = quantity * (reference_price - fill_price)
            account.cash_usdt += proceeds - fee
            account.btc -= quantity
            allocated_cost = (
                account.position_cost_basis_usdt
                if quantity == account.btc + quantity
                else account.position_cost_basis_usdt * quantity / (account.btc + quantity)
            )
            account.position_cost_basis_usdt = max(ZERO, account.position_cost_basis_usdt - allocated_cost)
            account.position_realized_pnl_usdt += proceeds - fee - allocated_cost
            account.fees_usdt += fee
            account.slippage_usdt += slippage_cost
            account.realized_pnl_usdt += proceeds - allocated_cost - fee
            if account.btc <= Decimal("0.00000000000001"):
                account.btc = ZERO
                if account.position_entry_cost_accum_usdt > ZERO:
                    result.completed_trade_returns_pct.append(
                        account.position_realized_pnl_usdt / account.position_entry_cost_accum_usdt
                    )
                account.position_cost_basis_usdt = ZERO
                account.position_entry_cost_accum_usdt = ZERO
                account.position_realized_pnl_usdt = ZERO
            self._fill(
                result,
                PaperFill(
                    timestamp,
                    OrderSide.SELL,
                    quantity,
                    reference_price,
                    fill_price,
                    fee,
                    slippage_cost,
                    reason,
                    correlation_id=correlation_id,
                    post_fill_equity_usdt=account.cash_usdt
                    + account.btc * reference_price,
                ),
                contributing_ids=contributing_ids,
            )
            try:
                self.ledger.record_execution(
                    correlation_id=correlation_id,
                    order_attempt_timestamp=timestamp.isoformat(),
                    fill={
                        "side": OrderSide.SELL.value,
                        "quantity_btc": str(quantity),
                        "reference_price": str(reference_price),
                        "fill_price": str(fill_price),
                        "fee_usdt": str(fee),
                        "slippage_usdt": str(slippage_cost),
                        "reason": reason,
                    },
                    rejection_code=None,
                    execution_timestamp=timestamp.isoformat(),
                )
                self.ledger.record_successful_write()
            except Exception:
                self.ledger.record_failed_write()

    def _evidence_for(self, candle: Candle) -> OpportunityEvidence | None:
        # Regime-based selection is shadow-only in v1; the active opportunity
        # evidence therefore uses the frozen default ensemble record.
        evidence = self.opportunity_evidence.get("default")
        if evidence is not None and evidence.strategy_set_version != self.strategy_set_version:
            return None
        return evidence

    def _validate_stream(self, candles: Sequence[Candle]) -> None:
        if len(candles) < 2:
            raise ValueError("paper replay requires at least two finalized hourly candles")
        for previous, current in zip(candles, candles[1:]):
            if current.utc_timestamp - previous.utc_timestamp != timedelta(hours=1):
                raise ValueError("missing or non-hourly candle; data-integrity halt required")

    @staticmethod
    def _audit(
        result: ReplayResult,
        timestamp: datetime,
        event_type: str,
        actor: str,
        details: dict[str, object],
        correlation_id: str | None = None,
    ) -> None:
        result.audit_events.append(AuditEvent(timestamp, event_type, actor, details, correlation_id=correlation_id))

    def _audit_risk(
        self,
        result: ReplayResult,
        timestamp: datetime,
        decision: RiskDecision,
        order: OrderIntent,
        correlation_id: str | None = None,
    ) -> None:
        self._audit(
            result,
            timestamp,
            "risk_decision",
            "risk_engine",
            {
                "approved": decision.approved,
                "reason_code": decision.reason_code,
                "message": decision.message,
                "side": order.side.value,
                "quantity_btc": str(order.quantity_btc),
                "policy_id": self.policy.policy_id,
                "policy_version": self.policy.version,
                "evidence_id": order.evidence_id,
                "correlation_id": correlation_id,
            },
            correlation_id=correlation_id,
        )

    def _reject(
        self, result: ReplayResult, timestamp: datetime, code: str, message: str
    ) -> None:
        result.rejected_orders.append((timestamp, code, message))
        self._audit(result, timestamp, "order_rejected", "risk_engine", {"reason_code": code, "message": message})

    def _suppress_order(
        self,
        result: ReplayResult,
        timestamp: datetime,
        code: str,
        message: str,
        correlation_id: str | None = None,
    ) -> None:
        """Record an opportunity-gate suppression (not a risk-engine rejection)."""
        result.opportunity_suppressions.append((timestamp, code, message, correlation_id))
        result.rejected_orders.append((timestamp, code, message))
        self._audit(
            result,
            timestamp,
            "order_suppressed",
            "opportunity_gate",
            {"reason_code": code, "message": message, "correlation_id": correlation_id},
            correlation_id=correlation_id,
        )

    def _fill(
        self,
        result: ReplayResult,
        fill: PaperFill,
        contributing_ids: frozenset[str] = frozenset(),
    ) -> None:
        result.fills.append(fill)
        result.account.consecutive_execution_errors = 0
        for strategy_id in contributing_ids:
            perf = result.strategy_performance.get(strategy_id)
            if perf is None:
                continue
            result.strategy_performance[strategy_id] = StrategyPerformance(
                strategy_id=strategy_id,
                signals=perf.signals,
                buy_signals=perf.buy_signals,
                sell_signals=perf.sell_signals,
                abstentions=perf.abstentions,
                valid_votes=perf.valid_votes,
                contributing_votes=perf.contributing_votes,
                fills_while_voting_with_selector=perf.fills_while_voting_with_selector + 1,
            )
        self._audit(
            result,
            fill.timestamp,
            "paper_fill",
            "paper_engine",
            {
                "side": fill.side.value,
                "quantity_btc": str(fill.quantity_btc),
                "reference_price": str(fill.reference_price),
                "fill_price": str(fill.fill_price),
                "fee_usdt": str(fill.fee_usdt),
                "slippage_usdt": str(fill.slippage_usdt),
                "reason": fill.reason,
                "correlation_id": fill.correlation_id,
            },
            correlation_id=fill.correlation_id,
        )

    def _record_outcomes(
        self,
        candles: list[Candle],
        result: ReplayResult,
        trade_start_index: int,
    ) -> None:
        """Record market and portfolio outcomes at defined horizons.

        Measurement anchoring:
        - Fill-anchored (anchor="fill"): excursions run from the fill candle
          (the candle whose open interval contains the fill timestamp) to the
          horizon candle, both inclusive. Baselines are the effective fill
          price (reference +/- slippage) and the post-fill equity marked at
          the fill reference price.
        - Decision-anchored (anchor="decision"): used when no fill exists.
          Baseline is the decision candle close; the excursion window is the
          same, [decision_index + 1, decision_index + w], because the
          decision candle close is the decision instant.

        Excursions are signed by the decision direction d (+1 long intent,
        -1 exit/cash intent, None for flat cash no-ops measured on raw price
        direction, stored as +1 with direction NULL).
        MAE/MFE are stored as positive magnitudes in bps of the baseline
        (market) or as positive USDT magnitudes (portfolio). The horizon
        return is signed by d and stored in bps of the baseline.

        Insert-only: no updates to existing records. Missing horizons are
        represented by status events with transition-unique status IDs.
        """
        HORIZON_CANDLES = {"1h": 1, "4h": 4, "24h": 24, "7d": 168}

        candle_index_by_ts = {
            c.utc_timestamp.isoformat(): i for i, c in enumerate(candles)
        }
        fills_by_corr = {
            fill.correlation_id: fill for fill in result.fills
        }
        snapshot_by_ts = {
            snap.timestamp.isoformat(): snap for snap in result.portfolio_snapshots
        }
        rejected_by_corr: dict[str, str] = {}
        for ts, code, message, corr_id in result.risk_rejections:
            if corr_id is not None:
                rejected_by_corr[corr_id] = code
        suppressed_by_corr: dict[str, str] = {}
        for ts, code, message, corr_id in result.opportunity_suppressions:
            if corr_id is not None:
                suppressed_by_corr[corr_id] = code
        order_requested_corr: set[str] = {
            order[3] for order in result.order_requests if order[3] is not None
        }

        def record_status(corr_id: str, horizon: str, status: str) -> None:
            try:
                self.ledger.record_outcome_status(
                    correlation_id=corr_id, horizon=horizon, status=status
                )
                self.ledger.record_successful_write()
            except Exception:
                self.ledger.record_failed_write()

        def record_market_outcome(
            corr_id: str,
            horizon: str,
            decision_candle: Candle,
            decision_index: int,
            fill: PaperFill | None,
            direction: int | None,
        ) -> None:
            horizon_index = decision_index + HORIZON_CANDLES[horizon]
            horizon_candle = candles[horizon_index]
            window_start = decision_index + 1
            if fill is not None:
                mapped = execution_candle_index(candles, fill.timestamp)
                window_start = mapped if mapped is not None else window_start
            window = candles[window_start : horizon_index + 1]
            if fill is not None:
                anchor = "fill"
                baseline = fill.fill_price
            else:
                anchor = "decision"
                baseline = Decimal(str(decision_candle.close))
            sign = Decimal(1 if direction is None else direction)
            adverse = ZERO
            favourable = ZERO
            for window_candle in window:
                low_delta = sign * (Decimal(str(window_candle.low)) - baseline)
                high_delta = sign * (Decimal(str(window_candle.high)) - baseline)
                adverse = min(adverse, low_delta, high_delta)
                favourable = max(favourable, low_delta, high_delta)
            mae_bps = (
                max(ZERO, -adverse) / baseline * BPS
            ).quantize(BPS_QUANTUM)
            mfe_bps = (
                max(ZERO, favourable) / baseline * BPS
            ).quantize(BPS_QUANTUM)
            horizon_return_bps = (
                sign
                * (Decimal(str(horizon_candle.close)) - baseline)
                / baseline
                * BPS
            ).quantize(BPS_QUANTUM)
            try:
                self.ledger.record_outcome(
                    correlation_id=corr_id,
                    horizon=horizon,
                    horizon_timestamp=horizon_candle.utc_timestamp.isoformat(),
                    outcome_type="market",
                    anchor=anchor,
                    direction=direction,
                    baseline_price=str(baseline),
                    baseline_equity=None,
                    horizon_return_bps=str(horizon_return_bps),
                    price_at_horizon=str(horizon_candle.close),
                    portfolio_equity_at_horizon=None,
                    realized_pnl_at_horizon=None,
                    unrealized_pnl_at_horizon=None,
                    max_adverse_excursion=str(mae_bps),
                    max_favourable_excursion=str(mfe_bps),
                )
                self.ledger.record_successful_write()
            except Exception:
                self.ledger.record_failed_write()

        def record_portfolio_outcome(
            corr_id: str,
            horizon: str,
            decision_index: int,
            fill: PaperFill,
        ) -> None:
            horizon_index = decision_index + HORIZON_CANDLES[horizon]
            horizon_candle = candles[horizon_index]
            snapshot = snapshot_by_ts.get(horizon_candle.utc_timestamp.isoformat())
            baseline = fill.post_fill_equity_usdt
            if snapshot is None or baseline is None:
                record_status(corr_id, horizon, "data_insufficient")
                return
            window_equities = []
            for candle_index in range(decision_index + 1, horizon_index + 1):
                window_snapshot = snapshot_by_ts.get(
                    candles[candle_index].utc_timestamp.isoformat()
                )
                if window_snapshot is None:
                    record_status(corr_id, horizon, "data_insufficient")
                    return
                window_equities.append(window_snapshot.equity_usdt)
            mae_usdt = (max(ZERO, baseline - min(window_equities))).quantize(USDT_QUANTUM)
            mfe_usdt = (max(ZERO, max(window_equities) - baseline)).quantize(USDT_QUANTUM)
            horizon_return_bps = (
                (snapshot.equity_usdt - baseline) / baseline * BPS
            ).quantize(BPS_QUANTUM)
            try:
                self.ledger.record_outcome(
                    correlation_id=corr_id,
                    horizon=horizon,
                    horizon_timestamp=snapshot.timestamp.isoformat(),
                    outcome_type="portfolio",
                    anchor="fill",
                    direction=1 if fill.side is OrderSide.BUY else -1,
                    baseline_price=str(fill.fill_price),
                    baseline_equity=str(baseline),
                    horizon_return_bps=str(horizon_return_bps),
                    price_at_horizon=str(horizon_candle.close),
                    portfolio_equity_at_horizon=str(snapshot.equity_usdt),
                    realized_pnl_at_horizon=str(snapshot.realized_pnl_usdt),
                    unrealized_pnl_at_horizon=str(snapshot.unrealized_pnl_usdt),
                    max_adverse_excursion=str(mae_usdt),
                    max_favourable_excursion=str(mfe_usdt),
                )
                self.ledger.record_successful_write()
            except Exception:
                self.ledger.record_failed_write()

        def record_no_fill_status(
            decision_index: int,
            corr_id: str,
            direction: int | None,
        ) -> None:
            if corr_id in suppressed_by_corr:
                record_status(
                    corr_id, horizon,
                    f"no_fill_suppressed:{suppressed_by_corr[corr_id]}",
                )
                return
            if corr_id in rejected_by_corr:
                record_status(
                    corr_id, horizon,
                    f"no_fill_rejected:{rejected_by_corr[corr_id]}",
                )
                return
            if decision_index == len(candles) - 1:
                record_status(corr_id, horizon, "no_order_end_of_replay")
                return
            if corr_id in order_requested_corr:
                record_status(corr_id, horizon, "no_fill")
                return
            if direction is None:
                record_status(corr_id, horizon, "no_action_flat")
                return
            record_status(corr_id, horizon, "no_fill")

        recorded_corr: set[str] = set()
        for candle_decision in result.decisions:
            decision_index = candle_index_by_ts.get(
                candle_decision.candle.utc_timestamp.isoformat()
            )
            if decision_index is None:
                continue
            corr_id = self._correlation_id(
                self.strategy_set_version,
                candle_decision.candle.utc_timestamp,
                decision_index,
            )
            recorded_corr.add(corr_id)
            fill = fills_by_corr.get(corr_id)
            if fill is not None:
                direction = 1 if fill.side is OrderSide.BUY else -1
            elif candle_decision.selection.target is TargetPosition.LONG:
                direction = 1
            else:
                direction = None
            for horizon in HORIZON_CANDLES:
                if decision_index + HORIZON_CANDLES[horizon] >= len(candles):
                    record_status(corr_id, horizon, "horizon_not_reached")
                    continue
                record_market_outcome(
                    corr_id,
                    horizon,
                    candle_decision.candle,
                    decision_index,
                    fill,
                    direction,
                )
                if fill is not None:
                    record_portfolio_outcome(
                        corr_id, horizon, decision_index, fill
                    )
                else:
                    record_no_fill_status(decision_index, corr_id, direction)

        risk_stop_exits = [
            fill
            for fill in result.fills
            if fill.correlation_id is not None
            and fill.correlation_id not in recorded_corr
        ]
        for fill in risk_stop_exits:
            corr_id = fill.correlation_id
            execution_index = execution_candle_index(candles, fill.timestamp)
            if execution_index is None or execution_index == 0:
                continue
            decision_index = execution_index - 1
            for horizon in HORIZON_CANDLES:
                if decision_index + HORIZON_CANDLES[horizon] >= len(candles):
                    record_status(corr_id, horizon, "horizon_not_reached")
                    continue
                record_market_outcome(
                    corr_id,
                    horizon,
                    candles[decision_index],
                    decision_index,
                    fill,
                    -1,
                )
                record_portfolio_outcome(
                    corr_id, horizon, decision_index, fill
                )


BPS_QUANTUM = Decimal("0.0001")
USDT_QUANTUM = Decimal("0.01")
BPS = Decimal("10000")


def execution_candle_index(candles: list[Candle], fill_ts: datetime) -> int | None:
    """First candle index whose interval contains the fill timestamp.

    Candle timestamps are close times, so the execution candle is the first
    candle with utc_timestamp >= fill timestamp + 1h. A naive lookup keyed on
    the fill close-ish timestamp resolves to the decision candle instead.
    """
    threshold = fill_ts + timedelta(hours=1)
    for index, candle in enumerate(candles):
        if candle.utc_timestamp >= threshold:
            return index
    return None


ZERO = Decimal("0")
ONE = Decimal("1")
