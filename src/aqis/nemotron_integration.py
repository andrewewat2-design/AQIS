"""Nemotron shadow integration module.

Provides a read-only research observer that analyses completed AQIS replay
results without influencing trading decisions, risk controls, portfolio state,
or paper execution. All Nemotron output is advisory and stored separately from
trading and risk events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from types import MappingProxyType

from aqis.models import AuditEvent, Candle, SelectorDecision, StrategySignal
from aqis.nemotron import (
    FindingKind,
    MockNemotronProvider,
    NemotronProvider,
    NemotronResearchAgent,
    ResearchReport,
)
from aqis.paper import ReplayResult


@dataclass(frozen=True, slots=True)
class ResearchSnapshot:
    """Read-only snapshot of replay results supplied to Nemotron for analysis.

    This is the only data Nemotron receives. It contains no order placement,
    position modification, or risk policy change capabilities.
    """

    timestamp_utc: datetime
    strategy_set_version: str
    total_candles: int
    traded_candles: int
    portfolio_snapshots: tuple[dict[str, object], ...]
    selector_decisions: tuple[dict[str, object], ...]
    signals: tuple[dict[str, object], ...]
    regimes: tuple[dict[str, object], ...]
    risk_rejections: tuple[dict[str, object], ...]
    opportunity_suppressions: tuple[dict[str, object], ...]
    risk_interventions: tuple[dict[str, object], ...]
    pause_resume_history: tuple[dict[str, object], ...]
    strategy_performance: tuple[dict[str, object], ...]
    selector_statistics: dict[str, object]
    starting_equity_usdt: str
    ending_equity_usdt: str
    halted: bool
    halt_reason: str | None


@dataclass(slots=True)
class NemotronObserver:
    """Read-only research observer for completed AQIS replay results.

    Runs after replay completion. Nemotron analyses immutable snapshots and
    produces research reports that are stored separately from trading and risk
    events. The observer cannot modify replay results, account state, risk
    policies, or execution state.
    """

    agent: NemotronResearchAgent = field(default_factory=lambda: NemotronResearchAgent())
    reports: list[ResearchReport] = field(default_factory=list)

    def observe(self, result: ReplayResult) -> list[ResearchReport]:
        """Analyse a completed replay result and produce research reports.

        Does not modify result, account, risk engine, or any trading state.
        Returns the newly produced reports (also appended to self.reports).
        """
        snapshot = self._build_snapshot(result)
        reports: list[ResearchReport] = []

        # Market conditions research from portfolio snapshots and regimes.
        reports.append(
            self._safe_research(
                self.agent.research_market_summary,
                {
                    "strategy_set_version": snapshot.strategy_set_version,
                    "total_candles": snapshot.total_candles,
                    "traded_candles": snapshot.traded_candles,
                    "regimes": [self._mutable_copy(r) for r in snapshot.regimes],
                    "portfolio": {
                        "starting_equity_usdt": snapshot.starting_equity_usdt,
                        "ending_equity_usdt": snapshot.ending_equity_usdt,
                        "halted": snapshot.halted,
                        "halt_reason": snapshot.halt_reason,
                    },
                },
            )
        )

        # Portfolio performance research.
        reports.append(
            self._safe_research(
                self.agent.research_portfolio_performance,
                {
                    "starting_equity_usdt": snapshot.starting_equity_usdt,
                    "ending_equity_usdt": snapshot.ending_equity_usdt,
                    "portfolio_snapshots": [self._mutable_copy(s) for s in snapshot.portfolio_snapshots],
                    "halted": snapshot.halted,
                },
            )
        )

        # Strategy disagreement research.
        reports.append(
            self._safe_research(
                lambda facts: self.agent.investigate_strategy_disagreement(
                    facts["signals"], facts["selections"]
                ),
                {
                    "signals": [self._mutable_copy(s) for s in snapshot.signals],
                    "selections": [self._mutable_copy(s) for s in snapshot.selector_decisions],
                },
            )
        )

        # Unusual conditions research.
        reports.append(
            self._safe_research(
                lambda facts: self.agent.identify_unusual_conditions(
                    facts["regimes"], facts["interventions"]
                ),
                {
                    "regimes": [self._mutable_copy(r) for r in snapshot.regimes],
                    "interventions": [self._mutable_copy(i) for i in snapshot.risk_interventions],
                },
            )
        )

        # Rejected orders and interventions research.
        reports.append(
            self._safe_research(
                lambda facts: self.agent.review_rejected_orders(
                    facts["risk_rejections"], facts["opportunity_suppressions"]
                ),
                {
                    "risk_rejections": [self._mutable_copy(r) for r in snapshot.risk_rejections],
                    "opportunity_suppressions": [self._mutable_copy(s) for s in snapshot.opportunity_suppressions],
                    "interventions": [self._mutable_copy(i) for i in snapshot.risk_interventions],
                    "pause_resume_history": [self._mutable_copy(p) for p in snapshot.pause_resume_history],
                },
            )
        )

        # Strategy experiment recommendations.
        reports.append(
            self._safe_research(
                self.agent.recommend_strategy_experiments,
                {
                    "strategy_performance": [self._mutable_copy(p) for p in snapshot.strategy_performance],
                    "selector_statistics": self._mutable_copy(snapshot.selector_statistics),
                },
            )
        )

        # Store reports separately from trading and risk events.
        self.reports.extend(reports)
        result.nemotron_reports.extend(reports)
        return reports

    def _safe_research(
        self,
        method: Any,
        facts: dict[str, object],
    ) -> ResearchReport:
        """Call a research method safely; failures produce fallback reports.

        Provider failures never interrupt replay or raise to the caller.
        """
        try:
            return method(facts)
        except Exception:
            # Provider failure: produce a fallback advisory report without
            # interrupting deterministic AQIS operation.
            model_info = self.agent.provider.model_info()
            analysis_id = NemotronResearchAgent._analysis_id(
                "fallback", model_info, "provider_failure"
            )
            return ResearchReport(
                analysis_id=analysis_id,
                model_name=model_info.get("model_name", "unknown"),
                model_version=model_info.get("model_version", "unknown"),
                timestamp_utc=datetime.now(timezone.utc),
                topic="fallback",
                findings=tuple(),
                summary="Provider failure; no model opinion formed.",
                confidence=0.0,
                raw_response={"error": "provider_failure", "fallback": True},
            )

    def _immutable_dict(self, data: dict[str, object]) -> MappingProxyType[str, object]:
        """Return a read-only proxy over a dict to prevent in-place mutation.

        MappingProxyType preserves serialization compatibility and provides a
        defensive view without copying the underlying data at snapshot build
        time. Fresh mutable copies are created at the provider boundary.
        """
        return MappingProxyType(data)

    def _immutable_tuple_of_dicts(
        self, items: list[dict[str, object]]
    ) -> tuple[MappingProxyType[str, object], ...]:
        """Return a tuple of read-only dict proxies."""
        return tuple(self._immutable_dict(item) for item in items)

    def _mutable_copy(self, data: MappingProxyType[str, object]) -> dict[str, object]:
        """Create a fresh mutable copy for the provider boundary.

        Ensures the provider receives its own copy and cannot mutate the
        snapshot or affect subsequent research calls.
        """
        return dict(data)

    def _build_snapshot(self, result: ReplayResult) -> ResearchSnapshot:
        """Build a read-only snapshot from a completed replay result.

        Only extracts data; never mutates result or any contained state.
        All nested dicts are wrapped in read-only proxies.
        """
        portfolio_list = [
            {
                "timestamp": snap.timestamp.isoformat(),
                "equity_usdt": str(snap.equity_usdt),
                "btc_exposure_usdt": str(snap.btc_exposure_usdt),
                "exposure_pct": str(snap.exposure_pct),
                "drawdown_pct": str(snap.drawdown_pct),
                "daily_loss_pct": str(snap.daily_loss_pct),
                "paused": snap.paused,
            }
            for snap in result.portfolio_snapshots
        ]
        decision_list = [
            {
                "timestamp": dec.candle.utc_timestamp.isoformat(),
                "target": dec.selection.target.value,
                "buy_votes": list(dec.selection.buy_votes),
                "sell_votes": list(dec.selection.sell_votes),
                "abstentions": list(dec.selection.abstentions),
                "reason": dec.selection.reason,
                "regime": dec.regime.regime,
            }
            for dec in result.decisions
        ]
        signal_list = [
            {
                "strategy_id": signal.strategy_id,
                "strategy_version": signal.strategy_version,
                "action": signal.action.value,
                "confidence": signal.confidence,
                "reason": signal.reason,
            }
            for dec in result.decisions
            for signal in dec.signals
        ]
        regime_list = [
            {
                "timestamp": dec.candle.utc_timestamp.isoformat(),
                "regime": dec.regime.regime,
                "trend_strength": dec.regime.trend_strength,
                "hourly_atr_fraction": dec.regime.hourly_atr_fraction,
            }
            for dec in result.decisions
        ]
        rejection_list = [
            {
                "timestamp": ts.isoformat(),
                "reason_code": code,
                "message": message,
                "correlation_id": cid,
            }
            for ts, code, message, cid in result.risk_rejections
        ]
        suppression_list = [
            {
                "timestamp": ts.isoformat(),
                "reason_code": code,
                "message": message,
                "correlation_id": cid,
            }
            for ts, code, message, cid in result.opportunity_suppressions
        ]
        intervention_list = [
            {
                "timestamp": intervention.timestamp.isoformat(),
                "type": intervention.intervention_type,
                "actor": intervention.actor,
                "reason": intervention.reason,
                "correlation_id": intervention.correlation_id,
            }
            for intervention in result.risk_interventions
        ]
        pause_resume_list = [
            {
                "timestamp": event.timestamp.isoformat(),
                "type": event.intervention_type,
                "actor": event.actor,
                "reason": event.reason,
                "correlation_id": event.correlation_id,
            }
            for event in result.pause_resume_history
        ]
        performance_list = [
            {
                "strategy_id": perf.strategy_id,
                "signals": perf.signals,
                "buy_signals": perf.buy_signals,
                "sell_signals": perf.sell_signals,
                "abstentions": perf.abstentions,
                "valid_votes": perf.valid_votes,
                "contributing_votes": perf.contributing_votes,
                "fills_while_voting_with_selector": perf.fills_while_voting_with_selector,
            }
            for perf in result.strategy_performance.values()
        ]
        selector_stats = dict(result.selector_statistics)

        return ResearchSnapshot(
            timestamp_utc=datetime.now(timezone.utc),
            strategy_set_version=getattr(result, "strategy_set_version", "aqis-four-agent-v1"),
            total_candles=0,
            traded_candles=len(result.decisions),
            portfolio_snapshots=self._immutable_tuple_of_dicts(portfolio_list),
            selector_decisions=self._immutable_tuple_of_dicts(decision_list),
            signals=self._immutable_tuple_of_dicts(signal_list),
            regimes=self._immutable_tuple_of_dicts(regime_list),
            risk_rejections=self._immutable_tuple_of_dicts(rejection_list),
            opportunity_suppressions=self._immutable_tuple_of_dicts(suppression_list),
            risk_interventions=self._immutable_tuple_of_dicts(intervention_list),
            pause_resume_history=self._immutable_tuple_of_dicts(pause_resume_list),
            strategy_performance=self._immutable_tuple_of_dicts(performance_list),
            selector_statistics=self._immutable_dict(selector_stats),
            starting_equity_usdt=str(result.starting_equity_usdt),
            ending_equity_usdt=str(result.ending_equity_usdt),
            halted=result.halted,
            halt_reason=result.halt_reason,
        )


def attach_nemotron_research(
    result: ReplayResult,
    agent: NemotronResearchAgent | None = None,
) -> list[ResearchReport]:
    """Convenience function: run Nemotron research on a completed replay result.

    Reports are stored on result.nemotron_reports separately from audit events.
    Does not modify trading or risk state.
    """
    if agent is None:
        agent = NemotronResearchAgent()
    observer = NemotronObserver(agent=agent)
    return observer.observe(result)
