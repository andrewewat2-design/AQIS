from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from aqis.models import OpportunityEvidence, RiskPolicy


@dataclass(frozen=True, slots=True)
class OpportunityDecision:
    eligible: bool
    reason_code: str
    message: str


class OpportunityGate:
    """Consumes frozen out-of-sample evidence; never estimates edge from confidence."""

    def __init__(
        self,
        policy: RiskPolicy,
        minimum_validation_trades: int = 30,
        strategy_set_version: str = "aqis-four-agent-v1",
    ) -> None:
        self.policy = policy
        self.minimum_validation_trades = minimum_validation_trades
        self.strategy_set_version = strategy_set_version

    def assess(self, evidence: OpportunityEvidence | None) -> OpportunityDecision:
        if evidence is None:
            return OpportunityDecision(False, "missing_evidence", "No frozen validation evidence")
        if evidence.strategy_set_version != self.strategy_set_version:
            return OpportunityDecision(False, "strategy_version_mismatch", "Evidence is for a different strategy set version")
        if evidence.risk_policy_id != self.policy.policy_id or evidence.risk_policy_version != self.policy.version:
            return OpportunityDecision(False, "risk_policy_version_mismatch", "Evidence was validated under a different risk policy")
        if evidence.validation_trade_count < self.minimum_validation_trades:
            return OpportunityDecision(
                False,
                "insufficient_validation_sample",
                f"Need at least {self.minimum_validation_trades} validation trades",
            )
        if evidence.net_edge_lower_bound_pct <= self.policy.min_net_edge_pct:
            return OpportunityDecision(
                False,
                "net_edge_below_hurdle",
                "Conservative expected net edge does not exceed the approved hurdle",
            )
        if evidence.stressed_loss_fraction <= Decimal("0"):
            return OpportunityDecision(
                False,
                "invalid_stress_loss",
                "Validation evidence must include a positive stressed-loss estimate",
            )
        return OpportunityDecision(True, "eligible", "Frozen validation evidence passes the opportunity hurdle")
