from __future__ import annotations

import hashlib
import json
import random
import statistics
from decimal import Decimal
from typing import Sequence

from aqis.models import OpportunityEvidence, ValidationReport


def build_validation_report(
    *,
    report_id: str,
    strategy_set_version: str,
    regime: str,
    net_trade_returns_pct: Sequence[Decimal],
    data_fingerprint: str,
    minimum_trades: int = 30,
    expected_shortfall_quantile: Decimal = Decimal("0.95"),
    minimum_net_edge_pct: Decimal = Decimal("0"),
    maximum_drawdown_pct: Decimal = Decimal("0"),
    maximum_allowed_drawdown_pct: Decimal = Decimal("0.05"),
    risk_policy_id: str = "aqis-paper-btcusdt",
    risk_policy_version: int = 1,
) -> ValidationReport:
    """Build conservative evidence from a frozen chronological holdout segment.

    A deterministic circular moving-block bootstrap is used for the lower bound
    so adjacent trade outcomes are not treated as fully independent.
    """
    returns = [float(value) for value in net_trade_returns_pct]
    if not returns:
        raise ValueError("validation requires completed net trade returns")
    count = len(returns)
    mean = statistics.fmean(returns)
    seed_bytes = hashlib.sha256(",".join(map(str, returns)).encode("utf-8")).digest()[:8]
    rng = random.Random(int.from_bytes(seed_bytes, "big"))
    block_length = max(1, int(count**0.5))
    bootstrap_means: list[float] = []
    for _ in range(1000):
        sample: list[float] = []
        while len(sample) < count:
            start = rng.randrange(count)
            sample.extend(returns[(start + offset) % count] for offset in range(block_length))
        bootstrap_means.append(statistics.fmean(sample[:count]))
    bootstrap_means.sort()
    lower_bound = Decimal(str(bootstrap_means[int(0.05 * (len(bootstrap_means) - 1))]))
    sorted_returns = sorted(returns)
    tail_count = max(1, int((1 - float(expected_shortfall_quantile)) * count + 0.999999))
    tail_returns = sorted_returns[:tail_count]
    expected_shortfall = Decimal(str(max(0.0, -statistics.fmean(tail_returns))))
    fingerprint = hashlib.sha256(
        json.dumps(
            [str(value) for value in net_trade_returns_pct], separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    data_hash = hashlib.sha256(f"{data_fingerprint}:{fingerprint}".encode()).hexdigest()
    evidence = OpportunityEvidence(
        evidence_id=f"{report_id}:{regime}",
        strategy_set_version=strategy_set_version,
        regime=regime,
        net_edge_lower_bound_pct=lower_bound,
        stressed_loss_fraction=expected_shortfall,
        validation_trade_count=count,
        data_hash=data_hash,
        risk_policy_id=risk_policy_id,
        risk_policy_version=risk_policy_version,
    )
    passed = (
        count >= minimum_trades
        and lower_bound > minimum_net_edge_pct
        and expected_shortfall > 0
        and maximum_drawdown_pct <= maximum_allowed_drawdown_pct
    )
    notes = []
    if count < minimum_trades:
        notes.append(f"insufficient sample: {count} < {minimum_trades}")
    if lower_bound <= minimum_net_edge_pct:
        notes.append("conservative expected net edge did not clear the hurdle")
    if expected_shortfall <= 0:
        notes.append("validation sample did not establish a positive stressed-loss estimate")
    if maximum_drawdown_pct > maximum_allowed_drawdown_pct:
        notes.append("validation drawdown exceeded the approved policy limit")
    notes.append("95% one-sided lower bound uses a deterministic circular moving-block bootstrap")
    notes.append("confidence scales and strategy rules must remain frozen for this evidence")
    return ValidationReport(
        report_id=report_id,
        strategy_set_version=strategy_set_version,
        evidence=evidence,
        passed=passed,
        net_expectancy_pct=Decimal(str(mean)),
        expected_shortfall_pct=expected_shortfall,
        maximum_drawdown_pct=maximum_drawdown_pct,
        notes=tuple(notes),
        risk_policy_id=risk_policy_id,
        risk_policy_version=risk_policy_version,
    )
