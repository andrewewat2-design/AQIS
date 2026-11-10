from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from aqis.marketdata import DataIntegrityError, download_binance_spot_klines, load_csv, validate_candles
from aqis.lifecycle import PersistentLifecycleManager, StrategyVersion
from aqis.models import Candle, OpportunityEvidence, RiskPolicy
from aqis.models import LifecycleStage
from aqis.paper import PaperReplayEngine
from aqis.risk import PersistentPolicyRegistry
from aqis.validation import build_validation_report


def _load_policy(path: str | None) -> RiskPolicy:
    if path is None:
        return RiskPolicy()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return RiskPolicy(
        policy_id=raw.get("policy_id", "aqis-paper-btcusdt"),
        version=int(raw.get("version", 1)),
        starting_equity_usdt=Decimal(str(raw.get("starting_equity_usdt", "10000"))),
        max_position_pct=Decimal(str(raw.get("max_position_pct", "0.20"))),
        max_order_pct=Decimal(str(raw.get("max_order_pct", "0.10"))),
        daily_loss_stop_pct=Decimal(str(raw.get("daily_loss_stop_pct", "0.02"))),
        max_drawdown_stop_pct=Decimal(str(raw.get("max_drawdown_stop_pct", "0.05"))),
        fee_rate=Decimal(str(raw.get("fee_rate", "0.001"))),
        slippage_bps=Decimal(str(raw.get("slippage_bps", "10"))),
        min_net_edge_pct=Decimal(str(raw.get("min_net_edge_pct", "0"))),
        max_candle_lag_seconds=int(raw.get("max_candle_lag_seconds", 300)),
        max_consecutive_execution_errors=int(raw.get("max_consecutive_execution_errors", 3)),
        volatility_review_multiple=Decimal(str(raw.get("volatility_review_multiple", "2"))),
        strategy_correlation_review=Decimal(str(raw.get("strategy_correlation_review", "0.85"))),
        slippage_deviation_review_bps=Decimal(str(raw.get("slippage_deviation_review_bps", "20"))),
        minimum_correlation_samples=int(raw.get("minimum_correlation_samples", 30)),
        market=raw.get("market", "BTC/USDT"),
        venue=raw.get("venue", "binance"),
        timeframe=raw.get("timeframe", "1h"),
        live_trading_enabled=bool(raw.get("live_trading_enabled", False)),
    )


def _synthetic_candles(count: int = 900, seed: int = 7) -> list[Candle]:
    rng = random.Random(seed)
    timestamp = datetime(2023, 1, 1, 1, tzinfo=timezone.utc)
    price = 16500.0
    candles = []
    for index in range(count):
        drift = 0.00035 if (index // 120) % 2 == 0 else -0.0002
        change = drift + rng.gauss(0, 0.004)
        open_price = price
        close = max(100.0, open_price * (1 + change))
        high = max(open_price, close) * (1 + abs(rng.gauss(0, 0.001)))
        low = min(open_price, close) * (1 - abs(rng.gauss(0, 0.001)))
        candles.append(Candle(timestamp, open_price, high, low, close, 10 + rng.random() * 100))
        timestamp += timedelta(hours=1)
        price = close
    return candles


def _load_evidence(path: str) -> OpportunityEvidence:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if not report.get("passed"):
        raise ValueError("validation report did not pass")
    raw = report["evidence"]
    return OpportunityEvidence(
        evidence_id=str(raw["evidence_id"]),
        strategy_set_version=str(raw["strategy_set_version"]),
        regime=str(raw["regime"]),
        net_edge_lower_bound_pct=Decimal(str(raw["net_edge_lower_bound_pct"])),
        stressed_loss_fraction=Decimal(str(raw["stressed_loss_fraction"])),
        validation_trade_count=int(raw["validation_trade_count"]),
        data_hash=str(raw["data_hash"]),
        risk_policy_id=str(raw.get("risk_policy_id", "aqis-paper-btcusdt")),
        risk_policy_version=int(raw.get("risk_policy_version", 1)),
    )


def _report_json(report) -> dict[str, object]:
    return {
        "report_id": report.report_id,
        "strategy_set_version": report.strategy_set_version,
        "passed": report.passed,
        "net_expectancy_pct": str(report.net_expectancy_pct),
        "expected_shortfall_pct": str(report.expected_shortfall_pct),
        "maximum_drawdown_pct": str(report.maximum_drawdown_pct),
        "risk_policy_id": report.risk_policy_id,
        "risk_policy_version": report.risk_policy_version,
        "notes": list(report.notes),
        "evidence": {
            "evidence_id": report.evidence.evidence_id,
            "strategy_set_version": report.evidence.strategy_set_version,
            "regime": report.evidence.regime,
            "net_edge_lower_bound_pct": str(report.evidence.net_edge_lower_bound_pct),
            "stressed_loss_fraction": str(report.evidence.stressed_loss_fraction),
            "validation_trade_count": report.evidence.validation_trade_count,
            "data_hash": report.evidence.data_hash,
            "risk_policy_id": report.evidence.risk_policy_id,
            "risk_policy_version": report.evidence.risk_policy_version,
        },
    }


def command_validate_data(args: argparse.Namespace) -> int:
    try:
        candles = load_csv(args.csv, require_contiguous=not args.allow_gaps)
    except (OSError, DataIntegrityError) as error:
        print(f"invalid: {error}", file=sys.stderr)
        return 2
    report = validate_candles(candles)
    print(json.dumps({
        "valid": report.valid,
        "candle_count": report.candle_count,
        "missing_intervals": [item.isoformat() for item in report.missing_intervals],
        "duplicate_timestamps": [item.isoformat() for item in report.duplicate_timestamps],
        "invalid_rows": list(report.invalid_rows),
    }, indent=2))
    return 0 if report.valid or (args.allow_gaps and not report.duplicate_timestamps and not report.invalid_rows) else 2


def command_replay(args: argparse.Namespace) -> int:
    try:
        candles = load_csv(args.csv)
    except (OSError, DataIntegrityError) as error:
        print(f"cannot replay: {error}", file=sys.stderr)
        return 2
    try:
        with PersistentPolicyRegistry(args.policy_database) as registry:
            policy = registry.active(args.policy_id)
    except (KeyError, OSError) as error:
        print(f"no human-approved active risk policy: {error}", file=sys.stderr)
        return 2
    evidence = {"default": _load_evidence(args.evidence)} if args.evidence else {}
    engine = PaperReplayEngine(policy, evidence, validation_mode=args.validation_mode)
    result = engine.run(candles)
    output = {
        "mode": "validation_only" if args.validation_mode else "paper_replay",
        "evidence_file": args.evidence,
        "starting_equity_usdt": str(result.starting_equity_usdt),
        "ending_equity_usdt": str(result.ending_equity_usdt),
        "net_pnl_usdt": str(result.ending_equity_usdt - result.starting_equity_usdt),
        "fills": len(result.fills),
        "rejected_orders": len(result.rejected_orders),
        "max_drawdown_pct": str(result.maximum_drawdown_pct),
        "halted": result.halted,
        "halt_reason": result.halt_reason,
        "selector_decisions": len(result.decisions),
        "regimes_shadow_only": sorted({decision.regime.regime for decision in result.decisions}),
    }
    text = json.dumps(output, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def command_demo(_: argparse.Namespace) -> int:
    candles = _synthetic_candles()
    engine = PaperReplayEngine(RiskPolicy(), validation_mode=True)
    result = engine.run(candles)
    print("Synthetic, non-promotable pipeline demo")
    print(json.dumps({
        "candles": len(candles),
        "signals": len(result.decisions),
        "fills": len(result.fills),
        "rejections": len(result.rejected_orders),
        "starting_equity_usdt": str(result.starting_equity_usdt),
        "ending_equity_usdt": str(result.ending_equity_usdt),
        "max_drawdown_pct": str(result.maximum_drawdown_pct),
        "halted": result.halted,
        "note": "Synthetic prices are a pipeline smoke test and cannot approve or promote a candidate.",
    }, indent=2))
    return 0


def command_validation(args: argparse.Namespace) -> int:
    try:
        train = load_csv(args.train_csv)
        holdout = load_csv(args.holdout_csv)
    except (OSError, DataIntegrityError) as error:
        print(f"cannot validate: {error}", file=sys.stderr)
        return 2
    if train[-1].utc_timestamp + timedelta(hours=1) != holdout[0].utc_timestamp:
        print("training and holdout files must be contiguous hourly series", file=sys.stderr)
        return 2

    policy = _load_policy(args.policy)
    holdout_material = "\n".join(
        f"{c.utc_timestamp.isoformat()},{c.open},{c.high},{c.low},{c.close},{c.volume}"
        for c in holdout
    ).encode("utf-8")
    data_hash = hashlib.sha256(holdout_material).hexdigest()
    replay = PaperReplayEngine(policy, validation_mode=True).run(
        train + holdout,
        trade_start_index=len(train),
    )
    if not replay.completed_trade_returns_pct:
        print("holdout produced no completed trades; candidate cannot be promoted", file=sys.stderr)
        return 1
    report = build_validation_report(
        report_id=args.report_id,
        strategy_set_version=args.strategy_set_version,
        regime="default",
        net_trade_returns_pct=replay.completed_trade_returns_pct,
        data_fingerprint=data_hash,
        minimum_trades=args.minimum_trades,
        maximum_drawdown_pct=replay.maximum_drawdown_pct,
        maximum_allowed_drawdown_pct=policy.max_drawdown_stop_pct,
        risk_policy_id=policy.policy_id,
        risk_policy_version=policy.version,
    )
    serialized = json.dumps(_report_json(report), indent=2)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if report.passed else 1


def command_download(args: argparse.Namespace) -> int:
    try:
        start = datetime.fromisoformat(args.start.replace("Z", "+00:00"))
        end = datetime.fromisoformat(args.end.replace("Z", "+00:00"))
        candles = download_binance_spot_klines(start, end, symbol=args.symbol)
    except (ValueError, OSError, DataIntegrityError) as error:
        print(f"download failed: {error}", file=sys.stderr)
        return 2
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        stream.write("timestamp,open,high,low,close,volume\n")
        for candle in candles:
            stream.write(
                f"{candle.utc_timestamp.isoformat()},{candle.open},{candle.high},"
                f"{candle.low},{candle.close},{candle.volume}\n"
            )
    print(json.dumps({"symbol": args.symbol, "candles": len(candles), "output": str(output)}, indent=2))
    return 0


def command_split(args: argparse.Namespace) -> int:
    try:
        candles = load_csv(args.csv)
    except (OSError, DataIntegrityError) as error:
        print(f"cannot split data: {error}", file=sys.stderr)
        return 2
    if not 0.5 <= args.train_fraction < 1:
        print("train fraction must be in [0.5, 1)", file=sys.stderr)
        return 2
    split_at = int(len(candles) * args.train_fraction)
    if split_at < 2 or len(candles) - split_at < 2:
        print("both train and holdout sets require at least two candles", file=sys.stderr)
        return 2

    def write(path: str, rows: list[Candle]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as stream:
            stream.write("timestamp,open,high,low,close,volume\n")
            for candle in rows:
                stream.write(
                    f"{candle.utc_timestamp.isoformat()},{candle.open},{candle.high},"
                    f"{candle.low},{candle.close},{candle.volume}\n"
                )

    write(args.train_output, candles[:split_at])
    write(args.holdout_output, candles[split_at:])
    print(json.dumps({"train_candles": split_at, "holdout_candles": len(candles) - split_at}, indent=2))
    return 0


def command_candidate_create(args: argparse.Namespace) -> int:
    try:
        parameters = json.loads(args.parameters)
        scales = json.loads(args.confidence_scales)
        if not isinstance(parameters, dict) or not isinstance(scales, dict):
            raise ValueError("parameters and confidence scales must be JSON objects")
        version = StrategyVersion(
            candidate_id=args.candidate_id,
            version=args.version,
            strategy_set_version=args.strategy_set_version,
            parameters=tuple(sorted((str(key), str(value)) for key, value in parameters.items())),
            confidence_scales=tuple(sorted((str(key), str(value)) for key, value in scales.items())),
            source_hash=args.source_hash,
            risk_policy_version=args.risk_policy_version,
        )
        with PersistentLifecycleManager(args.database) as manager:
            candidate = manager.create_candidate(version, actor=args.actor)
        print(json.dumps({"candidate_id": candidate.candidate_id, "stage": candidate.stage.value}, indent=2))
        return 0
    except (ValueError, json.JSONDecodeError) as error:
        print(f"candidate creation failed: {error}", file=sys.stderr)
        return 2


def command_candidate_show(args: argparse.Namespace) -> int:
    with PersistentLifecycleManager(args.database) as manager:
        if args.candidate_id:
            candidates = [manager.candidates[args.candidate_id]] if args.candidate_id in manager.candidates else []
        else:
            candidates = list(manager.candidates.values())
        payload = [
            {
                "candidate_id": candidate.candidate_id,
                "version": candidate.version.version,
                "strategy_set_version": candidate.version.strategy_set_version,
                "stage": candidate.stage.value,
                "validation_passed": candidate.validation_passed,
                "human_approved_for_paper": candidate.human_approved_for_paper,
                "source_hash": candidate.version.source_hash,
            }
            for candidate in candidates
        ]
        if args.events:
            payload = {"candidates": payload, "events": manager.event_history()}
    print(json.dumps(payload, indent=2))
    return 0


def command_candidate_advance(args: argparse.Namespace) -> int:
    try:
        with PersistentLifecycleManager(args.database) as manager:
            candidate = manager.transition(
                args.candidate_id,
                LifecycleStage(args.stage),
                args.actor,
                reason=args.reason,
            )
        print(json.dumps({"candidate_id": candidate.candidate_id, "stage": candidate.stage.value}, indent=2))
        return 0
    except (ValueError, PermissionError, KeyError) as error:
        print(f"transition rejected: {error}", file=sys.stderr)
        return 2


def command_candidate_validation(args: argparse.Namespace) -> int:
    try:
        with PersistentLifecycleManager(args.database) as manager:
            candidate = manager.record_validation(args.candidate_id, args.passed, args.report_id)
        print(json.dumps({"candidate_id": candidate.candidate_id, "validation_passed": candidate.validation_passed}, indent=2))
        return 0
    except (ValueError, KeyError) as error:
        print(f"validation record rejected: {error}", file=sys.stderr)
        return 2


def command_candidate_approve(args: argparse.Namespace) -> int:
    try:
        with PersistentLifecycleManager(args.database) as manager:
            candidate = manager.approve_for_paper(args.candidate_id, args.actor, args.report_id)
        print(json.dumps({"candidate_id": candidate.candidate_id, "human_approved_for_paper": candidate.human_approved_for_paper}, indent=2))
        return 0
    except (ValueError, PermissionError, KeyError) as error:
        print(f"approval rejected: {error}", file=sys.stderr)
        return 2


def command_policy_activate(args: argparse.Namespace) -> int:
    try:
        policy = _load_policy(args.policy_file)
        with PersistentPolicyRegistry(args.database) as registry:
            registry.activate(policy, args.approved_by)
        print(json.dumps({"policy_id": policy.policy_id, "version": policy.version, "approved_by": args.approved_by}, indent=2))
        return 0
    except (ValueError, PermissionError, OSError) as error:
        print(f"policy activation rejected: {error}", file=sys.stderr)
        return 2


def command_policy_show(args: argparse.Namespace) -> int:
    try:
        with PersistentPolicyRegistry(args.database) as registry:
            policies = registry.versions(args.policy_id)
            active = registry.active(args.policy_id)
    except KeyError:
        print(f"no active policy for {args.policy_id}", file=sys.stderr)
        return 2
    print(json.dumps({
        "active_policy_id": active.policy_id,
        "active_version": active.version,
        "versions": [
            {
                "policy_id": item.policy_id,
                "version": item.version,
                "max_position_pct": str(item.max_position_pct),
                "max_order_pct": str(item.max_order_pct),
                "daily_loss_stop_pct": str(item.daily_loss_stop_pct),
                "max_drawdown_stop_pct": str(item.max_drawdown_stop_pct),
            }
            for item in policies
        ],
    }, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aqis", description="AQIS strategy lifecycle prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-data", help="validate completed hourly OHLCV CSV")
    validate.add_argument("csv")
    validate.add_argument("--allow-gaps", action="store_true")
    validate.set_defaults(handler=command_validate_data)

    replay = subparsers.add_parser("replay", help="replay historical candles through the paper engine")
    replay.add_argument("csv")
    replay.add_argument("--policy-database", default="data/aqis.sqlite3")
    replay.add_argument("--policy-id", default="aqis-paper-btcusdt")
    replay.add_argument("--output")
    replay.add_argument("--evidence", help="passing frozen validation report JSON")
    replay.add_argument("--validation-mode", action="store_true", help="validation-only replay; cannot authorize paper deployment")
    replay.set_defaults(handler=command_replay)

    validation = subparsers.add_parser("validate", help="evaluate frozen strategy rules on a separate holdout CSV")
    validation.add_argument("--train-csv", required=True)
    validation.add_argument("--holdout-csv", required=True)
    validation.add_argument("--policy")
    validation.add_argument("--report-id", required=True)
    validation.add_argument("--strategy-set-version", default="aqis-four-agent-v1")
    validation.add_argument("--minimum-trades", type=int, default=30)
    validation.add_argument("--output", required=True)
    validation.set_defaults(handler=command_validation)

    download = subparsers.add_parser("download", help="download public Binance Spot hourly candles")
    download.add_argument("--start", required=True, help="UTC ISO timestamp, e.g. 2024-01-01T00:00:00Z")
    download.add_argument("--end", required=True, help="UTC ISO timestamp")
    download.add_argument("--symbol", default="BTCUSDT")
    download.add_argument("--output", required=True)
    download.set_defaults(handler=command_download)

    split = subparsers.add_parser("split", help="create chronological train/holdout CSV files")
    split.add_argument("csv")
    split.add_argument("--train-fraction", type=float, default=0.7)
    split.add_argument("--train-output", required=True)
    split.add_argument("--holdout-output", required=True)
    split.set_defaults(handler=command_split)

    lifecycle = subparsers.add_parser("lifecycle", help="manage persistent candidate lifecycle records")
    lifecycle_commands = lifecycle.add_subparsers(dest="lifecycle_command", required=True)

    candidate_create = lifecycle_commands.add_parser("create", help="create an immutable discovery candidate")
    candidate_create.add_argument("--database", default="data/aqis.sqlite3")
    candidate_create.add_argument("--candidate-id", required=True)
    candidate_create.add_argument("--version", type=int, default=1)
    candidate_create.add_argument("--strategy-set-version", default="aqis-four-agent-v1")
    candidate_create.add_argument("--parameters", default="{}")
    candidate_create.add_argument("--confidence-scales", required=True)
    candidate_create.add_argument("--source-hash", required=True)
    candidate_create.add_argument("--risk-policy-version", type=int, default=1)
    candidate_create.add_argument("--actor", default="research_agent")
    candidate_create.set_defaults(handler=command_candidate_create)

    candidate_show = lifecycle_commands.add_parser("show", help="show candidate state or audit history")
    candidate_show.add_argument("--database", default="data/aqis.sqlite3")
    candidate_show.add_argument("--candidate-id")
    candidate_show.add_argument("--events", action="store_true")
    candidate_show.set_defaults(handler=command_candidate_show)

    candidate_advance = lifecycle_commands.add_parser("advance", help="advance a candidate through a valid stage transition")
    candidate_advance.add_argument("--database", default="data/aqis.sqlite3")
    candidate_advance.add_argument("--candidate-id", required=True)
    candidate_advance.add_argument("--stage", required=True, choices=[stage.value for stage in LifecycleStage])
    candidate_advance.add_argument("--actor", required=True)
    candidate_advance.add_argument("--reason", default="")
    candidate_advance.set_defaults(handler=command_candidate_advance)

    candidate_validation = lifecycle_commands.add_parser("record-validation", help="record a completed holdout report")
    candidate_validation.add_argument("--database", default="data/aqis.sqlite3")
    candidate_validation.add_argument("--candidate-id", required=True)
    candidate_validation.add_argument("--report-id", required=True)
    validation_result = candidate_validation.add_mutually_exclusive_group(required=True)
    validation_result.add_argument("--passed", action="store_true")
    validation_result.add_argument("--failed", action="store_true")
    candidate_validation.set_defaults(handler=command_candidate_validation)

    candidate_approve = lifecycle_commands.add_parser("approve-paper", help="human approval for a validated candidate")
    candidate_approve.add_argument("--database", default="data/aqis.sqlite3")
    candidate_approve.add_argument("--candidate-id", required=True)
    candidate_approve.add_argument("--report-id", required=True)
    candidate_approve.add_argument("--actor", required=True)
    candidate_approve.set_defaults(handler=command_candidate_approve)

    policy = subparsers.add_parser("policy", help="manage human-approved immutable risk policy versions")
    policy_commands = policy.add_subparsers(dest="policy_command", required=True)
    policy_activate = policy_commands.add_parser("activate", help="activate a new human-approved policy version")
    policy_activate.add_argument("--policy-file", required=True)
    policy_activate.add_argument("--database", default="data/aqis.sqlite3")
    policy_activate.add_argument("--approved-by", required=True)
    policy_activate.set_defaults(handler=command_policy_activate)
    policy_show = policy_commands.add_parser("show", help="show active and historical policy versions")
    policy_show.add_argument("--database", default="data/aqis.sqlite3")
    policy_show.add_argument("--policy-id", default="aqis-paper-btcusdt")
    policy_show.set_defaults(handler=command_policy_show)

    demo = subparsers.add_parser("demo", help="exercise the pipeline using synthetic data")
    demo.set_defaults(handler=command_demo)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)
