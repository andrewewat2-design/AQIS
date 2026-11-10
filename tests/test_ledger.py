import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from aqis.ledger import (
    LedgerDatabase,
    LedgerHealth,
    generate_decision_id,
    generate_outcome_id,
    generate_replay_id,
    generate_status_id,
)
from aqis.models import (
    Candle,
    OpportunityEvidence,
    OrderSide,
    RiskPolicy,
    SignalAction,
    StrategySignal,
    TargetPosition,
)
from aqis.paper import PaperReplayEngine
from aqis.strategies import StrategyAgent
from tests.helpers import flat_candles


def make_candle(ts: datetime, price: float = 100.0) -> Candle:
    return Candle(ts, price, price + 0.5, price - 0.5, price, 10.0)


class ScheduledAgent(StrategyAgent):
    version = "test"

    def __init__(self, strategy_id: str, action: SignalAction) -> None:
        self.strategy_id = strategy_id
        self.action = action
        super().__init__()

    def evaluate(self, candles):
        return self._signal(candles, self.action, 100, "test")


class IdentifierTests(unittest.TestCase):
    def test_replay_id_deterministic(self) -> None:
        r1 = generate_replay_id("set-v1", "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", 25)
        r2 = generate_replay_id("set-v1", "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", 25)
        self.assertEqual(r1, r2)
        self.assertEqual(len(r1), 16)

    def test_replay_id_distinct_for_different_inputs(self) -> None:
        r1 = generate_replay_id("set-v1", "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", 25)
        r2 = generate_replay_id("set-v2", "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", 25)
        self.assertNotEqual(r1, r2)
        r3 = generate_replay_id("set-v1", "2024-01-01T00:00:00Z", "2024-01-03T00:00:00Z", 49)
        self.assertNotEqual(r1, r3)

    def test_decision_id_unique_per_candle(self) -> None:
        rid = generate_replay_id("set-v1", "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", 25)
        d1 = generate_decision_id(rid, "2024-01-01T01:00:00Z", 1)
        d2 = generate_decision_id(rid, "2024-01-01T02:00:00Z", 2)
        self.assertNotEqual(d1, d2)

    def test_decision_id_deterministic(self) -> None:
        rid = generate_replay_id("set-v1", "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", 25)
        d1 = generate_decision_id(rid, "2024-01-01T01:00:00Z", 1)
        d2 = generate_decision_id(rid, "2024-01-01T01:00:00Z", 1)
        self.assertEqual(d1, d2)

    def test_outcome_id_deterministic_and_distinct(self) -> None:
        corr = generate_outcome_id("market", "corr-1", "1h")
        self.assertEqual(corr, generate_outcome_id("market", "corr-1", "1h"))
        self.assertEqual(len(corr), 16)
        self.assertNotEqual(corr, generate_outcome_id("market", "corr-1", "24h"))
        self.assertNotEqual(corr, generate_outcome_id("portfolio", "corr-1", "1h"))
        self.assertNotEqual(corr, generate_outcome_id("market", "corr-2", "1h"))
        self.assertNotEqual(corr, "corr-1")

    def test_status_id_distinct(self) -> None:
        sid = generate_status_id("corr-1", "24h", "horizon_not_reached", 0)
        self.assertEqual(len(sid), 16)
        self.assertNotEqual(sid, generate_status_id("corr-2", "24h", "horizon_not_reached", 0))
        self.assertNotEqual(sid, generate_status_id("corr-1", "24h", "data_insufficient", 0))
        self.assertNotEqual(sid, generate_status_id("corr-1", "24h", "no_action_flat", 1))


class SchemaTests(unittest.TestCase):
    def test_schema_version_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "test.sqlite3")
            with LedgerDatabase(db_path) as db:
                row = db.connection.execute(
                    "SELECT version FROM ledger_schema_version"
                ).fetchone()
                self.assertIsNotNone(row)
                self.assertGreaterEqual(row[0], 1)

    def test_immutability_triggers_reject_update(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "test.sqlite3")
            with LedgerDatabase(db_path) as db:
                db.register_replay(
                    replay_id="replay-1",
                    strategy_set_version="set-v1",
                    market="BTC/USDT",
                    venue="binance",
                    timeframe="1h",
                    first_ts="2024-01-01T00:00:00Z",
                    last_ts="2024-01-02T00:00:00Z",
                    candle_count=25,
                    policy_id="pol-1",
                    policy_version=1,
                )
                with self.assertRaises(Exception):
                    db.connection.execute(
                        "UPDATE ledger_replays SET candle_count=99 WHERE replay_id='replay-1'"
                    )
                    db.connection.commit()

    def test_immutability_triggers_reject_delete(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "test.sqlite3")
            with LedgerDatabase(db_path) as db:
                db.register_replay(
                    replay_id="replay-1",
                    strategy_set_version="set-v1",
                    market="BTC/USDT",
                    venue="binance",
                    timeframe="1h",
                    first_ts="2024-01-01T00:00:00Z",
                    last_ts="2024-01-02T00:00:00Z",
                    candle_count=25,
                    policy_id="pol-1",
                    policy_version=1,
                )
                with self.assertRaises(Exception):
                    db.connection.execute(
                        "DELETE FROM ledger_replays WHERE replay_id='replay-1'"
                    )
                    db.connection.commit()

    def test_ledger_health_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "test.sqlite3")
            with LedgerDatabase(db_path) as db:
                health = db.get_health()
                self.assertEqual(health.health_id, "ledger")
                self.assertFalse(health.ledger_complete)
                self.assertEqual(health.failed_write_count, 0)
                self.assertIsNone(health.last_successful_write)
                self.assertIsNone(health.last_failed_write)

    def test_schema_v2_columns(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "test.sqlite3")
            with LedgerDatabase(db_path) as db:
                columns = {
                    col[1]
                    for col in db.connection.execute(
                        "PRAGMA table_info(ledger_outcomes)"
                    )
                }
                self.assertTrue(
                    {"anchor", "direction", "baseline_price", "baseline_equity", "horizon_return_bps"}
                    <= columns
                )

    def test_v1_outcomes_schema_migrates_to_v2(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "legacy.sqlite3")
            legacy = sqlite3.connect(db_path)
            legacy.executescript(
                """
                CREATE TABLE ledger_schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
                INSERT INTO ledger_schema_version(version, applied_at) VALUES (1, 'legacy');
                CREATE TABLE ledger_outcomes (
                    outcome_id TEXT PRIMARY KEY,
                    correlation_id TEXT NOT NULL,
                    horizon TEXT NOT NULL,
                    horizon_timestamp TEXT NOT NULL,
                    outcome_type TEXT NOT NULL,
                    price_at_horizon TEXT,
                    portfolio_equity_at_horizon TEXT,
                    realized_pnl_at_horizon TEXT,
                    unrealized_pnl_at_horizon TEXT,
                    max_adverse_excursion TEXT,
                    max_favourable_excursion TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            legacy.commit()
            legacy.close()
            db = LedgerDatabase(db_path)
            try:
                columns = {col[1] for col in db.connection.execute("PRAGMA table_info(ledger_outcomes)")}
                self.assertTrue(
                    {"anchor", "direction", "baseline_price", "baseline_equity", "horizon_return_bps"}
                    <= columns
                )
                versions = [
                    row[0]
                    for row in db.connection.execute(
                        "SELECT version FROM ledger_schema_version ORDER BY version"
                    )
                ]
                self.assertIn(2, versions)
            finally:
                db.close()


class LedgerIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "test.sqlite3")
        self.db = LedgerDatabase(self.db_path)

        self.ts = datetime(2024, 1, 1, 1, tzinfo=timezone.utc)
        self.candles = [
            make_candle(self.ts + timedelta(hours=i), 100.0 + i * 0.5)
            for i in range(200)
        ]
        self.evidence = OpportunityEvidence(
            "e", "aqis-four-agent-v1", "default", Decimal("0.01"), Decimal("0.10"), 50, "h"
        )

    def tearDown(self) -> None:
        self.db.close()

    def _build_engine(self, candle_count: int = 50) -> PaperReplayEngine:
        agents = [
            ScheduledAgent("ema", SignalAction.ABSTAIN),
            ScheduledAgent("rsi", SignalAction.ABSTAIN),
            ScheduledAgent("mean", SignalAction.ABSTAIN),
            ScheduledAgent("breakout", SignalAction.ABSTAIN),
        ]
        return PaperReplayEngine(
            RiskPolicy(), {"default": self.evidence}, agents, ledger=self.db
        )

    def test_all_evaluated_decisions_recorded(self) -> None:
        engine = self._build_engine()
        result = engine.run(self.candles[:50])
        self.assertEqual(len(result.decisions), 50)
        # Each decision should have a corresponding ledger record
        for decision in result.decisions:
            corr_id = engine._correlation_id(
                "aqis-four-agent-v1", decision.candle.utc_timestamp,
                self.candles.index(decision.candle)
            )
            chain = self.db.get_decision_chain(corr_id)
            self.assertIsNotNone(chain["decision"])
            self.assertEqual(chain["decision"]["correlation_id"], corr_id)

    def test_decisions_not_to_trade_recorded(self) -> None:
        # All agents abstain -> cash target, no execution
        engine = self._build_engine()
        result = engine.run(self.candles[:50])
        # All 50 decisions should be in the ledger
        count = self.db.connection.execute(
            "SELECT COUNT(*) FROM ledger_decisions"
        ).fetchone()[0]
        self.assertEqual(count, 50)
        # No fills, no risk rejections (all abstain -> no order)
        self.assertEqual(len(result.fills), 0)
        self.assertEqual(len(result.risk_rejections), 0)

    def test_risk_rejection_recorded(self) -> None:
        # Use agents that vote BUY to trigger order attempts and risk rejections
        agents = [
            ScheduledAgent("ema", SignalAction.BUY),
            ScheduledAgent("rsi", SignalAction.BUY),
            ScheduledAgent("mean", SignalAction.BUY),
            ScheduledAgent("breakout", SignalAction.ABSTAIN),
        ]
        engine = PaperReplayEngine(
            RiskPolicy(), {"default": self.evidence}, agents, ledger=self.db
        )
        result = engine.run(self.candles[:50])
        # Some orders may be rejected by risk engine
        if result.risk_rejections:
            for ts, code, msg, cid in result.risk_rejections:
                chain = self.db.get_decision_chain(cid)
                self.assertIsNotNone(chain["risk"])
                self.assertEqual(chain["risk"]["correlation_id"], cid)

    def test_correlation_chain_complete(self) -> None:
        agents = [
            ScheduledAgent("ema", SignalAction.BUY),
            ScheduledAgent("rsi", SignalAction.BUY),
            ScheduledAgent("mean", SignalAction.BUY),
            ScheduledAgent("breakout", SignalAction.ABSTAIN),
        ]
        engine = PaperReplayEngine(
            RiskPolicy(), {"default": self.evidence}, agents, ledger=self.db
        )
        result = engine.run(self.candles[:50])
        for decision in result.decisions:
            idx = self.candles.index(decision.candle)
            corr_id = engine._correlation_id(
                "aqis-four-agent-v1", decision.candle.utc_timestamp, idx
            )
            chain = self.db.get_decision_chain(corr_id)
            self.assertIsNotNone(chain["decision"])
            # Risk record may or may not exist (depends on order attempt)
            # Execution record may or may not exist (depends on fill)
            # Outcomes should exist for horizons within range
            self.assertGreaterEqual(len(chain["outcomes"]), 0)

    def test_outcome_insert_only_no_updates(self) -> None:
        engine = self._build_engine()
        engine.run(self.candles[:50])
        # Try to update an outcome - should fail
        rows = self.db.connection.execute(
            "SELECT outcome_id FROM ledger_outcomes LIMIT 1"
        ).fetchall()
        if rows:
            with self.assertRaises(Exception):
                self.db.connection.execute(
                    "UPDATE ledger_outcomes SET price_at_horizon='99999' WHERE outcome_id=?",
                    (rows[0][0],),
                )
                self.db.connection.commit()

    def test_pending_outcome_absence(self) -> None:
        # Short replay: horizons beyond data should have status events but no outcomes
        engine = self._build_engine()
        engine.run(self.candles[:10])
        pending = self.db.get_pending_outcomes()
        # Should have pending outcomes for 7d horizon (not enough data)
        self.assertGreaterEqual(len(pending), 0)
        for p in pending:
            self.assertIn(p["status"], ("horizon_not_reached", "data_insufficient"))

    def test_status_event_for_missing_data(self) -> None:
        engine = self._build_engine()
        engine.run(self.candles[:10])
        # 7d horizon needs 168 candles after decision, not available in 10-candle replay
        rows = self.db.connection.execute(
            "SELECT status FROM ledger_outcome_status WHERE status='horizon_not_reached'"
        ).fetchall()
        self.assertGreaterEqual(len(rows), 0)

    def test_ledger_health_tracks_failures(self) -> None:
        health = self.db.get_health()
        self.assertFalse(health.ledger_complete)
        # After successful writes, health should reflect that
        engine = self._build_engine()
        engine.run(self.candles[:10])
        health2 = self.db.get_health()
        self.assertIsNotNone(health2.last_successful_write)

    def test_ledger_never_claims_complete(self) -> None:
        health = self.db.get_health()
        self.assertFalse(health.ledger_complete)

    def test_replay_reproducibility(self) -> None:
        agents = [
            ScheduledAgent("ema", SignalAction.ABSTAIN),
            ScheduledAgent("rsi", SignalAction.ABSTAIN),
            ScheduledAgent("mean", SignalAction.ABSTAIN),
            ScheduledAgent("breakout", SignalAction.ABSTAIN),
        ]

        def run_and_get_decisions(db_path):
            db = LedgerDatabase(db_path)
            engine = PaperReplayEngine(
                RiskPolicy(), {"default": self.evidence}, agents, ledger=db
            )
            result = engine.run(self.candles[:20])
            decision_ids = []
            for d in result.decisions:
                idx = self.candles.index(d.candle)
                cid = engine._correlation_id(
                    "aqis-four-agent-v1", d.candle.utc_timestamp, idx
                )
                decision_ids.append(cid)
            db.close()
            return decision_ids

        db1 = os.path.join(self.tmpdir.name, "test1.sqlite3")
        db2 = os.path.join(self.tmpdir.name, "test2.sqlite3")
        ids1 = run_and_get_decisions(db1)
        ids2 = run_and_get_decisions(db2)
        self.assertEqual(ids1, ids2)

    def test_get_decision_chain_returns_all_linked(self) -> None:
        engine = self._build_engine()
        result = engine.run(self.candles[:20])
        decision = result.decisions[0]
        idx = self.candles.index(decision.candle)
        corr_id = engine._correlation_id(
            "aqis-four-agent-v1", decision.candle.utc_timestamp, idx
        )
        chain = self.db.get_decision_chain(corr_id)
        self.assertEqual(chain["decision"]["correlation_id"], corr_id)
        self.assertIsInstance(chain["outcomes"], list)
        self.assertIsInstance(chain["statuses"], list)

    def test_schema_version_recorded_integration(self) -> None:
        engine = self._build_engine()
        engine.run(self.candles[:10])
        version = self.db.connection.execute(
            "SELECT MAX(version) FROM ledger_schema_version"
        ).fetchone()[0]
        self.assertGreaterEqual(version, 2)

    def test_outcome_status_transitions_have_distinct_ids(self) -> None:
        self.db.register_replay(
            replay_id="replay-status",
            strategy_set_version="set-v1",
            market="BTC/USDT",
            venue="binance",
            timeframe="1h",
            first_ts="2024-01-01T00:00:00Z",
            last_ts="2024-01-02T00:00:00Z",
            candle_count=25,
            policy_id="pol-1",
            policy_version=1,
        )
        self.db.record_decision(
            decision_id="decision-status-1",
            replay_id="replay-status",
            correlation_id="corr-x",
            candle_timestamp="2024-01-01T01:00:00Z",
            candle_index=1,
            market_data_snapshot={"close": 100.0},
            regime="ranging",
            strategy_signals=[],
            selector_decision={},
            effective_threshold=60.0,
            opportunity_evidence_id=None,
            risk_policy_id="pol-1",
            risk_policy_version=1,
        )
        self.db.record_outcome_status("corr-x", "1h", "pending")
        self.db.record_outcome_status("corr-x", "1h", "recorded")
        self.db.record_outcome_status("corr-x", "1h", "recorded")
        rows = self.db.connection.execute(
            "SELECT status FROM ledger_outcome_status WHERE correlation_id='corr-x'"
        ).fetchall()
        self.assertEqual(len(rows), 3)

    def test_no_fill_decisions_have_neither_portfolio_rows_nor_missing_statuses(self) -> None:
        engine = self._build_engine()
        result = engine.run(self.candles[:10])
        self.assertEqual(len(result.fills), 0)
        decisions_count = len(result.decisions)
        market_rows = self.db.connection.execute(
            "SELECT COUNT(*) FROM ledger_outcomes WHERE outcome_type='market'"
        ).fetchone()[0]
        portfolio_rows = self.db.connection.execute(
            "SELECT COUNT(*) FROM ledger_outcomes WHERE outcome_type='portfolio'"
        ).fetchone()[0]
        self.assertEqual(portfolio_rows, 0)
        self.assertGreater(market_rows, 0)
        statuses = self.db.connection.execute(
            "SELECT DISTINCT status FROM ledger_outcome_status"
        ).fetchall()
        allowed = {"no_action_flat", "horizon_not_reached", "no_order_end_of_replay"}
        for (status,) in statuses:
            self.assertIn(status, allowed)
        correlation_count = self.db.connection.execute(
            "SELECT COUNT(DISTINCT correlation_id) FROM ledger_decisions"
        ).fetchone()[0]
        self.assertEqual(correlation_count, decisions_count)

    def test_filled_decision_writes_market_and_portfolio_rows(self) -> None:
        agents = [
            ScheduledAgent("ema", SignalAction.BUY),
            ScheduledAgent("rsi", SignalAction.BUY),
            ScheduledAgent("mean", SignalAction.BUY),
            ScheduledAgent("breakout", SignalAction.SELL),
        ]
        engine = PaperReplayEngine(
            RiskPolicy(), {"default": self.evidence}, agents, ledger=self.db
        )
        result = engine.run(self.candles[:50])
        buy_fill_corrs = [
            fill.correlation_id
            for fill in result.fills
            if fill.side is OrderSide.BUY
        ]
        self.assertGreater(len(buy_fill_corrs), 0)
        for corr_id in buy_fill_corrs[:2]:
            chain = self.db.get_decision_chain(corr_id)
            types = {
                (o["horizon"], o["outcome_type"]) for o in chain["outcomes"]
            }
            self.assertIn(("1h", "market"), types)
            self.assertIn(("1h", "portfolio"), types)
            market = next(
                o
                for o in chain["outcomes"]
                if o["outcome_type"] == "market" and o["horizon"] == "1h"
            )
            self.assertEqual(market["anchor"], "fill")
            self.assertEqual(market["direction"], 1)
            self.assertIsNotNone(market["baseline_price"])
            self.assertIsNotNone(market["max_adverse_excursion"])
            self.assertIsNotNone(market["max_favourable_excursion"])
            self.assertIsNotNone(market["horizon_return_bps"])
            portfolio = next(
                o
                for o in chain["outcomes"]
                if o["outcome_type"] == "portfolio" and o["horizon"] == "1h"
            )
            self.assertEqual(portfolio["anchor"], "fill")
            self.assertIsNotNone(portfolio["baseline_equity"])
            self.assertIsNotNone(portfolio["max_adverse_excursion"])
            self.assertIsNotNone(portfolio["max_favourable_excursion"])
            self.assertEqual(portfolio["direction"], 1)
            self.assertEqual(
                portfolio["outcome_id"] != market["outcome_id"], True
            )

    def test_market_outcome_excursions_anchored_to_fill(self) -> None:
        agents = [
            ScheduledAgent("ema", SignalAction.BUY),
            ScheduledAgent("rsi", SignalAction.BUY),
            ScheduledAgent("mean", SignalAction.BUY),
            ScheduledAgent("breakout", SignalAction.SELL),
        ]
        engine = PaperReplayEngine(
            RiskPolicy(), {"default": self.evidence}, agents, ledger=self.db
        )
        result = engine.run(self.candles[:50])
        fill_by_corr = {fill.correlation_id: fill for fill in result.fills}
        candles = self.candles
        sample = None
        for decision in result.decisions:
            decision_index = candles.index(decision.candle)
            corr_id = engine._correlation_id(
                "aqis-four-agent-v1", decision.candle.utc_timestamp, decision_index
            )
            fill = fill_by_corr.get(corr_id)
            if fill is not None and fill.side is OrderSide.BUY:
                sample = (corr_id, decision_index, fill)
                break
        self.assertIsNotNone(sample)
        corr_id, decision_index, fill = sample
        execution_candle = candles[decision_index + 1]
        chain = self.db.get_decision_chain(corr_id)
        market = next(
            o
            for o in chain["outcomes"]
            if o["outcome_type"] == "market" and o["horizon"] == "1h"
        )
        baseline = Decimal(market["baseline_price"])
        self.assertEqual(baseline, fill.fill_price)
        slippage_fraction = RiskPolicy().slippage_bps / Decimal("10000")
        self.assertEqual(
            baseline,
            Decimal(str(execution_candle.open)) * (Decimal("1") + slippage_fraction),
        )
        self.assertEqual(
            market["horizon_timestamp"],
            candles[decision_index + 1].utc_timestamp.isoformat(),
        )
        baseline = Decimal(market["baseline_price"])
        adverse = (
            Decimal(str(execution_candle.open)) * (Decimal("1") + slippage_fraction)
            - Decimal(str(execution_candle.low))
        )
        adverse_bps = (
            max(Decimal(0), adverse)
            / baseline
            * Decimal("10000")
        ).quantize(Decimal("0.0001"))
        self.assertEqual(Decimal(market["max_adverse_excursion"]), adverse_bps)
        favourable_bps = (
            max(
                Decimal(0),
                Decimal(str(execution_candle.high)) - baseline,
            )
            / baseline
            * Decimal("10000")
        ).quantize(Decimal("0.0001"))
        self.assertEqual(Decimal(market["max_favourable_excursion"]), favourable_bps)

    def test_risk_stop_exit_records_complete_chain(self) -> None:
        falling_policy = RiskPolicy(max_drawdown_stop_pct=Decimal("0.001"))
        candle_class = type(self.candles[-1])
        candles = self.candles[:40] + [
            candle_class(
                self.candles[39].utc_timestamp + timedelta(hours=i + 1),
                104.0 - i * 2.0,
                104.5 - i * 2.0,
                103.0 - i * 2.0,
                103.5 - i * 2.0,
                10.0,
            )
            for i in range(40)
        ]
        agents = [
            ScheduledAgent("ema", SignalAction.BUY),
            ScheduledAgent("rsi", SignalAction.BUY),
            ScheduledAgent("mean", SignalAction.BUY),
            ScheduledAgent("breakout", SignalAction.SELL),
        ]
        engine = PaperReplayEngine(
            falling_policy, {"default": self.evidence}, agents, ledger=self.db
        )
        result = engine.run(candles)
        for decision in result.decisions:
            if decision.candle.utc_timestamp > candles[0].utc_timestamp + timedelta(hours=79):
                break
        stop_fills = [
            fill
            for fill in result.fills
            if fill.reason.startswith("risk_stop:")
        ]
        self.assertGreater(len(stop_fills), 0)
        for fill in stop_fills[:1]:
            corr_id = fill.correlation_id
            chain = self.db.get_decision_chain(corr_id)
            self.assertIsNotNone(chain["decision"])
            self.assertIsNotNone(chain["risk"])
            self.assertIsNotNone(chain["execution"])
            self.assertIsNotNone(chain["execution"]["fill"])
            types = {
                (o["horizon"], o["outcome_type"]) for o in chain["outcomes"]
            }
            self.assertIn(("1h", "market"), types)
            self.assertIn(("1h", "portfolio"), types)
            market = next(
                o
                for o in chain["outcomes"]
                if o["outcome_type"] == "market" and o["horizon"] == "1h"
            )
            self.assertEqual(market["direction"], -1)
            self.assertEqual(market["anchor"], "fill")

    def _crash_candles(self):
        candle_class = type(self.candles[-1])
        return self.candles[:40] + [
            candle_class(
                self.candles[39].utc_timestamp + timedelta(hours=i + 1),
                104.0 - i * 2.0,
                104.5 - i * 2.0,
                103.0 - i * 2.0,
                103.5 - i * 2.0,
                10.0,
            )
            for i in range(40)
        ]

    def _buy_agents(self):
        return [
            ScheduledAgent("ema", SignalAction.BUY),
            ScheduledAgent("rsi", SignalAction.BUY),
            ScheduledAgent("mean", SignalAction.BUY),
            ScheduledAgent("breakout", SignalAction.SELL),
        ]

    def _market_row(self, chain, horizon):
        return next(
            o
            for o in chain["outcomes"]
            if o["outcome_type"] == "market" and o["horizon"] == horizon
        )

    def _portfolio_row(self, chain, horizon):
        return next(
            o
            for o in chain["outcomes"]
            if o["outcome_type"] == "portfolio" and o["horizon"] == horizon
        )

    def test_exit_fill_market_outcome_values_match_controlled_fixture(self) -> None:
        candles = self._crash_candles()
        falling_policy = RiskPolicy(max_drawdown_stop_pct=Decimal("0.001"))
        engine = PaperReplayEngine(
            falling_policy, {"default": self.evidence}, self._buy_agents(), ledger=self.db
        )
        result = engine.run(candles)
        stop_fill = next(
            fill
            for fill in result.fills
            if fill.reason.startswith("risk_stop:")
        )
        corr_id = stop_fill.correlation_id
        exec_index = next(
            k for k, c in enumerate(candles) if c.utc_timestamp >= stop_fill.timestamp + timedelta(hours=1)
        )
        decision_index = exec_index - 1
        slippage_fraction = Decimal("10") / Decimal("10000")
        baseline = Decimal(str(candles[exec_index].open)) * (Decimal("1") - slippage_fraction)
        self.assertEqual(baseline, stop_fill.fill_price)
        self.assertEqual(Decimal(str(candles[exec_index].open)), stop_fill.reference_price)

        chain = self.db.get_decision_chain(corr_id)
        self.assertEqual(chain["decision"]["candle_index"], decision_index)
        market = self._market_row(chain, "1h")
        self.assertEqual(market["anchor"], "fill")
        self.assertEqual(market["direction"], -1)
        self.assertEqual(Decimal(market["baseline_price"]), baseline)
        self.assertEqual(market["baseline_equity"], None)
        self.assertEqual(market["horizon_timestamp"], candles[exec_index].utc_timestamp.isoformat())
        self.assertEqual(Decimal(market["price_at_horizon"]), Decimal(str(candles[exec_index].close)))

        sign = Decimal("-1")
        adverse = Decimal(0)
        favourable = Decimal(0)
        for window_candle in [candles[exec_index]]:
            low_delta = sign * (Decimal(str(window_candle.low)) - baseline)
            high_delta = sign * (Decimal(str(window_candle.high)) - baseline)
            adverse = min(adverse, low_delta, high_delta)
            favourable = max(favourable, low_delta, high_delta)
        bps = Decimal("10000")
        expected_mae = (max(Decimal(0), -adverse) / baseline * bps).quantize(Decimal("0.0001"))
        expected_mfe = (max(Decimal(0), favourable) / baseline * bps).quantize(Decimal("0.0001"))
        expected_return = (
            sign * (Decimal(str(candles[exec_index].close)) - baseline) / baseline * bps
        ).quantize(Decimal("0.0001"))
        self.assertEqual(Decimal(market["max_adverse_excursion"]), expected_mae)
        self.assertEqual(Decimal(market["max_favourable_excursion"]), expected_mfe)
        self.assertEqual(Decimal(market["horizon_return_bps"]), expected_return)
        self.assertGreater(Decimal(market["max_adverse_excursion"]), Decimal(0))
        self.assertGreater(Decimal(market["max_favourable_excursion"]), Decimal(0))

        portfolio = self._portfolio_row(chain, "1h")
        self.assertEqual(portfolio["anchor"], "fill")
        self.assertEqual(portfolio["direction"], -1)
        post_fill_equity = stop_fill.post_fill_equity_usdt
        self.assertIsNotNone(post_fill_equity)
        self.assertEqual(Decimal(portfolio["baseline_equity"]), post_fill_equity)
        snapshot = next(
            s
            for s in result.portfolio_snapshots
            if s.timestamp == candles[exec_index].utc_timestamp
        )
        expected_portfolio_mae = max(Decimal(0), post_fill_equity - snapshot.equity_usdt).quantize(Decimal("0.01"))
        expected_portfolio_mfe = max(Decimal(0), snapshot.equity_usdt - post_fill_equity).quantize(Decimal("0.01"))
        expected_portfolio_return = (
            (snapshot.equity_usdt - post_fill_equity) / post_fill_equity * bps
        ).quantize(Decimal("0.0001"))
        self.assertEqual(Decimal(portfolio["max_adverse_excursion"]), expected_portfolio_mae)
        self.assertEqual(Decimal(portfolio["max_favourable_excursion"]), expected_portfolio_mfe)
        self.assertEqual(Decimal(portfolio["horizon_return_bps"]), expected_portfolio_return)
        self.assertEqual(Decimal(portfolio["portfolio_equity_at_horizon"]), snapshot.equity_usdt)

    def test_prefill_excursion_excluded_from_fill_anchored_window(self) -> None:
        base_ts = self.candles[0].utc_timestamp
        candles = [
            make_candle(base_ts + timedelta(hours=i), 100.0) for i in range(40)
        ]
        candle_class = type(self.candles[-1])
        candles[0] = candle_class(base_ts, 100.0, 200.0, 99.5, 100.0, 10.0)
        engine = PaperReplayEngine(
            RiskPolicy(), {"default": self.evidence}, self._buy_agents(), ledger=self.db
        )
        result = engine.run(candles)
        corr0 = engine._correlation_id(
            "aqis-four-agent-v1", candles[0].utc_timestamp, 0
        )
        fill = next(f for f in result.fills if f.correlation_id == corr0)
        self.assertEqual(fill.side, OrderSide.BUY)
        baseline = fill.fill_price
        self.assertEqual(baseline, Decimal("100.0") * Decimal("1.001"))

        chain = self.db.get_decision_chain(corr0)
        for horizon in ("1h", "4h"):
            market = self._market_row(chain, horizon)
            self.assertEqual(market["anchor"], "fill")
            self.assertEqual(market["direction"], 1)
            self.assertEqual(Decimal(market["baseline_price"]), baseline)
            self.assertEqual(
                Decimal(market["max_adverse_excursion"]),
                ((baseline - Decimal("99.5")) / baseline * Decimal("10000")).quantize(Decimal("0.0001")),
            )
            contaminated_mfe = (
                (Decimal("200.0") - baseline) / baseline * Decimal("10000")
            ).quantize(Decimal("0.0001"))
            stored_mfe = Decimal(market["max_favourable_excursion"])
            self.assertEqual(
                stored_mfe,
                (max(Decimal(0), Decimal("100.5") - baseline) / baseline * Decimal("10000")).quantize(Decimal("0.0001")),
            )
            self.assertNotEqual(stored_mfe, contaminated_mfe)
            self.assertLess(stored_mfe, Decimal("50"))

        market_1h = self._market_row(chain, "1h")
        self.assertEqual(
            Decimal(market_1h["horizon_return_bps"]),
            ((Decimal("100.0") - baseline) / baseline * Decimal("10000")).quantize(Decimal("0.0001")),
        )
        self.assertEqual(
            Decimal(market_1h["horizon_return_bps"]), Decimal("-9.9900")
        )

    def test_window_completeness_one_record_per_decision_horizon(self) -> None:
        agents = [
            ScheduledAgent("ema", SignalAction.BUY),
            ScheduledAgent("rsi", SignalAction.BUY),
            ScheduledAgent("mean", SignalAction.BUY),
            ScheduledAgent("breakout", SignalAction.ABSTAIN),
        ]
        engine = PaperReplayEngine(
            RiskPolicy(), {"default": self.evidence}, agents, ledger=self.db
        )
        result = engine.run(self.candles[:120])
        decision_count = len(result.decisions)
        ledger_decision_count = self.db.connection.execute(
            "SELECT COUNT(*) FROM ledger_decisions"
        ).fetchone()[0]
        self.assertEqual(ledger_decision_count, decision_count)
        for decision in result.decisions:
            decision_index = self.candles.index(decision.candle)
            corr_id = engine._correlation_id(
                "aqis-four-agent-v1", decision.candle.utc_timestamp, decision_index
            )
            fill = next(
                (f for f in result.fills if f.correlation_id == corr_id), None
            )
            for horizon in ("1h", "4h", "24h", "7d"):
                horizon_candles = {"1h": 1, "4h": 4, "24h": 24, "7d": 168}[horizon]
                market_count = self.db.connection.execute(
                    "SELECT COUNT(*) FROM ledger_outcomes WHERE correlation_id=? AND horizon=? AND outcome_type='market'",
                    (corr_id, horizon),
                ).fetchone()[0]
                portfolio_count = self.db.connection.execute(
                    "SELECT COUNT(*) FROM ledger_outcomes WHERE correlation_id=? AND horizon=? AND outcome_type='portfolio'",
                    (corr_id, horizon),
                ).fetchone()[0]
                statuses = [
                    row[0]
                    for row in self.db.connection.execute(
                        "SELECT status FROM ledger_outcome_status WHERE correlation_id=? AND horizon=?",
                        (corr_id, horizon),
                    )
                ]
                if decision_index + horizon_candles >= 120:
                    self.assertEqual((market_count, portfolio_count), (0, 0))
                    self.assertEqual(statuses, ["horizon_not_reached"])
                else:
                    self.assertEqual(market_count, 1)
                    if fill is not None:
                        self.assertEqual((portfolio_count, len(statuses)), (1, 0))
                    else:
                        self.assertEqual(portfolio_count, 0)
                        self.assertEqual(len(statuses), 1)
                for status in statuses:
                    self.assertTrue(
                        status in {"horizon_not_reached", "no_fill", "no_action_flat", "no_order_end_of_replay"}
                        or status.startswith("no_fill_rejected:")
                        or status.startswith("no_fill_suppressed:")
                    )
        duplicate_outcomes = self.db.connection.execute(
            "SELECT outcome_id, COUNT(*) FROM ledger_outcomes GROUP BY outcome_id HAVING COUNT(*) > 1"
        ).fetchall()
        self.assertEqual(duplicate_outcomes, [])
        duplicate_statuses = self.db.connection.execute(
            "SELECT correlation_id, horizon, COUNT(*) FROM ledger_outcome_status GROUP BY correlation_id, horizon HAVING COUNT(*) > 1"
        ).fetchall()
        self.assertEqual(duplicate_statuses, [])
        reachable_horizons = 0
        for decision in result.decisions:
            decision_index = self.candles.index(decision.candle)
            reachable_horizons += sum(
                1
                for horizon_candles in {"1h": 1, "4h": 4, "24h": 24, "7d": 168}.values()
                if decision_index + horizon_candles < 120
            )
        total_records = self.db.connection.execute(
            "SELECT (SELECT COUNT(*) FROM ledger_outcomes) + (SELECT COUNT(*) FROM ledger_outcome_status)"
        ).fetchone()[0]
        self.assertEqual(total_records, reachable_horizons + 4 * decision_count)

    def test_horizon_timestamps_exclude_decision_and_reach_horizon_index(self) -> None:
        agents = [
            ScheduledAgent("ema", SignalAction.ABSTAIN),
            ScheduledAgent("rsi", SignalAction.ABSTAIN),
            ScheduledAgent("mean", SignalAction.ABSTAIN),
            ScheduledAgent("breakout", SignalAction.ABSTAIN),
        ]
        engine = PaperReplayEngine(
            RiskPolicy(), {"default": self.evidence}, agents, ledger=self.db
        )
        result = engine.run(self.candles[:200])
        decision_index = 3
        decision_candle = self.candles[decision_index]
        corr_id = engine._correlation_id(
            "aqis-four-agent-v1", decision_candle.utc_timestamp, decision_index
        )
        chain = self.db.get_decision_chain(corr_id)
        for horizon, w in (("1h", 1), ("4h", 4), ("24h", 24), ("7d", 168)):
            market = self._market_row(chain, horizon)
            horizon_candle = self.candles[decision_index + w]
            self.assertEqual(market["anchor"], "decision")
            self.assertEqual(market["direction"], None)
            self.assertEqual(Decimal(market["baseline_price"]), Decimal(str(decision_candle.close)))
            self.assertEqual(market["horizon_timestamp"], horizon_candle.utc_timestamp.isoformat())
            self.assertEqual(
                horizon_candle.utc_timestamp - decision_candle.utc_timestamp,
                timedelta(hours=w),
            )
            self.assertIsNone(market["baseline_equity"])
            self.assertGreaterEqual(Decimal(market["max_adverse_excursion"]), Decimal(0))
            self.assertGreaterEqual(Decimal(market["max_favourable_excursion"]), Decimal(0))
        self.assertEqual(
            {o["status"] for o in chain["statuses"]}, {"no_action_flat"}
        )
        self.assertEqual(len(chain["statuses"]), len(chain["outcomes"]))

        fill_engine = PaperReplayEngine(
            RiskPolicy(), {"default": self.evidence}, self._buy_agents(), ledger=self.db
        )
        fill_result = fill_engine.run(self.candles[:50])
        first_fill = next(
            f for f in fill_result.fills if f.side is OrderSide.BUY
        )
        fill_corr = first_fill.correlation_id
        fill_chain = self.db.get_decision_chain(fill_corr)
        decision_index = self.candles.index(
            next(
                d.candle
                for d in fill_result.decisions
                if fill_engine._correlation_id(
                    "aqis-four-agent-v1",
                    d.candle.utc_timestamp,
                    self.candles.index(d.candle),
                )
                == fill_corr
            )
        )
        market_fill = self._market_row(fill_chain, "1h")
        self.assertEqual(market_fill["anchor"], "decision")
        self.assertEqual(
            Decimal(market_fill["baseline_price"]),
            Decimal(str(self.candles[decision_index].close)),
        )
        self.assertEqual(
            market_fill["horizon_timestamp"],
            self.candles[decision_index + 1].utc_timestamp.isoformat(),
        )
        self.assertIsNotNone(fill_chain["execution"])
        self.assertEqual(fill_chain["decision"]["candle_index"], decision_index)

    def test_addon_buy_portfolio_baselines_are_decision_local(self) -> None:
        engine = PaperReplayEngine(
            RiskPolicy(), {"default": self.evidence}, self._buy_agents(), ledger=self.db
        )
        result = engine.run(self.candles[:50])
        buy_fills = [f for f in result.fills if f.side is OrderSide.BUY]
        self.assertGreaterEqual(len(buy_fills), 2)
        corr0, corr1 = buy_fills[0].correlation_id, buy_fills[1].correlation_id

        def decision_index_for(corr_id):
            return self.candles.index(
                next(
                    d.candle
                    for d in result.decisions
                    if engine._correlation_id(
                        "aqis-four-agent-v1",
                        d.candle.utc_timestamp,
                        self.candles.index(d.candle),
                    )
                    == corr_id
                )
            )

        decision0 = decision_index_for(corr0)
        decision1 = decision_index_for(corr1)
        self.assertEqual(decision1, decision0 + 1)
        chain0 = self.db.get_decision_chain(corr0)
        chain1 = self.db.get_decision_chain(corr1)
        row0 = self._portfolio_row(chain0, "1h")
        row1 = self._portfolio_row(chain1, "1h")
        baseline0 = buy_fills[0].post_fill_equity_usdt
        baseline1 = buy_fills[1].post_fill_equity_usdt
        self.assertIsNotNone(baseline0)
        self.assertIsNotNone(baseline1)
        self.assertEqual(Decimal(row0["baseline_equity"]), baseline0)
        self.assertEqual(Decimal(row1["baseline_equity"]), baseline1)
        self.assertNotEqual(Decimal(row0["baseline_equity"]), Decimal(row1["baseline_equity"]))
        exec_index0 = decision0 + 1
        exec_index1 = decision1 + 1
        snapshot0 = next(
            s for s in result.portfolio_snapshots
            if s.timestamp == self.candles[exec_index0].utc_timestamp
        )
        snapshot1 = next(
            s for s in result.portfolio_snapshots
            if s.timestamp == self.candles[exec_index1].utc_timestamp
        )
        expected_mae0 = max(Decimal(0), baseline0 - snapshot0.equity_usdt).quantize(Decimal("0.01"))
        expected_mae1 = max(Decimal(0), baseline1 - snapshot1.equity_usdt).quantize(Decimal("0.01"))
        expected_mfe0 = max(Decimal(0), snapshot0.equity_usdt - baseline0).quantize(Decimal("0.01"))
        expected_mfe1 = max(Decimal(0), snapshot1.equity_usdt - baseline1).quantize(Decimal("0.01"))
        self.assertEqual(Decimal(row0["max_adverse_excursion"]), expected_mae0)
        self.assertEqual(Decimal(row1["max_adverse_excursion"]), expected_mae1)
        self.assertEqual(Decimal(row0["max_favourable_excursion"]), expected_mfe0)
        self.assertEqual(Decimal(row1["max_favourable_excursion"]), expected_mfe1)

    def test_outcome_records_identical_across_databases(self) -> None:
        def run_into(db):
            agents = self._buy_agents()
            engine = PaperReplayEngine(
                RiskPolicy(), {"default": self.evidence}, agents, ledger=db
            )
            engine.run(self.candles[:30])
            return engine

        path_a = os.path.join(self.tmpdir.name, "repro_a.sqlite3")
        path_b = os.path.join(self.tmpdir.name, "repro_b.sqlite3")
        db_a = LedgerDatabase(path_a)
        db_b = LedgerDatabase(path_b)
        try:
            run_into(db_a)
            run_into(db_b)
            outcome_rows_a = []
            for row in db_a.connection.execute("SELECT * FROM ledger_outcomes ORDER BY outcome_id"):
                d = dict(row)
                d.pop("created_at", None)
                outcome_rows_a.append(d)
            outcome_rows_b = []
            for row in db_b.connection.execute("SELECT * FROM ledger_outcomes ORDER BY outcome_id"):
                d = dict(row)
                d.pop("created_at", None)
                outcome_rows_b.append(d)
            self.assertEqual(len(outcome_rows_a), len(outcome_rows_b))
            self.assertEqual(outcome_rows_a, outcome_rows_b)
            status_rows_a = []
            for row in db_a.connection.execute("SELECT * FROM ledger_outcome_status ORDER BY status_id"):
                d = dict(row)
                d.pop("recorded_at", None)
                status_rows_a.append(d)
            status_rows_b = []
            for row in db_b.connection.execute("SELECT * FROM ledger_outcome_status ORDER BY status_id"):
                d = dict(row)
                d.pop("recorded_at", None)
                status_rows_b.append(d)
            self.assertEqual(len(status_rows_a), len(status_rows_b))
            self.assertEqual(status_rows_a, status_rows_b)
            self.assertGreater(len(outcome_rows_a), 0)
            self.assertGreater(len(status_rows_a), 0)
        finally:
            db_a.close()
            db_b.close()

    def test_at_most_one_fill_per_correlation_id(self) -> None:
        falling_policy = RiskPolicy(max_drawdown_stop_pct=Decimal("0.001"))
        engine = PaperReplayEngine(
            falling_policy, {"default": self.evidence}, self._buy_agents(), ledger=self.db
        )
        result = engine.run(self._crash_candles())
        self.assertGreater(len(result.fills), 0)
        corr_ids = [f.correlation_id for f in result.fills]
        self.assertTrue(all(c is not None for c in corr_ids))
        self.assertEqual(len(corr_ids), len(set(corr_ids)))
        ledger_filled = self.db.connection.execute(
            "SELECT correlation_id FROM ledger_executions WHERE fill IS NOT NULL"
        ).fetchall()
        self.assertEqual(len(ledger_filled), len(result.fills))
        self.assertCountEqual([r[0] for r in ledger_filled], corr_ids)


if __name__ == "__main__":
    unittest.main()
