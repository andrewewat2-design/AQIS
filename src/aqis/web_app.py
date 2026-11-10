from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# ---------------------------------------------------------------------------
# AQIS imports – the web app uses the same package under `src/aqis/`
# ---------------------------------------------------------------------------
# Ensure the project `src` directory is on the path so `aqis` is importable
# regardless of how streamlit is invoked.
_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from aqis.cli import build_parser
from aqis.risk import RiskEngine, PersistentPolicyRegistry, IntegrityStatus
from aqis.models import (
    RiskPolicy,
    PaperAccount,
    OrderIntent,
    OrderSide,
    StrategySignal,
    SignalAction,
    TargetPosition,
    ONE,
    ZERO,
)
from aqis.regime import ShadowRegimeClassifier, RegimeAssessment
from aqis.selector import ConsensusSelector
from aqis.validation import build_validation_report
from aqis.paper import PaperReplayEngine
from aqis.lifecycle import PersistentLifecycleManager
from aqis.opportunity import OpportunityGate
from aqis.audit import AuditLog
from aqis.indicators import atr, directional_index, ema, rsi, bollinger, donchian
from aqis.marketdata import validate_candles, load_csv, download_binance_spot_klines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_cmd(args: List[str]) -> str:
    """Run an aqis CLI subcommand and return stdout."""
    result = subprocess.run(
        [sys.executable, "-m", "aqis"] + args,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        st.error(f"CLI error: {result.stderr}")
    return result.stdout


def _parse_policy_json(path: str) -> Dict[str, any]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------------------------

PAGE_CONFIG = {
    "page_title": "AQIS Web UI",
    "page_icon": "📊",
    "layout": "wide",
}

st.set_page_config(**PAGE_CONFIG)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("AQIS Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Demo",
        "Data Validation",
        "Strategy Validation",
        "Paper Trading",
        "Policy Management",
        "Lifecycle",
        "Regime Analysis",
    ],
)


# ---------------------------------------------------------------------------
# Page: Demo
# ---------------------------------------------------------------------------

if page == "Demo":
    st.title("📈 AQIS Demo Pipeline")
    st.caption("Synthetic, non-promotable pipeline smoke test")

    if st.button("Run Demo", type="primary"):
        with st.spinner("Running synthetic pipeline..."):
            output = _run_cmd(["demo"])
        try:
            json_start = output.find("{")
            if json_start == -1:
                raise ValueError("No JSON object found in CLI output")
            data, _ = json.JSONDecoder().raw_decode(output[json_start:])
        except (ValueError, json.JSONDecodeError):
            data = {"raw_output": output}
        st.json(data)

    st.divider()
    st.markdown(
        "**Note:** The demo uses synthetic prices and cannot approve or promote "
        "a candidate. It exercises the full pipeline: data → strategies → selector → "
        "risk engine → paper replay."
    )


# ---------------------------------------------------------------------------
# Page: Data Validation
# ---------------------------------------------------------------------------

elif page == "Data Validation":
    st.title("📋 Data Validation")
    st.caption("Validate completed hourly OHLCV CSV files")

    csv_file = st.file_uploader("Upload BTCUSDT-1h CSV", type=["csv"])
    allow_gaps = st.checkbox("Allow gaps", value=False)

    if st.button("Validate CSV"):
        if csv_file is None:
            st.error("Please upload a CSV file.")
        else:
            tmp_path = Path("/tmp") / "btcusdt-1h.csv"
            with open(tmp_path, "wb") as f:
                f.write(csv_file.getbuffer())

            output = _run_cmd(["validate-data", str(tmp_path), "--allow-gaps" if allow_gaps else ""])
            try:
                data = json.loads(output)
            except Exception:
                data = {"raw": output}

            st.subheader("Validation Result")
            st.json(data)

            if data.get("valid"):
                st.success("CSV is valid and contiguous.")
            else:
                st.warning("CSV has issues - see details above.")


# ---------------------------------------------------------------------------
# Page: Strategy Validation
# ---------------------------------------------------------------------------

elif page == "Strategy Validation":
    st.title("🧪 Strategy Validation")
    st.caption("Evaluate frozen strategy rules on a holdout CSV")

    train_file = st.file_uploader("Train CSV (strategy generation)", type=["csv"], key="train")
    holdout_file = st.file_uploader("Holdout CSV (validation)", type=["csv"], key="holdout")
    min_trades = st.number_input("Minimum completed trades", min_value=10, value=30, step=5)

    if st.button("Validate Strategies"):
        if train_file is None or holdout_file is None:
            st.error("Please upload both train and holdout CSV files.")
        else:
            train_path = Path("/tmp") / "train.csv"
            holdout_path = Path("/tmp") / "holdout.csv"
            with open(train_path, "wb") as f:
                f.write(train_file.getbuffer())
            with open(holdout_path, "wb") as f:
                f.write(holdout_file.getbuffer())

            report_id = st.text_input("Report ID", value=f"streamlit-{datetime.utcnow().isoformat()}")
            output = _run_cmd(["validate", "--train-csv", str(train_path), "--holdout-csv", str(holdout_path), "--report-id", report_id])

            try:
                data = json.loads(output)
            except Exception:
                data = {"raw": output}

            st.subheader("Validation Report")
            st.json(data)

            if data.get("passed"):
                st.success("Validation passed - evidence is frozen.")
            else:
                st.info("Validation did not pass - adjust strategy parameters.")


# ---------------------------------------------------------------------------
# Page: Paper Trading
# ---------------------------------------------------------------------------

elif page == "Paper Trading":
    st.title("💹 Paper Trading Replay")
    st.caption("Replay historical candles through the paper engine")

    csv_file = st.file_uploader("Paper period CSV", type=["csv"], key="paper")
    policy_db = st.text_input("Policy database path", value="data/aqis.sqlite3")
    evidence_file = st.file_uploader("Validation evidence JSON (optional)", type=["json"], key="evidence")
    output_file = st.text_input("Output JSON path", value="outputs/paper-run.json")

    if st.button("Run Paper Replay"):
        if csv_file is None:
            st.error("Please upload a paper period CSV file.")
        else:
            paper_path = Path("/tmp") / "paper-period.csv"
            with open(paper_path, "wb") as f:
                f.write(csv_file.getbuffer())

            evidence_path = None
            if evidence_file is not None:
                evidence_path = Path("/tmp") / "validation.json"
                with open(evidence_path, "wb") as f:
                    f.write(evidence_file.getbuffer())

            args = ["replay", str(paper_path), "--policy-database", policy_db]
            if evidence_path:
                args += ["--evidence", str(evidence_path)]
            args += ["--output", output_file]

            with st.spinner("Running paper replay..."):
                output = _run_cmd(args)

            try:
                data = json.loads(output)
            except Exception:
                data = {"raw": output}

            st.subheader("Paper Run Result")
            st.json(data)

            if data.get("ending_equity_usdt"):
                ending = float(data["ending_equity_usdt"])
                starting = float(data.get("starting_equity_usdt", "10000"))
                pnl_pct = (ending - starting) / starting * 100
                col1, col2, col3 = st.columns(3)
                col1.metric("Starting Equity", f"${starting:,.2f}")
                col2.metric("Ending Equity", f"${ending:,.2f}")
                col3.metric("P&L %", f"{pnl_pct:+.2f}%")

            if data.get("halted"):
                st.warning("Replay was halted (risk limit breach or execution error).")
            else:
                st.success("Replay completed successfully.")


# ---------------------------------------------------------------------------
# Page: Policy Management
# ---------------------------------------------------------------------------

elif page == "Policy Management":
    st.title("🛡️ Policy Management")
    st.caption("Human-approved immutable risk policy versions")

    action = st.radio("Action", ["Show active", "Activate new version"])

    policy_file = st.text_input("Policy JSON file path", value="config/paper-policy.json")
    approved_by = st.text_input("Approved by (human name)", value="operator")

    if st.button("Execute"):
        if action == "Show active":
            # Show active policy
            if os.path.exists(policy_file):
                policy_dict = _parse_policy_json(policy_file)
                st.json(policy_dict)
            else:
                st.error(f"Policy file not found: {policy_file}")

        elif action == "Activate new version":
            if not os.path.exists(policy_file):
                st.error(f"Policy file not found: {policy_file}")
            elif not approved_by:
                st.error("Please enter a human actor name.")
            else:
                # Use the CLI to activate
                with st.spinner("Activating policy version..."):
                    # First, let's use the PersistentPolicyRegistry directly
                    from aqis.risk import PersistentPolicyRegistry

                    registry = PersistentPolicyRegistry(database=policy_db if False else "data/aqis.sqlite3")
                    # Read the policy from the JSON file
                    policy_dict = _parse_policy_json(policy_file)
                    from aqis.models import RiskPolicy
                    policy = RiskPolicy(**policy_dict)

                    try:
                        registry.activate(policy, approved_by)
                        st.success(f"Policy version {policy.version} activated by {approved_by}")
                    except (PermissionError, ValueError) as e:
                        st.error(f"Failed to activate: {e}")
                    finally:
                        registry.close()

                # Show active policies
                try:
                    active = registry.active("aqis-paper-btcusdt")
                    st.json({"active_policy": active, "policy_id": "aqis-paper-btcusdt"})
                except KeyError:
                    st.info("No active policy found - activate one first.")


# ---------------------------------------------------------------------------
# Page: Lifecycle
# ---------------------------------------------------------------------------

elif page == "Lifecycle":
    st.title("🔄 Candidate Lifecycle")
    st.caption("Manage persistent candidate state machine")

    sub_action = st.radio(
        "Action",
        [
            "Create candidate",
            "Show candidate",
            "Advance stage",
            "Record validation",
            "Approve paper",
        ],
    )

    candidate_id = st.text_input("Candidate ID", value="btc-four-agent-v1")

    if sub_action == "Create candidate":
        conf_scales = st.text_area(
            "Confidence scales (JSON)",
            value='{"ema":"1 ATR","rsi":"15 RSI points","mean_reversion":"1 ATR","breakout":"1 ATR"}',
        )
        parameters = st.text_area(
            "Parameters (JSON)",
            value='{"timeframe":"1h","selector":"3-of-4","confidence_threshold":"60"}',
        )
        source_hash = st.text_input("Source hash (optional)", value="")

        if st.button("Create"):
            # Generate a simple hash if not provided
            hash_val = source_hash or f"hash-{datetime.utcnow().isoformat()}"
            args = [
                "lifecycle",
                "create",
                f"--candidate-id",
                candidate_id,
                f"--confidence-scales",
                conf_scales,
                f"--parameters",
                parameters,
                f"--source-hash",
                hash_val,
            ]
            output = _run_cmd(args)
            st.code(output)

    elif sub_action == "Show candidate":
        if st.button("Show"):
            output = _run_cmd(["lifecycle", "show", f"--candidate-id", candidate_id])
            st.code(output)

    elif sub_action == "Advance stage":
        stage = st.selectbox("Stage", ["discovery", "research", "backtest", "validation", "paper_trading"])
        actor = st.text_input("Actor", value="strategy_agent")

        if st.button("Advance"):
            output = _run_cmd(["lifecycle", "advance", f"--candidate-id", candidate_id, f"--stage", stage, f"--actor", actor])
            st.code(output)

    elif sub_action == "Record validation":
        report_id = st.text_input("Report ID", value="btcusdt-v1-validation")
        passed = st.checkbox("Passed", value=True)

        if st.button("Record"):
            actor = st.text_input("Actor", value="backtesting_agent")
            output = _run_cmd([
                "lifecycle",
                "record-validation",
                f"--candidate-id",
                candidate_id,
                f"--report-id",
                report_id,
                f"--actor",
                actor,
                "--passed" if passed else "",
            ])
            st.code(output)

    elif sub_action == "Approve paper":
        report_id = st.text_input("Validation report ID", value="btcusdt-v1-validation")
        actor = st.text_input("Actor (human)", value="operator")

        if st.button("Approve"):
            output = _run_cmd(["lifecycle", "approve-paper", f"--candidate-id", candidate_id, f"--report-id", report_id, f"--actor", actor])
            st.code(output)


# ---------------------------------------------------------------------------
# Page: Regime Analysis
# ---------------------------------------------------------------------------

elif page == "Regime Analysis":
    st.title("🌊 Regime Analysis")
    st.caption("Shadow regime classification for the current candle window")

    st.markdown(
        "The regime classifier determines market conditions "
        "(trending, ranging, high_volatility, warming_up) "
        "and the ConsensusSelector adjusts its confidence threshold "
        "accordingly. The risk engine remains the authoritative hard gate."
    )

    csv_file = st.file_uploader("Upload candle CSV for regime analysis", type=["csv"], key="regime")

    if csv_file is not None:
        regime_path = Path("/tmp") / "regime-csv.csv"
        with open(regime_path, "wb") as f:
            f.write(csv_file.getbuffer())

        # Load candles
        from aqis.models import Candle

        candles = load_csv(regime_path)
        st.info(f"Loaded {len(candles)} candles")

        if st.button("Classify Regime"):
            assessment = ShadowRegimeClassifier().classify(candles)

            col1, col2, col3 = st.columns(3)
            col1.metric("Regime", assessment.regime)
            col2.metric("ADX", f"{assessment.trend_strength:.1f}" if assessment.trend_strength else "N/A")
            col3.metric("ATR Fraction", f"{assessment.hourly_atr_fraction:.4f}" if assessment.hourly_atr_fraction else "N/A")

            st.markdown(f"**Selector Effect:** `{assessment.selector_effect}`")

            # Show effective threshold
            selector = ConsensusSelector()
            selector.regime = assessment
            eff_thresh = selector._effective_threshold()
            st.info(f"Effective confidence threshold: **{eff_thresh:.1f}%**")

            # Demonstrate voting with sample signals
            st.markdown("#### Demo Voting with Regime-Aware Threshold")
            sample_signals = [
                StrategySignal("ema", "v1", candles[-1].timestamp, SignalAction.BUY, 70, "strong"),
                StrategySignal("rsi", "v1", candles[-1].timestamp, SignalAction.BUY, 65, "moderate"),
                StrategySignal("mean", "v1", candles[-1].timestamp, SignalAction.BUY, 55, "weak"),
                StrategySignal("breakout", "v1", candles[-1].timestamp, SignalAction.BUY, 45, "weak"),
            ]

            decision = selector.decide(sample_signals)
            col1.metric("Target", decision.target)
            col2.metric("Buy Votes", ", ".join(decision.buy_votes))
            col3.metric("Reason", decision.reason)

    else:
        st.info("Upload a CSV file to analyze regime conditions.")
        st.caption(
            "CSV format: timestamp,open,high,low,close,volume "
            "(UTC candle-close times, strictly increasing hourly finalized candles)"
        )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.sidebar.divider()
st.sidebar.markdown(
    "AQIS v1 prototype • Agent-supervised strategy lifecycle platform\n"
    "Single-market: BTC/USDT Spot, Binance, 1-hour completed candles"
)