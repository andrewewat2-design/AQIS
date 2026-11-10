# AQIS

AQIS is an agent-supervised strategy lifecycle platform. Its v1 prototype is deliberately single-market: BTC/USDT Spot, Binance, 1-hour completed candles, historical validation, and paper trading.

The architecture is multi-market by design. Market/engine adapters are isolated from AQIS-owned strategy versioning, lifecycle approvals, selection, risk policy, monitoring, and audit records.

## V1 boundaries

- Four independent deterministic strategy agents: 20/50 EMA crossover, RSI momentum, mean reversion, and breakout detection.
- Each signal carries a deterministic 0–100 rule-strength confidence. A vote is valid at 60 or above; confidence is not a probability of profit.
- Equal-weight selector: three or more valid BUY votes target long; three or more valid SELL votes target cash; otherwise target cash. A cash target exits an existing long.
- An opportunity gate requires frozen validation evidence for expected net edge and stressed loss. A separate deterministic risk engine evaluates every order.
- The Risk Evaluation Agent can analyze and recommend; it cannot alter policies, submit orders, promote strategies, or resume a pause.
- Regime-based selection is computed in shadow mode during v1 and cannot alter paper orders.
- No real-money execution or restricted deployment in v1.

## Working paper policy defaults

The policy is human-owned and immutable once activated. Changing it creates a new policy version requiring approval.

| Setting | V1 default |
| --- | ---: |
| Simulated equity | 10,000 USDT |
| Maximum aggregate BTC exposure | 20% of current equity |
| Maximum order notional | 10% of current equity |
| Daily loss stop | 2% of UTC day-start equity |
| Maximum drawdown stop | 5% from equity high-water mark |
| Fee model | 0.10% per fill, configurable per policy |
| Simulated slippage | 10 bps adverse per fill |
| Missing/stale hourly candle | Halt if not received within 5 minutes of expected close |
| Consecutive paper execution errors | Halt after 3 |
| Volatility review alert | Current hourly volatility ≥ 2× trailing baseline |
| Strategy correlation review alert | Pairwise correlation ≥ 0.85 (minimum 30 observations) |
| Slippage review alert | Actual slippage exceeds expected by 20 bps |

Equity/loss calculations include realized and unrealized P&L, fees, and simulated slippage. A loss-limit breach cancels pending orders, targets cash using reduce-only exits, and latches a pause until human approval. A data-integrity halt cancels orders and blocks new submissions; AQIS does not issue orders from an untrusted market or position state.

Volatility, correlation, and slippage review thresholds are monitoring alerts in v1; they do not silently change strategy votes or risk limits. The Risk Agent can recommend a pause or size reduction, while the deterministic engine and human-owned policy remain authoritative.

## Run locally

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
aqis demo
```

Run the tests with:

```sh
python -m unittest discover -s tests -v
```

Historical CSV input uses `timestamp,open,high,low,close,volume`; timestamps are UTC candle-close times and must be strictly increasing hourly finalized candles.

```sh
aqis download --start 2023-01-01T00:00:00Z --end 2024-01-01T00:00:00Z \
  --output data/binance/BTCUSDT-1h.csv
aqis validate-data path/to/BTCUSDT-1h.csv
aqis split data/binance/BTCUSDT-1h.csv --train-output data/train.csv \
  --holdout-output data/holdout.csv
aqis validate --train-csv data/train.csv --holdout-csv data/holdout.csv \
  --report-id btcusdt-v1-validation --output outputs/validation.json
aqis policy activate --policy-file config/paper-policy.json --approved-by operator
aqis replay data/paper-period.csv --policy-database data/aqis.sqlite3 \
  --evidence outputs/validation.json --output outputs/paper-run.json
```

The replay command fails closed on new entries unless it receives a passing, frozen validation evidence report. The validation command uses a separate chronological holdout segment and requires at least 30 completed trades by default. The demo command uses synthetic data and validation-only sizing; it is explicitly non-promotable.

Candidate state is stored in SQLite with append-only lifecycle events:

```sh
aqis lifecycle create --candidate-id btc-four-agent-v1 \
  --confidence-scales '{"ema":"1 ATR","rsi":"15 RSI points","mean_reversion":"1 ATR","breakout":"1 ATR"}' \
  --parameters '{"timeframe":"1h","selector":"3-of-4","confidence_threshold":"60"}' \
  --source-hash <source-hash>
aqis lifecycle advance --candidate-id btc-four-agent-v1 --stage research --actor strategy_agent
aqis lifecycle advance --candidate-id btc-four-agent-v1 --stage backtest --actor supervisor_agent
aqis lifecycle advance --candidate-id btc-four-agent-v1 --stage validation --actor backtesting_agent
aqis lifecycle record-validation --candidate-id btc-four-agent-v1 \
  --report-id btcusdt-v1-validation --passed
aqis lifecycle approve-paper --candidate-id btc-four-agent-v1 \
  --report-id btcusdt-v1-validation --actor operator
aqis lifecycle advance --candidate-id btc-four-agent-v1 --stage paper_trading --actor operator
```

## Architecture

- `src/aqis/strategies.py`: four independent strategy agents and their frozen mathematical signal-strength scales.
- `src/aqis/selector.py`: deterministic confidence filter and 3-of-4 target selector.
- `src/aqis/opportunity.py`: validates out-of-sample net-edge/stress-loss evidence before new risk is taken.
- `src/aqis/risk.py`: immutable policy, portfolio limits, loss stops, market-data integrity halt, and reduce-only close path.
- `src/aqis/lifecycle.py`: candidate state machine; modifications create new versions and do not inherit approval.
- `src/aqis/paper.py`: completed-candle replay, next-candle fills, fees, slippage, sizing, P&L, and monitoring metrics.
- `src/aqis/agents.py`: structured supervisor/risk-agent interfaces and an optional OpenAI-compatible provider. Provider output is advisory and cannot invoke order or policy tools.
- `src/aqis/engine.py`: restricted Freqtrade control adapter for health, pause, and order cancellation. The self-contained paper replay is the verified v1 execution path; no exchange credentials or live order path are implemented. The Freqtrade adapter deliberately does not expose force-entry or force-exit calls to agents.

## Strategy math

For each strategy, raw strength `m` is normalized by a fixed, versioned scale `q`:

```text
confidence = 100 * clamp(m / q, 0, 1)
```

The initial scales are frozen in code/config before validation:

- EMA: `abs(EMA20 - EMA50) / ATR14`, with `q = 1 ATR`.
- RSI: distance beyond the 55/45 trigger in RSI points, with `q = 15 points`.
- Mean reversion: the smaller of prior lower-band penetration and current re-entry distance, divided by ATR14, with `q = 1 ATR`.
- Breakout: close distance beyond the prior 20-hour high / 10-hour low, divided by ATR14, with `q = 1 ATR`.

Changing a signal rule, confidence formula, scale, or selector creates a new candidate version and requires backtesting, validation, and human approval again.

## Freqtrade and future markets

AQIS owns the policy/control plane. `FreqtradeAdapter` is a restricted REST control client for health, pause, and cancellation; it has no force-entry/force-exit methods. The verified v1 paper path is the local replay engine. A complete Freqtrade strategy/risk callback bridge is not yet validated in this environment, which has neither Freqtrade nor Docker installed. Do not connect an agent to Freqtrade's unrestricted REST API. Its GPL-3.0 license must be considered before distributing a combined deployment. LEAN-style signal → target → risk → execution boundaries are used as an architectural pattern; LEAN is not a v1 runtime dependency.

Regime-aware selection, extra exchanges/assets, live trading, and autonomous policy or strategy promotion are later phases.
