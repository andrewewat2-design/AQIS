# M4 — Decision Ledger and Outcome Evaluation

Status: **In progress — engineering review, not yet accepted**

Working branch: `m4-ledger-completion`

## Purpose

M4 makes AQIS decisions and their later outcomes traceable. It must preserve the original authority boundaries: the ledger observes and records; it does not select strategies, approve risk, alter policies, or execute orders.

## Current implementation observed on the working branch

- SQLite-backed replay, decision, risk, execution, outcome, outcome-status, and ledger-health tables.
- Update/delete prevention triggers on the primary ledger tables.
- Decision records include a market-data snapshot, regime, strategy signals, selector result, threshold, opportunity-evidence reference, and risk-policy identity.
- Risk and execution records are linked by a decision correlation ID.
- Market and portfolio outcomes are measured at 1h, 4h, 24h, and 7d horizons.
- Outcome measurements distinguish decision-anchored from fill-anchored baselines.
- Missing horizons and no-fill cases are represented with status records rather than fabricated zero returns.
- A read-only decision-chain query joins the decision, risk, execution, outcome, and status records.
- The repository test suite includes 36 ledger tests. The prior local run recorded 36 ledger tests and 134 total AQIS tests passing; these results must be rerun against the final branch head before M4 acceptance.

## Research principles applied

1. Preserve business intent and decision context, not only resulting state.
2. Keep historical records immutable; corrections should be new, attributable records rather than silent rewrites.
3. Make event/schema versions explicit and provide safe forward migration behavior.
4. Make retry and replay behavior idempotent.
5. Distinguish source records from derived projections and health summaries.
6. Treat replay as a deterministic, side-effect-free reconstruction process.
7. Report incomplete measurement as incomplete; never substitute zero for an unavailable horizon.
8. Keep strategy attribution honest: agreement with a selector is not proof that a strategy caused a portfolio outcome.

## Open engineering review items

These are review questions to resolve with tests and code inspection; they are not all confirmed defects.

### A. Replay identity and provenance
- Current replay ID inputs visibly include strategy-set version, first/last candle timestamps, and candle count.
- Confirm that market, venue, timeframe, policy identity/version, source-data fingerprint, and relevant engine/config version are either part of replay identity or immutably recorded and checked for conflicts.
- Confirm that two different datasets with the same time boundaries and count cannot be silently treated as the same replay.
- Confirm that a reused replay ID with different metadata fails loudly rather than being silently ignored.

### B. Idempotency and duplicate handling
- Review all insert paths for consistent retry semantics.
- Confirm duplicate decision, risk, execution, outcome, and status writes cannot silently change the meaning of an existing record.
- Confirm status transition sequence allocation is safe under concurrent writers, or explicitly document and enforce single-writer operation for the v1 SQLite implementation.
- Test replaying the same run into the same database and into a fresh database.

### C. Failure semantics and completeness
- Paper replay currently catches ledger write exceptions in several paths and continues.
- Define whether ledger failure is non-blocking observability degradation or a hard integrity halt for each run mode.
- Ensure every swallowed write failure is reflected in health and in the replay result/output.
- Ensure a run cannot be labelled ledger-complete when records are missing.
- Verify pending-outcome counts and completeness flags are derived correctly and cannot become stale.

### D. Outcome correctness
- Confirm the decision-time snapshot contains no future candle data.
- Confirm fill-anchored measurements begin at the correct execution candle and use the effective fill price.
- Confirm decision-anchored measurements use the decision candle close and only subsequent candles.
- Confirm MAE/MFE direction and units are consistent for long intent, exit intent, and flat/no-action decisions.
- Confirm portfolio outcomes include fees, slippage, realized and unrealized P&L as intended.
- Confirm incomplete 1h/4h/24h/7d horizons are explicitly represented and never interpreted as zero.
- Document the limitations of candle OHLC data: intrabar ordering and exact path are unknowable from hourly bars.

### E. Auditability and reconstruction
- A reviewer must be able to follow one correlation ID from decision through risk, order attempt, fill/rejection, and outcome/status.
- Retain exact strategy-set, policy, opportunity-evidence, market-data, and engine-version references needed to interpret the record.
- Confirm serialized records are stable, canonical, and versioned.
- Confirm audit output makes missing links and ledger degradation visible.

### F. Database lifecycle and operational safety
- Verify all connections close reliably and context-manager behavior is used.
- Test schema migration from every supported prior version.
- Test SQLite foreign keys, WAL/busy timeout behavior, backup/restore, and integrity checks.
- Confirm the ledger cannot affect order sizing, selector output, risk decisions, or execution behavior.
- Consider hash chaining or periodic external anchoring only if the threat model requires tamper evidence beyond local SQLite triggers; do not add blockchain or unnecessary infrastructure by default.

## M4 acceptance gates

M4 is complete only when all of the following are true:

- [ ] Replay and decision provenance is sufficient to distinguish materially different runs.
- [ ] All decision outcomes, including no-trade, rejected, suppressed, and not-yet-reached cases, have explicit status semantics.
- [ ] Duplicate/retry behavior is deterministic and tested.
- [ ] Ledger failures are visible, classified, and reflected in run completeness.
- [ ] Decision-to-outcome reconstruction works using stable identifiers.
- [ ] Outcome calculations pass controlled fixtures for all horizons, directions, and fill/no-fill cases.
- [ ] Schema migrations and database lifecycle tests pass.
- [ ] Full AQIS test suite passes on the final branch head.
- [ ] README and operator documentation explain M4 capabilities, limitations, and recovery procedures.
- [ ] M4 changes are reviewed and merged to `main` only after acceptance.

## Explicit non-goals

- No live trading or exchange order submission.
- No change to the four-agent selector or strategy confidence model.
- No autonomous strategy promotion or risk-policy changes.
- No regime-based influence on paper orders.
- No claims that per-strategy vote agreement proves causal performance.
- No blockchain dependency or distributed event-store migration for this prototype.

## Research references

- Microsoft Azure Architecture Center, Event Sourcing pattern: https://learn.microsoft.com/en-us/azure/architecture/patterns/event-sourcing
- AWS Prescriptive Guidance, Event Sourcing pattern: https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/event-sourcing-pattern.html

These references are architectural guidance, not a requirement to adopt full event sourcing. AQIS should use append-only recording selectively where auditability and historical reconstruction justify the complexity.
