---
name: trading-safety
description: The non-negotiable safety gates of Adaptive-Trader-AI — how to verify them and how to extend the system without breaking any of them.
when_to_use: Any change touching rule ingestion, model promotion, order dispatch, or execution mode.
---

# Trading Safety (Adaptive-Trader-AI)

## The gates (verify, never bypass)
1. `EXECUTION_MODE=SIGNAL_ONLY` + `DEMO_TRADING_ENABLED=False` — checked in
   worker start (`market_worker.py`) and demo dispatch
   (`demo_executor.py::assert_safe_to_execute_demo_order`).
2. Human approval: rules enter execution ONLY with `approval_status=approved`
   (`get_active_strategy_rules`); the knowledge pipeline creates PENDING
   candidates and never approves.
3. Knowledge classification gate (2026-11-07): ONLY chunks classified
   EXPLICIT_RULE become rule candidates — a 102-row GLOSSARY now yields ZERO
   registry rules (verified end-to-end; ambiguous rule-intent stays out for
   human review; the audit log records the class summary).
4. Model promotion: 10 ordered gates incl. walk-forward provenance and a
   persisted ML-WF run binding; evidence validation rejects NaN/±inf and
   out-of-range values.
5. Order dispatch: stale quote → refuse; account must be demo; `order_send`
   under the global MT5 lock; ambiguous outcomes → EXECUTION_UNKNOWN (never
   guessed), retained forever by retention (buglist item 2).

## How to verify quickly (real commands)
- Rules status: read-only
  `SELECT approval_status, COUNT(*) FROM trading_rules GROUP BY 1`.
- Promotion pre-check: `GET /api/v1/ml/models/{id}/promotion-check`.
- Emergency stop: `POST /api/v1/system/emergency-stop/activate` (token) —
  never by editing config.

## Evidence (2026-11-07)
- glossary ingest → 0 rules created (was 102), all 102 chunks preserved,
  3 ambiguous kept out with an audit record; a real rule sentence
  ("Buy when RSI < 30 on XAUUSD M15...") still extracts. Zero auto-approvals
  at every step.
