# Adaptive Trader AI — Master Project Memory (Phase A onward)

> **Purpose:** Single durable handover for the Adaptive Trader AI project. Update this file when a material decision, code change, test result, blocker, or phase transition occurs.
>
> **Evidence labels**
> - **Reported:** stated in a user/Agent report or planning document; not independently rerun here.
> - **Observed in source/report:** directly visible in a file/report available during consolidation.
> - **Verified now:** directly checked in the current work session.
> - **Unresolved:** evidence is incomplete or conflicting. Never promote a reported claim to verified without evidence.

Last consolidated: 2026-10-13  
Repository: https://github.com/rezamxs/Adaptive-Trader-AI  
Local workspace: `G:\mash`  
Working branch referenced by recent work: `fix/core-correctness-audit`  
Related PR: https://github.com/rezamxs/Adaptive-Trader-AI/pull/1  
Memory file path: `docs/PROJECT_MEMORY.md`

---

## 1. Project purpose and intended operating model

Adaptive Trader AI is a personal, adaptive trading research system. Its intended workflow is:

1. Read the user's trading education and strategy documents.
2. Extract explicit rules and cite the original page/section.
3. Separate explicit rules from interpretations, ambiguities, and contradictions.
4. Obtain the user's approval for rules before using them.
5. Collect and validate historical market data from MetaTrader 5 (MT5), currently focused on XAUUSD / M1.
6. Build time-safe datasets and features; avoid future-data leakage.
7. Train and compare simple ML baselines and candidate models.
8. Backtest with realistic spread, commission, slippage, contract specifications, and position sizing.
9. Observe new market data in `SIGNAL_ONLY` without placing orders.
10. Consider demo execution only after all prior gates pass and the user explicitly authorizes it.
11. Keep model promotion controlled, reproducible, auditable, and reversible.

A model or strategy is not considered successful merely because code runs or a test suite passes. There is no proven profitability claim in the available evidence.

## 2. Hard constraints and collaboration rules

- Keep `EXECUTION_MODE=SIGNAL_ONLY` and `DEMO_TRADING_ENABLED=False` unless the user explicitly authorizes a later, narrowly scoped demo test.
- Do not place orders during audits, strategy extraction, data diagnosis, or MT5 timestamp investigation. Do not call `order_send` for these tasks.
- Real-money trading is outside the approved scope.
- Extracted strategy rules remain `PENDING/UNAPPROVED` until the user explicitly approves them.
- Missing risk/account context must fail closed. Do not replace missing values with fake zeroes or synthetic data.
- Do not silently fall back to Mock/Synthetic market data in a production path.
- Do not merge a PR or change protected trading settings without explicit user approval.
- Before edits, inspect the actual branch, HEAD, working tree, diff, and relevant tests. Prefer one coherent review/change pass over repeated small loops.
- Explain technical reports in plain Persian using four parts: **what was done / what is actually proven / what remains unknown or risky / next step**.
- The user calls the assistant “جک”. Be candid about uncertainty and coding limitations. Prefer architecture review, evidence checks, test design, and reproducibility over unsupported claims.
- Do not write credentials, tokens, passwords, account identifiers, or other secrets into this file.

## 3. Environment and architecture

### Environment reported in project history

- Windows machine; local project at `G:\mash`.
- MT5 terminal connected to broker/provider reported as YourBroker-MT5.
- Primary instrument: XAUUSD; primary timeframe: M1.
- Hardware reported: Intel i7-2630QM and 12 GB RAM.
- Python backend, FastAPI, SQLAlchemy/Alembic, SQLite, scikit-learn, and a single-page dashboard built with HTML/CSS/vanilla JavaScript. The frontend is not React/Next.js.
- MT5 is intended to be read-only except for a future, explicitly authorized demo phase.
- Reported components include document/knowledge ingestion, BM25 retrieval, market-data service, dataset pipeline, ML registry, backtest engine, risk controls, signal worker, API, dashboard, audit logging, and a separately gated demo executor.

These architecture details come from prior project reports and must be checked against the current repository before major changes.

### Knowledge base and ML reports

- A previous report described 6 education documents and approximately 318 knowledge chunks using BM25 retrieval. Coverage and retrieval quality are not fully verified; previous questions involving POC/Delta and slippage reportedly had weak retrieval.
- Baseline, Logistic Regression, and Random Forest were reported as the initial ML models. More complex methods (including XGBoost, Optuna, reinforcement learning, FinRL, NautilusTrader, backtesting.py, vectorbt, and heavy OCR) should not be added without a demonstrated need.
- One historical dataset report described about 94,999 closed XAUUSD M1 candles, past-only features, a triple-barrier label horizon of about 24 minutes, and a purged 70/15/15 chronological split. Reconcile these figures with actual artifacts before reuse.
- Older model reports were weak/uncertain, with ROC-AUC around 0.48–0.50 and reported net losses. These results do not justify deployment or claims of edge.
- A prior report said 47 strategy rules were pending. Another later task report referenced six structured rules in the database. This discrepancy must be reconciled by counting records per source in the current database; do not assume either count is current.

---

## 4. Phase history: A–M (initial build)

The project plan records phases A–M as completed **according to the submitted report**, not independently verified in this consolidation. The plan reported 50/50 tests passing at that stage and explicitly called for independent reruns.

| Phase | Topic | Reported goal/result | Required re-check |
|---|---|---|---|
| A | Environment and MT5 audit | Environment and MT5 inspection reported complete | Re-read the original audit evidence; verify terminal, symbol, clock, and read-only behavior |
| B | Architecture and execution plan | Architecture/plan reported complete | Compare documentation to current source tree and actual entry points |
| C | Database and migrations | Database/migration work reported complete | Create a fresh database and run all migrations from zero |
| D | Document ingestion and rule extraction | Document ingestion/rule extraction reported complete | Use the actual strategy source; check page coverage, formulas, tables, and source citations |
| E | Read-only MT5 market data | Market-data integration reported complete | Independently validate symbol, timestamps, quote freshness, candle completeness, and timezone |
| F | Dataset and features | Dataset/feature pipeline reported complete | Check labels, chronological split, leakage, missing/duplicate bars, and data lineage |
| G | ML models and registry | Models/registry reported complete | Reproduce training and compare to a baseline using a held-out time period |
| H | Backtest engine | Backtest work reported complete | Validate event order, open-position closure, spread/commission/slippage, tick value, volume, and drawdown |
| I | Risk controls and emergency stop | Risk/emergency-stop work reported complete | Test boundary cases, missing context, persistence across restart, and fail-closed behavior |
| J | SIGNAL_ONLY | Signal-only flow reported complete | Verify no order path is reachable in this mode and all signals are auditable |
| K | Limited demo execution | Demo executor reported implemented but locked | Keep disabled; any later test requires explicit authorization and proof of demo account, idempotency, and emergency stop |
| L | Dashboard and API | Dashboard/API reported complete | Test real API paths and distinguish live data from fixtures/mocks |
| M | Tests and documentation | Tests/docs reported complete; plan says 50/50 passed | Rerun the suite on current HEAD and inspect skips, warnings, CI, and migration evidence |

### Important interpretation of A–M

“Completed” in the phase table means **reported complete in the earlier plan**, not that each acceptance criterion has been re-proven now. The plan itself required rechecking the audit, database migration, source citations, MT5 timestamps, dataset leakage, backtest costs, risk boundaries, order-path isolation, and the full test suite.

---

## 5. Subsequent milestones and reported work

### Strategy/knowledge audit

A later task prompt requested a source-by-source audit of documents and rules, including:
- exact file name/path/format/size and page count;
- pages/sections actually readable;
- explicit count of extracted rules, executable rules, ambiguous/conflicting rules, and educational-only statements;
- reconciliation between source documents and database records;
- evidence for every claim of complete document processing.

A previous report described six pending structured rules. Other context referred to 47 pending rules. This remains an explicit reconciliation task.

### Security audit (read-only report dated 2026-10-11)

A saved audit report described the working tree as clean on branch `master`, HEAD `fef584f8fde6f291bccbe8d28ac0a01d681fd29b`, with Python 3.14, FastAPI 0.110, SQLAlchemy 2.0, Pydantic 2.6, SQLite, and a vanilla-JavaScript frontend. It also reported:
- `EXECUTION_MODE=SIGNAL_ONLY`, `DEMO_TRADING_ENABLED=False`, `DEBUG=False`;
- MT5 read-only access with a global `_mt5_lock`;
- Backend Security and Frontend Security skills detected/readable; frontend guidance was adapted because the UI is not React/Next.js.

The audit reported these findings; the report says no live filesystem exploit test was run because the audit was read-only:

| ID | Finding | Reported severity | Status in report |
|---|---|---|---|
| SEC-001 | Path traversal risk in document storage due to untrusted filename being joined to storage path | High | Confirmed from source |
| SEC-002 | Stored XSS risk from unescaped values inserted into `innerHTML` in rules/audit-log tables | High | Confirmed from source |
| SEC-003 | Sensitive state-changing endpoints lacked authentication/access control | Medium | Confirmed from source |
| SEC-004 | Joblib/Pickle artifact loading could execute code if the artifact is tampered with | Medium | Source-inferred risk |
| SEC-005 | Missing baseline HTTP security headers/CSP | Low | Confirmed from source |
| SEC-006 | Upload bytes were read fully into memory before a service-level size check | Low | Confirmed from source |

**Do not assume these findings were fixed.** Check current source and the latest security tests/CI. If fixed, record the exact commit and regression test. Do not reintroduce vulnerable patterns.

### Phase A–G plan proposed during core-correctness audit

A later review organized core correctness into:
- A: MT5 time/data validity;
- B: ML target and signal semantics;
- C: risk/account/position integration;
- D: realistic XAUUSD backtest;
- E: model/knowledge reproducibility;
- F: full tests plus real MT5 validation in `SIGNAL_ONLY`;
- G: dashboard UI/UX redesign.

This A–G list is a **later corrective plan**, not the same as the original A–M build phases. Keep these two phase schemes distinct in reports. Dashboard redesign was intentionally deferred until core correctness and data validity are addressed.

### Core-correctness findings reported during review

Reported issues and fixes across the later audit included:
- MT5 timestamp conversion and quote freshness; future/stale/invalid timestamp handling.
- BUY/SELL semantic correctness and signal aggregation.
- Risk engine fail-closed behavior when open-position count, daily loss, account data, quote, or contract information is absent.
- Contract specification, tick value, volume sizing, spread, and costs in backtests.
- Correct treatment of open positions at the end of a backtest.
- Time-based/purged split, OHLC validation, labels that account for spread, and separation of validation from test data.
- Avoiding silent Mock/Synthetic fallback in production paths.
- Persistence of knowledge chunks, safe upload names/paths, and upload size limits.
- Signal idempotency based on symbol/timeframe/candle and compatibility checks for older keys.
- Rechecking quote freshness, risk, contract details, and volume before a demo execution path.
- Focused regression tests and CI workflow.

This is a list of reported issues/changes, not proof that all fixes exist on the current HEAD. Verify source and tests before relying on any item.

### Phase N — delivery and reproducibility audit

A report from the user/Agent stated:
- full PyTest run: 175 passed, 0 failed, 0 skipped, around 36.27 seconds;
- 58 focused CI tests passed;
- `compileall` and migrations against a fresh SQLite database passed;
- commit reported as `99fba8d` for Phase N fixes;
- `@dataclass` restored for `TimestampValidationResult`;
- 14 time-conversion tests aligned with the current `MT5TimeConverter` contract;
- five risk tests were fixed by supplying valid `open_trades_count=0` and `daily_loss_percent=0.0`, while retaining a fail-closed test for missing context;
- model-promotion test fixture was expanded to 600 candles and synthetic Walk-Forward evidence (5 folds, 50 trades, positive net profit, PF 1.25, DD 4%, costs included);
- API backtest tests were isolated from live MT5, with a separate 503 test for MT5 unavailability;
- `SIGNAL_ONLY` and `DEMO_TRADING_ENABLED=False` were retained.

All figures above are **reported, not independently rerun in this consolidation**. A synthetic test fixture proves only that the test path accepts that fixture; it does not prove real strategy profitability or that production evaluation automatically generates valid Walk-Forward evidence. Recheck actual branch HEAD and CI.

### Q1 Worker milestone

Earlier transfer notes reported:
- Q1 Worker implementation commit `b2f1cda`;
- 85 tests passed;
- 18 live cycles observed;
- zero orders sent;
- the next step was Q2: verify processing of fresh candles, restart behavior, MT5 disconnection handling, and persistence/idempotency.

Treat this as reported evidence. Verify the commit, logs, and cycle records before claiming live operation is currently healthy. No live orders are authorized by this report.

---

## 6. Unresolved MT5 timestamp/freshness blocker

A read-only report from YourBroker-MT5 described five ticks and five historical M1 candles whose timestamps appeared approximately three hours ahead of the system UTC clock. Windows timezone was reported as Iran Standard Time (UTC+03:30), and Windows time synchronization was reported as enabled. One reported raw epoch, `1791845712`, converts to `2026-10-12 22:55:12 UTC`, while the report stated the system UTC time was around `19:55:12`.

The current code was reported to reject future timestamps with `FUTURE_TIMESTAMP_DETECTED` and not apply a global fixed offset.

Required investigation:
1. Capture the raw MT5 epoch values for ticks and bars and independently convert them to UTC.
2. Confirm the machine UTC clock and synchronization status at the same instant.
3. Compare the latest tick time, last closed candle time, and server/provider behavior.
4. Confirm whether the discrepancy is a display conversion, host clock issue, or provider-specific behavior.
5. Test that a genuinely fresh tick is accepted and stale/future ticks are rejected.
6. Do not subtract three hours globally based on one observation.

This is a blocker for data-dependent work. Do not claim the live market-data pipeline is validated until this is reproducibly resolved.

---

## 7. Model governance and acceptance gates

Reported promotion criteria require cost-aware Walk-Forward evidence:
- at least 3 folds and 30 trades;
- positive net profit after costs;
- profit factor greater than 1;
- drawdown within the configured limit;
- spread, slippage, and commission explicitly included;
- evaluation on data not used to tune/select the model;
- traceable dataset/model/config versions.

The actual production evaluator and automatic evidence generation were reported as incomplete or not yet proven. A test fixture alone is insufficient. A model must remain a candidate unless the production evaluation path produces verifiable evidence and all acceptance criteria pass. A model promotion must not alter the active model mid-run; rollback must be possible.

---

## 8. Roadmap from Phase N onward

This roadmap is recorded from the project plan and later reports. Current status must be updated only with evidence.

| Phase | Purpose | Acceptance focus | Current note |
|---|---|---|---|
| N | Delivery/reproducibility audit | Reproducible tests, clean secrets handling, safe execution lock | Results reported; verify current HEAD/CI |
| O | Extract strategy from the source document in Antigravity | Full readable source coverage, page/section citations, rule registry, ambiguity list; no rules activated | **Next strategy workflow**; original source document must be attached to Agent chat |
| P | Human review/approval of rules | Every testable rule has ID, source, measurable condition, and explicit approval; ambiguities remain pending | Not complete until user approval |
| Q | Data and target-market readiness | Sufficient historical data; missing/duplicate bars, timestamps, contract specs and costs audited | Blocked by unresolved MT5 timestamp issue |
| R | Strategy-related labels and model training | Reproducible data/model versions, no leakage, candidate-only status | Wait for O/P and data gate |
| S | Real-strategy backtest | Out-of-sample evaluation with costs, stress scenarios, baseline comparison and drawdown | Wait for strategy approval and valid data |
| T | Forward observation in SIGNAL_ONLY | Fresh-data signal logs, fail-closed behavior, enough observations | No orders |
| U | Controlled demo activation | Explicit user approval, verified demo account, volume/spread/freshness checks, idempotency and tested emergency stop | **Disabled until explicit authorization** |
| V | Champion/candidate learning cycle | Candidate trained off-path, objective promotion gate, rollback, no in-run model swaps | Future |
| W | Operational hardening and handover | Backup/restore test, secure logs, monitoring, dependency versions, incident/runbook documentation | Future |

### Phase O expected artifacts

If equivalent files do not already exist, create/update:
- `docs/strategy/strategy_spec.md` — explicit rules with source page/section;
- `docs/strategy/rule_registry.json` — stable IDs, conditions, parameters, source, approval state;
- `docs/strategy/ambiguities.md` — questions, contradictions, and missing definitions;
- `docs/strategy/data_requirements.md` — symbols, timeframes, history and cost requirements;
- `reports/strategy_review.md` — coverage, unreadable parts, validation evidence.

Read tables, formulas, images/diagrams as well as selectable text. Keep direct source statements separate from Agent interpretation. Rules remain `UNAPPROVED`. Do not train, backtest, or place orders as part of extraction.

---

## 9. Additional technical debt and risks to reconcile

- A previous report said 11 Python scripts inside a ZIP were excluded from ingestion. Verify whether they are strategy source, tooling, or unrelated files and whether coverage should include them.
- A previous report said `semantic-fixtures.yaml` was truncated at 50,000 characters. Verify current file size and parsing behavior.
- Reconcile dataset row counts, timezone semantics, spread/slippage/commission, normalized one-lot assumptions versus actual position volume, Max Drawdown definitions, and differences in trade counts across reports.
- Verify knowledge ingestion covers all intended sources and that citations point to the original document/page.
- Verify `.env`, keys, logs, model artifacts, databases, and Git history do not expose secrets. A previous command showed tracked files including `.env.example`, `data/processed/.gitkeep`, `data/raw/.gitkeep`, and `logs/app.log`; this list alone does not prove a secret leak, but the tracked log and Git history should be reviewed.
- Dashboard UI/UX was disliked by the user; redesign is lower priority than MT5 correctness, strategy extraction, data validity, and model evaluation.
- Keep the first model stack lightweight for the reported hardware. Do not add complex models or infrastructure before baseline evidence supports it.

---

## 10. Git/PR status and evidence discipline

Known recent facts:
- Repository: `rezamxs/Adaptive-Trader-AI`.
- Working branch referenced in recent work: `fix/core-correctness-audit`.
- PR #1 was previously described as Draft/Open and not merged. Check live status before acting.
- A memory-file commit previously created this file on that branch: `5ff9dbe58b3c9bc6674c1babaec8a1f3459888fa`.
- The prior Phase N report referenced commit `99fba8d`; the actual current branch HEAD must be checked before treating it as current.

Do not infer that the branch, PR, local working tree, or test results remain unchanged. Before the next code change:
1. Fetch the live repository metadata and branch HEAD.
2. Confirm PR state and CI checks.
3. Inspect the relevant files and tests at that exact SHA.
4. Record exact commit/test commands and whether tests were local, CI, mocked, or live.
5. Keep reported results distinct from independently observed results.

## 11. Current next actions (priority order)

1. Reconcile the actual current branch/HEAD/PR/CI against the reported Phase N commit.
2. Investigate and resolve MT5 timestamp/freshness semantics using read-only evidence.
3. Run the existing regression suite and focused tests from current HEAD; record skipped tests and warnings.
4. Start Phase O by attaching the original strategy file to Antigravity and extracting all readable content with source citations.
5. Reconcile rule counts and knowledge-source coverage against the database and original files.
6. Keep all extracted rules unapproved until the user reviews them.
7. Do not proceed to data-dependent training/backtest or demo activation until the relevant gates pass.

## 12. Update protocol for this file

For each material project event, add:
- date and phase;
- what changed and why;
- exact paths/commit/PR where available;
- exact test command and result;
- evidence type (reported, local, CI, mocked, or live read-only);
- what is proven, what is not proven, risks/blockers, and next action.

Preserve history when a status changes. Do not silently overwrite a failed result with a later success without retaining enough context to understand the correction. Never include credentials or secrets.

---

## 13. Second verification pass — 2026-10-13

This section records a second pass against the current GitHub state and the project/library reports available during this session. It supplements earlier sections and preserves discrepancies rather than silently choosing one version.

### Current GitHub state — verified now

- Repository default branch: `main`.
- PR #1: `OPEN`, `DRAFT`, not merged; base branch `main`.
- PR head branch: `fix/core-correctness-audit`.
- PR base SHA: `f60ed647956068c6d6423846dd93399a09ec682a`.
- PR head SHA at this check: `88bfc9eef8a25790d835670697681fd5e4775cdf`, the commit that updated this memory file.
- PR is reported mergeable, but this does not mean it should be merged.
- A GitHub Actions workflow run associated with this head was found: run ID `38311139568`, workflow `Core Regression Tests`, conclusion `success`.
- The combined commit-status endpoint returned no status entries. The workflow run above is the direct evidence available for the current head; do not infer that every CI job or the full test suite ran.
- The PR description states that 58 focused regression/security tests passed, compilation/import smoke checks passed, and a fresh SQLite Alembic upgrade passed. It explicitly says the complete repository test suite is **not claimed as passing**. It also says Windows/Python 3.14 and real the configured broker/MT5 symbol economics, account history, quote freshness, margin calculation, and demo execution remain to be verified locally.
- Therefore, the older report of 175/175 tests passing conflicts with the current PR's narrower validation statement. Preserve the 175-test figure as an older reported result only; do not present it as current verified evidence until its exact commit, command, and raw output are recovered and reconciled.

### Additional knowledge-ingestion evidence recovered from project history

A prior QA report described:
- approximately 99.6% coverage of textual educational content;
- 318 knowledge chunks;
- the cover page of a price-action PDF excluded from the extracted text;
- 11 Python scripts inside a ZIP not ingested;
- `semantic-fixtures.yaml` truncated after 50,000 characters;
- remaining retrieval quality problems caused in part by coarse chunking/length bias, including questions about POC/Delta and slippage.

These are historical report claims, not a fresh count of the current database. They mean “all educational material fully ingested” must not be stated without qualification. Check whether the cover page, ZIP scripts, and truncated YAML are required knowledge sources. If they are, repair ingestion and add source-coverage tests; otherwise document the exclusions and their rationale.

### Strategy-source and database count discrepancies

Project reports have used several different counts:
- one report said six source documents were read and six structured rules were stored as `pending`, with 55 tests passing and strategy-related documentation created;
- another report described 47 rules pending;
- other notes describe six education documents and 318 knowledge chunks.

These figures measure different things and may come from different revisions. Do not treat them as equivalent. Reconcile them by listing source documents, document IDs, extracted rules by status, duplicate/unsupported rules, and persisted chunks from the current database. Preserve the exact database snapshot and commit used for the count.

### Earlier repository audit evidence and limits

A prior read-only review reported inspecting 144 tracked files at commit `f60ed647956068c6d6423846dd93399a09ec682a`. It confirmed the broad code/module structure but did not run the Windows test suite or independently verify the local dataset, UTC timestamps, local database, `.env`, live MT5 connection, or complete end-to-end execution.

That review also reported:
- the tracked `data/processed` directory contained only `.gitkeep`, so a processed dataset/spec artifact was not present in that repository snapshot;
- a Stage C report expected 94,926 rows while another artifact/report showed 94,927;
- the KnowledgeChunk persistence/insertion path was incomplete in that snapshot.

The current PR description says knowledge-ingestion chunks are now persisted and focused tests cover ingestion chunks. This is evidence of an intended fix and a focused test, not proof that all old dataset artifacts/count mismatches are resolved. Check the current implementation, migration/database, and test output at the current HEAD before closing these items.

### Limits of the five-chat extraction

The user identified five chats titled `تعریف و آموزش ماشین لرنینگ` as the source set. During this second pass, the available project conversation summaries and saved reports were searchable, but the five complete raw chat transcripts were not exposed as individual readable files in the current file search. Therefore:
- this memory file consolidates the available summaries, reports, project plan, and GitHub evidence;
- it must **not** be described as a verified, message-by-message extraction of all five complete transcripts;
- if the raw chat text becomes available as project files or exports, compare it against this memory file and add any missing decisions, commands, commits, failures, and test outputs.

### Updated priority list after this verification

1. Resolve the conflict between the historical 175-test claim and the current PR's explicitly narrower 58-test CI scope.
2. Reconcile source-document/rule/chunk counts and the exclusions listed above.
3. Verify whether the earlier Stage C row-count mismatch and missing processed artifacts were corrected.
4. Independently test MT5 timestamp/freshness and broker symbol economics on Windows in read-only mode.
5. Continue Phase O strategy extraction only with the original source document accessible; keep all extracted rules unapproved.
6. Keep PR #1 open and draft until the user explicitly chooses otherwise. Do not enable demo execution or send orders.
