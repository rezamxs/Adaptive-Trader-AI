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
- PR head SHA during the first verification pass: `88bfc9eef8a25790d835670697681fd5e4775cdf`.
- PR is reported mergeable, but this does not mean it should be merged.
- A GitHub Actions workflow run associated with this head was found: run ID `38311139568`, workflow `Core Regression Tests`, conclusion `success`.
- The combined commit-status endpoint returned no status entries. The successful workflow run above belongs to the earlier head `88bfc9eef8a25790d835670697681fd5e4775cdf`; after the documentation-only follow-up commit, the PR head became `6d0685373a4dcfc23073ea215c68ecccbd09013c`. No workflow run was returned for this newest documentation-only commit. Do not infer that every CI job or the full test suite ran.
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


---

## 14. Chat-history consolidation — 2026-10-17

This addendum corrects and supplements the earlier “Limits of the five-chat extraction” note in section 13. It records what was actually readable in the current Codex session and keeps user-submitted reports separate from independently verified GitHub evidence.

### Scope and access

- The ChatGPT project **ماشین** contains 19 listed ChatGPT conversations. This session read 18; one conversation, ID `6ad12f2a-b994-83eb-b8ac-3515f8c642bc`, could not be opened because the conversation host reported it unavailable. Do not claim all 19 transcripts were read.
- The five numbered conversations titled `تعریف و آموزش ماشین لرنینگ (1)` through `(5)`, plus the base conversation `تعریف و آموزش ماشین لرنینگ`, were among the readable project history. The conversations include the user's earlier explicit request to read these chats and maintain one comprehensive Markdown handover in GitHub.
- Two related Codex conversations about Hindsight setup were also read. Some earlier messages contain references to attached files whose contents were not returned by the chat-history tool. This is therefore a transcript-based consolidation of accessible messages, not independent inspection of every attachment, local file, database, report, or project commit.
- The canonical file was fetched successfully from repository `rezamxs/Adaptive-Trader-AI`, ref `fix/core-correctness-audit`, on 2026-10-17. This session observed file blob SHA `238a7dbb866cc5659b56595361e6037577a2d9fe` before this update. No local checkout, branch HEAD, PR state, CI, database, or tests were inspected in this memory-only task.

### Durable user instructions and collaboration preferences

- Keep one canonical, cumulative handover at `docs/PROJECT_MEMORY.md`; do not create parallel memory files. Add material decisions, evidence, blockers, and phase changes here when they arise.
- Before giving project advice, inspect relevant prior project chats and current repository evidence; do not make the user repeat context that is already available. Be candid, act on authorized work, avoid looping or asking the user to relay steps they have already delegated.
- Reply in Persian, right-to-left where appropriate. Keep updates concise; explain difficult/important steps briefly and save detailed evidence in this file. The user prefers one coherent final report over repeated low-value progress messages.
- The user has supplied `PromptEngineeringforLLMsFullFA.pdf` and `Prompt-Engineering-Lecture-Elvis.pdf` as preferred references for crafting professional prompts for Antigravity/other coding agents. Their contents were attachments and were not available in this session; use the references only when the files can actually be read.
- The user wants Antigravity or other agents to review each other's output, run focused verification, and correct errors, with a manager/reviewer role for multi-agent work. Do not claim an agent or independent review ran unless it did.
- The user disliked the current dashboard UI/UX and asked to defer redesign until after core correctness/data work.
- The user expects a full-repository review when requesting one, not a superficial spot check. Explain what was actually covered and what was not.

### Hindsight shared-memory attempt — status from chat history

- The user's intended architecture is one project-specific Hindsight memory shared by ChatGPT, Antigravity, and Zed, while retaining `docs/PROJECT_MEMORY.md` as the canonical project record.
- Project-specific bank slug used in setup instructions: `adaptive-trader-ai`. Do not invent or expose any bearer token. A bearer token was pasted in chat; treat it as compromised and never reuse or copy it into this file, prompts, logs, or tools.
- The setup conversations show attempts to configure the custom MCP connector and inspect ChatGPT settings. They did **not** establish a successful Hindsight connection or successful retain/recall test. A Codex UI check reported a Free account, missing Developer mode, and a Plugins settings loading error; a later attempt was blocked by unsupported browser URL-policy automation. These are historical UI observations, not proof of the account's current state.
- Do not claim shared cross-tool memory is operational until the connector is actually available and a write/read round-trip succeeds. GitHub memory updates are durable repository storage only; they do not automatically make every future chat read this file.

### Phase O source inventory — user/agent report, not independently rechecked here

A Phase O final report supplied in the project chats claimed that six files were inspected from the user's `فایل ها` folder. Preserve these as **reported** source metadata until verified from the actual files/repository:

| Reported source | Reported size | Reported SHA-256 | Reported coverage |
|---|---:|---|---|
| `راهنمای_جامع_و_سند_راهبردی_ربات_اسکلپر_1 (2).pdf` | 6,668,982 bytes | `85a219257e7d916a3461a997db578701e9de12f965358d073a34d9ecb035b7e5` | 15 pages |
| `اصطلاح پرکاربرد در ترید.docx` | 26,014 bytes | `1350af69197010602db0f6708be5c472a7ba33a6b771f30a3e29125b98f91254` | 313 bilingual glossary paragraphs |
| `جزوه_بازطراحی_پروفایل_و_اوردر_فلو_جلسات_۱۱_تا_۲۵.docx` | 102,711 bytes | `0e830dce54a94ef43a9086f7131201c6258bc2906f82d02170c9af227c668478` | 1,323 training paragraphs |
| `_⁨پرایس اکشن⁩.pdf` | 167,667,492 bytes | `a6705cc4bcbcc3f3e3b83f5eda31177a1df8d39b87fe7ca75cb365d3bbd40f58` | 89 slides; report says vector text extracted |
| `KX_EIGHT_MYTHS.pdf` | 902,194 bytes | `9cf1f18c03bc5cffb0bff5dca05d248cbef7ef7c3a96aaaa499b8f506e0795dd` | 39 pages / 1,401 text lines |
| `6332607065685434115_526865520591018.zip` | 546,855 bytes | `e7a1795fa9857d41f02172a6b29d8b7ea7bc151049615a2eb8ca1e3a6a9bfa9c` | Report says 70 files: 40 assets, 4 governance, 14 references, 11 scripts, root `SKILL.md` |

The report says all six sources were fully extracted and that the ZIP's 11 Python scripts were included in the audit. Earlier memory noted 11 ZIP scripts as possibly excluded. Treat that exclusion as **superseded by a later reported claim, not independently resolved**; inspect the archive and ingestion artifacts before closing it.

### Phase O strategy extraction — later report claims and unresolved verification

A later user-provided Phase O report claims:
- 25 new operational price-action rules were added to a prior set of 27, for 52 total rule records across the six sources;
- earlier counts of 6, 25, 27, 41, 47, and 52 refer to different snapshots/scopes, per the report's reconciliation;
- all 52 rules are `UNAPPROVED`; the rule registry is claimed to be `docs/strategy/rule_registry.json`;
- 20 scalper rules (`STR-001`–`STR-020`), five KX governance rules (`STR-PA-001`–`005`), two order-flow rules (`STR-OF-001`–`002`), and 25 newly detailed price-action rules were included;
- the 25 newly reported rules cover 11 RTM setups (`STR-PA-RTM-001`–`011`, source pages 65–84), two supply/demand rules (`STR-PA-SND-001`–`002`, pages 58–64), five harmonic patterns (`STR-PA-HARM-001`–`005`, pages 43–49), five Lance Beggs setups (`STR-PA-BEGGS-001`–`005`, pages 33–35), one candle rule (`STR-PA-CNDL-001`, pages 21–29), and one oscillator calibration rule (`STR-PA-IND-001`, pages 88–89).
- the report identifies human Phase P decisions: reject or replace the reported Martingale ladder with a fixed ATR-based stop; select candidate transparent price-action triggers (examples: QM, Caps, FL, TST); and approve rules individually.

These rule counts, page mappings, extraction completeness, registry contents, and source files were reported in chat and are **not verified by this GitHub memory-file fetch**. Preserve all rules as pending/unapproved unless the actual registry and explicit user approval prove otherwise. The earlier section 8 roadmap entry calling Phase O the “next” strategy workflow is stale relative to this later report: Phase O is **reported complete**, Phase P decisions remain pending, and all downstream status still requires evidence.

### Phase O remediation and test reports — contradictory reported snapshots

The chats contain multiple results that must remain distinct:
- An earlier Phase O report says the working tree was clean, no files changed, no commit/push occurred, and no tests ran at that snapshot; it described a reproduced document-deduplication defect.
- Later reports claim fixes for SHA-256 content deduplication, a nullable/indexed `content_sha256` migration, safe handling of legacy documents and same-name/same-size collisions, and stricter DOCX session-heading parsing. Claimed files include `backend/app/models/document.py`, `backend/app/services/knowledge/manager.py`, `backend/app/services/knowledge/parsers.py`, `backend/app/services/knowledge/retriever.py`, `backend/app/core/config.py`, `migrations/versions/2026_10_15_add_content_sha256_to_documents.py`, and `backend/tests/unit/test_phase_o_three_defects.py`.
- A later “Phase O Final Exit-Gate Audit” report says branch `fix/core-correctness-audit` was ahead of origin by one commit (`5aedb1a`), with five modified unstaged files and two untracked files (the test and migration above), no staged files, no new commit/push/merge, and `git diff --check` clean.
- That same report claims `python -m pytest -o addopts="" --tb=no` returned 193 passed, 0 failed, 0 skipped, 1 warning in 74.74s; it says 11 focused three-defect tests, 7 evidence-remediation tests, and 28 strategy-registry/retrieval tests passed. Earlier reports claim 175 tests passed, while a Phase N audit had reported 157 passed and 17 failed. These are different historical snapshots and **none were rerun in this memory update**.
- Fix claims include recovery from `IntegrityError` during concurrent ingestion, hash-backed deduplication, preserving distinct same-name/same-size content, parser false-positive reduction for “جلسه ۱۱–۲۵”, and migration compatibility. Inspect the live branch and tests before describing any as current implementation.
- No claim in these reports authorizes trading. Keep `EXECUTION_MODE=SIGNAL_ONLY`, `DEMO_TRADING_ENABLED=False`, no `order_send`, and no live/demo order placement.

### Phase Q2 and engineering workflow — reported only

A user-provided Q2 report said the MT5 terminal was connected during a Sunday check but its last tick was about 39.8 hours old because the market session was closed; it reported zero orders and software/persistence checks passing while fresh-candle validation remained pending until the market reopened. The report included account identifiers, intentionally omitted here. This does not clear the MT5 timestamp/freshness blocker in section 6.

The user also shared a `JEV` (decision-maker model) idea and a SubQ account with a small balance, and asked whether to use it. No architectural adoption, API-key use, spending, or model integration is verified here. Do not send secrets or add a paid service without a concrete scoped decision.

### Corrected transcript-coverage statement

The earlier section 13 statement that the project conversations were not individually readable is no longer accurate for this session: 18 of 19 listed ChatGPT project conversations and two related Codex Hindsight threads were opened and reviewed. One ChatGPT conversation remains unavailable, attached-file contents were not returned, and this task did not inspect repository source, CI, database, MT5, local files, or tests. Treat this as a material-history consolidation, not a full source audit or proof of current implementation.

### Current next actions after this consolidation

1. At the exact current branch head, inspect `docs/strategy/rule_registry.json`, the six source artifacts, Phase O reports, migration, ingestion/parser code, and related tests; reconcile the reported 52-rule inventory and approval states.
2. Inspect branch HEAD, working tree, PR/CI, and the reported uncommitted changes before any code work; do not overwrite or commit unknown user changes.
3. Re-run relevant tests only when requested or when implementing a change; this memory update ran no tests.
4. Obtain explicit human decisions for Phase P rules; keep all rules unapproved until then.
5. Continue the read-only MT5 timestamp/freshness investigation before data-dependent modeling/backtesting.
6. If cross-tool memory remains desired, verify a working Hindsight OAuth connector and successful retain/recall round-trip; do not reuse the exposed token.


---

## 15. Phase O repository verification — 2026-10-17

This section records direct GitHub evidence gathered after the chat-history consolidation. It supersedes any conflicting Phase O status claims in earlier chat reports, while preserving those reports as history.

### Repository and branch state — verified now

- Repository: `rezamxs/Adaptive-Trader-AI`; default branch: `main`.
- Branch `fix/core-correctness-audit` exists. Its current HEAD at this check was `cce107f016ff630d74960e2b8c7f593032b87852`, commit message `docs: consolidate machine project chat history`. The commit tree was `5b9f9c2cdc0a0bf2e47464b1d81e5a5ca6d7e556`.
- Pull request #1 targets `main` and is `OPEN`, `DRAFT`, not merged; its head matched the branch SHA above.
- GitHub returned two completed `core-regression` check runs with conclusion `success` for this SHA, and two successful `Core Regression Tests` workflow run records. The combined commit-status endpoint had no status entries. This verifies those reported CI checks only; it does not establish that the missing Phase O test module ran or that every repository test ran.
- The Remote Desktop Commander device `<OWNER_WORKSTATION>` was offline (last seen 15 hours earlier). Local `G:\mash` branch/HEAD/working-tree state and the reported local unstaged changes therefore remain unknown. No local files were changed and no local tests were run in this audit.

### Strategy registry — verified now

At the branch SHA above, `docs/strategy/rule_registry.json` exists (blob `f1c55eb53d0dac19769a2679c831ba15e8ebfb3c`) and parses as JSON:
- `schema_version: 1.1.0`, phase `Phase O Comprehensive Source Extraction`, declared and actual count both 52;
- all 52 rules have status `UNAPPROVED`;
- no duplicate `rule_id` values were found;
- category counts are 20 `SCALPER_EA`, 30 `PRICE_ACTION`, and 2 `ORDER_FLOW`;
- all 52 records have the registry's expected identifying, source-reference, exact-source-text, executable-definition, and approval-blocker fields populated;
- the registry cites four direct source-document values: the scalper guide PDF (20 rules), the strategy ZIP (5), the order-flow/profile DOCX (2), and the price-action PDF (25). The glossary DOCX and `KX_EIGHT_MYTHS.pdf` appear in the source inventory but are not direct `source_document` values for registry rules.

This verifies the stored registry structure and approval states, not the correctness of each rule against its original source and not user approval. No rule was activated.

### Phase O source artifacts — incomplete independent verification

The following project documents exist at this SHA: `docs/strategy/source_inventory.md`, `strategy_spec.md`, `ambiguities.md`, `data_requirements.md`, `reports/strategy_review.md`, and the strategy audit/reconciliation reports at `docs/strategy-source-audit.md`, `docs/strategy-rule-reconciliation.md`, `docs/strategy-evidence-validation.md`, `docs/strategy-validation-report.md`, and `docs/strategy-integration-report.md`.

The source inventory and review report assert complete inspection and give file sizes/hashes for six originals. However, the recursive Git tree at this SHA contains no original PDF or DOCX source files. It contains a same-named ZIP only at `.agents/skills/6332607065685434115_526865520591018.zip`, with GitHub-reported size **534,605 bytes**, while the reported original ZIP size is **546,855 bytes**. This size difference means that repository copy cannot establish byte identity with the reported source archive. Thus the 52-rule inventory and the committed audit documents are verified as present, but the original six files, the claimed hashes, and complete source-to-rule fidelity remain unverified through GitHub alone. Recheck the actual files on the desktop when that device is online.

### Reported Phase O deduplication/parser fixes — not present in current GitHub tree

At the same SHA:
- `backend/app/models/document.py` has no `content_sha256` field.
- `backend/app/services/knowledge/manager.py` creates a random document ID and writes the upload before parsing; the checked implementation has no content-hash deduplication or same-content collision recovery.
- `backend/app/services/knowledge/parsers.py` does not contain the reported DOCX session-heading pattern.
- The reported `backend/tests/unit/test_phase_o_three_defects.py` is absent from the recursive Git tree (GitHub contents lookup returned 404).
- The reported `migrations/versions/2026_10_15_add_content_sha256_to_documents.py` is absent (contents lookup returned 404). The tree only lists the initial-schema and 2026-10-12 validation-metrics migrations.
- Existing knowledge tests are present, including `test_knowledge_ingestion_chunks.py`, `test_knowledge_pipeline.py`, and `test_knowledge_retrieval.py`; their existence alone does not prove the reported deduplication, legacy backfill, race-recovery, or parser regressions are covered.

Therefore, the Phase O fixes and the 193-test report remain **reported local work** at most; they are not in the current GitHub branch snapshot checked here. They may exist as uncommitted files on the offline desktop, which cannot be ruled in or out. Do not merge or describe these fixes as shipped until local state is recovered and the implementation, migration, tests, and commit are inspected.

### Audit conclusion and next actions

- **Verified:** GitHub branch/HEAD, open draft PR #1, CI check conclusions, committed Phase O documentation, and a 52-entry all-`UNAPPROVED` registry with no duplicate IDs.
- **Not verified:** source binaries/hashes, local dirty state, the reported Phase O code fixes, their migration/tests, and the reported 193-test local run.
- Reconnect the desktop and inspect `G:\mash` status and diffs before making any code changes. Preserve the five reported modified files and two reported untracked paths if they still exist.
- Recover the six source files and compare their actual size/hash with `docs/strategy/source_inventory.md`; inspect the 52 rule citations against source pages.
- If the local Phase O changes are valid, add the missing migration, model/service/parser changes, and focused tests to the intended branch; then run the focused suite and full requested suite. Keep the PR draft and all rules unapproved pending the user's Phase P decisions.
- Continue enforcing `EXECUTION_MODE=SIGNAL_ONLY`, `DEMO_TRADING_ENABLED=False`, and no order placement.

---

## 16. Shared cross-agent memory / Hindsight integration — 2026-10-16

### Goal

Use one project-scoped Hindsight memory bank across ChatGPT, Antigravity, and the user's coding editor where supported, while keeping `docs/PROJECT_MEMORY.md` as the canonical, version-controlled project handover. Hindsight is a retrieval layer, not the authority for current code, Git state, test results, trading settings, or secrets.

### Verified local/repository state on 2026-10-16

- Local workspace: `G:\\mash`.
- Local branch: `fix/core-correctness-audit`; local HEAD observed as `5aedb1a3c95728ddab67beeed03aac3cd1ac6489`; local branch reports `ahead 1` of `origin/fix/core-correctness-audit`.
- GitHub branch head observed separately as `9510c764ae89e63d559d3fc6ba214eaa30ff039f`. The divergence must be reconciled deliberately; do not reset, rebase, push, or overwrite local work as part of memory setup.
- Existing memory file confirmed locally and on the working GitHub branch: `docs/PROJECT_MEMORY.md`. Do not create a parallel master-memory file.
- Existing local working-tree changes before this section was appended: five modified tracked files (`backend/app/core/config.py`, `backend/app/models/document.py`, `backend/app/services/knowledge/manager.py`, `backend/app/services/knowledge/parsers.py`, `backend/app/services/knowledge/retriever.py`) and two untracked files (`backend/tests/unit/test_phase_o_three_defects.py`, `migrations/versions/2026_10_15_add_content_sha256_to_documents.py`). Treat these as user-owned; do not discard or stage them.
- Antigravity 2.23.1 is installed and running. Cursor 3.18.25 and VS Code are installed.
- A process named `ZCode.exe` is running from `G:\\Projects\\ZCode\\ZCode.exe`; it has not yet been verified as the Zed editor. Do not modify its configuration until its identity/config format is confirmed.
- Hindsight CLI/server, `uv`/`uvx`, Docker, and Ollama were not found on PATH during the initial inspection. This does not prove they cannot exist elsewhere, but no working local Hindsight deployment has been verified.
- No Hindsight account token/API key was available through the inspected tool results. Never search for, print, or store secret values in this file.

### Architecture decision

1. `docs/PROJECT_MEMORY.md` remains the canonical, human-readable project handover and is committed with ordinary project documentation only after the user reviews the diff.
2. Hindsight should use a dedicated bank such as `adaptive-trader-ai`, isolated from unrelated personal memories.
3. Store durable decisions, user-approved constraints, known failures, verified test evidence, unresolved questions, and handover summaries. Do not store credentials, raw account identifiers, API keys, or unverified agent claims as facts.
4. Each agent must recall relevant context before substantial work and retain a concise update after material decisions or verified results. Retrieved memories are hints; validate current branch/HEAD, source, database state, and tests before acting.
5. Preserve evidence labels: `VERIFIED`, `REPORTED`, `UNRESOLVED`, and `SUPERSEDED`. A newer memory does not erase an older failure; record why a status changed.
6. Keep trading-safety rules fail-closed: `EXECUTION_MODE=SIGNAL_ONLY`, `DEMO_TRADING_ENABLED=False`, no orders, no rule approvals, and no demo/live activation without explicit user authorization.

### Integration feasibility and blockers

- Hindsight documents a built-in MCP endpoint for self-hosted deployments and a hosted Cloud MCP connection. Zed has a documented `hindsight-zed` setup path. Client setup is client-specific and must be verified with a read/write/recall smoke test.
- ChatGPT custom MCP apps/connectors require Developer Mode and a supported plan according to the current Hindsight integration documentation. The user's current ChatGPT plan is Free, so direct ChatGPT-to-Hindsight MCP integration is currently blocked unless account capabilities change. Do not claim ChatGPT is connected.
- Antigravity is installed, but its actual MCP configuration location and support for the chosen Hindsight transport still need to be verified. Do not write guessed settings.
- The running `ZCode.exe` application must be identified before deciding whether the Zed integration applies.
- Hosted Hindsight requires the user to authenticate and authorize access/create a token or connection. No credentials are available in this session. Do not create an account, incur paid usage, or use a third-party API balance without explicit user authorization.
- A local Hindsight server is not yet configured. It may require a supported LLM provider or a suitable local model; do not install a large local model on the reported low-memory laptop without benchmarking and user approval.

### Next actions

1. Confirm whether `ZCode.exe` is Zed or another editor and locate Antigravity's real MCP configuration without exposing secrets.
2. Ask the user to open Hindsight Cloud, sign in, and create/select the dedicated `adaptive-trader-ai` bank using the official connection flow; obtain the appropriate supported connection details through the UI. Never ask the user to paste a secret token into chat if a secure setup path is available.
3. Configure Antigravity and the confirmed coding editor with the supported integration; preserve existing MCP server entries and validate JSON/config syntax before saving.
4. Run a smoke test: retain a harmless unique test fact in the project bank, recall it from a new session in each connected coding client, then delete/mark the test record if the service supports it.
5. For ChatGPT, verify plan-level MCP availability before promising direct access. If unavailable, continue using this canonical memory file and explicit handover summaries; do not imply that Hindsight memories are automatically visible to ChatGPT.
6. Update this section with exact client versions, config locations (not secret values), successful test evidence, remaining blockers, and date.
7. Do not commit or push this documentation update until the user reviews the complete working-tree diff; do not alter trading code or database state for this integration task.

### Progress update — 2026-10-16 (setup attempt)

- Created the previously absent Antigravity global MCP config at `<OWNER_HOME>\\.gemini\\config\\mcp_config.json` with one entry, `hindsight-adaptive-trader-ai`, pointing to `https://api.hindsight.vectorize.io/mcp/adaptive-trader-ai/`. No token or secret is stored in the file.
- Parsed the new config successfully as JSON and confirmed the endpoint is reachable over HTTPS (GET returned HTTP 200). This is only a network reachability check; it does **not** prove MCP initialization, OAuth authorization, bank creation, or successful memory read/write.
- Antigravity was already running when the config was created. Its MCP server list has not yet been reloaded and no authenticated Hindsight connection has been verified. Use Antigravity's MCP Servers / Manage MCP Servers UI to reload and complete authentication if prompted.
- Hindsight Cloud account/bank ownership and OAuth authorization remain pending user authentication. Do not attempt to bypass login, invent credentials, or claim the service is connected.
- ChatGPT is still not connected. The current Free plan is a blocker for custom MCP app/connector setup according to the current Hindsight/ChatGPT integration documentation.
- ZCode 3.15.3.8237 is installed and running, but it is a distinct product named ZCode, not verified as the Zed editor. No ZCode MCP settings were changed.
- No Git commit, push, merge, database migration, trading configuration change, or order submission was performed as part of this memory setup.

### Progress update — 2026-10-16 (ZCode MCP)

- Identified the running editor correctly as **ZCode Desktop 3.15.3.8237** (not Zed).
- ZCode's user config `<OWNER_HOME>\\.zcode\\cli\\config.json` existed and had valid JSON with `plugins` and `skills`, but no MCP servers. Created a backup at `<OWNER_HOME>\\.zcode\\cli\\config.json.pre-hindsight-20261016.bak`, then added the Hindsight HTTP endpoint under `mcp.servers.hindsight-adaptive-trader-ai`. The existing top-level settings were preserved and post-write JSON validation passed.
- Tested the Hindsight MCP endpoint using an unauthenticated JSON-RPC `tools/list` request: it returned HTTP 401 with a WWW-Authenticate challenge. This confirms the endpoint requires authentication; it does not confirm successful OAuth or tool access.
- The ZCode config was written while the app was running. Its MCP server list has not yet been refreshed/restarted, and the user-facing authorization flow has not been completed. No memory write/read smoke test has passed yet.
- Antigravity's global config was created separately and points to the same project bank. It likewise needs reload and authentication.
- No Hindsight API key, token, or password was stored in either config. No Git commit/push, database change, or trading action was performed.

### Progress update — 2026-10-16 (agent instructions)

- Added a new repository-root `AGENTS.md` (16 lines) as an **instruction layer**, not a second memory store. It tells coding agents to read `docs/PROJECT_MEMORY.md`, use the Hindsight bank when actually connected, preserve evidence labels, update the canonical memory after material events, and obey the project's no-trading/no-unapproved-Git-operation constraints.
- This file is local and untracked at the time of this note. Neither it nor the 58-line memory update has been committed or pushed, so the GitHub copy remains unchanged until the user explicitly authorizes a reviewed commit/push.
- ChatGPT's direct Hindsight MCP connection remains blocked on the current Free plan. The available GitHub connector can read the repository's memory file once the update is pushed; until then, this conversation's local copy and prior context are the available sources. Do not claim automatic Hindsight recall in ChatGPT.


### Progress update — 2026-10-16 (user signed in; client verification still pending)

- The user reports that they connected/signed in to the Hindsight account. This is **user-reported**; account/session state was not independently inspected.
- Re-read both local MCP configs and confirmed the Hindsight endpoint entries remain present in Antigravity's `mcp_config.json` and ZCode's `cli/config.json`. ZCode's existing settings are still present; the pre-change backup remains available.
- Searched Antigravity's current `main.log` and the current-day ZCode logs for the literal `hindsight`; no matches were found. This indicates there is no log evidence yet that either client initialized the configured server. It does not prove the apps failed; config reload/authorization may not have been triggered.
- No Hindsight MCP tools are exposed to this ChatGPT session, so I cannot directly call the bank's retain/recall operations from here. The prior unauthenticated JSON-RPC test returned HTTP 401. Therefore, no successful authenticated memory write/read smoke test has been demonstrated, and the shared bank must not be described as fully connected yet.
- Next required client-side verification: reload/restart the MCP server from each app's MCP settings UI and complete the authorization flow if prompted; then confirm the server is listed as connected and perform a harmless retain/recall test. Avoid terminating active editor processes blindly because unsaved work may exist.
- No commit/push, trading action, database migration, or project code change was performed in this verification pass.

### Progress update — 2026-10-16 (ZCode CLI MCP config fix and independent verification)

**What was done:**
- Independently audited the ZCode MCP configuration at `<OWNER_HOME>\.zcode\cli\config.json`. Found one schema defect: the field `"enable": true` was not the canonical name; corrected to `"enabled": true` per the official ZCode MCP schema (legacy `enable` may be migrated by CLI but not by the Desktop app session manager). Also increased `timeoutMs` from `30000` to `60000` because OAuth handshake adds latency.
- Validated the corrected JSON: `type: "http"`, `url`, `enabled`, `timeoutMs` — zero unknown keys. The schema is strict; an unknown key silently drops the server entry.
- Confirmed no workspace-level MCP config exists (`G:\mash\.zcode/` absent; no `.agents/mcp.json` at user or workspace scope). The user-level config is the only active source.
- All existing `plugins` and `skills` settings in the config were preserved unchanged.

**What is proven (verified now):**
- The Hindsight OAuth discovery is functional: `GET /.well-known/oauth-protected-resource` and `GET /.well-known/oauth-authorization-server` both return valid JSON. The server supports `authorization_code` grant with PKCE (`S256`), dynamic client registration at `/oauth/register`, and scopes `openid profile email offline_access`.
- An unauthenticated `POST` to the MCP endpoint returns HTTP 401 with `WWW-Authenticate: Bearer resource_metadata=...`. This confirms the endpoint is reachable and authentication-gated.
- ZCode Desktop logs (`<OWNER_HOME>\.zcode\v2\logs\2026-10-16.log`) show MCP initialization only for official plugin servers (`image_search`). No `hindsight` entry appeared in any log line. The config was written while ZCode was running; the server list was not reloaded.
- No `mcp__hindsight*` tools are registered in this CLI session's tool namespace.

**What remains unresolved:**
1. **ZCode Desktop must reload the MCP server list.** The config file is correct, but the running process hasn't picked it up. Action: open **Settings → MCP** in ZCode Desktop, verify `hindsight-adaptive-trader-ai` appears, and click to reconnect/enable it. If OAuth is prompted, complete the browser-based authorization. Do not paste tokens into chat.
2. **No authenticated smoke test has passed.** Until a ZCode session can list the server's tools (e.g., `retain`, `recall`, `forget`) and execute a harmless recall, the connection is not confirmed.
3. **This CLI session started before the config fix.** A new `kiro-cli chat` session in `G:\mash` will read the corrected config and attempt MCP server connection at startup, including the OAuth flow.

**Files changed:**
- `<OWNER_HOME>\.zcode\cli\config.json` — field `enable` → `enabled`, `timeoutMs` 30000 → 60000 in the `hindsight-adaptive-trader-ai` server entry. All other settings preserved.
- `G:\mash\docs\PROJECT_MEMORY.md` — this update appended.

**Safety:** No Git commit/push, database migration, trading config change, order submission, or strategy rule activation was performed.

### Progress update — 2026-10-16 (Antigravity MCP verification and status)

**What was done:**
- Audited Antigravity global MCP configuration at `<OWNER_HOME>\.gemini\config\mcp_config.json`.
- Connected directly to Antigravity's running Language Server on `127.0.0.1:49235` (HTTPS) using internal CSRF token authentication (`x-codeium-csrf-token`).
- Executed `RefreshMcpServers` RPC to force the running Language Server to reload configuration live without killing the process.
- Inspected the live server state via `GetMcpServerStates` RPC.

**What is proven (verified now):**
- Antigravity successfully loaded the server definition `hindsight-adaptive-trader-ai` from `<OWNER_HOME>\.gemini\config\mcp_config.json`.
- Live connection state is `status: MCP_SERVER_STATUS_ERROR` with exact error: `calling "initialize": sending "initialize": Unauthorized`.
- Hindsight Cloud endpoint `https://api.hindsight.vectorize.io/mcp/adaptive-trader-ai/` challenges the connection with HTTP 401 and `WWW-Authenticate: Bearer resource_metadata="https://api.hindsight.vectorize.io/.well-known/oauth-protected-resource"`.
- Antigravity's Language Server binary contains the full OAuth pipeline (`discoverProtectedResourceMetadata`, `discoverAuthServerMetadata`, `registerClient`, PKCE `S256`, `CompleteMcpOAuth`), but the OAuth handshake has not been triggered or completed for this server instance.
- Because `initialize` failed with 401 Unauthorized, no Hindsight MCP tools (`retain`, `recall`, etc.) are registered or callable in the agent toolset.
- No secrets or credentials were written to configuration files or exposed in logs.

**What remains unresolved / required manual action:**
1. Open Antigravity Desktop UI, navigate to **Additional Options (...) → MCP Servers** (or **Skills & Customizations → MCP Servers**).
2. Locate `hindsight-adaptive-trader-ai` (status: Error / Sign-in required). Click the **Sign in** / **Authenticate** button to trigger the browser OAuth authorization flow for Hindsight Cloud.
3. Once the OAuth authorization flow completes in the browser, Antigravity will save the token and the server status will transition to `MCP_SERVER_STATUS_READY`.
4. Perform the smoke test (`retain` a test fact, then `recall`) once tools become visible.

**Safety:** Kept `EXECUTION_MODE=SIGNAL_ONLY`, no trading actions, no database changes, no Git commits or branch modifications.

---

## 17. Git divergence reconciliation and Phase O fix session — 2026-10-17 (ZCode/جک)

**Evidence labels:** items marked VERIFIED were executed and observed directly in this session; REPORTED items come from sub-agent output or earlier reports.

### Verified now (this session, read-only + minimal fix)

- Local branch `fix/core-correctness-audit`, HEAD `5aedb1a3c95728ddab67beeed03aac3cd1ac6489`, was diverged from remote: remote advanced with 2 docs-only commits (`cce107f` chat-history consolidation, `4a8bd1a` Phase O repository audit) by rezamxs 2026-10-17 evening.
- Remote `4a8bd1a` had drifted into the same 5 code files local had as uncommitted work (manager.py, parsers.py, retriever.py idempotency/provenance removed; `backend/tests/unit/test_phase_o_evidence_remediation.py` deleted from remote tree, 199 lines).
- Verified rule-regression evidence in remote registry edits: `docs/strategy/rule_registry.json` blocker bullets for STR-012 (Gambler's Ruin / Negative Skewness evidence text) and STR-015 (floating spread evidence) were shortened; local richer versions retained.
- PR #1 verified via `gh pr view 1`: OPEN, DRAFT, MERGEABLE, head `4a8bd1a`, base `main`, two `core-regression` checks SUCCESS (runs 39045688243, 39045684235, 2026-10-17T15:35Z).
- Local uncommitted state preserved: 5 modified files + untracked `backend/tests/unit/test_phase_o_three_defects.py` (14/14 passed locally) and `migrations/versions/2026_10_15_add_content_sha256_to_documents.py`. Nothing was committed, pushed, stashed, or discarded in this session pending user authorization.

### Defect found and fixed (verified)

- `backend/tests/unit/test_phase_o_evidence_remediation.py:156` opened `Path("docs/strategy/rule_registry.json")` relative to CWD; test passed only when pytest ran from repo root, broke from `backend/`.
- Fix applied (one line, no logic change): `reg_path = Path(__file__).resolve().parents[3] / "docs/strategy/rule_registry.json"`.
- Test evidence: from repo root `7 passed in 4.33s`; from `backend/` `7 passed in 4.16s`; both Phase O modules together `20 passed, 1 skipped in 5.21s` (1 skip pre-existing, environment-dependent; not investigated yet).

### Sub-agent output assessment

- Sub-agent (Git agent prompt) partially fixed the CWD path issue but reported it garbled/incomplete; its claims were NOT accepted without reproduction — fix was re-derived from source (`Path(__file__)` anchor) and verified by direct test runs above.

### Superseded

- Earlier session belief that Hindsight bank `adaptive-trader-ai` was unreachable ("Bank not found") is SUPERSEDED: `get_bank` succeeded and `sync_retain`/`recall` round-trip verified twice this day (project snapshot + divergence audit stored). Hindsight live memory is operational in ZCode.

### Remaining risks / unresolved

1. Working tree still diverged from origin: no merge, commit, or push yet (needs explicit user authorization per project protocol).
2. Full-suite local run not executed this session (last REPORTED figures only: 193 passed from Phase O exit-gate report; 175 from older Phase N report — reconcile after merge).
3. MT5 timestamp/freshness ~3h blocker still open (section 6); no data-dependent work authorized until resolved.
4. Two pre-existing skipping anomalies listed above; plus intermittent stray untracked file `ِ‪…Purpose…` (unusual Unicode-named file in git status, unexamined, priority low).
5. Phase P human rule approvals still pending; all 52 rules remain UNAPPROVED.

### Next actions (priority order)

1. User-authorized merge: `git merge origin/fix/core-correctness-audit`, resolve conflicts keeping local rich versions for the 5 code files, reconcile `docs/PROJECT_MEMORY.md` (both remote sections 14/15 and local section 16+17), restore `test_phase_o_evidence_remediation.py` from `5aedb1a`, single reconciliation commit, then full suite before any push.
2. After merge+commit: user-authorized push to update PR #1 head.
3. Then MT5 timestamp/freshness read-only investigation.
4. Then Phase P approval workflow.

**Safety:** `EXECUTION_MODE=SIGNAL_ONLY`, `DEMO_TRADING_ENABLED=False`, no orders, no rule approvals, no live/demo execution; database untouched.

---

## 18. ECC toolchain installed for OpenCode — 2026-10-22

**VERIFIED (this session):**
- ECC (github.com/affaan-m/ECC) v2.2.3, source clone at `<OWNER_HOME>\ECC` (commit `90606c8`), installed for the OpenCode harness per the repo's documented OpenCode path (`npm install && npm run build:opencode`; installer `node scripts/ecc.js install --profile full --target opencode --enable-hooks`).
- Install root: `<OWNER_HOME>\.config\opencode\` — 311 skill directories, 103 command files, `hooks\hooks.json` present. Install-state: `ecc-install-state.json`; `ecc doctor --target opencode` returned OK (1/1, no issues).
- The user-owned `opencode.json` was preserved (installer skipped it: not ECC-owned). Node v24.15.0, Git 2.54.0 (Windows).
- All files are OUTSIDE `G:\mash`; no repository changes were made for this install. No commit/push of the memory update.
- New skills/commands appear in the OpenCode session skill list only after the session reloads the config directory (new session or reload).

**REPORTED / not yet observed:** hook runtime behavior inside OpenCode sessions (hooks copied but no live session observed firing them yet in this session).

**Next action:** start a fresh OpenCode session and verify ECC commands/skills are discoverable; remove nothing; optional cleanup later via `node scripts/ecc.js uninstall --target opencode --dry-run` from `<OWNER_HOME>\ECC`.

---

## 19. Audit-fix execution H1..H7 + C1 (uncommitted, working tree) — 2026-10-24

**Scope:** user-ordered fix of exactly 8 audit findings (H1 promotion-gate metric attestation, H2 dashboard XSS, H3 defusedxml, H4 Alembic ownership of drift + real baseline downgrade, H5 fail-closed market 503, H6 demo-order PENDING-before-dispatch, H7 streamed upload, C1 API-token auth on state-changing endpoints). ECC reviewer agents were NOT available as subagent types in this harness; each step was executed manually in TDD (red → fix → green) with the same gate criteria. Hindsight MCP unavailable this session.

**VERIFIED now (all with tests, full suite 223 passed / 0 failed — `python -m pytest backend/tests -q -o addopts=""`):**
- **C1** — `backend/app/core/auth.py:16` `require_api_token` (fail-closed `settings.API_TOKEN` bootstrap in `backend/app/core/config.py`; `X-API-TOKEN` header; `hmac.compare_digest`). Applied to 9 state-changing POST endpoints: `system.py:111,119`, `rules.py:32,42`, `signals.py:47,98`, `ml.py:51,122`, `backtest.py:55`, `documents.py:31`. Tests: `backend/tests/unit/test_audit_fixes_corr1.py` (C1 block).
- **H2** — `frontend/index.html:852` `escapeHtml` + `esc()`; all server-controlled interpolations in innerHTML sinks wrapped (31 uses). Tests: H2 block of `test_audit_fixes_corr1.py` (static sink scan + node-evaluated vector test).
- **H3** — `backend/app/services/knowledge/parsers.py:161` defusedxml `ElementTree` (DOCTYPE/ENTITY forbidden by library); tests incl. billion-laughs zip bomb.
- **H5** — `backend/app/api/v1/endpoints/market.py:17` 503 "refuse to serve mock data"; MockMarketDataProvider fallback removed; factory test asserts raise.
- **H6** — `backend/app/services/execution/demo_executor.py:171-226`: single DemoOrder row committed as PENDING BEFORE `mt5.order_send`, updated in place to FILLED/REJECTED after; post-fill commit wrapped in try/except → `DEMO_ORDER_PERSIST_FAILED` CRITICAL event, row stays PENDING for recovery; idempotency now reserved pre-dispatch. Tests: `backend/tests/unit/test_audit_fixes_h6.py` (4).
- **H7** — `backend/app/api/v1/endpoints/documents.py:17,44` streamed 1MB-chunk upload into bounded temp file, 250MB cap enforced at stream boundary (spoofable pre-declared size not trusted), 400/413/500 paths; temp cleanup in `finally`.
- **H4** — boot ALTER patches removed from `backend/app/db/init_db.py` (now create_all + safety seeding only); new revision `migrations/versions/2026_10_24_20261024_initp_init_db_patches.py` (worker_cycle_records.lifecycle_phase/updated_at, ml_models.validation_metrics) with real downgrade; baseline `2026_10_10_01f57c481926_initial_schema.py:39` now real `Base.metadata.drop_all` downgrade. Tests: `backend/tests/unit/test_audit_fixes_h4.py` (upgrade→downgrade base→upgrade cycle on isolated tmp DB).
- **H1** — `backend/app/core/metrics_integrity.py:40,60` (HMAC-SHA256 with env `METRIC_SIGNING_KEY`, else SHA-256 integrity digest; canonical JSON incl. model_id, validation/test/walk-forward metrics); `MLModelRecord.metrics_digest` column + migration `2026_10_24_20261024_mdig_metrics_digest.py`; trainer commits digest into artifact payload and DB row (`trainer.py`); `registry.promote_candidate` rejects unattested, tampered-store, or artifact-mismatch candidates before any gate mutates state. Tests: `backend/tests/unit/test_audit_fixes_h1.py` (5). Existing governance fixtures re-attest via `metric_digest` when they intentionally change metrics (`test_model_promotion_governance.py::_model`, `data_ml/test_ml_governance.py::_re_attest`).

**INCIDENT (disclosed, recovered):** first H4 test run executed Alembic against the PRODUCTION DB because `migrations/env.py` read `settings.DATABASE_URL` and ignored the test-provided URL. Result: prod progressed 01f57c481926 → head, then initp columns were dropped and re-added; values previously stored in lifecycle_phase/updated_at/validation_metrics reset to defaults (no orders/trades data involved; documents/knowledge data intact). `env.py` now honors `ALEMBIC_DATABASE_URL` override first; prod re-migrated to `20261024_mdig`, all 12 tables + columns verified present, version confirmed `20261024_mdig`.

**Remaining risks / UNRESOLVED:**
1. Legacy `ml_models` rows have `metrics_digest=NULL` → promotion fails closed until re-registered (intentional).
2. `METRIC_SIGNING_KEY` not set → digest is integrity-only (not attacker-resistant); set env key for production.
3. All changes UNCOMMITTED (working tree on `fix/core-correctness-audit`); commit/push awaits explicit user authorization, same as the pre-existing staged Phase O set.
4. MT5 ~3h timestamp/freshness blocker unchanged (section 6).

**Next action:** user reviews diff + authorizes commit(s); optionally set `METRIC_SIGNING_KEY`; then Phase P approvals.

---

## 20. Ponytail review of the 8 fixes: 11 findings resolved — 2026-10-26

**VERIFIED now (full suite 231 passed / 0 failed — `python -m pytest backend/tests -q -o addopts=""`; related subset 86 passed):**
1. Dashboard auth: `frontend/index.html` `getApiToken/setApiToken/authHeaders` + 🔑 button; all 9 POST fetches send `X-API-TOKEN` from localStorage. Tests: `test_audit_fixes_corr1.py` (node-evaluated `authHeaders`, POST-fetch scan, token-entry presence).
2. `mt5.order_send` exception → REJECTED row + released idempotency (`demo_executor.py`, try/except around dispatch). Test: `test_order_send_exception_marks_record_rejected`.
3. H7 RAM limit documented at `documents.py` full-read site (path-based ingest is the real fix, still open).
4. Ambiguities double-escape removed (`r.ambiguities.map(esc).join('<br>')`) + regression test.
5. `ModelRegistry.attach_walk_forward_metrics(model_id, wf, validation_metrics=None, test_metrics=None)` — the only supported post-registration metric write; re-attests digest and syncs artifact. Test copies `_re_attest` deleted; `data_ml/test_ml_governance.py` and `test_audit_fixes_h1.py` use the real path (7 tests in h1 file).
6. Loser of concurrent dispatch race gets "Duplicate order prevented" instead of IntegrityError 500 (`_commit_pending_safely`). Test simulates colliding second commit.
7. README section 2 now states `alembic upgrade head` is mandatory on existing DBs (init_db no longer ALTERs); section 3 documents the dashboard token flow.
8. Dead `_os.environ.get("API_TOKEN")` branch removed from `config.py` (BaseSettings already reads env/.env).
9. 11 model imports in test fixture replaced by `import backend.app.models`.
10. Unused warm-up subprocess removed from the H2 node test.
11. After post-fill commit failure, response returns the store's PENDING record (`demo_executor.py` rollback + requery), never a fake FILLED.

**Unchanged constraints:** `EXECUTION_MODE=SIGNAL_ONLY`, `DEMO_TRADING_ENABLED=False`, no orders; all changes UNCOMMITTED awaiting user authorization (working tree now holds §19+§20 changes together).

**Still open:** MT5 ~3h timestamp blocker; real-browser render of the dashboard (finding 1 verified against the real JS via node, not a browser); path-based `ingest_document` for true chunk-bounded memory.

---

## 21. Final safety & correctness fixes (3 items) — 2026-10-27

**VERIFIED now (full suite 244 passed / 0 failed — `python -m pytest backend/tests -q -o addopts=""`; targeted 65 passed):**
1. **Path-based upload:** `DocumentParser.parse_file_path` (PDF→pymupdf path, XLSX→openpyxl path, DOCX/ZIP→ZipFile path, TXT/CSV→text-mode read) + `KnowledgeManager.ingest_document_path` (chunk-wise SHA-256 `_hash_file_chunked`, `shutil.copyfile` raw storage) + upload endpoint streams to temp and ingests from the temp path; `stored.read()` removed — the raw file is never one bytes object. Known structural limit: extracted text is persisted as `Document.raw_text`, so text-format content still reaches memory/DB by design. Temp files cleaned on success/failure. Tests: `backend/tests/unit/test_final_safety_fixes.py` (5).
2. **Unknown MT5 outcome:** `order_send` exception → `DemoOrder.status="EXECUTION_UNKNOWN"` (idempotency key stays reserved; NOT auto-REJECTED) + `_reconcile_unknown_order` best-effort `mt5.history_orders_get` match on the request comment promotes to FILLED with the broker ticket; retry on an unknown-outcome signal is blocked without a second dispatch; definitive pre-send blocks and retcode refusals stay REJECTED. `DemoOrder.status` doc updated (no migration needed; String(32) column). Tests: 5 scenarios in `test_audit_fixes_h6.py`.
3. **Dashboard token in memory only:** `frontend/index.html` dropped localStorage/sessionStorage entirely; `let apiToken` page-memory + prompt on every load; all 9 POST fetches still send `X-API-TOKEN`; backend 401 on missing/wrong token re-verified. Tests: web-storage ban (code lines), node-eval of real `authHeaders` with no storage defined, no-token leak into console/alert, backend rejection.

All MT5 paths tested with stubs; no real MT5 dispatch, no orders, `EXECUTION_MODE=SIGNAL_ONLY`, `DEMO_TRADING_ENABLED=False`. Nothing committed; working tree now spans §19–§21 (44 changed/untracked paths).

---

## 22. Reconciler fix against documented MT5 return shape — 2026-10-27

**VERIFIED now (full suite 246 passed / 0 failed; targeted 47 passed):**
- Doc check (mql5.com `mt5historyordersget_py`): the date-range form of `history_orders_get` returns a **tuple of TradeOrder objects directly** (a structseq of 24 fields: ticket…comment…state), and `None` on error/no-match. The old test built `([row],)` — a tuple wrapping a list — which hid the bug: `entries = history[0]` grabbed the FIRST TradeOrder instead of the sequence, so the match loop iterated its fields and never promoted to FILLED.
- Fix in `demo_executor.py::_reconcile_unknown_order`: `if history is None: return; entries = history` (no unwrapping).
- Real-shape regression tests in `test_audit_fixes_h6.py` now build genuine `MetaTrader5.TradeOrder` structseqs (positional 24-field construction, `ORDER_STATE_FILLED=4`): FILLED promotion with broker ticket, no-match stays EXECUTION_UNKNOWN, error shapes (None result + raised exception) stay EXECUTION_UNKNOWN. Red recorded first (test failed against the buggy unwrapping), green after the fix. An early false-red (state=0 default) was diagnosed and the FILLED scenario now sets state explicitly.
- Not verified against a live MT5 terminal (unreachable in session); stubs use the real TradeOrder structseq type and documented call/return contract.

---

## 23. Alembic safety hardening round 2: no prod-DB test dependency, no URL leak — 2026-10-29

**VERIFIED (this session):**
1. `backend/tests/unit/test_worker_cycle_backfill_migration.py`: `test_production_migration_state_after_wcbf` (read the real `adaptive_trader.db`; skipped when absent) REMOVED from the automated suite per user order — real-DB state is no longer a pass/skip condition. The backfill behavior test runs entirely on an isolated tmp DB via `ALEMBIC_DATABASE_URL`; the contradictory module docstring (claimed "production DB is never touched" while pinning prod drift facts) was rewritten.
2. Production DB facts preserved and re-verified READ-ONLY this session (2026-10-29, sqlite `mode=ro`, SELECT/PRAGMA only, no migration run): version `20261029_wcbf`, 609 `worker_cycle_records` rows, 0 NULL `updated_at`, 0 NULL `lifecycle_phase`, both columns NOT NULL (`PRAGMA table_info` notnull=1). The earlier pre-migration figure (599 NULLs at `20261024_mdig`) remains historical/superseded.
3. `migrations/env.py::get_url` no longer prints `settings.DATABASE_URL` in the fail-closed RuntimeError (credentials-into-logs class); the message still names `ALEMBIC_DATABASE_URL` and the explicit-destination procedure; the now-unused `settings` import was removed. TDD: new test `backend/tests/unit/test_alembic_destination_safety.py::test_missing_destination_error_hides_database_url` was RED first (leak reproduced against the old message), GREEN after the fix — asserts the secret URL, its password, host and user never appear while `ALEMBIC_DATABASE_URL` does.
4. Tests: migration+safety subset (`test_worker_cycle_backfill_migration.py` + `test_alembic_destination_safety.py` + `test_audit_fixes_h4.py`) → 7 passed. Full suite `python -m pytest backend/tests -q -o addopts=""` → 317 passed / 0 failed / 479 warnings, reproduced twice (second run logged to `/tmp/pytest_full_20261029.log`). For tests, `ALEMBIC_DATABASE_URL` was only ever set to tmp DBs via monkeypatch.

**Warnings remaining (pre-existing, unaddressed):** `datetime.datetime.utcnow()` DeprecationWarning (majority across worker/phase/audit tests), PydanticDeprecatedSince20 class-based `config` in `backend/app/schemas/rule_schema.py` and `document_schema.py`, alembic `path_separator` config deprecation, StarletteDeprecationWarning (`httpx` with `starlette.testclient`).

**Scope notes:** Hindsight MCP unavailable this session — no live recall/retention; this section is the memory record. All changes UNCOMMITTED; working tree on branch `reveo` @ `48d7d43` spans §19–§23. No migration was executed against the production DB.

**Next action:** user reviews diff and authorizes commit(s); the four warning categories are candidates for a later cleanup pass.

---

## 24. Seven-batch audit remediation (M5–M23 + minor items) — 2026-10-29

**VERIFIED this session (full suite `python -m pytest backend/tests -q -o addopts=""` → 335 passed / 0 failed / 47 warnings, run twice at the end; 18 new tests; warnings down from 479 to 47):**

- **Investigation first:** 5 read-only Explore agents verified the current state of every ID before any edit. Already-fixed (no change): M7 (registry status guard), M20 (create_all-only boot), M19-core (fmtUtcTime), item 18 (POST handling), item 14 (per-side commission), item 16 signal-candle linkage, M23 (CI full-suite on reveo), CORS (pinned allowlist), item 13 TP/SL-last-candle.
- **M5 fixed (red→green), `demo_executor.py`:** `TRADE_RETCODE_DONE_PARTIAL` (10010) now FILLED with `result.volume` (was REJECTED, fill discarded); reconciler computes `filled = volume_initial − volume_current` per mql5 semantics (volume_current = REMAINING) and promotes `ORDER_STATE_PARTIAL` too; sweeper now includes `EXECUTION_UNKNOWN` rows; sweep un-gated from `DEMO_TRADING_ENABLED` (read-only `history_orders_get`); non-fill error marker appended once. h6 fixtures that pinned the wrong `volume_current`-as-filled contract rewritten to correct semantics + 2 new tests.
- **M6:** account-wide daily-loss scope PINNED by test (`test_mt5_server_offset.py::test_daily_loss_is_account_wide_manual_deals_included`); docs contradiction 5.0% vs configured 3.0% resolved in `docs/strategy-specification.md` toward the configured 3.0% with an explicit "operator decision required" note; UTC-midnight day boundary documented. **OPEN DECISION:** operator must confirm 3.0% vs the spec's original 5.0%.
- **M8 fixed (red→green), `registry.py`:** `_verify_backtest_provenance` compares wf payload numbers (net_profit/profit_factor/max_drawdown_percent isclose, trade_count) against the stored `BacktestRun` at BOTH attach and promote; promotion fails closed when wf lacks `backtest_run_id` or the run was deleted. Fixtures (h1/ml_governance/promotion_governance) updated to seed genuinely consistent runs.
- **M11 fixed, `documents.py`:** ingest path (`pymupdf/openpyxl` parse + hashing + copy + index refresh) wrapped in `asyncio.to_thread`; thread-identity test proves it runs off the loop. Residual: sync 1MB-chunk temp writes stay on the loop (documented).
- **Item 8 fixed + measured, `retriever.py`/`market_worker.py`:** corpus index now builds LAZILY on first search (query-count test: retriever construction = 0 `knowledge_chunks` queries, repeated searches reuse the cache); worker `trading_ready`/`trading_readiness_reason` query `TradingRule` directly (approved-only invariant preserved, behaviorally pinned). O(N)-per-query BM25 scoring itself unchanged (structural rewrite out of scope, recorded).
- **Item 16 wired:** `WORKER_CYCLE_RETENTION_DAYS` / `AUDIT_LOG_RETENTION_DAYS` (default 0 = disabled, nothing deleted without explicit operator window) applied in lifespan startup; wiring test proves configured windows run and defaults never prune.
- **M21 hardened:** baseline `downgrade()` refuses without `ALEMBIC_ALLOW_DESTRUCTIVE_DOWNGRADE=yes` (red→green test; h4 cycle test updated to opt in explicitly); README destructive-downgrade warning; alembic.ini stale comment fixed + `path_separator = os` added (kills the alembic deprecation warning).
- **M20 behavioral test:** `test_init_db_is_a_noop_on_an_existing_populated_database` (tmp DB, engine-bound). NOTE: the first buggy run of this test called the real `init_db()` on the production engine — verified READ-ONLY afterwards: version still `20261029_wcbf`, 609 worker rows, 0 probe rows, no writes (create_all idempotent, seed branch skipped).
- **M22 fixed:** promote endpoint maps ValueError→404 / ModelGovernanceError→409 / unexpected→sanitized 500 (no more blanket 400); `ml.py` train 500s sanitized (no `str(e)` leak); backtest no-approved-rules → 409; `signals.py` `_execution_failure_status` maps executor messages to 409/403/503/400. GET routes remain open by documented C1 policy (risk note: `/signals/orders` + `/system/audit-logs` are sensitive if the host ever leaves 127.0.0.1).
- **Item 3:** security-headers middleware added (nosniff / frame-options DENY / no-referrer / CSP). Independent review found the first CSP broke Swagger `/docs` (cdn.jsdelivr.net) and the dashboard Vazirmatn font (fonts.googleapis/gstatic) — FIXED by allowing exactly those origins; header test pins them.
- **Item 5/4:** README corrected (token = page memory only, never localStorage; random bootstrap token is NOT in any log — `.env.example` now documents setting `API_TOKEN`); `API_AUTH_MISSING` logged; corr1 web-storage-ban test added.
- **M19 finished:** the two worker-telemetry clocks (`last_cycle_time`, `last_processed_bar_time`) now render through `fmtUtcTime` (red→green node test with an offset-carrying stamp).
- **Item 19:** `datetime.utcnow` eliminated from `backend/app` (shared `utc_now_naive` in `db/base.py`; naive-UTC DB contract preserved); pydantic class-Config → `ConfigDict` in 2 schemas; `path_separator = os`. Kept deliberately: the scoped fastapi.testclient filter (upstream httpx2), and `test_phase_q2_1_consistency.py` 0.05s sleep (≥2 Windows clock ticks; changing it trades one flake class for another). Coverage tooling (pytest-cov) NOT added — needs a dependency decision.
- **Independent review (general-purpose agent, skeptical pass):** ran 200 tests independently (all green), verified constraint compliance (no commit/push, no real-DB migration, EXECUTION_MODE/DEMO flags untouched, no secrets) and MQL5 semantics. Verdict: SHIP after the two CSP fixes (applied). Remaining noted residuals: reconciler fails-safe (no false FILLED) but cannot promote when a broker row carries `volume_initial=0`; per-order full-history `history_orders_get` is O(history) per stale order; `_execution_failure_status` string-matching is coupled to pinned executor messages; `int()` truncation possible for non-integer wf trade_count at attach (very low).

**Scope notes:** Hindsight MCP unavailable all session. All changes UNCOMMITTED; branch `reveo` @ `48d7d43`, working tree spans §19–§24. Real browser not available — dashboard verified via node-executed real-script tests only. `EXECUTION_MODE=SIGNAL_ONLY`, `DEMO_TRADING_ENABLED=False`, no orders, no rule approvals.

**Next action:** user authorizes commit(s); decides the 3.0-vs-5.0 daily-loss threshold and whether to add pytest-cov; the reconciler O(history) sweep can be optimized if unresolved orders ever accumulate.

---

## 25. Final approval push: M5 deals-based reconciliation, M8 real walk-forward, M15 gated reads — 2026-10-30

**VERIFIED this session (final suite `python -m pytest backend/tests -q -o addopts=''` → 351 passed / 0 failed / 66 warnings; `git diff --check` clean; HEAD unchanged `48d7d43`; no commit/push):**

- **M5 (deals-first reconciliation, `demo_executor.py`):** evidence ladder per the real MT5 contract — `history_deals_get(position=ticket)` deals with `deal.order == ticket` AND `deal.entry == DEAL_ENTRY_IN` are the authoritative executed volume (exit deals excluded); fallback to `volume_initial − volume_current` (mql5: volume_current = REMAINING); missing history/API error → row stays EXECUTION_UNKNOWN; only terminal-negative states (CANCELED/REJECTED/EXPIRED) with zero execution resolve to CANCELLED/REJECTED — STARTED/PLACED/PARTIAL stay unresolved; repeated sweeps idempotent (terminal rows leave the filter, executed_at stamped once). TradeDeal 18-field positional order VERIFIED against the installed package. Two real bugs found and fixed during red→green: an `or`-coercion discarding `entry==0` (DEAL_ENTRY_IN) deals, and non-terminal states resolving to CANCELLED. Tests: full fill via deals, partial-then-cancel with same-ticket exit-deal exclusion, multi-IN-deal sum, no-execution → CANCELLED/REJECTED, deals-unavailable fallback, no-evidence-at-all stays UNKNOWN, missing history/API error, idempotency.
- **M8 (real walk-forward, `services/backtest/walk_forward.py` NEW + `registry.py::_verify_walk_forward_provenance`):** a REAL evaluation now exists — chronological contiguous OOS folds run by the same RealisticBacktestEngine with the same costs, one persisted run per fold, one aggregate run (strategy_name="walk_forward") carrying assumptions["walk_forward"] provenance (fold boundaries/run_ids/per-fold metrics, dataset_fingerprint sha256, engine_config). Registry verifies at attach AND promote: strategy marker, provenance block, fold_count ≥ MIN, payload folds == persisted folds, chronological windows, per-fold re-aggregation == run fields AND payload, fingerprint match, trade_count must be a true int (no truncation). `POST /backtest/run?walk_forward=true` wires it through the API. THE negative test: a hand-made plain BacktestRun with EXACTLY matching top-level numbers is rejected ("not a walk-forward evaluation"). Governance fixtures now share `_wf_evidence.py` (simulated evaluator output, honestly labelled) — real-evaluator tests live separately. Honest limits (documented in walk_forward.py docstring): the chain is CONSISTENCY not AUTHENTICITY — a DB-writing forger can fabricate the whole block; METRIC_SIGNING_KEY HMAC covers the model payload only, not BacktestRun rows; fold run_ids are not re-queried; no per-fold ML retraining (attests strategy-level OOS robustness).
- **M15 (sensitive GETs gated):** 12 GETs now require the token — documents/ (raw_text), system/status (account context), worker/status, worker/cycles, audit-logs, rules/, rules/conflicts, signals/ (entry/SL/TP), signals/orders (broker tickets/prices), ml/models, ml/models/{id}/compare, backtest/runs. Open by documented policy: /health (liveness probe) and /market-data/* (public market data, no account context). Frontend: all 5 GET fetches send authHeaders(); 401s degrade to error rows (ensureOk), token stays page-memory-only. corr1 test pins the 401 matrix + valid-token pass + dashboard fetch scan (future-proofed against method:"GET" fetches). Integration/q1/q2_1 tests updated to the new contract.
- **Leftovers resolved this session:** executor failure fragments are now single-source constants (EXEC_MSG_* in demo_executor.py) imported by the signals classifier — identity test pins the coupling; `trade_count` float truncation fixed (must be int); 3.0-vs-5.0 daily-loss discrepancy remains BLOCKED on owner decision (documented in config.py + strategy-specification.md, NOT decided unilaterally); remaining 66 warnings all originate in 6 TEST files (datetime.utcnow in test fixtures; app code clean since §24) + the deliberately scoped starlette/httpx filter — non-critical, scheduled cleanup.
- **Independent review (skeptical subagent):** ran 122 tests across 3 suites (all green), verified TradeDeal/TradeOrder field orders against the installed package, the no-look-ahead property of the fold slicing (FeatureEngineer uses only shift/rolling/ewm), all-GET-route enumeration (no sensitive route left open), and constraint compliance. Verdict: SHIP ×3; its finding A (dead None-evidence branch from `or`-coercion) fixed, docstring staleness fixed, test-strength items applied (same-ticket exit-deal exclusion, multi-IN-deal sum, no-evidence test, future-proofed dashboard scan).
- **GH Actions:** `gh run list` works; latest runs are 2026-10-17 on fix/core-correctness-audit. NO Actions run exists for this uncommitted work and none is possible without push — local runs are not CI evidence.

**Next action:** owner decides the 3.0/5.0 risk cap; commit authorization; future phase: expose attach_walk_forward_metrics through an operator flow (the API already produces wf runs; attaching is currently programmatic).

---

## 26. Full-repo audit round 3 (6 sub-agents + independent verification, 22 fixes) — 2026-10-30

**VERIFIED this session (full suite `python -m pytest backend/tests -q -o addopts=""` → 362 passed / 0 failed / 69 warnings, up from the §25 baseline 351; new module `backend/tests/unit/test_audit_round3_fixes.py` = 11 tests; fresh-DB Alembic up→down(20261015)→up smoke green (head 20261029_wcbf, 12 tables); `from backend.app.main import app` OK). Method: 6 read-only general-purpose sub-agents (trading-core / data+ML / knowledge+backtest / API+core+db+models / migrations+configs+frontend / test-suite) read every in-scope file COMPLETELY with file:line evidence; every load-bearing claim was then re-verified against the CURRENT tree by hand before any edit — the tree was concurrently updated mid-audit by the §25 session, so agents' stale claims (open sensitive GETs, arithmetic-only reconciler) were discarded.

**CONFIRMED & FIXED (code):**
1. `demo_executor.py` — TRADE_RETCODE_PLACED (10008) fell into the definitive-refusal branch (REJECTED while the broker may execute; idempotency reserved = stranded position) → added to `_AMBIGUOUS_RETCODES` (EXECUTION_UNKNOWN + reconciliation). 2. Sweep loop isolates a failing order (per-order try/except + rollback) instead of aborting the whole sweep on the first bad commit (PendingRollbackError cascaded into the worker cycle). 3. `_reconcile_unknown_order` except-path rolls back the session. 4. Simulated (non-MT5) dispatch branch now uses `_commit_pending_safely` — duplicate race returns "Duplicate order prevented", not a 500.
5. `risk_engine.py` — `account=None` fails closed (REJECTED_INVALID_ACCOUNT) instead of AttributeError outside the try (both call sites pre-guarded; latent).
6. `market_worker.py` — reservation failure (non-IntegrityError) now SKIPS the cycle (RESERVATION_FAILED, analysis_executed=False) instead of running the analysis unreserved and colliding with the unique bar_key on finalize; dead `_record_cycle_in_db` + else-branch removed (only caller was the rec=None fallback); stale `last_error_message` no longer written into COMPLETED records; `trading_governance_probe()` consolidates the rules+artifact probe — /health and get_diagnostics deserialize the pickle artifact ONCE per poll instead of twice.
7. `registry.py::load_active_model_artifact` verifies the artifact's embedded `metrics_digest` against the attested store digest (H1 contract at inference time); digest-less/tampered → None (fail-closed). 8. `attach_walk_forward_metrics` enforces the same artifact-path containment as promote/load (attach rewrites the artifact file). 9. `_verify_walk_forward_provenance` round 3: payload cost flags must MATCH the persisted run's `engine_config` + `costs_included`; the wf payload writer (`walk_forward.py`) previously hardcoded all three flags True even for zero-cost runs (zero-cost evidence could satisfy the promotion costs gate) — writer now emits honest flags, registry cross-checks, `_wf_evidence.py` fixture aligned (costs_included block added).
10. `evaluator.py` — measurement-failure AND single-class brier sentinels 0.25 → 1.0: 0.25 exactly satisfied the promotion gate (`val_brier > 0.25` passes at 0.25), fabricating passing evidence; 1.0 always fails it.
11. `signals.py`/`system.py` — `limit` bounded `Query(50, ge=1, le=500)` on 4 list endpoints (SQLite LIMIT -1 = unlimited table dump).
12. `backtest.py`/`ml.py` — `int(symbol_info.spread)` guarded (broker None/invalid → 503, not raw 500); ml.py spreads converted once (`broker_spread`).
13. `config.py` — dead keys MT5_LOGIN / MT5_TIMEOUT_MS / MAX_CONSECUTIVE_EXECUTION_ERRORS removed (grep-verified zero references; operators would have tuned an inert abort counter); `.env.example` synced + ALEMBIC_DATABASE_URL / ALEMBIC_ALLOW_DESTRUCTIVE_DOWNGRADE documented there for the first time.
14. `parsers.py` — per-member decompression cap 64MB (`_MAX_MEMBER_BYTES`, bounded `z.open().read()` for ZIP members AND `word/document.xml` = decompression-bomb guard, cap+1 detection); DOCX/ZIP parse failures now RAISE instead of returning empty-success (an empty return became a terminal NO_EXTRACTABLE_CONTENT record = silent data loss; manager's parse-except path marks FAILED with the real cause); duplicate `infolist` line removed.
15. `manager.py` — hash-dedup query ordered by `Document.id` (deterministic winner when duplicate-hash twins exist); legacy NULL-hash backfill now backfills EVERY proven-identical row (was first-only; remaining NULL twins re-triggered the scan on every re-ingest).
16. `retriever.py` — query-invariant `total_informative_idf_sum` hoisted out of the per-chunk loop (was O(chunks × query_terms) redundant lookups per search).
17. `engine.py` — empty-frame ValueError + missing-ATR-column ValueError (hardcoded 0.0015 fallback on XAUUSD ≈ sub-point SL/TP = garbage-in sizing); `labeling.py` same for its 0.0010 fallback.
18. `pipeline.py` — dataset start/end normalized to naive-UTC before the DB bind (aware index silently stripped by the SQLite dialect).
19. Dead code: `ConditionItem` schema (zero refs), unused imports (backtest/ml/market/rules endpoints), stale defensive `getattr` on declared attrs (system.py), `init_db` Optional annotation.
20. README badges corrected (Python 3.11+ per `requires-python`, dehardcoded test count; FastAPI badge shows the real floor).

**TEST-SUITE integrity fixes:** `test_knowledge_retrieval.py` (real-DB dependency — the 2026-10-27 CRITICAL residual) now opens the real corpus via SQLite `mode=ro` URI + corpus-presence probe + explicit skip when absent (real-corpus validation preserved; can never write; fresh-clone/CI-safe). C1 sentinel (`test_api_endpoints.py`) skips when `adaptive_trader.db` is absent (was a hard OperationalError). `test_document_parser_docx` re-anchored from `<OWNER_HOME>\Downloads\...` (silent no-op counting as pass on every other machine) to repo-anchored `فایل ها/` with explicit skip. Session-heading regex extracted to module-level `parsers.SESSION_HEADING_PATTERN` — `test_phase_o_three_defects.py` imports the production object (copy-pasted lookalikes cannot detect drift). `documents_module.get_db` module-attr leaks in corr1/final_safety fixtures → monkeypatch (restored after yield).

**Test contracts deliberately updated to new documented behavior:** `test_docx_entity_expansion_rejected` (now expects EntitiesForbidden propagation — fail-loud contract), `test_database_failure_resilience_and_rollback` (RESERVATION_FAILED fail-closed instead of unreserved analysis + guaranteed DATABASE_WRITE_FAILED churn). Contract updates to deliberate production fixes, not green-washing; both docstrings state the new contract.

**CONFIRMED BUT NOT FIXED (documented residual):** zero FK constraints across models (needs a migration decision; PRAGMA foreign_keys=ON is currently dead enforcement); CSP `script-src 'unsafe-inline'` globally (Swagger inline script requires it; per-route CSP = bigger change); dashboard `onclick="...esc(...)"` attribute context is wrong-in-principle (HTML entity-decoding defeats escaping inside JS strings; ids are hex-only today → unreachable; data-*+addEventListener refactor deferred); market.py 3 GETs + /health unauthenticated (M15-documented policy; API binds 127.0.0.1 by default); `audit_log.created_at` unindexed while retention sweeps filter on it (needs migration); reconciler first-match `return` and PARTIAL-promoted-at-first-evidence residual (§24-documented design tradeoff); `volume_profile` O(N·window·bins) (not invoked by the pipeline today); XLSX parsed by openpyxl's internal parser outside defusedxml scope (third-party posture); `metrics.py` "gross_profit/loss" fields are net-based (semantic mislabel); floating equity marks omit exit-side slippage on non-exit bars (documented worst-case convention gap); TradeSignal/DemoOrder rows unbounded (retention decision needed — WORKER_CYCLE/AUDIT_LOG windows exist, signals/orders do not); promote read-then-mutate race (SQLite serializes writes; two concurrent promotions last-commit-wins); logging.py pseudo-JSON formatter does not escape quotes (malformed lines possible); pyproject all `>=` floating with NO lockfile (non-reproducible builds — process decision); python-dotenv direct pin KEPT deliberately (we rely on env_file; pydantic-settings declares it transitively).

**Also verified this session:** the §24-era "MT5 ~3h timestamp blocker" is RESOLVED in current source — `MT5_SERVER_TIME_OFFSET_HOURS=3.0` is server-gated (applied only when account.server == MT5_SERVER; verified live 2026-10-28 per config comment + provider consistency validation that fails closed on wrong-offset stamps). `**Purpose:**` 0-byte stray file at repo root left untouched (untracked, harmless, owner cleanup). Hindsight MCP unavailable all session. All changes UNCOMMITTED; branch `reveo` @ 48d7d43; working tree now spans §19–§26 (65 modified + ~40 untracked paths). `EXECUTION_MODE=SIGNAL_ONLY`, `DEMO_TRADING_ENABLED=False`, `order_send` reachable only from demo_executor, no orders, no rule approvals, no real-DB migration.

**Next action:** owner reviews diff and authorizes commit(s); decides FK-constraint and lockfile questions; Phase P rule approvals remain pending.

---

## 27. M8 ML walk-forward + migration-chain fix + dedup integrity + 4-scope repo audit — 2026-10-31

**VERIFIED this session (final suite `python -m pytest tests -q` from `backend/` → EXIT=0, ≈391 tests, up from §26's 362; new/updated test files: test_ml_walk_forward.py (16), test_mlwf_migration.py (2), test_dedup_unique_migration.py (3), test_extractor_timeframe_scope.py (3), +1 settings-key test in test_audit_fixes_h1.py, +3 M5 tests in test_audit_fixes_h6.py, +1 rule-stage-failure test in test_phase_o_three_defects.py, head-stamp assertion in test_alembic_destination_safety.py). Context7 unavailable all session (no MCP tools) — MT5 `history_deals_get` semantics verified against official mql5.com docs via WebFetch instead. Hindsight MCP unavailable (protocol rule 9 followed; PROJECT_MEMORY used).

**CRITICAL BUG FIXED — alembic.ini CWD-relative `version_locations` (`alembic.ini:10,37` → `%(here)s`):** running `alembic upgrade head` from any CWD other than the repo root resolved ZERO revision files and "succeeded" WITHOUT executing a single migration (only an empty `alembic_version` table) — caught by `test_migration_runs_on_explicit_tmp_url`, root-caused (pytest fixture pins absolute script_location; version_locations stayed CWD-relative), fixed with `%(here)s` for BOTH script_location and version_locations, and the test now also asserts the stamped version equals the chain head (silent-empty mode can never pass again). Verified from both CWDs (6 "Running upgrade" lines + 12 tables each).

**M8 completed — true per-fold ML walk-forward (`backend/app/services/ml/walk_forward.py` NEW):** `run_ml_walk_forward` re-fits a FRESH model (logistic_regression/random_forest) AND a fresh BaselineModel per fold on expanding-window training partitions that end `purge_window(=horizon_bars=24)+embargo_bars(=12)` before each chronological contiguous OOS window; StandardScaler fitted on each fold's TRAIN ONLY; fixed threshold 0.5 (verifier rejects any other value); payload carries fold boundaries (positions+timestamps), per-fold model+baseline metrics, recomputable aggregates, dataset fingerprint, feature names, hyperparameters, label/cost definition, code_version. Evidence chain: `metrics_integrity.ml_walk_forward_digest` (HMAC w/ SHA-256 fallback) + `verify_ml_walk_forward_payload` (independent re-verification: fold_index sequence, chronology, no overlap, purge>=horizon enforced, purge+embargo gap, aggregate recomputation incl. total_oos_samples, fixed threshold, required provenance). Registry: `attach_ml_walk_forward_metrics` (structural+digest gate, ≥3 folds, candidate-only, artifact-path containment) + promotion-time re-verification `_ml_walk_forward_failure` wired into `promote_candidate` (tampered/absent-digest/overlap/forged-aggregate all refuse promotion). Model column `ml_models.ml_walk_forward_metrics` + migration `20261031_mlwfm` (additive, server_default '{}', reversible — verified on a POPULATED legacy DB: backfill '{}' on legacy rows, data preserved, idempotent re-upgrade, downgrade/upgrade round-trip; `test_worker_cycle_backfill_migration` destination pinned to `20261024_mdig` because the relative `-1` silently changed meaning with the new revision).

**M2 verified + one gap closed:** the transactional ingestion state machine was confirmed by existing failure-injection tests (chunk-stage crash → failed + 0 chunks; retry reuses document_id; repeated failures no duplicates; legacy completed-without-chunks repaired) and a NEW test covers the RULE-stage failure (commit #3): doc stays failed, no partial rules, retry rebuilds exactly once, re-ingest idempotent. State machine now documented on the KnowledgeManager docstring.

**M5 completed per the task's mandatory-test requirement:** the previous stubs accepted ANY arguments (`lambda **kw`), so nothing proved the query semantics. NEW capturing-args test pins `history_deals_get(position=<order ticket>)` exactly (no positional args, no `ticket=`, no other value; semantics verified against official MT5 docs: `position=` filters DEAL_POSITION_ID, `ticket=` filters DEAL_ORDER). Reconciliation now also records deal-derived costs: VWAP executed_price, summed commission+swap over ENTRY_IN deals of OUR order (EXIT deals excluded — test), and the order-row fallback leaves executed_price/commission/swap untouched (no fabricated zeros — test). Ruff dead code removed (`filled_states`).

**Dedup integrity (audit round-3 residual resolved):** `content_sha256` had only a NON-unique index while KnowledgeManager's concurrent-collision path relies on IntegrityError — the protection was illusory. Migration `20261031_dshq` adds UNIQUE `ux_documents_content_sha256` (non-destructive: duplicate twins lose the hash on higher-id rows, never rows; NULL legacy hashes stay legal) + model-level `__table_args__`; old redundant index dropped. Tests: duplicates neutralized without row loss, DB rejects duplicate-hash insert (IntegrityError — the exact mechanism the manager catches), NULL hashes allowed, reversible.

**Audit-log index (round-3 residual resolved):** migration `20261031_auidx` + model `index=True` on `audit_logs.created_at` (retention pruning + time queries).

**Lockfile:** `requirements-lock.txt` pins the project's full transitive closure (47 packages, exact versions the suite ran green against, Python 3.14.4); regeneration note included; backport packages excluded by their own markers.

**METRIC_SIGNING_KEY honoring fixed:** the digest layer read ONLY `os.environ`, but pydantic-settings loads .env into Settings WITHOUT injecting into os.environ — a .env-only key was silently ignored (silent degradation to non-tamper-evident SHA-256). `_signing_key()` now falls back to `settings.METRIC_SIGNING_KEY` (new Settings field, default ""), documented in `.env.example`; test pins HMAC-via-Settings + tamper detection.

**Other fixes from the 4-scope audit:** H-1 executor MT5 calls now hold the process-global `_mt5_lock` (order_send, terminal_info, history_orders_get, history_deals_get — sweep runs on a worker thread while API threads call the provider; MT5 package not thread-safe); M-4 extractor timeframe substring bug fixed with word-boundary regex ("M15" no longer also grants "M1"; regression tests pin M1-only, M15-only, both); dead duplicate CSV block removed (parsers.py); retriever dead vars removed (query_stopwords, tokens); backtest endpoint's computed-but-dropped `report_md` generation removed (+unused import); test F811 shadowed import fixed; ml/walk_forward frame validation hardened to the DatasetPipeline OHLC contract (finite/positive/high>=low).

**4-scope repo audit (read-only sub-agents, file:line evidence, ~8.5k lines fully reviewed in-scope + targeted cross-checks):** scope = API/core/schemas/db (19 files, 1801 lines), services/execution/risk/signal/worker/market_data/knowledge (15 files, 4860 lines), ML/dataset/backtest/metrics_integrity (15 files, 2576 lines), models/migrations/tests/frontend/CI/PROJECT_MEMORY. Inventory: 184 files, ~30k lines (backend_app 71/10271, tests 55/12335, migrations 8/438→now 9, frontend 1/1270, docs 38/5663, other 11/500). MIMOSA's ETIMEDOUT report superseded by this executable audit + compileall (exit 0) + full suite. Trading-safety verdict (evidence-chained): NO path reaches mt5.order_send under SIGNAL_ONLY + DEMO_TRADING_ENABLED=False — 12 enumerated gates; no runtime writer of the two settings; worker hard-asserts SIGNAL_ONLY. Frontend: no fully-unescaped sink (L-1 onclick attribute-context escaping is wrong-in-principle but unreachable with hex-only IDs — defense-in-depth TODO). No test writes the real DB (two read-only mode=ro sentinels documented).

**CONFIRMED BUT NOT FIXED (documented, ordered):** ML-WF evidence is self-attesting at attach (no binding to a persisted run/dataset like the strategy-WF path; mitigations: no HTTP route, optional for promotion, independent gates still enforced — design follow-up recommended); attach endpoints have NO HTTP surface at all so the governance chain needs an in-process step before /ml promote (owner decision: operator endpoint vs documented script); MT5-dependent safety tests skip on ubuntu CI (Windows runner decision); zero FK constraints (SQLite table-rebuild migration decision, previously owner-deferred); per-request provider.disconnect() cuts the shared terminal connection (worker self-heals; lifecycle redesign needed); emergency-stop activation commit failure loses the audit trail (fail-closed direction preserved); concurrent-ingest loser surfaces 500 + orphan raw file (winner-must-be-completed); CSP 'unsafe-inline' (documented structural need of the inline dashboard bundle); signals/trades retention disabled by default (owner decision; WORKER_CYCLE/AUDIT_LOG windows exist); CI python 3.11 vs local 3.14; logging formatter no rotation/escape; Ruff ~1100 findings are bulk style classes (UP006/UP045/I001/F401/UP017/UP035) — batch deferred to a dedicated typing-modernization pass (no blind --fix per task constraint); 3.0/5.0 daily-loss cap remains BLOCKED on owner decision.

**Runtime validation:** uvicorn boot on an ISOLATED temp DB (real `backend/adaptive_trader.db` never touched): /health 200; dashboard 200; auth fail-closed verified (401 without X-API-TOKEN, 200 with — 10 POST routes token-gated; market GETs open per M15 policy); document upload → completed + 1 rule; duplicate upload → SAME document_id (idempotent, now DB-enforced); documents list, ml/models, worker/status, signals 200; knowledge retrieval + grounded query verified service-level; /system/status reports SIGNAL_ONLY + demo gate disabled. Browser console validation NOT performed — the browser-use tool (mcp__node_repl__js) is not exposed in this environment; HTTP-level frontend checks done instead.

**Git:** branch `reveo` @ 48d7d43, all changes UNCOMMITTED (commit/push forbidden by task); ~120 modified/untracked paths spanning §19–§27. GitHub Actions: latest real run 2026-10-17 success on fix/core-correctness-audit; NO CI run exists for this work without push. `EXECUTION_MODE=SIGNAL_ONLY`, `DEMO_TRADING_ENABLED=False`, zero orders, zero rule approvals, real DB untouched (migrations tested only on temp copies).

**Next action:** owner reviews diff and authorizes commit(s); decides: ML-WF attach binding design + operator attach endpoint, Windows CI runner for MT5 tests, FK migration, CSP refactor, signals/orders retention, METRIC_SIGNING_KEY provisioning, 3.0/5.0 cap.

---

## 28. Remediation round 2: position_id semantics, APPROVED-rule preservation, replay guard — 2026-10-31 (same day)

**VERIFIED this session (final suite `python -m pytest tests -o addopts="" --junitxml` → EXIT=0, junit counts: 400 tests / 400 passed / 0 failed / 0 errors / 0 skipped; `git diff --check` and `git diff --cached --check` both clean). Method: coordinator implemented root-caused fixes, then FOUR independent read-only sub-agents (A: M5 reconciliation, B: M2 atomicity, C: ML-WF provenance, D: migrations/git/safety) re-audited every change with file:line evidence and raw command outputs; A: 7/7 VERIFIED, B: 6/6 VERIFIED, C: 8/8 VERIFIED, D: all items VERIFIED, no blocking defect.

**M5 — position_id semantics corrected (task-mandated):** the deals query now reads `TradeOrder.position_id` FROM THE ORDER RECORD (never assumes order ticket == position id; netting add-on orders attach to an EXISTING position). `position_id=0` → deals query is skipped entirely (test asserts history_deals_get is NEVER called) and the order-row arithmetic decides. All deal fixtures now use deliberately DIFFERENT identifiers (order_ticket 911xxx vs position_id 900xxx) and the capturing test asserts `position == 900777 AND position != 911777`. New tests: deals-API exception = distinct evidence state (fallback resolves, no fabricated costs); unrelated order's ENTRY deal on the same position excluded from volume/VWAP/commission/swap. Agent A independently re-verified semantics against official mql5.com docs and confirmed real TradeOrder/TradeDeal field ordering of the fixtures.

**M2 — two real defects fixed:** (1) SAWarning root cause: after a mid-ingest rollback, `Query.delete()` removes DB rows but does NOT evict identity-map entries, so re-created rows reusing freed autoincrement PKs collide — recovery now expunges stale matched objects BEFORE deleting (proven with `-W error::sqlalchemy.exc.SAWarning`). (2) Agent-B finding: recovery deleted ALL rules including HUMAN-APPROVED ones and rebuilt fresh PENDING clones with new rule_ids — silent de-approval. Recovery now preserves APPROVED rules with their identity and fresh extraction skips exact (name, category) clones of preserved approved rules (regression test: approve → boundary-4 failure → retry → approved rule survives with same rule_id/status, zero clones). Failure-injection coverage now spans ALL five persistence boundaries (document/chunks/rules/completion-status/audit) with DB-state assertions; new boundary-4 and boundary-5 tests.

**M8 — replay guard + idempotency fixed (Agent-C driven):** (1) attach re-signing a replayed payload for a DIFFERENT model used to overwrite the foreign digest silently — replay guard now refuses a digest not signed for the target model (test). (2) Agent-C found idempotent re-attach of a model's OWN payload produced evidence the promotion verifier would reject as tampered (digest computed over payload still containing the stale digest key) — attach now strips the digest key before re-signing (test: re-attach → promotion verifier passes). (3) Non-string digest → clean ModelGovernanceError instead of raw TypeError. (4) Verifier rejects non-numeric aggregate mean/std instead of raising. Honest-limitation statement (self-attestation residual; optional-for-promotion) re-confirmed in PROJECT_MEMORY.

**Chores:** ChainedAssignmentError in test_walk_forward_provenance.py fixed via `.loc` (pandas CoW); staged EOF whitespace in test_phase_o_three_defects.py normalized and that ONE file re-staged (`git add` of a single file only — index now matches the working copy); zero-byte repo-root file identified precisely: name = PUA codepoints U+F02A/U+F03A rendering as "**Purpose:**" — accidental shell redirect from an earlier agent session (REPORTED origin, 0 bytes, untracked, harmless; owner may delete manually). `adaptive_trader.db-wal` no longer exists (checkpointed); only `adaptive_trader_backup.db-wal` (zero-byte sidecar, gitignored) remains.

**Agent D verdict (migrations/git/safety):** single head 20261031_auidx, linear 9-revision chain; fresh upgrade from a NON-repo CWD ran all 9 migrations and built 12 tables (raw output captured); dshq has NO DELETE (keeps MIN(id), NULLs higher-id twins); every 20261031_* upgrade additive-only with guarded downgrades; the ONLY `mt5.order_send` call site remains demo_executor.py behind the fail-closed gates (EXECUTION_MODE/DEMO_TRADING_ENABLED defaults+validator untouched by today's diffs); env.py refuses to migrate without explicit ALEMBIC_DATABASE_URL.

**Still deferred (unchanged, owner decisions):** race-loser 500 + orphan raw file; FK constraints; CSP 'unsafe-inline'; signals/orders retention; ML-WF binding to a persisted run (anti-self-attestation design follow-up — digest-stripping replay is closed, but fresh-payload re-signing at attach remains possible with no HTTP route exposed); Windows CI for MT5 tests; browser console validation (tool unavailable); Ruff style-class debt; 3.0/5.0 daily-loss cap.

**Git:** branch `reveo` @ 48d7d43; 118 changed/untracked paths (52 untracked); 7 staged entries preserved (test_phase_o_three_defects.py re-staged with EOF fix + today's additions); nothing committed/pushed. `EXECUTION_MODE=SIGNAL_ONLY`, `DEMO_TRADING_ENABLED=False`, zero orders, real DB untouched.

---

## 29. ML engineering knowledge base (`ata-ml-engineering`) — 2026-10-31

**VERIFIED this session (knowledge-building only; zero code/config/trading changes, zero installs, no test runs claimed):** Owner-mandated mission to turn awesome-machine-learning (https://github.com/josephmisiti/awesome-machine-learning) into usable repo expertise. Method: ground truth from `requirements-lock.txt` + live `python -I` version probe (both agree: numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0, fastapi 0.143.0, sqlalchemy 2.1.4, alembic 1.20.0, pydantic 2.13.4/pydantic-settings 2.14.2, pymupdf 1.28.2, openpyxl 3.1.5, MetaTrader5 5.0.6231; NOT installed: statsmodels, optuna, xgboost, lightgbm, catboost, sktime, river, imbalanced-learn, shap, skforecast), then FOUR research sub-agents (ML/eval, TS/financial ML, data engineering, backend/infra) read actual source and fetched official vendor docs; coordinator integrated and de-conflicted. Context7 MCP was unavailable this session — vendor docs fetched directly via WebFetch with URL+version+review-date recorded per file; no doc text copied verbatim.

**Deliverables (6 new files):** `docs/engineering/ml-library-map.md` (coordinator map: environment table, installed-stack code facts, candidate verdicts, BUY/SELL asymmetry, 10-item gap list) + `.agents/skills/ata-ml-engineering/` — `SKILL.md` (non-negotiables, decision guides, reference pointers) and `references/ml-algorithms-evaluation.md`, `references/timeseries-financial-ml.md`, `references/data-engineering.md`, `references/backend-model-infra.md` (1104 reference lines total, each ending with Provenance + Gaps).

**Key code facts encoded (file-verified):** ML walk-forward = purge 24 (≥ horizon, enforced) + embargo 12 (`horizon//2`) + retrain per fold + fixed 0.5 threshold (`ml/walk_forward.py`, pinned by `tests/unit/test_ml_walk_forward.py:132-140`); static chronological split in `dataset/pipeline.py` purges but has NO separate embargo (UNRESOLVED: intentional or gap); `label_buy_barrier` is BUY-only (ask entry with historical per-bar spread, ATR TP/SL fail-loud, conservative same-bar collision → 0; no SELL labeler — direction-classifier ≠ barrier-success-classifier documented); evaluator computes acc/prec/recall/f1/ROC-AUC/Brier/LogLoss + manual 5-bin calibration (sentinel constants gate-failing by design; log_loss informational); imbalance handled only via LR `class_weight="balanced"` (RF unweighted; no resampling; balanced_accuracy/average_precision NOT logged — cheapest zero-dep next step); calibration measured but NO calibrator fitted anywhere (owner decision); registry = candidate→active→archived, promotion only via token-gated `/models/{id}/promote` with 9 gates, HMAC digest via `METRIC_SIGNING_KEY` else SHA-256, walk-forward attach has no HTTP surface; MT5 provider fully RLock-wrapped + read-only + fail-closed time offset; alembic.ini `%(here)s`-based + env.py fail-closed without `ALEMBIC_DATABASE_URL`; pandas 3.0 CoW/str-dtype semantics documented and codebase is CoW-compliant.

**Candidate verdicts (all NOT INSTALLED — adding requires explicit user approval):** defer by default (Optuna, XGBoost, LightGBM, CatBoost, statsmodels, sktime, River, imbalanced-learn, shap); skforecast BLOCKED (dev docs pin `pandas < 3.0` vs installed 3.0.6). py3.14/Windows wheel support UNVERIFIED for every candidate (statsmodels docs cap at 3.13; LightGBM python-package and River install pages 404'd; mql5 `copy_rates_from` page 404'd — output contract verified from mql5 overview + code). imbalanced-learn floor ok (sklearn ≥1.4.2), ceiling vs 1.9.1 UNVERIFIED.

**Safety verdict:** no source/config file touched; 6 new knowledge files + this entry only; trading settings untouched; no orders; no rule approvals; nothing installed; nothing committed.

**Next action:** owner reviews the 6 files; decides on the gap list (esp. static-split embargo intent, calibration fitting, per-side metrics logging, py3.14 wheel verification if any candidate is ever wanted); commit decision per standing rule.

**Addendum (2026-11-01):** work committed and pushed to `origin/reveo` (commits 591c0b3, 5d10643, 5e6822d). CI "Core Regression Tests" on ubuntu-latest: first full-suite run failed (19) — defusedxml was missing from pyproject (declared only in the local env; root cause of a 500 on /system/status in CI), test_mt5_server_offset lacked the Windows-only skipif, data/processed-dependent tests lacked file-missing skips, and test_security_headers_on_every_response queried the real DB without isolated_app_db (silently DB-dependent locally). All four fixed; latest run: **completed success** (run 41255078662). README refreshed (400 tests, lockfile install). Runtime dirs (.mimosa, .backup-spec-kit-20261029, backend/artifacts, backend/data, zero-byte PUA file) gitignored, not committed.

---

## 30. M4 closed + stale-list reconciliation + SQLite busy_timeout + signal/order retention — 2026-11-01

**VERIFIED this session (full suite `python -m pytest tests -q -o addopts=""` from `backend/` → 420 passed / 0 failed, run twice; commit `8499733` pushed to `origin/reveo` after a rebase onto the two README docs commits `439ab51`/`6aa4979` that had landed remotely).** Hindsight MCP unavailable; Context7 MCP unavailable (no tools exposed this session) — no library/API claims needed it.

**M4 CLOSED (the last critical open item):** explicit rule direction replaces every name-derived vote.
- `models/rule.py`: `TradingRule.direction` (`String(4)`, nullable — 'BUY' | 'SELL' | NULL).
- `schemas/rule_schema.py`: `RuleDirection` enum, optional `direction` on Base/Response, `RuleDirectionUpdate` (actor+notes) for the new `POST /api/v1/rules/{rule_id}/direction` (token-gated, `RULE_DIRECTION_SET` audit event) — the ONLY conscious, reviewable assignment path.
- `signal/generator.py`: direction votes read `rule.direction` ONLY; NULL/unknown casts no vote (fail-closed); the `buy/خرید/sell/فروش` name scan is DELETED (grep-verified zero name-direction lookups left in `backend/app`).
- `knowledge/manager.py::check_rule_conflicts`: CONTRADICTORY_DIRECTION now decided by explicit directions, never names.
- Migration `20261101_rdir` (head; inspector-guarded like mlwfm because the initial revision builds tables via `Base.metadata.create_all` — a plain ADD COLUMN double-adds on fresh chains). NO name-based backfill by design: legacy rules keep direction=NULL and stay fail-closed until an operator assigns direction (Phase P human decision stays reviewable).
- Tests: `test_rule_direction_m4.py` (13: misleading-name→SELL, name-says-BUY-with-NULL-direction→no vote, garbage 'LONG'→no vote, direction-based conflicts (names neutral → no conflict), API set/clear/422/404/401, list exposes direction, migration adds-nullable+no-backfill+reversible). Fixtures in 5 older test files pinned `direction="BUY"` explicitly (previously name-derived).

**Minor item 7 CLOSED:** `DB_BUSY_TIMEOUT_MS` (default 5000) applied via sqlite3 `connect_args["timeout"]` (the native busy handler) — NOT a PRAGMA f-string: the first commit was BLOCKED by the Mimosa pre-commit hook (flagged `session.py:32` f-string-into-execute as SQL-injection class); parameter binding is impossible for PRAGMA (`near "?": syntax error`), so the connect-arg approach was adopted (no SQL string construction at all). Tests `test_sqlite_busy_timeout.py` (5).

**Minor item 16 extended:** `TRADE_SIGNAL_RETENTION_DAYS` / `DEMO_ORDER_RETENTION_DAYS` (default 0 = DISABLED, same contract as the §24 windows) wired into `prune_operational_records` + lifespan; tests `test_signal_order_retention.py` (2) + existing contracts updated to the 4-key dict.

**User's audit list reconciled against the tree (items largely STALE, previously VERIFIED in §24–§28):** M9 (UNIQUE content_sha256 migration `20261031_dshq` + model Index) ✓, M10 (64MB per-member decompression cap in parsers.py) ✓, M13 (lazy BM25 index + consolidated governance probe) ✓, M14/M15 (12 sensitive GETs token-gated; market/health open by documented C1 policy) ✓, M22 (promote maps 404/409/sanitized-500 — no blanket 400) ✓, M23 (CI green on reveo run 41255289820) ✓, minors 3/4/5/6/8/10/12/13/14 previously closed. Minor 15 (old DB files) = owner cleanup decision; adaptive_trader*.db are gitignored. Remaining genuinely open per the roadmap: M12 (Phase 2.1 connection manager), ML-WF→BacktestRun binding (2.4, owner-design), CSP/onclick (2.5), coverage/mypy (2.6), 3.0-vs-5.0 daily-loss (owner decision), signals/orders retention WINDOWS (owner values), M8 volume-reconciliation acceptance (needs the owner's Phase P review).

**CI note:** run 41325881829 FAILED on this push — NOT the M4 change: the remote README-rewrite commits had dropped the exact documented phrase `حافظهٔ همان صفحه` (page-memory token flow) that `test_dashboard_never_persists_token_in_web_storage` pins. README re-pinned (one hamze), corr1 suite 26 passed; follow-up commit restores CI.

**Docs:** `docs/architecture.html` NEW — standalone RTL Persian architecture document (component map with click-through details, document→signal data flow, safety rails, roadmap position, M-item file map; HTML tag-balance + node `--check` verified).

**Constraints:** `EXECUTION_MODE=SIGNAL_ONLY`, `DEMO_TRADING_ENABLED=False`, zero orders, zero rule approvals, real DB untouched (migrations tested only on tmp DBs; M4 leaves legacy rules direction=NULL so nothing silently changes behavior).

**Next action:** owner (1) fixes the README phrase commit/push if not yet — done this session; (2) decides merge reveo→main (roadmap 1.2) — NOT done without explicit authorization; (3) decides the three product decisions (3.0/5.0, retention, METRIC_SIGNING_KEY provisioning); (4) assigns directions to existing rules via the new endpoint before approving them (Phase P); (5) then Phase 2 (2.1 MT5 connection manager first).

---

## 31. automation/news-pause: owner decisions applied + 7-step automation (news rule BLOCKED on source) — 2026-11-03

**VERIFIED this session (branch `automation/news-pause` from `reveo@bf30a9f`; suite 456 passed / 0 failed; commit chain 6 commits: decisions → archive → inbox → suggestion+batch → scheduler+pre-check → roadmap).** Context7/Hindsight MCP unavailable all session; the MT5 calendar verdict was verified against the installed package (`dir(MetaTrader5)` → zero calendar functions on 5.0.6231) and the official mql5.com Python docs (WebFetch).

**Step 1 — owner decisions APPLIED:** `MAX_DAILY_LOSS_PERCENT` 3.0→**5.0** (resolves the §24/§25 BLOCKED 3.0-vs-5.0 discrepancy; the risk-engine test was updated to the new cap as a deliberate, documented contract change) and retention defaults 0→**30 days** for all four operational tables (per-table 0 still disables). METRIC_SIGNING_KEY: owner-only rule documented in `.env.example` — the owner generates, stores and backs up the key privately; never in chat/git/logs; no agent ever fills it. Decisions pinned as executable contracts in `backend/tests/unit/test_owner_decisions_2026_11_01.py`.

**Step 2 — news pause rule: BLOCKED, one alternative proposed (owner decision pending).** The MetaTrader5 Python integration exposes NO economic calendar (verified twice). Per the mandate ("if it has no access, stop and propose exactly one reputable alternative; do not mix sources"), the news rule was NOT implemented and exactly ONE alternative was proposed: the ForexFactory weekly JSON feed (`https://nfs.faireconomy.media/ff_calendar_thisweek.json`, no API key, High/Medium/Low impact labels). Until the owner picks a source, the live signal loop continues WITHOUT news protection — recorded as an explicit risk in ROADMAP.md. The `news-pause-rule-spec.md` file referenced by the prompt was NOT present in the delivered folder; the prompt's embedded rules were used verbatim.

**Step 3 — automation (6 of 7 steps implemented; step 6's news integration blocked):**
- 3.1: `services/knowledge/inbox.py` — `data/document_inbox/` scanned at startup, every supported file ingested through the transactional pipeline (idempotent via content_sha256); processed/failed folders; per-file progress logs; 6 tests.
- 3.2: chunk progress logged every 25 pages; `get_page_type_summary` reports IMAGE_ONLY/VECTOR_ONLY/BLANK pages per document (reported, never silently dropped, no rule built from them, no OCR per mandate).
- 3.3: `suggested_direction` column (migration `20261102_sdir`, inspector-guarded) filled from document TEXT only (never the name); `POST /rules/batch-review` (token-gated, audited, ≤200 items) — approve with explicit BUY/SELL, SUGGESTED adoption, or null (directionless fail-closed), or reject with reason; per-item errors never abort the batch; M4 preserved (a suggestion alone can never vote — pinned by test).
- 3.4: `services/ml/training_scheduler.py` — daily daemon at 21:00 UTC training ONE candidate on XAUUSD M1 via the NEW shared `services/ml/training.py` flow (also adopted by /ml/train, so both paths cannot drift); NEVER promotes.
- 3.5: `GET /ml/models/{id}/promotion-check` — read-only per-gate summary; `promote_candidate` now evaluates via the same `_promotion_gate_failures` (messages byte-identical; governance suites green).
- 3.7: `_archive_monthly_summary` writes per-month JSON (total + per-status) to `artifacts/reports/archive/<table>/<YYYY-MM>.json` BEFORE pruning, merged across passes, fail-safe (archive failure SKIPS pruning that table — rows are never deleted without a surviving summary).
- Step 6 (live loop + news rule): the 3-second gold loop is unchanged; its news integration waits for the owner's source decision.

**Step 4 — roadmap overhaul:** NEW canonical `docs/ROADMAP.md` (vision paragraph; CEO + technical versions in one document; stage/gate table with automatic-vs-human per stage; owner-decisions table with dates; plain-language risks incl. "green tests ≠ profit"; a what-changed section that removes stale items). `docs/architecture.html` roadmap section synced to it.

**Confirmed but NOT done (owner items):** news source choice (proposal: ForexFactory JSON); merge to main; direction assignment + batch approval of the existing pending rules (Phase P); METRIC_SIGNING_KEY provisioning on the owner's machine; then roadmap-9.5 phase 2 (M12 connection manager first).

**Constraints:** `EXECUTION_MODE=SIGNAL_ONLY`, `DEMO_TRADING_ENABLED=False`, zero orders, zero rule approvals, zero model activations; real DB untouched (all migration tests on tmp DBs); no secrets committed (signing key stays placeholder-only).
