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
