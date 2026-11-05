# Shared Project Context and Memory Protocol

Before substantial work in this repository:

1. Read `docs/PROJECT_MEMORY.md` first. Treat it as the canonical human-readable project handover; do not create a parallel master-memory file.
2. If the Hindsight MCP server `hindsight-adaptive-trader-ai` is connected, recall memories relevant to the current task before planning or editing. After a material decision, verified fix, test run, or blocker, retain a concise factual memory with evidence status and date.
3. Use memory for continuity, not as proof. Verify current Git branch/HEAD/status, source files, database state, and test output before making claims or taking actions.
4. When a material project fact changes, update `docs/PROJECT_MEMORY.md` with exact paths/commits/commands, actual test results, evidence type, remaining risks, and next action. Preserve significant failed results and explain superseded claims.
5. Clearly label information as **VERIFIED**, **REPORTED**, **UNRESOLVED**, or **SUPERSEDED**. Never present another agent's report as independently verified unless you reproduced the evidence.
6. Do not store credentials, API keys, passwords, access tokens, account identifiers, or other secrets in Hindsight, Markdown files, logs, or commits.
7. Preserve existing user changes. Before edits, inspect `git status`, branch, HEAD, and the relevant diff. Never reset, discard, stage, commit, push, or merge without explicit authorization.
8. Trading safety is mandatory: keep `EXECUTION_MODE=SIGNAL_ONLY` and `DEMO_TRADING_ENABLED=False`; place no orders; do not activate/approve strategy rules or enable demo/live execution unless the user explicitly authorizes a narrowly scoped change.
9. If Hindsight is disconnected or unavailable, continue using `docs/PROJECT_MEMORY.md`; state that live memory recall/retention was unavailable rather than pretending it occurred.
10. Use the project's preferred concise report format: what changed, what is proven, what remains unknown/risky, and the next action.

Hindsight is a shared retrieval layer only. The repository, actual runtime state, and reproducible test evidence remain authoritative for current technical facts.
