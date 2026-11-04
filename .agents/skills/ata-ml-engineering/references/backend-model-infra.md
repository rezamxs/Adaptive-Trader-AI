# Backend & Model Infrastructure Reference — Adaptive Trader AI

Scope: FastAPI assembly, ORM/migrations, settings, model registry/governance, MT5 integration.
Verified against branch `reveo` (HEAD 48d7d43), installed packages per `requirements-lock.txt`.

Fixed trading-safety constraints (never change without explicit user authorization):
`EXECUTION_MODE=SIGNAL_ONLY` (default; `LIVE`/`REAL` strings rejected at config validation),
`DEMO_TRADING_ENABLED=False`, `EXECUTION_MODE.LIVE_TRADING` intentionally absent from the enum
(`backend/app/core/config.py`).

---

### FastAPI 0.143.0

- **What it does:** ASGI web framework with dependency injection, lifespan events, automatic OpenAPI, and middleware.
- **Where in this project:** `backend/app/main.py` (app assembly, lifespan, CORS, security headers), `backend/app/api/v1/endpoints/ml.py` (endpoint pattern), `backend/app/db/session.py::get_db` (session dependency), `backend/app/core/auth.py::require_api_token` (token gate).
- **Essential APIs:**
  - `FastAPI(lifespan=...)` — startup/shutdown as one `@asynccontextmanager` function.
  - `Depends(fn)` — per-request dependency resolution; `def` (sync) dependencies run in a threadpool.
  - `APIRouter` + `app.include_router(router, prefix="/api/v1")`.
  - `HTTPException(status_code=..., detail=...)` for typed error mapping.
  - `app.add_middleware(CORSMiddleware, allow_origins=...)`.
  - `@app.middleware("http")` — raw HTTP middleware for response headers.
- **Best patterns (as implemented):** lifespan owns `init_db()` + worker start/stop + retention sweep, so tests or module imports never pay the startup cost; endpoints are declared `def` (not `async def`) because they do blocking DB/ML/MT5 work — FastAPI runs them in its threadpool; `get_db` is a yield-dependency that closes the session in `finally`; error classes are mapped per exception type (`ValueError`→404, `ModelGovernanceError`→409, sanitized `Exception`→500) in `POST /models/{model_id}/promote`. Docs note lifespan is all-or-nothing vs deprecated `on_event` handlers — this project uses lifespan only, correctly.
- **Anti-patterns / common mistakes:** using `async def` handlers that call blocking libraries (blocks the event loop); loading models or connecting to MT5 at module import instead of via dependency/lifespan; keeping an MT5 connection open across requests (this project connects → fetches → disconnects in `finally` inside the `/train` endpoint); leaking raw internals in error details.
- **Minimal example (project-consistent):**
  ```python
  @router.post("/models/{model_id}/promote")
  def promote_model(model_id: str, req: PromoteModelRequest,
                    db: Session = Depends(get_db),
                    _token: str = Depends(require_api_token)):
      try:
          return ModelRegistry(db).promote_candidate(model_id, actor=req.actor)
      except ModelGovernanceError as e:
          raise HTTPException(status_code=409, detail=str(e))
  ```
- **Official docs:** https://fastapi.tiangolo.com/advanced/events/ and https://fastapi.tiangolo.com/tutorial/dependencies/ (docs track latest; FastAPI 0.143.0 installed, reviewed 2026-10-31)

---

### SQLAlchemy 2.1.4

- **What it does:** ORM with 2.0-style declarative mapping, typed columns, and `Session` unit-of-work persistence.
- **Where in this project:** `backend/app/db/base.py` (`Base(DeclarativeBase)`, `TimestampMixin`, `utc_now_naive`), `backend/app/db/session.py` (engine, `SessionLocal`, `get_db`), `backend/app/db/init_db.py` (`create_all` + safety seeding), `backend/app/models/ml_model.py` (`MLModelRecord`).
- **Essential APIs:**
  - `class Base(DeclarativeBase)` + `Mapped[...]` / `mapped_column(...)` for typed declarations.
  - `create_engine(url, connect_args=..., pool_pre_ping=True, echo=DEBUG)`.
  - `sessionmaker(bind=engine, autocommit=False, autoflush=False)`.
  - `Session.add / commit / rollback / close / refresh`, `db.query(Model).filter_by(...)` (legacy 1.x Query API, still supported).
  - `event.listens_for(engine, "connect")` for SQLite pragmas (`PRAGMA foreign_keys=ON`, `journal_mode=WAL`).
  - `Base.metadata.create_all(bind=engine)` — missing-tables only; ALTERs are owned by Alembic.
- **Best patterns (as implemented):** one module-level engine + `sessionmaker`; per-request session via `get_db` yield-dependency; services commit explicitly and `promote_candidate` wraps `commit()` in `try/except` with `rollback()` (flush failure leaves the Session in an "inactive" state until `rollback()` is called — per docs); `check_same_thread=False` for SQLite plus WAL for reliability; timestamps stored naive-UTC by contract (`utc_now_naive`), the dashboard normalizes to ISO.
- **Anti-patterns / common mistakes:** `autocommit=False` in `sessionmaker(...)` is accepted only as a backwards-compat no-op in 2.1 (source: `session.py` line 1706: "the 'autocommit' keyword is present for backwards compatibility"); `autocommit=True` raises. Sessions are not thread-safe — one per thread/request; `db.query(...)` is the legacy API while 2.0 recommends `select()`; forgetting `rollback()` after a failed flush makes the session unusable.
- **Minimal example (project-consistent):**
  ```python
  class MLModelRecord(Base):
      __tablename__ = "ml_models"
      model_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
      status: Mapped[str] = mapped_column(String(32), default="candidate", index=True)

  db = SessionLocal()
  try:
      rec = db.query(MLModelRecord).filter_by(model_id=cid).first()
      rec.status = "active"
      db.commit()
  except Exception:
      db.rollback()
      raise
  finally:
      db.close()
  ```
- **Official docs:** https://docs.sqlalchemy.org/en/21/orm/session_basics.html and https://docs.sqlalchemy.org/en/21/orm/quickstart.html (2.1 series, reviewed 2026-10-31)

---

### Alembic 1.20.0

- **What it does:** schema migration tool: revision chain, autogenerate, online (live DB) and offline (`--sql` script) modes.
- **Where in this project:** `alembic.ini` (repo root), `migrations/env.py`, `migrations/versions/` (9 dated revisions 2026-10-10 … 2026-10-31).
- **Essential APIs:**
  - `alembic upgrade head` / `alembic revision --autogenerate -m "..."`.
  - `script_location`, `version_locations`, `path_separator` ini keys.
  - `context.is_offline_mode()`, `run_migrations_offline()` (literal binds, no DB connection) vs `run_migrations_online()` (`engine_from_config`, NullPool) in `env.py`.
- **Best patterns (as implemented):** `%(here)s` pins `script_location = %(here)s/migrations` and `version_locations = %(here)s/migrations/versions`, making both CWD-independent. **The past CRITICAL fix is in place and extended:** the empty/CWD-relative version-location problem is fixed — `version_locations` is explicitly set with `%(here)s`, and `path_separator = os` is set explicitly so alembic>=1.13 does not warn. `sqlalchemy.url` in the ini is an intentional **empty placeholder**: `env.py::get_url()` reads only the `ALEMBIC_DATABASE_URL` env var and raises `RuntimeError` when unset (H4-incident class prevention); it never falls back to `settings.DATABASE_URL` and never echoes the URL. `init_db()` only does `create_all` for missing tables; schema drift on existing tables is owned by the migration chain (boot never ALTERs).
- **Anti-patterns / common mistakes:** CWD-relative `script_location` silently finds zero revisions ("upgrade head" succeeds doing nothing); falling back to an app-settings URL lets a stray migration hit a wrong database; offline (`--sql`) mode cannot run per-row SELECT logic and assumes transactional DDL; mixing `create_all` ALTERs with the migration chain.
- **Minimal example (project-consistent):**
  ```bash
  export ALEMBIC_DATABASE_URL='sqlite:///G:/mash/adaptive_trader.db'
  alembic upgrade head   # fails closed if the env var is unset
  ```
- **Official docs:** https://alembic.sqlalchemy.org/en/latest/tutorial.html and https://alembic.sqlalchemy.org/en/latest/offline.html (latest ≈ 1.20, reviewed 2026-10-31)

---

### Pydantic 2.13.4 / pydantic-settings 2.14.2

- **What it does:** typed, validated configuration (`BaseSettings` reads env vars → `.env` → defaults, with validators).
- **Where in this project:** `backend/app/core/config.py` (Settings, ExecutionMode, validators, token bootstrap), `backend/app/core/metrics_integrity.py` (`METRIC_SIGNING_KEY` usage), `backend/app/core/errors.py` (`LiveTradingProhibitedError`).
- **Essential APIs:**
  - `SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore")`.
  - `@field_validator("EXECUTION_MODE", mode="before")` + `@classmethod` — raw-string gate before enum coercion.
  - `field_validator(...)` (after-mode) for numeric sanity (`math.isfinite`, bounds).
  - `BaseSettings` precedence: init kwargs > env vars > dotenv > secrets_dir.
- **Best patterns (as implemented):** fail-closed design — the `ExecutionMode` enum omits `LIVE_TRADING`; the before-validator rejects any string containing `LIVE`/`REAL` with `LiveTradingProhibitedError` before Pydantic coerces to the enum (matches the docs' before-validator pattern of normalizing/rejecting raw input); `MT5_SERVER_TIME_OFFSET_HOURS` validator rejects NaN/non-finite explicitly (NaN bypasses every `>`/`<` comparison) and bounds `|offset| <= 12h`; empty `API_TOKEN` falls back to a random per-process `secrets.token_urlsafe(32)` so no guessable default exists; `METRIC_SIGNING_KEY` is read from `os.environ` first, then the Settings field, because pydantic-settings loads `.env` into the model but does **not** inject it into `os.environ` (documented in `metrics_integrity.py`).
- **Anti-patterns / common mistakes:** forgetting `@classmethod` under `@field_validator` or forgetting to `return` the value; using after-mode validators that need raw strings; trusting `abs(v) > limit` to catch NaN; assuming `case_sensitive=True` affects `os.environ` on Windows — per docs it is normalized to uppercase there (it still applies to dotenv values), so this project's `case_sensitive=True` is only fully effective for `.env`; `extra="ignore"` deviates from the docs default `'forbid'` (chosen deliberately; unknown env vars are always ignored anyway, only dotenv extras differ).
- **Minimal example (project-consistent):**
  ```python
  class ExecutionMode(str, Enum):
      BACKTEST = "BACKTEST"; SIGNAL_ONLY = "SIGNAL_ONLY"; DEMO_TRADING = "DEMO_TRADING"

  @field_validator("EXECUTION_MODE", mode="before")
  @classmethod
  def validate_execution_mode(cls, v):
      s = str(v).strip().upper()
      if "LIVE" in s or "REAL" in s:
          raise LiveTradingProhibitedError(f"'{s}' is forbidden.")
      return ExecutionMode(s)
  ```
- **Official docs:** https://docs.pydantic.dev/latest/concepts/pydantic_settings/ and https://docs.pydantic.dev/latest/concepts/validators/ (redirect to pydantic.dev docs; latest ≈ 2.13/2.14, reviewed 2026-10-31)

---

### MetaTrader5 5.0.6231

- **What it does:** Python bridge to a running MT5 terminal via IPC: initialize/shutdown, quotes, bars, account info, deal history, margin calc.
- **Where in this project:** `backend/app/services/market_data/mt5_provider.py` (sole wrapper), `backend/app/api/v1/endpoints/ml.py` (short-lived connect→fetch→disconnect), `backend/app/core/config.py` (`MT5_PATH`, `MT5_SERVER`, `MT5_SERVER_TIME_OFFSET_HOURS`).
- **Essential APIs (all called under `_mt5_lock`, a module-level `threading.RLock`):**
  - `mt5.initialize(path=...)` / `mt5.shutdown()` — connection lifecycle, inside `connect()`/`disconnect()`.
  - `mt5.terminal_info()` — liveness check (`term.connected`).
  - `mt5.account_info()` — account details; server name gates the broker-clock offset; `trade_mode` checked against `MT5_ACCOUNT_TRADE_MODE_DEMO`.
  - `mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)` — bars from an index; `start_pos=1` in `get_historical_dataframe` returns closed bars only.
  - `mt5.symbol_info(symbol)` / `mt5.symbol_info_tick(symbol)` — specs and last tick.
  - `mt5.history_deals_get(from_dt, to_dt)` — daily-loss reconstruction from deal cash flows.
  - `mt5.order_calc_margin(...)` — read-only margin estimate (no order placement anywhere in the provider).
  - `mt5.last_error()` — error detail on every failure path.
- **Best patterns (as implemented):** every C-extension call is serialized by one re-entrant lock; timestamps converted to tz-aware UTC with the server-gated measured offset and rejected on ambiguity (raw stamp AND corrected stamp both fresh ⇒ refuse); staleness/finiteness checks fail closed (return `None`/`[]` instead of guessing); deal history with unparseable or out-of-window stamps aborts daily-loss reconstruction; provider is read-only — the safety posture (`SIGNAL_ONLY`, no orders) is architectural.
- **Anti-patterns / common mistakes:** calling MT5 functions from multiple threads without a lock; using `start_pos=0` for training data (includes the currently open bar); interpreting raw MT5 epoch as UTC unconditionally for a broker that stamps server-time; keeping the terminal connection open for the whole request; trusting `None` returns without checking `last_error()`.
- **Minimal example (project-consistent):**
  ```python
  with _mt5_lock:                       # module-level threading.RLock
      if not mt5.initialize(path=settings.MT5_PATH):
          return False
      try:
          rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 1, 500)
      finally:
          mt5.shutdown()
  ```
- **Official docs:** https://www.mql5.com/en/docs/python_metatrader5 (5.0.x, reviewed 2026-10-31). The docs page documents initialize/shutdown, `copy_rates_from_pos(symbol, timeframe, start_pos, count)`, account_info, symbol_info_tick, and states the package works "via interprocessor communication directly from the MetaTrader 5 terminal." It does **not** document threading/thread-safety constraints — the RLock here is a project-level guarantee; the process-global single-terminal nature is implied by the IPC design (UNVERIFIED as an explicit docs statement).

---

## Model registry & governance (project-specific)

Files: `backend/app/services/ml/registry.py` (806 lines), `backend/app/services/ml/trainer.py`, `backend/app/core/metrics_integrity.py`, `backend/app/models/ml_model.py`, endpoint `backend/app/api/v1/endpoints/ml.py`.

**Status lifecycle (actual):** `candidate` (set by trainer; never auto-active) → `active` (only via `promote_candidate`) → `archived` (previous active, set during a successful promotion). `rejected` exists as a status value and is *blocked* from promotion, but **no registry method ever sets it** — the only lifecycle writer is `attach_*` (candidate-only) and `promote_candidate`.

**Registration (trainer):** `train_candidate()` trains on a purged/embargoed dataset split, validates two-class finite partitions, computes train/validation/test metrics + calibration curve, writes the joblib artifact, and inserts `MLModelRecord(status="candidate")` with `metrics_digest` attested at registration. Artifact payload keys: `model_id, algorithm, model, scaler, feature_names, hyperparameters, dataset_version_tag, validation_metrics, test_metrics, metrics_digest` (walk-forward metrics are added later by attach; the artifact is then rewritten to stay in sync with the store).

**Digests (H1/M8):** `metric_digest(model_id, validation_metrics, test_metrics, walk_forward_metrics)` = HMAC-SHA256 when `METRIC_SIGNING_KEY` is set, else plain SHA-256 (corruption detection only, not tamper-evident). Canonical JSON with sorted keys. A separate `ml_walk_forward_digest` attests the ML walk-forward payload over the digest-stripped payload. Attach paths re-attest the digest and rewrite the artifact; the store and artifact must always move together.

**Promotion gates (all must pass, in `promote_candidate`):**
1. Status is `candidate` (rejected is refused outright).
2. Stored `metrics_digest` present and `verify_metric_digest` passes (tamper detection).
3. Artifact containment: `Path.resolve()` + `relative_to(settings.ARTIFACTS_MODELS_DIR)` and `.joblib` suffix (blocks arbitrary-file load and arbitrary-file write in attach).
4. Metric gates: ROC-AUC > 0.52 and Brier ≤ 0.25 on **both** validation and independent test partitions; ≥ 30 samples each.
5. Strategy walk-forward gate: ≥ 3 folds, ≥ 30 trades, finite metrics, net_profit > 0, profit factor > 1, max drawdown within `settings.MAX_DRAWDOWN_PERCENT` (6.0), and all three cost flags (`spread/slippage/commission_costs_included`) true.
6. Backtest provenance: `backtest_run_id` must exist; claimed numbers must match the stored `BacktestRun` (`math.isclose`, 1e-9); folds chronological; per-fold values re-aggregate to the stored run; `dataset_fingerprint.sha256` present; cost flags cross-checked against the run's persisted `engine_config` (a fabricated `True` cannot beat a zero-cost run).
7. ML walk-forward evidence (when attached): ≥ 3 retrained folds, independent structural re-verification, digest re-verified; replay guard refuses re-signing a digest belonging to another model.
8. Active-model comparison: candidate validation ROC-AUC must strictly exceed the active model's validation ROC-AUC (active must itself carry a valid re-evaluated validation partition; test values are reserved for absolute gates to avoid tuning against test).
9. Artifact load + inference contract: payload is a dict with a callable `model.predict_proba` and non-empty `feature_names`; embedded `metrics_digest` must equal the store digest; embedded validation/test metrics must equal store values.

**Human approval point:** promotion is only reachable via `POST /api/v1/models/{model_id}/promote` behind `require_api_token`, carrying `actor`/`notes`; there is no scheduled or automatic promotion path. `log_security_event("MODEL_PROMOTED", ...)` and `"ML_WALK_FORWARD_EVIDENCE_ATTACHED"` write the audit trail (`AuditLog` retained by default — retention windows default to 0 = never pruned).

**Artifact loading safety:** `load_active_model_artifact()` and all attach/promote paths restrict `joblib.load` to resolved paths inside `ARTIFACTS_MODELS_DIR` with `.joblib` suffix and a structural payload check plus digest cross-check. Trust model: joblib/pickle executes code on load, so only trusted locally-produced artifacts in the governed directory are ever loaded — never user uploads or remote sources.

**Reproducibility:** `dataset_version_tag`, `hyperparameters`, and `features_used` are stored in both the DB record and the artifact payload; target is `target_tp_before_sl`; walk-forward evidence binds evaluations to a dataset fingerprint (SHA-256) and to exact persisted fold records.

---

## Provenance

- All project facts read directly from source files at HEAD 48d7d43, branch `reveo`, on 2026-10-31: `config.py`, `registry.py`, `trainer.py`, `base.py`, `session.py`, `init_db.py`, `main.py`, `ml.py`, `mt5_provider.py`, `metrics_integrity.py`, `ml_model.py`, `alembic.ini`, `migrations/env.py`, migration filenames in `migrations/versions/`.
- The `sessionmaker(autocommit=False)` legacy-kwarg finding verified against the installed source: `<OWNER_HOME>\AppData\Local\Programs\Python\Python314\Lib\site-packages\sqlalchemy\orm\session.py` (line ~1706 docstring, ~1803-1809 logic).
- Sibling-agent facts reused without re-verification (flagged as REPORTED): RLock + `copy_rates_from_pos(start_pos=1)` + `MT5_SERVER_TIME_OFFSET_HOURS=3.0` fail-closed ambiguity rejection (independently visible in this file's reads); walk-forward purge 24 / embargo 12 with per-fold retraining in `backend/app/services/ml/walk_forward.py` (`MLWF_EVIDENCE_TYPE = "ml_walk_forward"`, `MLWF_THRESHOLD = 0.5` confirmed by grep); calibration measured (Brier/LogLoss) but no calibrator fitted.
- Docs fetched 2026-10-31 via WebFetch (https, official domains): fastapi.tiangolo.com (events, dependencies), docs.sqlalchemy.org/en/21 (session_basics, quickstart), alembic.sqlalchemy.org/en/latest (tutorial, offline), docs.pydantic.dev → pydantic.dev (pydantic_settings, validators), mql5.com python_metatrader5. No doc text copied verbatim; summaries are original. Context7 MCP was NOT available in this session — no claim of its usage is made.

## Gaps / open questions

1. `status="rejected"` is checked by `promote_candidate` but no code path sets it — rejection is currently only reachable by direct DB writes. Intentional workflow hole or missing feature (UNRESOLVED).
2. `MT5_SERVER_TIME_OFFSET_HOURS = 3.0` is a measured offset for server `YourBroker-MT5` (config comment: verified live 2026-10-28). The ~3h MT5 timestamp question remains documented as UNRESOLVED in the data-engineering reference; here it is a config fact, not a general solution.
3. MT5 Python threading constraints are not stated in the mql5.com docs page; the RLock is a project guarantee, and "MT5 package is process-global" is an inference from the IPC design (UNVERIFIED as official documentation).
4. Attach paths (`attach_walk_forward_metrics`, `attach_ml_walk_forward_metrics`) `joblib.load` and rewrite the artifact without the payload-structure validation used elsewhere (only `isinstance(dict)`), and their `db.commit()` lacks the try/rollback framing that `promote_candidate` has — a failed commit mid-rewrite could leave the artifact and store momentarily inconsistent.
5. `metrics_digest` degrades to plain SHA-256 (corruption detection, not tamper-evidence) whenever `METRIC_SIGNING_KEY` is unset; `.env`-only values are honored via the Settings fallback, but production tamper-evidence requires the env key. Whether the operator has set it in the runtime environment is UNRESOLVED from this read-only pass.
6. The trainer's registration digest covers `walk_forward_metrics={}` (empty); a candidate can only pass promotion after `attach_walk_forward_metrics` re-attests — direct registration with fabricated walk-forward numbers is blocked by provenance gate 6, by design.
