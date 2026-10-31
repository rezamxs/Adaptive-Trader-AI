# ML Library Map — Adaptive Trader AI

> **Purpose:** Single entry point for ML engineering knowledge in this repo. Per-library depth lives in
> `.agents/skills/ata-ml-engineering/references/` — this file is the coordinator-level map.
> **Built:** 2026-10-31. Method: 4 independent research sub-agents (ML/eval, time-series/financial ML,
> data engineering, backend/infra) read the actual source and fetched official vendor docs; the
> coordinator integrated and de-conflicted their outputs. Primary discovery index:
> https://github.com/josephmisiti/awesome-machine-learning (used to locate canonical libraries; API
> facts below come from official docs and this repo's code, never from the index itself).

## 1. Environment ground truth (VERIFIED 2026-10-31)

Python 3.14.4, Windows x64. Versions verified live via `python -I` probe AND `requirements-lock.txt`
(both agree). Any claim in references that contradicts this table is stale.

| Library | Version | Status | Primary project usage |
|---|---|---|---|
| scikit-learn | 1.9.1 | installed | `ml/models.py`, `ml/trainer.py`, `ml/evaluator.py`, `ml/walk_forward.py` |
| joblib | 1.6.0 | installed | artifact serialization (`trainer.py`), model loading (`ml/registry.py`) |
| numpy | 2.5.3 | installed | everywhere (labels, finiteness guards, metrics) |
| pandas | 3.0.6 | installed | dataset pipeline, labeling, MT5 frames — **CoW + str-dtype semantics** |
| scipy | 1.18.1 | installed | transitive base (no direct heavy use yet) |
| fastapi | 0.143.0 | installed | `app/main.py`, `api/v1/endpoints/*` (sync-def endpoints, lifespan) |
| sqlalchemy | 2.1.4 | installed | `db/base.py`, `db/session.py`, models, registry persistence |
| alembic | 1.20.0 | installed | `migrations/` (9 revisions, single head) |
| pydantic / pydantic-settings | 2.13.4 / 2.14.2 | installed | `core/config.py` fail-closed settings |
| pymupdf | 1.28.2 | installed | `knowledge/parsers.py` — canonical `import pymupdf` (no `fitz` in codebase) |
| openpyxl | 3.1.5 | installed | `knowledge/parsers.py` (xlsx) |
| MetaTrader5 | 5.0.6231 | installed | `market_data/mt5_provider.py` (read-only, RLock-wrapped) |
| optuna, xgboost, lightgbm, catboost, sktime, river, statsmodels, imbalanced-learn, shap, skforecast | — | **NOT INSTALLED** | none — candidates only, see §3 |

## 2. Installed-stack facts every agent must know (code-verified)

- **Walk-forward (VERIFIED):** `backend/app/services/ml/walk_forward.py` — purge = 24 bars (must be
  ≥ label horizon, rejects smaller), embargo = 12 (`horizon // 2`), **retrain per fold** (fresh model,
  fresh baseline, fresh scaler), fixed 0.5 threshold, no OOS threshold tuning.
  Test: `backend/tests/unit/test_ml_walk_forward.py:132-140`. Two distinct walk-forwards exist: this
  ML one (retrains) vs the strategy one (`services/backtest/walk_forward.py`, deliberately does NOT
  retrain — rule-based robustness only).
- **Static split gap:** `services/dataset/pipeline.py` chronological split applies purge (2×horizon
  buffering) but **no separate embargo** — embargo exists only in ML walk-forward (UNRESOLVED whether
  intentional; flagged in references).
- **Labeling (VERIFIED):** `services/dataset/labeling.py::label_buy_barrier` is **BUY-only**: entry at
  ask (`close + historical per-bar spread`), TP/SL from ATR with fail-loud ATR requirement,
  conservative same-bar collision → label 0, NaN ATR → label stays NaN. No SELL labeler exists.
- **Evaluation metrics (VERIFIED):** `ml/evaluator.py` — accuracy/precision/recall/f1
  (`zero_division=0`), ROC-AUC, Brier, LogLoss on `proba[:,1]`, manual 5-bin calibration curve.
  Sentinel constants (roc_auc=0.5, brier=1.0) are deliberately gate-failing. `log_loss` is
  informational, read by no gate.
- **Imbalance:** handled only via `class_weight="balanced"` on LogisticRegression; RF is unweighted;
  no resampling; `balanced_accuracy` / `average_precision` are NOT logged (cheap no-dependency next
  step). Whether weights suffice is currently unanswerable.
- **Calibration:** measured (Brier/LogLoss + curve) but **no calibrator is fitted anywhere**; sklearn
  1.9.1 options (sigmoid vs isotonic sample counts, `FrozenEstimator` pattern) documented in
  `ml-algorithms-evaluation.md`. Owner decision pending.
- **Backtest costs (VERIFIED):** signal on bar i → execution at open of bar i+1; SELL exits priced on
  ask-shifted OHLC; commission per-side; `CENTS_AT_EVENT` contract; position sizing hard cap 1.5×.
- **Registry governance (VERIFIED):** lifecycle `candidate → active → archived`; promotion ONLY via
  `POST /models/{id}/promote` (`require_api_token`, `actor`/`notes`); 9 gates including digest
  verification, artifact-path containment (`ARTIFACTS_MODELS_DIR`, `.joblib` only), ROC-AUC > 0.52 and
  Brier ≤ 0.25 on BOTH partitions (≥30 samples each), walk-forward evidence (≥3 folds, ≥30 trades,
  PF > 1, DD ≤ 6%, three cost flags), provenance `backtest_run_id`, dataset fingerprint, and
  validation-AUC comparison vs the active model. `metrics_digest` = HMAC-SHA256 when
  `METRIC_SIGNING_KEY` is set, else plain SHA-256 (corruption-detection only — provisioning gap).
  Walk-forward attach endpoints have **no HTTP surface**; the chain needs an in-process step before
  promotion (owner decision documented in PROJECT_MEMORY §27–28).
- **MT5 provider (VERIFIED):** every C-extension call under a module-level `threading.RLock`;
  read-only (no order placement; only `order_calc_margin`); `MT5_SERVER_TIME_OFFSET_HOURS=3.0`
  configured per-server with fail-closed rejection on offset ambiguity; the historical ~3h timestamp
  issue remains UNRESOLVED at evidence level — never claim live-data validation.
- **Migrations (VERIFIED):** `alembic.ini` uses CWD-independent `%(here)s` paths, `path_separator=os`,
  empty `sqlalchemy.url`; `migrations/env.py` raises without `ALEMBIC_DATABASE_URL` (fail-closed).
- **pandas 3.0 semantics:** Copy-on-Write default (chained assignment is silently ineffective),
  default `str` dtype, default `datetime64[us]`. Codebase is CoW-compliant (`.loc` assignment,
  `.copy()` on slices). Do not import 2.x habits.

## 3. Candidate libraries — coordinator verdicts (all NOT INSTALLED)

Full evaluations with trigger criteria: `references/ml-algorithms-evaluation.md` and
`references/timeseries-financial-ml.md`. Summary:

| Candidate | Coordinator verdict (2026-10-31) | Blocking/trigger |
|---|---|---|
| Optuna | Defer. sklearn built-ins (`RandomizedSearchCV`, `HalvingRandomSearchCV`) cover current HPO scale. Trigger: multi-model HPO campaigns with pruning needs. | py3.14 wheel support UNVERIFIED; user approval required |
| XGBoost | Defer. No evidence LR/RF plateaued; wheel availability for Windows x86_64 confirmed in docs (needs VC++ redistributable; `xgboost-cpu` variant) but py3.14 wheels UNVERIFIED. | documented need + user approval |
| LightGBM | Defer. Fetched Windows docs describe source-build only; python-package page 404'd — wheel status open. Weakest install story of the three boosters. | resolve wheel status first |
| CatBoost | Defer. `auto_class_weights` mirrors the existing LR policy; native categorical support not currently needed (features are numeric). | documented need + user approval |
| skforecast | **Incompatible today** — dev docs pin `pandas < 3.0`, repo runs pandas 3.0.6. Revisit only after upstream supports pandas 3.x. | blocking incompatibility |
| statsmodels | Defer. Official docs (0.15.0) list Python 3.9–3.13 — py3.14 unconfirmed. No current need (no ARIMA/cointegration work). | py3.14 support + need |
| sktime | Defer. Project already has a purpose-built walk-forward/purge/embargo pipeline; sktime would duplicate it. | duplication of existing verified code |
| River | Defer. Online/incremental ML has no demonstrated need in batch M1 research flow. | streaming requirement |
| imbalanced-learn | Defer with care. Floor ok (needs sklearn ≥1.4.2), ceiling vs 1.9.1 UNVERIFIED. Cheapest first step is NO dependency: log `balanced_accuracy`/`average_precision`. If later used, SMOTE only INSIDE folds (leakage). | measurement first |
| shap | Defer. Interpretability currently served by calibration curves + per-metric gates; add only when model explanations become a governance requirement. | governance need |

**Rule (from PROJECT_MEMORY):** no new dependency enters without demonstrated need AND explicit user
approval. "Installed in another project" is not a justification.

## 4. BUY-success vs SELL-direction (the asymmetry the owner flagged)

Details: `references/timeseries-financial-ml.md` §BUY-success vs SELL-direction and
`references/ml-algorithms-evaluation.md` §Evaluation asymmetry. Core points:

- A **direction classifier** (P(price down)) is NOT a **barrier-success classifier** (P(TP-before-SL
  for a short trade)). Different target, different base rates, different calibration.
- Only `label_buy_barrier` exists. A SELL side needs mirrored semantics: entry at bid (`close -
  spread`), TP below entry, SL above, same conservative collision rule, same historical per-bar spread.
- Evaluation must become per-side regardless: per-side thresholds and per-side metric reporting —
  a single global ROC-AUC hides opposite-side behavior.

## 5. Knowledge gaps (compiled from all four agents; full lists in each reference)

1. Python 3.14 wheel support UNVERIFIED for every candidate library (statsmodels docs cap at 3.13;
   River and LightGBM pages 404'd during review).
2. Static-split embargo absence — intentional design or gap? UNRESOLVED.
3. Calibration is measured but never fitted — owner decision required before any calibrator enters
   the walk-forward loop.
4. `balanced_accuracy` / `average_precision` not logged — cheapest improvement, zero dependencies.
5. ML walk-forward evidence is self-attesting at attach (no binding to a persisted run) — known
   PROJECT_MEMORY residual.
6. MT5 ~3h timestamp matter UNRESOLVED at evidence level (code has configured offset + fail-closed).
7. `mql5.com/en/docs/python_metatrader5/copy_rates_from` page 404'd during review; the 8-field output
   contract was verified from the mql5 overview page + project code, exact dtypes UNVERIFIED.
8. imbalanced-learn upper bound vs sklearn 1.9.1 UNVERIFIED.
9. `METRIC_SIGNING_KEY` provisioning in operations UNRESOLVED (without it digests are not
   tamper-evident).
10. Historical baseline numbers in `docs/core-model-baseline.md` (AUC ≈ 0.50) come from an older
    snapshot (master @ 2f2ed9e) — REPORTED, never reproduced on current HEAD.

## 6. Provenance

- Primary reference index: https://github.com/josephmisiti/awesome-machine-learning (reviewed 2026-10-31;
  raw README cached at `~/.tmp-aml-ref/aml-readme.md`).
- Official docs fetched per library with URL + version + review date: see `## Provenance` in each
  reference file (`ml-algorithms-evaluation.md`, `timeseries-financial-ml.md`, `data-engineering.md`,
  `backend-model-infra.md`).
- All code facts verified by reading source at `reveo` @ 48d7d43 on 2026-10-31.
- Context7 MCP was NOT available in this session; vendor docs were fetched directly via WebFetch.
  No usage of Context7 is claimed anywhere in this knowledge base.
- No doc text was copied verbatim; all content is original technical summary with references.
