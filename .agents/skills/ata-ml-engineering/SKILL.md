---
name: ata-ml-engineering
description: ML engineering expertise for the Adaptive Trader AI repo — model training/evaluation, time-series validation (walk-forward, purging, embargo, triple-barrier), pandas 3.0 data handling, model registry governance, and dependency evaluation discipline. Use when doing ANY ML, dataset, backtest-evaluation, or model-governance work in this repository, or when evaluating whether to add an ML library.
---

# ATA ML Engineering

Project-grounded ML knowledge for Adaptive Trader AI (XAUUSD M1 research system).
Everything here was verified against actual source at `reveo` @ 48d7d43 and official vendor docs
(reviewed 2026-10-31). Full detail lives in `references/` — read the relevant file before non-trivial work.

## Non-negotiables (violating any of these is a defect, not a style choice)

1. **Trading safety:** `EXECUTION_MODE=SIGNAL_ONLY`, `DEMO_TRADING_ENABLED=False`. Never place orders,
   never touch these settings, never call `mt5.order_send` outside the gated demo executor.
2. **No new dependency** without demonstrated need AND explicit user approval (see `references/`
   candidate verdicts; the default answer is "defer" — check §3 of `docs/engineering/ml-library-map.md`).
3. **Models are born `candidate`** — promotion to `active` only through
   `POST /models/{id}/promote` with token + actor + notes; never auto-promote, never hand-edit status.
4. **Fail closed:** missing ATR column, one-class training partition, non-finite features, ambiguous
   MT5 time offset, missing `ALEMBIC_DATABASE_URL` — all raise. Keep it that way; no silent fallbacks,
   no synthetic data substitutes.
5. **Never claim test success without running the tests.** Record versions/URLs/dates for every doc
   claim; never copy vendor docs verbatim — summarize with references.
6. **pandas 3.0 semantics:** Copy-on-Write default — chained assignment is silently ineffective; use
   `.loc`/`.copy()`. Default `str` dtype; default `datetime64[us]`.

## Environment (VERIFIED 2026-10-31, `requirements-lock.txt` + live probe)

Python 3.14.4, Windows x64 · scikit-learn 1.9.1 · joblib 1.6.0 · numpy 2.5.3 · pandas 3.0.6 ·
scipy 1.18.1 · fastapi 0.143.0 · sqlalchemy 2.1.4 · alembic 1.20.0 · pydantic 2.13.4 /
pydantic-settings 2.14.2 · pymupdf 1.28.2 (`import pymupdf`, never `fitz`) · openpyxl 3.1.5 ·
MetaTrader5 5.0.6231.
**NOT installed (candidates only):** statsmodels, optuna, xgboost, lightgbm, catboost, sktime, river,
imbalanced-learn, shap, skforecast. Note: skforecast pins `pandas < 3.0` → currently incompatible.

## Decision guides

### Choosing / changing a model
- Order of adoption: `baseline` → `logistic_regression` → `random_forest`. Anything beyond requires a
  documented plateau of the current stack + user approval (PROJECT_MEMORY governance rule).
- Baseline comparison is mandatory context for every candidate metric; never report a candidate
  metric without the baseline metric next to it.
- Imbalance: today only `class_weight="balanced"` on LR (RF unweighted). Before reaching for
  resampling libraries, log `balanced_accuracy` + `average_precision` — zero new dependencies.
- Calibration: currently measured (Brier/LogLoss + 5-bin curve), never fitted. Any calibrator
  (sigmoid for small samples, isotonic for ~1000+) must fit INSIDE walk-forward folds; owner approval first.

### Time-series validation (the core discipline)
- Walk-forward is implemented in `services/ml/walk_forward.py`: **purge = 24** (≥ label horizon,
  enforced), **embargo = 12** (`horizon // 2`), **retrain every fold** (fresh model + baseline +
  scaler), fixed 0.5 threshold, no OOS threshold tuning. Do not weaken any of these.
- The static chronological split in `services/dataset/pipeline.py` purges but has no separate embargo
  — if you touch that file, flag this gap to the owner rather than "fixing" silently.
- Every new feature must be computed with information available at bar close i only. Any use of bar
  i's high/low/close for features consumed at bar i's open, or rolling windows that include the
  current bar, is look-ahead bias.
- Labeling is `label_buy_barrier` (BUY-only): entry at ask with historical per-bar spread, ATR-based
  TP/SL, conservative same-bar collision → 0. A "SELL-direction" model is a DIFFERENT target than a
  "SELL barrier-success" model — see `references/timeseries-financial-ml.md` before designing one.
- Evaluation is per-metric and must stay cost-aware: spread, per-side commission, slippage
  (`CENTS_AT_EVENT`), sizing cap 1.5×. Signals execute at open of bar i+1.

### Model governance
- Read `references/backend-model-infra.md` §Model registry before touching `ml/registry.py`.
  9 promotion gates (digest verification, path containment, ROC-AUC > 0.52 / Brier ≤ 0.25 both
  partitions ≥30 samples, walk-forward ≥3 folds ≥30 trades PF>1 DD≤6% + 3 cost flags, provenance,
  dataset fingerprint, comparison vs active). `metrics_digest` is HMAC only when
  `METRIC_SIGNING_KEY` is set — otherwise corruption-detection only.
- Walk-forward attach has no HTTP surface by design; do not add one without an owner decision.

### Fetching library documentation
- Check the installed version in `requirements-lock.txt` FIRST, then fetch official docs and record
  URL + version + review date in whatever you produce. If a page doesn't confirm py3.14/Windows
  support, write UNVERIFIED — never guess. Context7 MCP may be unavailable; if so, say so and fetch
  vendor docs directly (https, official domains). Discovery of candidate libraries starts at
  https://github.com/josephmisiti/awesome-machine-learning, but API facts never come from it.

## References (read the one you need)

- [`references/ml-algorithms-evaluation.md`](references/ml-algorithms-evaluation.md) — sklearn 1.9.1,
  joblib, and candidate evaluations (Optuna, XGBoost, LightGBM, CatBoost, imbalanced-learn); the
  evaluation contract of `ml/evaluator.py`; BUY/SELL evaluation asymmetry.
- [`references/timeseries-financial-ml.md`](references/timeseries-financial-ml.md) — walk-forward,
  purge/embargo, leakage vectors, triple-barrier (actual semantics), BUY-success vs SELL-direction,
  candidates (statsmodels, sktime, River, skforecast), PSR/DSR concepts.
- [`references/data-engineering.md`](references/data-engineering.md) — numpy 2.5 / pandas 3.0 (CoW,
  str dtype), scipy, pymupdf 1.28 (untrusted-PDF guards), openpyxl, MT5 frame contract.
- [`references/backend-model-infra.md`](references/backend-model-infra.md) — FastAPI/SQLAlchemy/
  Alembic/Pydantic patterns as implemented, MT5 provider locking, registry governance deep-dive.
- Map + gaps overview: [`docs/engineering/ml-library-map.md`](../../../docs/engineering/ml-library-map.md)

## Known open issues (do not re-derive; from `docs/PROJECT_MEMORY.md` + reference gaps)

Static split lacks separate embargo · calibration never fitted · py3.14 wheels UNVERIFIED for all
candidates · MT5 ~3h timestamp UNRESOLVED at evidence level · ML-WF attach self-attestation residual ·
`METRIC_SIGNING_KEY` ops provisioning · historical baseline numbers (AUC ≈ 0.50) REPORTED not VERIFIED.
