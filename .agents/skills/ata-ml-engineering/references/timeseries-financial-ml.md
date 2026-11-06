# Time-Series & Financial ML — Project Reference (Adaptive Trader AI)

Scope: knowledge reference for the core time-series/financial-ML machinery of this repo.
All file references verified against branch `reveo`, HEAD `48d7d43`, on 2026-10-31.
Environment (VERIFIED via requirements-lock + live probe): scikit-learn 1.9.1,
numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, joblib 1.6.0 on Python 3.14.4 / Windows x64.
Governance rule: complex/new methods must NOT be added without demonstrated need.

## Core concepts (project-grounded)

### 1. Walk-forward validation & time-series CV (incl. retrain-per-fold)
- Standard k-fold CV randomly shuffles rows; on time-ordered financial data this trains on the
  future and tests on the past. Any chronological split (single holdout or folds) is mandatory.
- Two DISTINCT walk-forwards exist in this codebase (do not confuse them):
  1) Strategy walk-forward (`backend/app/services/backtest/walk_forward.py`): runs the approved-rule
     strategy through `RealisticBacktestEngine` with real costs across chronological OOS folds. It
     deliberately does NOT retrain the ML model per fold; each fold is a persisted `BacktestRun`
     whose provenance block is re-aggregated at attach/promotion.
  2) ML walk-forward (`backend/app/services/ml/walk_forward.py`, "M8"): TRUE per-fold retraining —
     VERIFIED: `run_ml_walk_forward` fits a FRESH model, a FRESH `BaselineModel(strategy="prior")`,
     and a FRESH `StandardScaler` on every fold's training partition (walk_forward.py:228-259),
     then evaluates only on that fold's later OOS window. A real pytest asserts
     `purge_window == 24`, `embargo_bars == 12`, and per-fold
     `train_end_pos + 24 + 12 <= test_start_pos` (test_ml_walk_forward.py:132-140).
- scikit-learn equivalent: `TimeSeriesSplit(n_splits, test_size, max_train_size, gap)` (docs 1.9.1).
  Default is an expanding train window; `max_train_size` converts it to sliding; `gap` = samples
  excluded from the end of each train set before the test set — sklearn's built-in purge/embargo
  knob. The project's `build_ml_walk_forward_folds` (walk_forward.py:92-141) reimplements this by
  hand because it also needs: expanding-prefix training `[0, test_start - purge - embargo)`,
  contiguous non-overlapping OOS windows (last fold absorbs the remainder), and a hard refusal of
  zero-gap geometry (walk_forward.py:111-112).
- Common mistakes: averaging per-fold metrics without recording variance (project stores mean/std,
  walk_forward.py:144-150, 277-288); tuning the threshold on OOS (project pins `MLWF_THRESHOLD = 0.5`,
  walk_forward.py:58, 307-308); conflating the strategy WF (no retraining) with the model WF.
- What "good" looks like: fresh scaler per fold, expanding chronological windows, baseline run on
  the SAME folds so uplift is attributable, single-class fold raises (walk_forward.py:242-247).

### 2. Purging & embargo
- Definitions: PURGE removes training samples whose label outcome window overlaps the OOS period
  (labels look forward, so a train row near the boundary "knows" test-period outcomes). EMBARGO
  adds an extra information gap after the purge buffer so boundary features cannot straddle it
  (López de Prado convention; adopted in the WF module docstring, walk_forward.py:11-18).
- ML walk-forward (VERIFIED): `purge_window` defaults to `horizon_bars` (24) and is fail-loud
  rejected if `< horizon_bars` (walk_forward.py:189-198). `embargo_bars` defaults to
  `max(1, horizon_bars // 2)` = 12 at horizon 24 (walk_forward.py:190). Training cutoff:
  `train_end = test_start - purge - embargo`.
- Static pipeline split (pipeline.py:94-121): single train/val/test with TWO purge buffers of
  `horizon_bars` (train|purge|val|purge|test). NO separate embargo here — purge-only; see Gaps
  for the open question about the asymmetric embargo policy between pipeline and WF.
- TimeSeriesSplit's `gap` maps to `purge + embargo` combined; sklearn does not distinguish the roles.
- Common mistakes: purge measured on cleaned-frame positions while labels were defined on raw
  positions (project builds folds on the CLEANED frame, walk_forward.py:213-222); forgetting the
  NaN label tail; zero-gap CV silently "passing" because leakage is undetectable after the fact
  (hence the explicit refusal, walk_forward.py:111-112).
- What "good" looks like: purge >= label horizon (enforced), embargo as belt-and-suspenders,
  geometry re-verifiable from stored positions (verify_ml_walk_forward_payload recomputes geometry
  and aggregates; HMAC/SHA-256 digest guards tampering, walk_forward.py:333-457).

### 3. Data leakage & look-ahead bias
- Concrete leak vectors and their guards in this codebase (all VERIFIED in code/tests):
  - Forward-looking features: `FeatureEngineer` is "strict zero look-ahead" (features.py:1-2;
    backward shifts only). Guard: `test_anti_lookahead_bias_guarantee` mutates 20 future bars and
    asserts every earlier-bar feature is identical (test_features_and_leakage.py:11-34).
  - Label leakage across split boundary: purge guards (pipeline.py:98-115; walk_forward.py:191-198);
    the WF verifier re-checks `train_end + purge + embargo <= test_start` (walk_forward.py:390-395).
  - Scaler leakage: `StandardScaler` fit ONLY on train (pipeline.py:124-127) and per-fold in WF
    (walk_forward.py:236-238); the pipeline test asserts train-only fitting.
  - Threshold leakage: fixed 0.5 in WF; protocol forbids OOS threshold search (protocol doc section 4).
  - Entry-timing leakage: signal on bar i fills at the OPEN of bar i+1 (engine.py:275-291).
  - Silent-cost leakage: labeler pays historical per-bar spread into the entry price
    (labeling.py:50-63); a current-day spread fallback would misprice all history (labeling.py:48-49).
  - Dirty-data leakage: fail-loud OHLC sanity (pipeline.py:50-58, walk_forward.py:74-89),
    ATR-column ValueError (labeling.py:38-45, engine.py:125-132), inf→NaN + dropna (pipeline.py:84-88).
  - Outcome-only optimism (equity curve): floating marks use a documented conservative convention
    (favorable extreme booked before adverse in-bar; TP/SL caps the respective excursion) because
    OHLC cannot prove intra-bar tick order (engine.py:199-239).
- What "good" looks like: leakage guards expressed as executable tests, not comments; fail-loud
  ValueError over silent fallbacks (both labeler and engine refuse hardcoded ATR fallbacks with an
  XAUUSD-sub-tick rationale, labeling.py:39-44, engine.py:126-131).

### 4. Triple-barrier labeling (actual `label_buy_barrier` behavior)
- Concept: for each event bar, set three barriers — profit barrier, stop barrier, and a time
  barrier (horizon). The label is which barrier is touched first. This repo implements the BUY
  side only (`backend/app/services/dataset/labeling.py`, class `BarrierLabeler`).
- Exact semantics (VERIFIED, labeling.py):
  - BUY-only; entry price = `close[i] + bar_spread * point_value` (pay the ask) where `bar_spread`
    comes from the historical `spread` column when present (numeric-coerced; non-finite/negative
    falls back to `spread_points`), else the `spread_points` default (labeling.py:50-63).
  - TP = entry + `tp_atr_mult * ATR[i]`; SL = entry - `sl_atr_mult * ATR[i]`; defaults 1.5x / 1.0x
    ATR_14 (labeling.py:68-69; protocol doc fixes these numbers).
  - Time barrier: scan bars `i+1 .. i+max_horizon_bars` (default 24). First touch wins:
    SL first or TP first → 0/1; BOTH touched in the same bar → conservative collision, label 0,
    stop scanning (labeling.py:80-85) — assumes the adverse path, unknown intra-bar tick order.
  - Horizon expires without TP → label 0 ("missed opportunity" is merged with "loss", i.e. no
    separate abstain class).
  - ATR column required and fail-loud; NaN or non-positive ATR → row skipped, label stays NaN
    (labeling.py:64-66). Last `max_horizon_bars` rows remain NaN by construction (loop bound,
    labeling.py:58). Output Series named `target_tp_before_sl` (labeling.py:95).
  - Parameter validation: `max_horizon_bars >= 1`, finite positive `point_value`,
    non-negative `spread_points` (labeling.py:26-31). Conservative collision is asserted by test
    (test_features_and_leakage.py:36-64).
- Downstream: pipeline drops NaN targets and casts to int (pipeline.py:87-88); symbol-aware cost
  defaults — XAUUSD point 0.01 / spread 17 pts, FX point 0.00001 / spread 15 pts (pipeline.py:65-67).
- Common mistakes: labeling without cost (inflates label base rate); optimistic collision (label 1
  whenever TP was touched at all); reusing today's spread for all history; using OHLC closes only
  (ignoring high/low barrier touches); forgetting the last-horizon NaN tail before splitting.
- What "good" looks like: this implementation — historical spread, conservative collision,
  NaN propagation, fail-loud ATR.

### 5. Class imbalance & probability calibration for barrier labels
- Barrier labels are naturally imbalanced and their base rate depends on barrier geometry: with
  TP = 1.5xATR vs SL = 1.0xATR, a zero-drift random walk hits SL first more often than TP (larger
  distance to TP), so the positive rate is typically well below 50% — accuracy misleads (protocol
  mandates ROC-AUC primary, balanced accuracy / PR-AUC / macro-F1 secondary, protocol doc sections 2, 5).
- Calibration matters because the economic use is probability-thresholded signals: Brier Score and
  Log Loss are protocol secondary metrics; `ModelEvaluator` reports them.
- sklearn 1.9.1 calibration knowledge (docs, summarized): `CalibratedClassifierCV` fits a
  calibrator mapping model scores to P(y=1); `ensemble=True` averages k (classifier, calibrator)
  pairs from cv folds, `ensemble=False` fits one calibrator on cross-val predictions. Sigmoid
  (Platt) assumes a roughly symmetric correction and suits small data; isotonic is non-parametric,
  needs ~1000+ samples, and can introduce probability ties (hurts ROC-AUC ranking). Calibrators
  should be fit on data disjoint from the classifier's training data (`FrozenEstimator` when
  wrapping a fitted model). Caveats directly relevant: the sigmoid assumption breaks under high
  imbalance; isotonic/sigmoid require every class present in each cv split.
- Project status: calibration is MEASURED (Brier, reliability data) but no calibrator is fitted in
  the walk-forward; threshold stays fixed at 0.5 (see Gaps).
- Common mistakes: fitting the calibrator on training predictions (biased toward 0/1); isotonic on
  small folds; comparing Brier scores across different class base rates; treating a well-ranked but
  badly calibrated model as economically equivalent.
- What "good" looks like: calibrator fit on a purged validation partition (never OOS test),
  calibration curve per fold, threshold chosen on train/val only per protocol.

### 6. Baseline comparison discipline
- Protocol baselines (docs/core-model-evaluation-protocol.md section 3): (A) majority-class always
  predictor; (B) constant prior-probability predictor; (C) same model with 24 technical features
  only; (D) same model + Volume Profile features (research-only).
- Implementation: `BaselineModel(strategy="prior")` is run per WF fold on identical windows
  (walk_forward.py:255-259), and the aggregate counts `folds_beating_baseline_roc_auc`
  (walk_forward.py:282-286); the verifier recomputes it (walk_forward.py:425-430).
- Acceptance rules (protocol section 5): OOS ROC-AUC >= ~0.55 (meaningfully above 0.50), balanced
  accuracy > 50%, no temporal collapse across folds, positive net return after full spread +
  commission when signals are used in backtest. Rejection rule: AUC stuck in the 0.48-0.52 noise
  band → model rejected for promotion.
- Prior-phase evidence (docs/core-model-baseline.md, REPORTED from that phase): EURUSD M15 showed
  high accuracy (60-74%) with ROC-AUC near 0.50 (accuracy without ranking skill — the classic
  imbalance trap); XAUUSD random-forest recall 4-6%; Volume Profile: `NO_CLEAR_IMPROVEMENT`.
- Common mistakes: comparing against a single global baseline instead of per-fold; baselining on
  accuracy under imbalance; changing the baseline after seeing results (the protocol freezes
  baselines before tests).
- What "good" looks like: baseline retrained per fold on the same data, uplift counted fold-by-fold,
  frozen baseline definitions.

### 7. Statistical significance & overfitting (PSR/DSR as knowledge concepts)
- Definitions (no dependency in this project): Probabilistic Sharpe Ratio (PSR) estimates the
  probability that a strategy's true Sharpe exceeds a benchmark given observed Sharpe, skew, and
  kurtosis — correcting for short samples where one lucky fold dominates. Deflated Sharpe Ratio
  (DSR) further shrinks Sharpe for the NUMBER of trials performed (multiple-testing bias: with 100
  hyperparameter sets tried, the best backtest Sharpe is an inflated order statistic, not an
  expectation). Both come from López de Prado's work on backtest overfitting.
- Practical meaning here: every WF run, hyperparameter variant, and feature-set variant is a
  "trial"; promotion based on the best of many trials overstates skill. Existing mitigations
  (frozen protocol, frozen baselines, fixed threshold, tamper-evident digests) address provenance
  but do not yet count trials or deflate metrics (see Gaps).
- Non-parametric checks available with the installed stack (sklearn 1.9.1 / scipy 1.18.1): per-fold
  AUC std is already recorded (walk_forward.py:144-150); scipy provides sign-test/Wilcoxon
  primitives for "model beats baseline in k of n folds" inference without new deps.
- Overfitting signals: train-metric >> OOS-metric; fold variance >> fold mean uplift; improvements
  only when the fold's regime matches training; test-partition tuning (forbidden by protocol).
- sklearn 1.9.1 permutation importance (docs, summarized): model-agnostic — shuffle one feature,
  measure score degradation; `n_repeats` gives importance variance. Compute on held-out data:
  train-important/held-out-unimportant features indicate overfitting. Correlated features
  understate each other (information still flows via the twin) — cluster and keep one. Scorer
  choice matters, especially under imbalanced classification. Unlike impurity-based (MDI), it is
  not biased toward high-cardinality features and works with any metric. Project mapping: auditing
  the 24 technical features and Volume Profile deltas (protocol C vs D).

### 8. Economic evaluation: spread, slippage, commission (engine + labeling costs)
- Labeling cost (one-way, embedded in labels): BUY entry pays ask = close + historical bar spread
  (labeling.py:50-63); barriers are ATR multiples of that entry, so labels already embed a one-way
  entry cost.
- Backtest cost model (`backend/app/services/backtest/engine.py`, VERIFIED):
  - Fills: signal on bar i → execution at open of bar i+1 (engine.py:275-291).
  - BUY entry = next_open + spread + slippage (ask side); SELL entry = next_open - slippage
    (bid side, bid-based MT5 OHLC) (engine.py:287-296).
  - SELL exits shift triggers to Ask (closing a short is a buy): ask_high = high + spread,
    ask_low = low + spread; TP when `ask_low <= tp`, SL when `ask_high >= sl` (engine.py:178-197).
    BUY exits trigger on raw bid OHLC (engine.py:160-177).
  - Same-bar collision: SL assumed first for BOTH directions, with gap-aware fill
    (`min(open, sl) - slippage` BUY; `max(open + spread, sl) + slippage` SELL)
    (engine.py:164-171, 186-191).
  - Commission: per-lot PER SIDE at entry and exit; CENTS_AT_EVENT rounding contract — every money
    event rounded at event time, net derives from rounded components (engine.py:240-252, 300-304).
  - Sizing: risk % of balance with tick economics, lot-step flooring, min/max lot, hard cap 1.5x
    intended risk; degenerate inputs → 0 lots (engine.py:52-89).
  - Time exit: `max_holding_bars` (default 50) closes at close ± costs; open position force-
    liquidated on the final bar (engine.py:109, 175-197, 316-329).
- Cost defaults are symbol-aware at both the labeler call site (pipeline.py:65-67) and engine
  constructor (engine.py:23-33).
- Common mistakes: symmetric cost application to SELL (with bid-based OHLC the spread belongs on
  the ask at short exit, not entry); ignoring gap fills at stops; equity computed from post-close
  balances (documented review regression, engine.py:331-335); commission charged once instead of per side.
- What "good" looks like: costs in BOTH the label and the simulator (this repo does), conservative
  collision, gap-aware stop fills, explicit rounding contract, documented intra-bar assumptions.

## BUY-success vs SELL-direction prediction asymmetry

THE key nuance: `labeling.py` implements ONLY `label_buy_barrier`. The backtest engine supports
BUY and SELL, but the ML target exists only for longs.

- What a hypothetical SELL barrier labeler must mirror:
  - Entry: a short entered at bid. With bid-based MT5 OHLC, entry = `close[i]` (NO spread added);
    the spread is paid at EXIT because covering a short is a buy at Ask. The engine's exit triggers
    show the mechanics: `ask_low = low + spread` must fall to/below TP; `ask_high = high + spread`
    must rise to/above SL (engine.py:178-197). A naive "flip the sign" labeler that adds spread at
    entry and uses raw low/high exits would misprice every short label.
  - Barriers: TP BELOW entry (`entry - tp_mult*ATR`), SL ABOVE (`entry + sl_mult*ATR`). Touch tests
    flip: TP uses the ask-adjusted bar LOW, SL the ask-adjusted bar HIGH.
  - Collision conservatism stays IDENTICAL: both barriers in one bar → assume SL first (label 0).
    Conservatism is direction-agnostic; only the geometry flips.
  - Horizon, NaN semantics (NaN ATR skip, tail rows NaN), fail-loud ATR requirement, historical
    per-bar spread preference: all stay symmetric.
- Why a "direction classifier" is NOT a "barrier-success classifier":
  - Different targets: a direction model predicts the SIGN of future return; a SELL-success model
    predicts TP-before-SL for an actual short position with cost-adjusted entry/exit — including
    bars where price fell but not enough to reach TP before horizon expiry, and bars where an
    adverse bounce hit SL first.
  - Different base rates: direction under drift-free conditions is ~50/50; barrier success with
    TP=1.5xATR vs SL=1.0xATR is structurally below 50% (asymmetric distances), further shifted by
    spread cost. Calibration curves and thresholds do NOT transfer between the two targets.
  - Different feature relevance: some features are direction-specific in meaning (e.g.
    `is_bullish_bar`), so feature ranking is not sign-symmetric either.
  - Consequence: ROC-AUC on one target says nothing about the other; each target needs its own
    baselines (protocol A/B), calibration, and threshold policy. The high-accuracy/low-AUC pattern
    recorded in docs/core-model-baseline.md (EURUSD M15, accuracy 60-74% with AUC ~0.50) is exactly
    this trap: accuracy under imbalance can look good while ranking skill is nil.
- Where a SELL-success classifier would slot in (knowledge only, NO code changes): a
  `label_sell_barrier` sibling mirroring label_buy_barrier, a pipeline flag, mirrored signal
  encoding (-1 already exists in the engine), and per-target walk-forward evidence. Adoption
  requires demonstrated need + explicit user approval per project governance.

## Candidate libraries

Evaluation-only. None are installed. All additions require explicit user approval
(project governance: no complex/new methods without demonstrated need).

### statsmodels
- Necessity vs current stack: LOW. The pipeline is supervised classification with barrier labels;
  statsmodels serves classical inference (ADF stationarity, ACF/PACF, ARIMA/SARIMAX baselines,
  OLS/GLM with hypothesis tests). Nothing in the current protocol requires it.
- Pros: mature, results validated against other statistical packages, rich diagnostics; BSD.
- Cons: classical-statistics orientation, not ML feature engineering; large surface for a narrow
  need; py3.14 UNVERIFIED — official 0.15.0 install page lists Python 3.9-3.13 only.
- Compute cost: modest. Maintenance: stable, long-running (docs reviewed 2026-10-31, v0.15.0).
- Added value if adopted: stationarity/cointegration pre-checks; an honest classical forecast baseline.
- **Trigger criteria for future adoption:** (1) an evaluation step formally requires
  stationarity/cointegration tests or an ARIMA-family baseline; (2) official py3.14 Windows wheels
  confirmed. Status: **NOT INSTALLED — adding requires explicit user approval.**

### sktime
- Necessity vs current stack: LOW-MEDIUM. Unified fit/predict framework for forecasting, detection,
  classification, transformation ("500+ models"). The project already hand-built the hard
  financial-ML part (purged+embargoed expanding-window WF with per-fold retraining), so sktime
  would largely duplicate existing machinery.
- Pros: one API across tasks; native time-indexed data handling sklearn lacks; probabilistic
  forecasting; adapters for private estimators.
- Cons: heavy soft-dependency ecosystem; its temporal CV does not implement financial
  purge/embargo for barrier labels out of the box; pandas 3.0.6 / py3.14 compat UNVERIFIED.
- Compute cost: adoption itself cheap; real cost is dependency + learning surface.
- Maintenance: active, large community (docs reviewed 2026-10-31; version not shown on page).
- Added value if adopted: only for a future price-path FORECASTING workstream, not the current
  barrier-classification objective.
- **Trigger criteria for future adoption:** (1) a formal forecasting task (beyond classification)
  enters the protocol; (2) py3.14/Windows + pandas 3 wheels verified. Status: **NOT INSTALLED —
  adding requires explicit user approval.**

### River (online/incremental ML)
- Necessity vs current stack: LOW today. The pipeline is batch (CSV artifacts, SQLite records,
  persisted .joblib models). River's streaming `learn_one` design matters only if M1-timeframe
  streaming inference or drift-driven adaptation becomes a requirement.
- Pros: true incremental learning, cheap progressive validation, drift-detection concepts;
  models serialize via pickle/dill (FAQ).
- Cons: online models typically underperform batch models on stationary windows; small model zoo;
  py3.14/Windows UNVERIFIED (landing page confirms release 0.26.1; install/compat pages unresolved).
- Compute cost: per-sample updates cheap; real cost is stream-wiring engineering.
- Maintenance: active research-driven project (docs reviewed 2026-10-31).
- Added value if adopted: drift monitoring and adaptive retraining cadence for live signal quality.
- **Trigger criteria for future adoption:** (1) demonstrated need for streaming/adaptive inference
  on live M1 data; (2) py3.14/Windows wheels verified. Status: **NOT INSTALLED — adding requires
  explicit user approval.**

### skforecast
- Necessity vs current stack: LOW. Wraps any sklearn-compatible regressor into a forecaster
  (`ForecasterRecursive` / `ForecasterDirect`) with a backtesting module (refits, gaps, fixed or
  expanding windows) — close in spirit to this repo's WF, but aimed at FORECASTING, not
  barrier-success classification.
- Pros: lightweight vs sktime; sklearn-native; pandas-in/pandas-out; BSD; NumFOCUS-affiliated.
- Cons (blocking): dev-version install docs pin `pandas >= 2.2, < 3.0` — INCOMPATIBLE with the
  installed pandas 3.0.6 until upstream updates; core deps pull optuna + numba (heavy);
  Python >= 3.10 stated but 3.14 specifically UNVERIFIED; Windows not explicitly documented.
- Compute cost: same order as wrapped sklearn models; numba JIT adds first-run latency.
- Maintenance: active since 2021 (docs reviewed 2026-10-31, unstable dev version fetched).
- Added value if adopted: only for a future multi-step forecasting workstream with existing
  sklearn models; not needed for the current label/eval stack.
- **Trigger criteria for future adoption:** (1) a formal forecasting workstream is approved;
  (2) upstream drops the pandas<3.0 pin and confirms py3.14 wheels. Status: **NOT INSTALLED —
  adding requires explicit user approval.**

## Provenance
All fetches via WebFetch, https only, official domains, reviewed 2026-10-31.
Context7 MCP was NOT available in this session (stated explicitly per protocol).
- scikit-learn 1.9.1 docs:
  - TimeSeriesSplit — https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
  - Probability calibration — https://scikit-learn.org/stable/modules/calibration.html
  - Permutation importance — https://scikit-learn.org/stable/modules/permutation_importance.html
- statsmodels 0.15.0 — https://www.statsmodels.org/stable/index.html and /stable/install.html
- sktime (version not shown on page) — https://www.sktime.net/en/stable/index.html
- River latest 0.26.1 — https://riverml.xyz/latest/ and /latest/faq/ (install page unresolved, 404)
- skforecast (dev-version docs) — https://skforecast.org/latest/index.html and
  /latest/quick-start/how-to-install.html
Project code (read-only, branch reveo @ 48d7d43):
- backend/app/services/dataset/labeling.py — BarrierLabeler.label_buy_barrier (95 lines)
- backend/app/services/dataset/pipeline.py — DatasetPipeline.build_dataset (207 lines)
- backend/app/services/dataset/features.py — FeatureEngineer (backward-looking)
- backend/app/services/ml/walk_forward.py — run_ml_walk_forward + verifier (M8)
- backend/app/services/backtest/walk_forward.py — strategy WF (no per-fold retraining)
- backend/app/services/backtest/engine.py — RealisticBacktestEngine (465 lines)
- backend/tests/unit/test_ml_walk_forward.py — purge/embargo/retrain assertions
- backend/tests/data_ml/test_features_and_leakage.py — anti-lookahead, collision, purge tests
- docs/core-model-evaluation-protocol.md, docs/core-model-baseline.md

## Gaps / open questions
1. UNRESOLVED — purge/embargo verified where? VERIFIED in code + tests for the ML walk-forward
   (purge 24 = horizon, embargo 12 = horizon//2; retrain-per-fold asserted by test). UNRESOLVED
   whether the STATIC DatasetPipeline split intentionally omits a separate embargo buffer
   (purge-only there) or whether alignment is a pending task.
2. UNVERIFIED — Python 3.14 / Windows wheels for all four candidate libraries (statsmodels official
   docs stop at 3.13; sktime/River version compat not on fetched pages; skforecast >= 3.10 in dev docs).
3. UNRESOLVED — skforecast pandas pin: dev docs show `pandas < 3.0`; compatibility with installed
   pandas 3.0.6 unconfirmed and likely broken until upstream changes.
4. UNRESOLVED — calibration is measured (Brier/LogLoss) but no calibrator is fitted in the WF;
   whether to add `CalibratedClassifierCV` (on a purged validation partition, sigmoid preferred
   under imbalance/small folds) is an open design decision requiring user approval.
5. UNRESOLVED — trial-count/DSR discipline: evidence digests prevent tampering but the number of
   hyperparameter/feature trials is not recorded, so best-of-N promotion bias is unaccounted.
6. REPORTED (not re-verified by this review) — historical results in docs/core-model-baseline.md
   (EURUSD AUC ~0.50, XAUUSD recall 4-6%, Volume Profile NO_CLEAR_IMPROVEMENT) date from an older
   snapshot (branch master @ 2f2ed9e) and have not been reproduced here.
