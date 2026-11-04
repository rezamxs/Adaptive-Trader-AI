# ML Algorithms & Evaluation — Adaptive Trader AI reference

Scope: what the evaluation layer computes (algorithms, metrics, imbalance, calibration),
version-matched API facts, evaluation-only notes for candidate libraries. Companion:
`backend-model-infra.md`, `timeseries-financial-ml.md`; walk-forward (purge=24/embargo=12, fixed
0.5 threshold) and registry mechanics are documented elsewhere, not repeated.

Ground truth (verified in code, 2026-10-31): scikit-learn 1.9.1, joblib 1.6.0 installed;
statsmodels, xgboost, lightgbm, catboost, optuna, sktime, river, imbalanced-learn, shap NOT
INSTALLED. Python 3.14.4, Windows x64.

## Current evaluation contract (code facts)

- `ModelEvaluator.evaluate` (`backend/app/services/ml/evaluator.py`): per partition computes
  `accuracy`, `precision`, `recall`, `f1` (all `zero_division=0`, 4 dp) plus probability metrics
  `roc_auc`, `brier_score`, `log_loss` on positive-class probability (`proba[:, 1]` when 2-D),
  `labels=[0,1]` for log_loss.
- Fail-closed sentinels: single-class labels or a metric exception writes `roc_auc=0.5`,
  `brier_score=1.0`, `log_loss=0.693`. Deliberate: the promotion gate is `roc_auc > 0.52` (strict)
  and `brier_score <= 0.25`, so sentinels fail it; a 0.25 sentinel would fabricate passing
  evidence. `0.693` is the no-information log-loss baseline (no gate reads it).
- `calculate_calibration_curve`: manual 5-bin `np.linspace(0,1)` split; per bin stores
  `avg_predicted_prob`, `true_positive_rate`, `sample_count`. Calibration is **measured but no
  calibrator is ever fitted** — no `CalibratedClassifierCV` call exists in the codebase.
- Baseline: `BaselineModel` wraps `DummyClassifier(strategy="prior")`. Promotion compares
  candidates to the *current active model* on validation ROC-AUC only; test metrics are reserved
  for absolute gates (avoids tuning against test). No separate "must beat baseline" arithmetic
  gate; the baseline is a training option for governance review.
- Imbalance: only `LogisticRegressionStrategyModel` uses `class_weight="balanced"`;
  `RandomForestStrategyModel` sets **no** class weight; no resampling exists anywhere.
  `zero_division=0` makes a degenerate all-negative prediction report precision/F1 of 0.0 silently.
- Trainer guards (`trainer.py`): training partition must contain both classes {0,1}; all
  partitions finite; `sample_count` stored per evaluation partition.

---

### scikit-learn 1.9.1 (installed)
- **What it does:** classification algorithms (LogisticRegression, RandomForestClassifier,
  DummyClassifier), metrics suite, calibration tooling.
- **Where in this project:** `backend/app/services/ml/models.py` (wrappers), `evaluator.py`
  (metrics), `trainer.py` (fit/evaluate); walk-forward folds reuse the same per-fit pattern.
- **Essential APIs:**
  - `roc_auc_score(y_true, y_score)` — pass `predict_proba(X)[:, 1]` (greater-label class) or
    `decision_function(X)`; never the 2-column matrix in binary use.
  - `brier_score_loss(y_true, prob_pos)` — binary range [0,1]; strictly proper; `scale_by_half`
    selects the original [0,2] Brier (1950) definition.
  - `log_loss(y_true, proba, labels=[0,1])` — `labels` keeps shape valid on single-class slices.
  - `precision/recall/f1_score(..., zero_division=0)` — default F1 of an all-true-negative set
    is 0; `zero_division` controls the degenerate value.
  - `balanced_accuracy_score` — macro recall; the imbalance-aware accuracy not logged today.
  - `average_precision_score` — PR-AUC; random baseline = positive prevalence; the more
    informative AUC under imbalance.
  - `CalibratedClassifierCV(estimator, method="sigmoid"|"isotonic", cv)` — `ensemble=True`
    (default) averages k (classifier, calibrator) couples; `ensemble=False` cheaper.
  - `calibration_curve(y_true, y_prob, n_bins=5, strategy="uniform"|"quantile")` — library
    equivalent of the manual curve; `"quantile"` fixes sparse high-prob bins.
  - `LogisticRegression(C, max_iter, class_weight)` — `"balanced"` =
    `n_samples / (n_classes * np.bincount(y))`.
- **Best patterns:** probability metrics only on `predict_proba`, never `predict` labels; keep
  `labels=[0,1]` for single-class slices; isotonic calibration needs ~1000+ samples (sigmoid is
  the small-sample choice; isotonic can shift AUC via ties, sigmoid preserves ranking); the 4 dp
  rounding in `ModelEvaluator` is load-bearing for the metrics digest — keep deterministic.
- **Anti-patterns / common mistakes:**
  - Reading low Brier as "well calibrated": Brier mixes calibration + discrimination +
    uncertainty (Murphy decomposition); judge calibration from the curve bins, not Brier alone.
  - `roc_curve`/`roc_auc_score` with non-{0,1} labels needs `pos_label`.
  - Version notes (1.9.x): LogisticRegression `penalty` **deprecated in 1.8, removal in 1.10**
    (use `l1_ratio` and `C=np.inf`); `n_jobs` deprecated with no effect; `max_iter` default is
    only 100 (project sets 1000, which matters under `class_weight="balanced"`).
  - Calibrating an already-fitted model requires
    `CalibratedClassifierCV(estimator=FrozenEstimator(m))` with data disjoint from the model's
    training rows.
  - Python 3.14: 1.9.1 installed and running under 3.14.4 — confirmed by operation, not only docs.
- **Minimal example (project-consistent):** a new metric slots into `ModelEvaluator.evaluate` as:
  ```python
  if len(np.unique(y_true)) > 1:
      try:
          metrics["average_precision"] = float(round(average_precision_score(y_true, prob_pos), 4))
      except Exception:
          metrics["average_precision"] = float(np.mean(y_true))  # random baseline = prevalence
  ```
- **Official docs:** https://scikit-learn.org/stable/modules/model_evaluation.html ,
  .../modules/calibration.html ,
  .../modules/generated/sklearn.linear_model.LogisticRegression.html (all scikit-learn 1.9.1,
  matching installed; reviewed 2026-10-31).

### joblib 1.6.0 (installed)
- **What it does:** persists the artifact payload (dict: fitted model, scaler, feature names,
  attested metrics digest) into `settings.ARTIFACTS_MODELS_DIR / f"{model_id}.joblib"`.
- **Where in this project:** `trainer.py` (`joblib.dump(artifact_payload, artifact_path)`);
  `registry.py` (`joblib.load` at promotion to verify the inference contract and digest).
- **Essential APIs:**
  - `joblib.dump(value, filename, compress=0, protocol=None)` — returns list of written files.
  - `joblib.load(filename)` — reassembles numpy arrays (optionally memmapped).
  - `compress`: `True` (level 3), int 0–9, or `(compressor, level)` with `zlib|gzip|bz2|lzma|xz`;
    filename extensions auto-select the compressor. Compression disables memmapping on load.
- **Best patterns:** keep the payload a plain dict of picklable values (as `trainer.py` does) so
  promotion checks can `getattr(artifact["model"], "predict_proba")` safely; only the same
  sklearn version that trained an artifact should load it (class layouts pickle by reference) —
  the registry already fails closed on load errors.
- **Anti-patterns / common mistakes:**
  - `joblib.load` executes arbitrary code — the promotion digest cross-check is the trust
    boundary; never load `.joblib` from outside `ARTIFACTS_MODELS_DIR`.
  - Don't mix plain `pickle` writers with joblib dumps.
  - Models defined in `__main__` (notebooks) cloudpickle and fail to load in the FastAPI process —
    all project models live in importable `models.py`; keep it that way.
  - Python 3.14 / Windows: 1.6.0 installed and round-trips in governance tests — confirmed by
    operation; the "stable" docs page does not state a version number.
- **Minimal example (project-consistent):**
  ```python
  payload = {"model_id": model_id, "algorithm": algorithm, "model": model,
             "scaler": scaler, "feature_names": feature_names,
             "metrics_digest": metrics_digest, "status": "candidate"}
  joblib.dump(payload, artifact_dir / f"{model_id}.joblib")
  ```
- **Official docs:** https://joblib.readthedocs.io/en/stable/generated/joblib.dump.html
  (stable docs, exact version not stated; installed 1.6.0; reviewed 2026-10-31).

### Optuna NOT INSTALLED — candidate
- **What it does:** define-by-run hyperparameter optimization: objective builds its search space
  at runtime; samplers (TPE etc.) propose parameters, pruners kill unpromising trials, RDB
  storage persists studies for resume/parallel work.
- **## Critical evaluation**
  - Necessity: the project has no search — fixed defaults (`C=1.0`, `max_iter=1000`,
    `n_estimators=100`, `max_depth=5`) and no OOS tuning. sklearn's built-in
    `RandomizedSearchCV`/`HalvingRandomSearchCV` cover small spaces without a new dependency;
    Optuna adds value only for guided TPE search, early pruning, persistent studies.
  - Pros: define-by-run spaces; pruning; persistence; an objective can wrap the existing
    walk-forward validation folds. Cons: each trial = full fit + walk-forward evaluation; an
    objective touching test folds would violate the "test reserved for absolute gates" design;
    auto-tuned models still must pass registry gates as candidates.
  - Compute cost: moderate-to-high (tens of trials × per-fold retraining); CPU.
  - Python 3.14 / Windows: **UNVERIFIED** — stable docs state no Python floor; 3.14 appears only
    in a tooltip string; no Windows caveats found, but 3.14 wheel support unconfirmed.
  - Maintenance: active; exact release number UNVERIFIED from fetched docs. Added value today: low.
  - **Trigger criteria for future adoption:** (1) repeated manual retuning across ≥2 dataset
    versions, (2) a third algorithm family with ≥5 tunable parameters and measured AUC
    sensitivity, or (3) fixed defaults failing the ROC-AUC > 0.52 gate while tuned parameters
    pass it on validation folds only.
  - **Status: NOT INSTALLED — adding requires explicit user approval.**
- **Essential APIs:** `create_study(direction, sampler, pruner, storage)` / `study.optimize(obj,
  n_trials)` / `samplers.TPESampler(seed=...)` / `pruners.MedianPruner()` /
  `trial.suggest_float/suggest_int` / `study.best_params` / `storages.RDBStorage(url)`.
- **Best patterns:** seed the sampler (project pins `random_state=42`); objective reads
  walk-forward validation folds only; log per-trial metrics beside the eventual MLModelRecord.
- **Anti-patterns / common mistakes:** tuning against the test partition; unseeded samplers
  breaking digest reproducibility; concurrent trials on one SQLite file on Windows (FAQ-noted
  locking issues).
- **Minimal example (project-consistent):**
  ```python
  # candidate pseudocode — not present in the codebase
  study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
  study.optimize(objective_over_validation_folds, n_trials=20)
  # then: trainer.train_candidate(dataset, hyperparameters=study.best_params) -> "candidate"
  ```
- **Official docs:** https://optuna.readthedocs.io/en/stable/ + /faq.html (stable docs; version
  and Python 3.14 support UNVERIFIED; reviewed 2026-10-31).

### XGBoost NOT INSTALLED — candidate
- **What it does:** regularized gradient-boosted trees with histogram methods, native
  missing-value handling, sklearn-compatible `XGBClassifier`.
- **## Critical evaluation**
  - Necessity: the bounded RF already provides a non-linear candidate; XGBoost's gains are speed
    and regularization, not a new capability class, at this dataset scale.
  - Pros: strong tabular baseline; `scale_pos_weight` for imbalance; native NaN handling. Cons:
    many hyperparameters (overfit risk without a search loop — see Optuna triggers); default
    wheels ship CUDA (larger footprint; `xgboost-cpu` variant exists).
  - Compute cost: fast on small CPU data; multiplied by walk-forward folds.
  - Python 3.14 / Windows: Windows x86_64 wheels supported (needs Visual C++ Redistributable);
    **Python 3.14 UNVERIFIED** — the install page names no Python version.
  - Maintenance: very active; release number UNVERIFIED. Added value today: unproven — the
    ROC-AUC > 0.52 gate makes any new family an empirical question, not a default upgrade.
  - **Trigger criteria for future adoption:** (1) RF/LR plateau below the 0.52 validation gate
    across ≥2 dataset versions, (2) a documented walk-forward comparison beating the active model
    on validation AUC without Brier regression, (3) user approval plus a tuning story.
  - **Status: NOT INSTALLED — adding requires explicit user approval.**
- **Essential APIs:** `XGBClassifier(n_estimators, max_depth, learning_rate, scale_pos_weight)` /
  `.fit` / `.predict_proba` / `early_stopping_rounds` + `eval_set` / `tree_method="hist"` /
  `Booster.save_model` (prefer over pickle for long-lived artifacts).
- **Best patterns:** set `scale_pos_weight = n_negative/n_positive` (any boosting candidate must
  state its imbalance policy explicitly — LR does, RF does not); check calibration before
  trusting `predict_proba` (boosted trees are often miscalibrated; the Brier gate would catch it).
- **Anti-patterns / common mistakes:** default probabilities straight through the Brier gate
  without a calibration check; full-CUDA wheel when CPU-only suffices; assuming XGBoost is exempt
  from walk-forward evidence at promotion.
- **Minimal example (project-consistent):**
  ```python
  # candidate only — not installed; would live beside models.py as an IStrategyModel wrapper
  # trainer.train_candidate(dataset, algorithm="xgboost") -> MLModelRecord(status="candidate")
  ```
- **Official docs:** https://xgboost.readthedocs.io/en/stable/install.html (stable docs; version
  UNVERIFIED; reviewed 2026-10-31).

### LightGBM NOT INSTALLED — candidate
- **What it does:** fast GBDT with leaf-wise growth (`num_leaves`), native categorical features,
  sklearn-compatible `LGBMClassifier`.
- **## Critical evaluation**
  - Necessity: same family as XGBoost; differentiators are training speed and categorical
    handling — the project has no categorical layer and small folds, so speed is immaterial.
  - Pros: fastest CPU of the boosters; `is_unbalance`/`scale_pos_weight`; low memory. Cons:
    leaf-wise growth overfits small samples fast (lower `num_leaves`, raise `min_data_in_leaf` —
    more tuning burden, reinforcing the Optuna trigger); CUDA/ROCm builds Linux-only.
  - Compute cost: lowest CPU cost of the boosters; still multiplied by folds.
  - Python 3.14 / Windows: **UNVERIFIED** — the fetched install guide documents Windows only as
    source builds (CMake + VS/MinGW) and defers Python packaging to a page that 404'd; wheel and
    Python 3.14 facts unconfirmed by any fetched official page.
  - Maintenance: active (EffVer scheme noted; exact version UNVERIFIED). Added value today:
    unproven; first ask whether the existing RF clears the gates.
  - **Trigger criteria for future adoption:** XGBoost's triggers plus (4) measured CPU-training
    pain inside walk-forward retraining where histogram speed is the fix.
  - **Status: NOT INSTALLED — adding requires explicit user approval.**
- **Essential APIs:** `LGBMClassifier(num_leaves, min_data_in_leaf, learning_rate,
  scale_pos_weight, is_unbalance)` / `.fit` / `.predict_proba` / `early_stopping_rounds` /
  `eval_metric` / `lgb.plot_importance` (interpretability hook).
- **Best patterns:** constrain `num_leaves`, generous `min_data_in_leaf` for small samples;
  always pass `eval_set` for early stopping; verify calibration (leaf-wise models distort
  probabilities).
- **Anti-patterns / common mistakes:** assuming pip wheels exist for the target Python version
  (UNVERIFIED here — verify before adoption); leaving `num_leaves` at default (127) on hundreds
  of rows; treating it as calibration-free when the Brier gate will judge it.
- **Minimal example (project-consistent):**
  ```python
  # candidate only — not installed; wrapper must satisfy the IStrategyModel contract
  # trainer.train_candidate(dataset, algorithm="lightgbm") -> MLModelRecord(status="candidate")
  ```
- **Official docs:** https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html (latest
  docs; python-package page 404'd — wheel/Python facts UNVERIFIED; reviewed 2026-10-31).

### CatBoost NOT INSTALLED — candidate
- **What it does:** GBDT with ordered boosting and ordered target statistics that reduce target
  leakage on categorical features; strong defaults, sklearn-compatible `CatBoostClassifier`.
- **## Critical evaluation**
  - Necessity: its headline (leakage-resistant categorical encoding) targets a problem the
    project does not have — features are numeric/derived and leakage is handled structurally by
    walk-forward purge/embargo. Its one genuine advantage over the other boosters: fewer knobs to
    get wrong for a governance-heavy codebase.
  - Pros: good out-of-the-box calibration; default-heavy API reduces tuning burden. Cons: third
    booster competing for one role; heavier binaries; slower CPU training than LightGBM.
  - Compute cost: moderate; GPU binaries statically link CUDA (driver-only), but CPU is the
    realistic path here.
  - Python 3.14 / Windows: Windows pip/conda builds exist; only 64-bit CPython supported (docs
    reference CatBoost 1.2.10 for GPU requirements); **Python 3.14 UNVERIFIED** — no Python
    version stated on the fetched page.
  - Maintenance: active (exact version beyond the 1.2.10 reference UNVERIFIED). Added value
    today: none demonstrated; three boosters is two too many for the governance overhead.
  - **Trigger criteria for future adoption:** the booster trigger above, plus a documented reason
    to prefer ordered boosting (e.g., measured calibration difference at equal validation AUC).
    User approval required regardless.
  - **Status: NOT INSTALLED — adding requires explicit user approval.**
- **Essential APIs:** `CatBoostClassifier(iterations, depth, learning_rate,
  loss_function="Logloss")` / `.fit(X, y, eval_set)` / `.predict_proba` / `cat_features` /
  `auto_class_weights` / `save_model`/`load_model` / `get_feature_importance`.
- **Best patterns:** `auto_class_weights="Balanced"` mirrors the LR policy if adopted; evaluate
  Brier first — ordered boosting tends to produce better-calibrated probabilities, aligned with
  the ≤0.25 gate.
- **Anti-patterns / common mistakes:** adopting "just in case"; assuming the GPU wheel needs a
  CUDA toolkit install (it does not); its serialization format differs from the project's joblib
  dict pattern — a wrapper is required for the inference contract.
- **Minimal example (project-consistent):**
  ```python
  # candidate only — not installed; wrapper exposes fit/predict_proba/get_params
  # to satisfy the IStrategyModel contract checked at promotion.
  ```
- **Official docs:** https://catboost.ai/docs/en/concepts/installation and
  .../python-installation (docs rev r21892739 referencing 1.2.10; Python 3.14 UNVERIFIED;
  reviewed 2026-10-31).

### imbalanced-learn NOT INSTALLED — candidate
- **What it does:** resampling toolkit (`SMOTE`, over/under-samplers, sampler-aware Pipeline) for
  imbalanced classification.
- **## Critical evaluation**
  - Necessity: the only imbalance handling is `class_weight="balanced"` on LR; RF is unweighted.
    Whether that is insufficient is currently *unmeasurable* — no per-class recall or PR-AUC is
    logged, so the problem itself is unmeasured before any resampler is justified. Measurement
    must precede resampling.
  - Pros: standard, sklearn-pipeline-friendly; documented floors (Python ≥3.10, sklearn ≥1.4.2)
    satisfied by the installed stack. Cons: synthetic oversampling on time-series-derived
    features can fabricate implausible samples and leak temporal structure if applied before the
    temporal split.
  - Compute cost: negligible (CPU, in-memory).
  - Python 3.14 / Windows: latest documented version 0.14.2; deps Python ≥3.10, NumPy ≥1.25.2,
    SciPy ≥1.11.4, scikit-learn ≥1.4.2; Windows pip/conda standard. **Explicit Python 3.14
    confirmation UNVERIFIED**, and whether 0.14.2 pins an upper bound on scikit-learn (accepts
    1.9.1) is UNVERIFIED from the fetched page — verify package metadata first.
  - Maintenance: active (scikit-learn-contrib). Added value today: low; measurement gap first.
  - **Trigger criteria for future adoption:** (1) positive-class recall collapse
    (precision/recall of 0 from `zero_division=0`) on validation/test, or (2)
    balanced_accuracy/average_precision measurement shows minority-class degradation class
    weights do not fix — any resampler must be applied inside folds (post-split), never across
    temporal boundaries. Requires user approval.
  - **Status: NOT INSTALLED — adding requires explicit user approval.**
- **Essential APIs:** `SMOTE(k_neighbors)` / `RandomOverSampler` / `RandomUnderSampler` /
  `SMOTEENN` / `imblearn.pipeline.Pipeline` (re-samples on `fit`, not `transform`).
- **Best patterns:** use `imblearn.pipeline.Pipeline` so resampling occurs per CV fold; prefer
  under-sampling + class weights on small financial data over SMOTE.
- **Anti-patterns / common mistakes:** resampling before the temporal split (leakage); SMOTE on
  inter-bar time-series features (synthetic neighbors may be impossible sequences); measuring
  success only by accuracy.
- **Minimal example (project-consistent):**
  ```python
  # candidate only — not installed
  # from imblearn.pipeline import Pipeline
  # pipe = Pipeline([("smote", SMOTE(random_state=42)), ("clf", LogisticRegression(...))])
  # ... then ModelEvaluator.evaluate as usual; artifact gains a resampler component.
  ```
- **Official docs:** https://imbalanced-learn.org/stable/install.html (0.14.2 docs; reviewed
  2026-10-31).

### Interpretability note: SHAP (NOT INSTALLED)
`shap` (TreeExplainer/LinearExplainer) maps feature names to attribution; the project's random
forest maps naturally onto TreeExplainer, and a zero-dependency first step already exists
(`RandomForestClassifier.feature_importances_`). Not fetched in this review; Python 3.14 wheel
support UNVERIFIED (compiled extensions historically lag new CPython on Windows). Same governance
rule: future-evaluation item, explicit user approval required.

## Evaluation asymmetry: BUY-success vs SELL-direction

Verified code fact: `backend/app/services/dataset/labeling.py` exposes only `label_buy_barrier`
(positive class = buy-side barrier success: TP hit before SL within `horizon_bars`, e.g. 12 in
governance tests). There is **no SELL-side labeler** — the ML layer learns one conditional
distribution and every metric in `ModelEvaluator` describes BUY-success probability only.

- The promotion gates (ROC-AUC > 0.52, Brier ≤ 0.25 on both partitions) certify BUY-signal
  probability quality; they say nothing about short-side decisions, which are currently
  rule-based rather than model-driven.
- If per-side evaluation were ever required, it would minimally mean: (a) a symmetric
  `label_sell_barrier` target (down-barrier first) so the same evaluator can score a second
  classifier; (b) per-side metric reporting — separate `validation_metrics`/`test_metrics`
  namespaces (e.g. `buy_*`/`sell_*` keys) so one MLModelRecord carries both without overwriting;
  (c) per-side gate thresholds, since long/short edge in FX is not symmetric under costs/spread
  and a shared 0.52 AUC floor is an assumption, not a fact; (d) the metrics digest schema extended
  to attest both payloads together. That is governance surface (registry, digest, promotion
  notes), not just a labeling function.
- Until then, any report should say "BUY-success probability model" — presenting single-sided
  evaluation as all-direction performance is an overclaim.

## Provenance

All URLs fetched 2026-10-31 (https, official domains). UNVERIFIED means no fetched page confirmed
the claim; such claims must not be asserted elsewhere without checking.

| Library | URL | Version documented | Review date |
|---|---|---|---|
| scikit-learn metrics | https://scikit-learn.org/stable/modules/model_evaluation.html | 1.9.1 (matches installed) | 2026-10-31 |
| scikit-learn calibration | https://scikit-learn.org/stable/modules/calibration.html | 1.9.1 | 2026-10-31 |
| scikit-learn LogisticRegression | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html | 1.9.1 | 2026-10-31 |
| joblib dump/load | https://joblib.readthedocs.io/en/stable/generated/joblib.dump.html | "stable" (version not stated; installed 1.6.0) | 2026-10-31 |
| Optuna overview/FAQ | https://optuna.readthedocs.io/en/stable/ + /faq.html | UNVERIFIED (stable tag) | 2026-10-31 |
| XGBoost install | https://xgboost.readthedocs.io/en/stable/install.html | UNVERIFIED (stable tag) | 2026-10-31 |
| LightGBM install | https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html | UNVERIFIED (latest tag) | 2026-10-31 |
| CatBoost install | https://catboost.ai/docs/en/concepts/installation + /python-installation | refs 1.2.10 (GPU reqs) | 2026-10-31 |
| imbalanced-learn install | https://imbalanced-learn.org/stable/install.html | 0.14.2 | 2026-10-31 |

Code evidence (read 2026-10-31): `backend/app/services/ml/evaluator.py`, `trainer.py`,
`models.py`, `registry.py` (gates lines 30–33, 192–233, 725–806),
`backend/tests/data_ml/test_ml_governance.py`, `backend/app/services/dataset/labeling.py`
(`label_buy_barrier` only).

## Gaps / open questions

1. **Python 3.14 wheel availability for ALL candidates is UNVERIFIED** — no candidate doc page
   states a 3.14 floor; only scikit-learn/joblib are confirmed by running under 3.14.4. Any
   adoption must first verify wheels on this exact interpreter (Windows x64).
2. **imbalanced-learn 0.14.2 upper-bound scikit-learn pin UNVERIFIED** — floor (≥1.4.2) is
   satisfied; whether it accepts 1.9.1 was not confirmed. Check package metadata first.
3. **OptunaSearchCV existence/version not confirmed** in fetched docs; the sklearn-comparison
   rests on sklearn's own built-in search classes, which are confirmed.
4. **No imbalance measurement exists yet** — `balanced_accuracy_score` and
   `average_precision_score` are not logged anywhere, so "is class_weight='balanced' enough?" is
   unanswerable from stored metrics. Adding those two (evaluation-only, no new dependency) is the
   cheapest next evidence step.
5. **Calibration is measured but never corrected** — the 5-bin curve is stored, yet no calibrator
   is fitted and no calibration-only metric (e.g. ECE) is computed; whether the fixed 0.5
   threshold plus Brier gate suffices downstream is an open governance question.
6. **LightGBM python-package docs 404'd during fetch** — the Windows wheel story is documented
   only as source build on the fetched page; wheel facts remain open.
