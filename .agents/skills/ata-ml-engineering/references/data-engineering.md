# Data Engineering Reference — Adaptive Trader AI

Scope: document parsing (PyMuPDF, openpyxl) and market-data frames (numpy, pandas, MT5 output contract).
Versions are the ones actually installed per `requirements-lock.txt` + live probe: numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, pymupdf 1.28.2, openpyxl 3.1.5, MetaTrader5 5.0.6231. Python 3.14.4, Windows x64.

---

## numpy 2.5.3

- **What it does:** n-dimensional arrays, vectorized numeric math, finiteness checks, dtype promotion under NEP 50 rules.
- **Where in this project:** `backend/app/services/dataset/pipeline.py` — `np.isfinite(ohlc).all()`, OHLC → `to_numpy(dtype=float)`, `np.inf`/`-np.inf` → NaN replacement, `np.maximum/np.minimum` candle-sanity checks; `backend/app/services/market_data/mt5_provider.py` consumes MT5's numpy structured arrays.
- **Essential APIs:**
  - `np.isfinite(arr).all()` — the project's canonical guard before any scaler/model fit (NaN and ±inf both fail it).
  - `np.asarray(..., dtype=float)` — force float64 before math; `copy=` accepts None/True/False since 2.0.
  - `np.copy(a)` — explicit copy; `a.view()` never owns data.
  - `np.maximum` / `np.minimum` — element-wise candle envelope checks.
  - `arr.replace` is pandas; for numpy use boolean assignment on a copied array.
- **Best patterns:**
  - Validate finite + positive OHLC right after numeric coercion, before feature math (project does this in `build_dataset`).
  - Rely on NEP 50: Python scalars are "weak"; `float32_array + 1.0` stays float32. If you need float64, cast explicitly.
  - `copy=False` on `np.asarray` now raises `ValueError` when a copy is unavoidable — use `copy=None` (default) for "copy if needed".
- **Anti-patterns / common mistakes:**
  - 1.x habits that died in 2.x: `np.float_`, `np.NaN`, `np.Inf`, `np.in1d`, `np.trapz`, `np.asfarray`, `numpy.core` — all removed; use `np.float64`, `np.nan`, `np.inf`, `np.isin`, `np.trapezoid`, `numpy._core`.
  - Expecting a Python int to widen a small-dtype array: under NEP 50 an out-of-range scalar raises OverflowError instead of promoting.
  - `np.can_cast()` with Python `int/float` instances — no longer allowed; compare `type(obj)` explicitly.
  - Trusting "Windows int is int32": since 2.0 Windows default integer is int64.
- **Minimal example (project-consistent):**
  ```python
  ohlc = raw_df[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
  if not np.isfinite(ohlc).all() or (ohlc <= 0).any():
      raise ValueError("Raw OHLC values must be finite and strictly positive.")
  ```
- **Official docs:** https://numpy.org/doc/stable/release/2.0.0-notes.html (NEP 50 + copy semantics; 2.5.x inherits these rules), reviewed 2026-10-31.

## pandas 3.0.6

- **What it does:** market-data DataFrames, feature construction, temporal purged/embargoed splitting, CSV persistence. **3.0 defaults Copy-on-Write (CoW) and the dedicated `str` dtype — several 2.x idioms are gone.**
- **Where in this project:** `backend/app/services/dataset/pipeline.py` (feature frame, purge/embargo split, scaler split, fingerprinting, CSV output), `backend/app/services/market_data/mt5_provider.py` (`get_historical_dataframe`), MT5 → features flow.
- **Essential APIs:**
  - `pd.DataFrame(records)` + `df.set_index("time")` + `df.sort_index()` — MT5 → frame path in `get_historical_dataframe`.
  - `pd.to_numeric(col, errors="coerce")` — coercion before `.to_numpy(dtype=float)`.
  - `df.loc[:, cols] = values` — the only assignment form that works reliably under CoW.
  - `df.replace([np.inf, -np.inf], np.nan)` then `df.dropna(subset=...)` — project's inf/NaN cleanup.
  - `df.iloc[a:b].copy()` — explicit materialization at partition boundaries.
  - `pd.util.hash_pandas_object(df, index=True)` — deterministic dataset-version fingerprint.
  - `pd.DatetimeIndex` checks: `index.has_duplicates`, `index.is_monotonic_increasing` — hard preconditions in `build_dataset`.
- **Best patterns (CoW-compliant, already used in pipeline.py):**
  - One-shot `.loc` assignment (`df.loc[:, feature_names] = finite_features`), never `df["x"][i] = v`.
  - `.copy()` at split boundaries (`train_df = clean_df.iloc[...].copy()`) — cheap under CoW, makes ownership explicit.
  - Explicit dtype checks before scalers: `clean_df["target"].astype(int)`.
  - Index integrity first: unique + ascending DatetimeIndex is a raised error, not a silent assumption.
- **Anti-patterns / common mistakes (2.x habits that break under 3.0):**
  - **Chained assignment silently does nothing** (`df[df.a > 0]["b"] = 1`); `SettingWithCopyWarning` is removed, so nothing warns you. Use single-call `.loc`.
  - `dtype == object` string checks: columns are now `str` dtype (PyArrow-backed if available, else object-backed); `str` columns hold only strings/NaN. Check `pd.api.types.is_string_dtype` instead.
  - Casting datetimes with `astype("int64")`: 3.0 infers `datetime64[us]` from strings (not ns) — values are 1000x smaller than ns expectations; use `.as_unit()` or `pd.to_datetime(..., unit=...)` explicitly.
  - `mode.copy_on_write` option and the `copy=` kwarg on `astype/reindex/rename/merge/tz_localize/transpose` are deprecated (no effect in 3.0; removal in 4.0) — call `.copy()` for an eager copy.
  - `read_excel/read_json` reject raw str/bytes; wrap in `StringIO`/`BytesIO`.
  - Old offset aliases `M/Q/Y` → `ME/QE/YE`; pytz replaced by stdlib `zoneinfo`.
  - `inplace=True` on `replace/fillna/clip/...` now returns `self` (not None) — do not rely on the old None return.
- **Minimal example (project-consistent):**
  ```python
  df_feat["target"] = target_series
  finite_features = df_feat[feature_names].replace([np.inf, -np.inf], np.nan)
  df_feat.loc[:, feature_names] = finite_features          # CoW-safe
  clean_df = df_feat.dropna(subset=feature_names + ["target"]).copy()
  clean_df["target"] = clean_df["target"].astype(int)
  ```
- **Official docs:** https://pandas.pydata.org/docs/whatsnew/v3.0.0.html (CoW default, str dtype, dtype-inference and removal list), reviewed 2026-10-31.

## scipy 1.18.1 (brief)

- **What it does:** statistical tests and signal processing; not imported by project code today, but the natural source for distribution/normality/filter work that statsmodels (not installed) would otherwise cover.
- **Where in this project:** none currently; `pipeline.py` imports only `sklearn.preprocessing.StandardScaler` from the ML stack. Candidate uses: `scipy.stats.skew/kurtosis/normaltest` on feature distributions, `scipy.signal` filters on tick noise.
- **Best patterns:** pin usage to `scipy.stats` / `scipy.signal` subpackages only; keep scalers/tests deterministic (fixed input arrays, no sampling without a seed).
- **Anti-patterns:** importing the top-level `scipy` for subpackage functions (`from scipy import stats` is required); assuming statsmodels APIs exist — they are NOT installed.
- **Minimal example:** `from scipy import stats; stats.normaltest(np.asarray(x, dtype=float))`
- **Official docs:** https://docs.scipy.org/doc/scipy/ (1.18.0 docs, June 2026; installed 1.18.1 is a patch within 1.18.x), reviewed 2026-10-31.

## PyMuPDF 1.28.2

- **What it does:** PDF text/image/vector extraction with per-page metadata; core of knowledge-document ingestion.
- **Where in this project:** `backend/app/services/knowledge/parsers.py` (`parse_pdf`, `parse_pdf_path`) — canonical `import pymupdf` at module top (NOT `import fitz`; the fitz alias is deprecated in 1.28 per verified project facts). Tests mirror this: `backend/tests/unit/test_knowledge_pipeline.py` imports pymupdf to synthesize sample PDFs.
- **Essential APIs:**
  - `pymupdf.open(filename=str(path), filetype="pdf")` — open from disk path.
  - `pymupdf.open(stream=source.read(), filetype="pdf")` — open from bytes/BytesIO.
  - `len(doc)` — page count; `doc.load_page(i)` — page accessor.
  - `page.get_text("text")` — plain-text extraction (empty string on image-only pages).
  - `page.get_images(full=True)` — embedded images; tuple indices used: `img[0]`=xref, `img[2]`=width, `img[3]`=height.
  - `page.get_drawings()` — vector paths (project distinguishes VECTOR_ONLY_PAGE).
  - `doc.close()` — always called in `parse_pdf_path` (finally-style correctness matters for untrusted files).
  - Test-side: `doc.new_page()`, `page.insert_text(pt, s)`, `doc.tobytes()`.
- **Best patterns (as implemented):**
  - Accept path OR stream (`parse_pdf_path`), so raw bytes never have to be materialized from disk.
  - Classify every page into TEXT_VECTOR_PAGE / IMAGE_ONLY_PAGE / VECTOR_ONLY_PAGE / BLANK_PAGE and mark non-text pages `is_semantic_content=False` — scanned PDFs surface as metadata, not silent emptiness.
  - Emit `--- [Page N] ---` anchors so downstream chunks keep page provenance.
- **Anti-patterns / common mistakes:**
  - `import fitz` — deprecated alias in 1.28; grep of `backend/app` shows zero fitz usage (only `pymupdf`).
  - Returning `("", 0, [])` on parse failure — the project explicitly re-raises instead, because an empty success is indistinguishable from a blank file and becomes a terminal `NO_EXTRACTABLE_CONTENT` record (silent data loss).
  - Unbounded loops over `get_drawings()` output on pathologically dense pages: count what you need (project keeps only `len(drawings)` unless the page is text-bearing).
  - No page-count/file-size cap exists inside `parse_pdf` itself — file-level caps are enforced by the ingestion layer upstream of this module; PDFs opened here are still fully decompressed into text, so the upstream cap is the real guard.
- **Minimal example (project-consistent):**
  ```python
  doc = pymupdf.open(stream=file_bytes, filetype="pdf")
  page = doc.load_page(0)
  text = page.get_text("text").strip()
  images = page.get_images(full=True)
  doc.close()
  ```
- **Official docs:** https://pymupdf.readthedocs.io/en/latest/ and https://pymupdf.readthedocs.io/en/latest/the-basics.html (all samples use `import pymupdf`; fitz deprecation statement per project verified facts — the docs pages fetched did not restate it), reviewed 2026-10-31.

## openpyxl 3.1.5

- **What it does:** .xlsx workbook parsing (values, sheets) for knowledge ingestion.
- **Where in this project:** `backend/app/services/knowledge/parsers.py` (`parse_excel`, `parse_excel_path`, `_excel_from_workbook`) — `load_workbook(..., data_only=True)`, iterate `wb.sheetnames`, `sheet.iter_rows(values_only=True)`, `wb.close()`.
- **Essential APIs:**
  - `load_workbook(str(path) | io.BytesIO(bytes), data_only=True)` — cached formula results instead of formulas.
  - `wb.sheetnames`, `wb[name]` — sheet access keyed for chunk `page` numbering.
  - `sheet.iter_rows(values_only=True)` — raw values, skips cell-object overhead.
  - `wb.close()` — called in `_excel_from_workbook`.
  - `load_workbook(read_only=True)` — constant-memory streaming mode (NOT currently used).
- **Best patterns:**
  - Path-based load (`parse_excel_path`) avoids holding the whole archive as bytes.
  - Filter empty cells before joining (`val is not None and str(val).strip()`) — produces one pipe-joined line per non-empty row.
- **Anti-patterns / common mistakes:**
  - Default mode loads the whole workbook into memory; for huge/untrusted files the official guidance is `read_only=True` (near-constant memory) — but note read-only cells are `ReadOnlyCell`, and `wb.close()` is mandatory there. The project trades memory for simplicity; keep an eye on file-size caps upstream.
  - `data_only=False` would return formula strings instead of values; `data_only=True` returns None for cells whose cache is missing (file saved without recalculation) — do not treat that None as a parse bug.
  - .xlsx is a ZIP container: decompression-bomb exposure is the same class as the DOCX/ZIP guards in this file; openpyxl itself performs no member-size cap.
- **Minimal example (project-consistent):**
  ```python
  wb = openpyxl.load_workbook(str(path), data_only=True)
  for name in wb.sheetnames:
      for row in wb[name].iter_rows(values_only=True):
          non_empty = [str(v).strip() for v in row if v is not None and str(v).strip()]
  wb.close()
  ```
- **Official docs:** https://openpyxl.readthedocs.io/en/stable/optimized.html (read-only/write-only modes), reviewed 2026-10-31.

## MetaTrader5 output contract (brief; deep dive owned by backend agent)

- **What it does:** `mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)` (and `copy_rates_from`/`copy_rates_range`) return a **numpy structured array** whose records are `(time, open, high, low, close, tick_volume, spread, real_volume)`; `time` is **seconds since 1970** (epoch int). The mql5 overview page confirms the 8-field tuple order and shows the official conversion `pd.DataFrame(rates)` + `pd.to_datetime(df["time"], unit="s")`.
- **Where in this project:** `backend/app/services/market_data/mt5_provider.py` — imports MT5 under try/except (`MT5_AVAILABLE` flag), serializes every C-extension call behind a module-level `threading.RLock()`.
- **Essential APIs used here:** `mt5.initialize/shutdown/last_error`, `mt5.copy_rates_from_pos`, `mt5.symbol_info`, `mt5.symbol_info_tick`, `mt5.account_info`, `mt5.history_deals_get`, `mt5.order_calc_margin`.
- **Best patterns (as implemented):**
  - Field-name access on the structured array (`r['time']`, `r['open']`, ..., `r['spread']`), then explicit casts per value into `Candle` dataclasses — safer than `pd.DataFrame(rates)` which depends on structured-array dtype layout.
  - `datetime.fromtimestamp(int(r['time']) - self._offset_seconds, tz=timezone.utc)` → tz-aware UTC timestamps; final frame gets `set_index("time")` + `sort_index()`.
  - `start_pos=1` fetches only closed bars; `start_pos=0` may include the still-open bar (marked `is_closed=False`).
  - Empty result → typed empty frame `pd.DataFrame(columns=["open","high","low","close","tick_volume","spread"])`.
  - Fail-closed timestamp validation: invalid/unparseable stamps are skipped (candles) or rejected (quotes); when a configured broker-clock offset is active, a tick whose RAW stamp is also fresh makes both interpretations ambiguous and the quote is refused.
- **Anti-patterns / common mistakes:**
  - Timezone naivety: `pd.to_datetime(df["time"], unit="s")` yields a naive index; MT5 stamps carry the broker server clock for the configured server, not guaranteed UTC. The project centralizes this in `MT5_SERVER_TIME_OFFSET_HOURS` (default 3.0, applied only when `account.server == settings.MT5_SERVER`, else zero offset + warning).
  - Calling MT5 C-extension functions without the shared lock from worker threads.
  - Building the frame directly from the structured array and silently inheriting its dtypes instead of validating numeric fields (quotes validate `math.isfinite` on bid/ask/point).
- **Minimal example (project-consistent):**
  ```python
  rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 1, 500)  # closed bars
  recs = [{"time": datetime.fromtimestamp(int(r['time']) - self._offset_seconds, tz=timezone.utc),
           "open": float(r['open']), "high": float(r['high']),
           "low": float(r['low']), "close": float(r['close'])} for r in rates]
  df = pd.DataFrame(recs).set_index("time").sort_index(ascending=True)
  ```
- **Official docs:** https://www.mql5.com/en/docs/python_metatrader5 (overview: copy_rates_* family, tuple fields, DataFrame conversion pattern; per-function page `copy_rates_from` returned 404 at fetch time), reviewed 2026-10-31.

---

## Provenance

Fetched official documentation (https only; all summaries original, no verbatim copying; review date 2026-10-31):

1. pandas 3.0.0 what's new — https://pandas.pydata.org/docs/whatsnew/v3.0.0.html (CoW default, str dtype, dtype inference, removals) — fetched OK.
2. pandas Copy-on-Write user guide — https://pandas.pydata.org/docs/user_guide/copy_on_write.html — returned only a redirect stub to a migration page that 404'd; CoW content is sourced from (1) plus project code, not from this page.
3. numpy 2.0 release notes (NEP 50, `copy=` keyword, removed aliases; inherited by 2.5.x) — https://numpy.org/doc/stable/release/2.0.0-notes.html — fetched OK.
4. scipy docs landing (version 1.18.0, dated 2026-06-19) — https://docs.scipy.org/doc/scipy/ — fetched OK.
5. PyMuPDF docs home + The Basics — https://pymupdf.readthedocs.io/en/latest/ , https://pymupdf.readthedocs.io/en/latest/the-basics.html — fetched OK (canonical `import pymupdf` shown in all samples; the znames.html and a fitz-deprecation statement were NOT found on fetched pages).
6. openpyxl optimized (read-only) mode — https://openpyxl.readthedocs.io/en/stable/optimized.html — fetched OK.
7. MetaTrader5 Python docs overview — https://www.mql5.com/en/docs/python_metatrader5 — fetched OK (copy_rates_* family, 8-field records, `pd.DataFrame` + `pd.to_datetime(unit='s')` pattern); https://www.mql5.com/en/docs/python_metatrader5/copy_rates_from returned 404.

Project sources read (no files modified): `backend/app/services/knowledge/parsers.py` (486 lines), `backend/app/services/dataset/pipeline.py` (207), `backend/app/services/market_data/mt5_provider.py` (456), `backend/app/core/config.py` (offset + limits), `backend/tests/unit/test_knowledge_pipeline.py`, `backend/tests/unit/test_final_safety_fixes.py` (path/bytes parity tests), grep of `fitz|pymupdf` across `backend/app` (only `pymupdf`, parsers.py lines 7/54/56). Installed versions from `requirements-lock.txt` + live probe as provided by the coordinator (not re-probed by this agent).

## Gaps / open questions

1. **UNRESOLVED — MT5 ~3h timestamp discrepancy.** The mission brief marks this open; `docs/PROJECT_MEMORY.md` is internally inconsistent (older entries at lines 679/725/729 call it an open blocker; a later 2026-10-28 entry claims it is resolved in source with a server-gated offset, "verified live" by another agent — not independently reproduced here). What is VERIFIED from code: `MT5_SERVER_TIME_OFFSET_HOURS=3.0` (config.py:72, with a plausibility validator), applied only when `account.server == settings.MT5_SERVER`, plus a fail-closed "raw stamp also fresh" refusal in `get_current_quote`. No live MT5 validation was performed by this agent; treat broker-clock semantics for any OTHER server as unknown.
2. pandas user-guide Copy-on-Write page could not be fetched (redirect stub → 404); the CoW rules above rest on the v3.0.0 what's-new page. A later fetch of the migration page would close this.
3. The `fitz` deprecation statement for 1.28 comes from the coordinator's verified facts; fetched PyMuPDF pages demonstrate `import pymupdf` but do not contain an explicit deprecation sentence. Project grep confirms zero `fitz` usage either way.
4. File-size / page-count caps for PDF ingestion are not present in `parsers.py`; the comments reference file-level caps enforced upstream (ingestion manager), which this agent did not read. Verify where those caps live before trusting them for untrusted uploads.
5. openpyxl usage is full-load (`read_only=False`); no per-workbook size cap inside the parser. Zip-bomb exposure for xlsx relies on the same upstream caps as DOCX/ZIP members.
6. MetaTrader5 per-function page (`copy_rates_from`) 404'd; field names/dtypes are corroborated by the mql5 overview example and by project code that accesses `r['time']`, `r['open']`, etc. successfully. Exact numeric dtypes of the structured array (e.g., uint32 vs int64 for `tick_volume`) remain unconfirmed from official docs.
