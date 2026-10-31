"""Stage-2 dashboard regression tests (audit M17/M18/M19 + UI minor bugs).

Each test executes the REAL dashboard JavaScript in node with a minimal DOM
stub and a scripted fetch queue, so assertions cover actual handler behavior
(alerts shown, forms kept, fetch targets, button state, rendered HTML) —
never just the presence of strings like "res.ok".
"""
import base64
import calendar
import json
import re
import subprocess
import tempfile
import os
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[3] / "frontend" / "index.html"


def _dashboard_script() -> str:
    src = FRONTEND.read_text(encoding="utf-8")
    match = re.search(r"<script>\n(.*)</script>", src, re.S)
    assert match, "dashboard <script> block not found"
    return match.group(1)


# DOM / fetch stubs prepended before the real dashboard script.
_STUBS = r"""
const __state = { alerts: [], fetches: [], queue: [] };
function alert(msg) { __state.alerts.push(String(msg)); }
function prompt(msg, def) { return null; }
const __elements = {};
function __el(id) {
  if (!__elements[id]) {
    __elements[id] = { id, innerText: "", innerHTML: "", value: "",
                       disabled: false, className: "", style: {}, files: [] };
  }
  return __elements[id];
}
const document = { getElementById: __el, querySelectorAll: () => [] };
const window = { onload: null };
class FormData { append(k, v) { (this._e = this._e || []).push([k, v]); } }
function __respond(status, body) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}
function __pend() {
  let resolve;
  const p = new Promise(r => { resolve = r; });
  p.__resolve = resolve;
  return p;
}
async function fetch(url, options) {
  __state.fetches.push({ url: String(url), options });
  const next = __state.queue.shift();
  if (!next) throw new Error("no scripted response for " + url);
  return next;
}
function __statusData(active) {
  return { app_name: "t", environment: "dev", execution_mode: "SIGNAL_ONLY",
           demo_trading_gate_enabled: false, database_connected: true,
           liveness: "LIVE", readiness: "READY",
           worker: { liveness: "LIVE", readiness: "READY", symbol: "XAUUSD", timeframe: "M1" },
           emergency_stop: { is_active: active, reason: "r", activated_at: null },
           mt5: { terminal_path: "p", is_connected: false, account_verified_demo: false,
                  account_trade_mode: null, server: "s", currency: "USD",
                  balance: 100, equity: 100, login_masked: "x" },
           server_time_utc: "2026-10-15T10:00:00.000Z" };
}
"""

_DRIVER_HEAD = """
(async () => {
  const results = {};
"""
_DRIVER_TAIL = """
  console.log("RESULT:" + Buffer.from(JSON.stringify(results), "utf-8").toString("base64"));
})().catch(e => { console.error("DRIVER_FAIL: " + ((e && e.stack) || e)); process.exit(1); });
"""


def _run_dashboard(driver_body: str) -> dict:
    js = _STUBS + "\n" + _dashboard_script() + "\n" + _DRIVER_HEAD + driver_body + _DRIVER_TAIL
    fd, path = tempfile.mkstemp(suffix=".js")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(js)
        res = subprocess.run(
            ["node", path], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=60,
        )
    finally:
        os.unlink(path)
    assert res.returncode == 0, f"node driver failed:\n{res.stderr[-3000:]}"
    result_line = next(line for line in res.stdout.splitlines() if line.startswith("RESULT:"))
    return json.loads(base64.b64decode(result_line[len("RESULT:"):]))


def _utc_seconds(*args) -> int:
    return calendar.timegm(args)


# ---------------------------------------------------------------------------
# M17: failed HTTP responses must surface the backend detail, never a fake
# success, and the form must survive for retry.
# ---------------------------------------------------------------------------

def test_failed_http_responses_show_backend_detail_and_keep_form():
    results = _run_dashboard(r"""
    // upload 401: error alert, form NOT cleared, no success message
    __el("input-doc-file").files = [{}];
    __el("input-doc-file").value = "C:/fake/report.pdf";
    __el("input-doc-desc").value = "desc";
    __state.queue.push(__respond(401, { detail: "Invalid API token." }));
    await uploadDocument(__el("btn-upload"));
    results.uploadAlert = __state.alerts[0];
    results.fileKept = __el("input-doc-file").value;
    results.descKept = __el("input-doc-desc").value;
    results.uploadBtnEnabled = !__el("btn-upload").disabled;

    // train 400: backend detail shown
    __state.alerts.length = 0;
    __el("ml-symbol").value = "EURUSD"; __el("ml-algo").value = "baseline"; __el("ml-bars").value = "300";
    __state.queue.push(__respond(400, { detail: "No approved strategy rules" }));
    await trainModel(__el("btn-train"));
    results.trainAlert = __state.alerts[0];
    results.trainBtnEnabled = !__el("btn-train").disabled;

    // backtest 400: readable error instead of TypeError on undefined metrics
    __state.alerts.length = 0;
    __el("bt-symbol").value = "EURUSD"; __el("bt-spread").value = "15";
    __el("bt-slip").value = "3"; __el("bt-comm").value = "7";
    __state.queue.push(__respond(400, { detail: "No approved strategy rules for this symbol" }));
    await runBacktest(__el("btn-bt"));
    results.btAlert = __state.alerts[0];

    // 422 pydantic array detail stays readable
    __state.alerts.length = 0;
    __state.queue.push(__respond(422, { detail: [
        { msg: "ensure this value is greater than or equal to 100" }] }));
    await trainModel(__el("btn-train"));
    results.val422Alert = __state.alerts[0];
    """)
    assert "Invalid API token." in results["uploadAlert"]
    assert "موفقیت" not in results["uploadAlert"]
    assert results["fileKept"] == "C:/fake/report.pdf", "form must not be cleared on failure"
    assert results["descKept"] == "desc"
    assert results["uploadBtnEnabled"] is True, "retry must be possible after failure"
    assert "No approved strategy rules" in results["trainAlert"]
    assert results["trainBtnEnabled"] is True
    assert "No approved strategy rules for this symbol" in results["btAlert"]
    assert "greater than or equal to 100" in results["val422Alert"]


# ---------------------------------------------------------------------------
# M17: numeric metric access must be robust against incomplete payloads.
# ---------------------------------------------------------------------------

def test_backtest_metrics_robust_to_missing_fields():
    results = _run_dashboard(r"""
    __el("bt-symbol").value = "EURUSD"; __el("bt-spread").value = "15";
    __el("bt-slip").value = "3"; __el("bt-comm").value = "7";
    __state.queue.push(__respond(200, { run_id: "r1",
        metrics: { net_profit: 123.456 },
        stress_test: { scenarios: { base: { net_profit: 5, profit_factor: 1.2,
                                            max_drawdown_percent: 3, win_rate: 55 } } } }));
    await runBacktest(__el("btn-bt"));
    results.net = __el("bt-net-profit").innerText;
    results.dd = __el("bt-max-dd").innerText;
    results.pf = __el("bt-profit-factor").innerText;
    results.wr = __el("bt-win-rate").innerText;

    // Response without metrics at all must not throw.
    __state.queue.push(__respond(200, { run_id: "r2" }));
    await runBacktest(__el("btn-bt"));
    results.net2 = __el("bt-net-profit").innerText;

    // Null stress scenario entry must not throw either.
    __state.queue.push(__respond(200, { run_id: "r3", metrics: { net_profit: 1 },
        stress_test: { scenarios: { base: null } } }));
    await runBacktest(__el("btn-bt"));
    results.stressHtml = __el("tbody-stress").innerHTML;
    results.errorAlertCount = __state.alerts.length;
    """)
    assert results["net"] == "$123.46"
    assert results["dd"].startswith("-"), "missing drawdown must render '-' not crash"
    assert results["pf"] == "-"
    assert results["wr"].startswith("-") and "-/-" in results["wr"]
    assert results["net2"] == "-"
    assert results["errorAlertCount"] == 0, (
        "missing metrics and null stress scenarios must render, never alert")


# ---------------------------------------------------------------------------
# M18: double submit must be blocked; button must be restored for retry.
# ---------------------------------------------------------------------------

def test_double_submit_blocked_and_button_restored():
    results = _run_dashboard(r"""
    const pend = __pend();
    __state.queue.push(pend);                                // the POST itself
    __state.queue.push(__respond(200, __statusData(false))); // the GET in finally
    const p1 = toggleEmergencyStop();
    const p2 = toggleEmergencyStop();                        // must early-return
    pend.__resolve(__respond(200, {}));
    await Promise.all([p1, p2]);
    results.postCount = __state.fetches.filter(f => f.options && f.options.method === "POST").length;
    results.btnEnabled = !__el("btn-emergency-toggle").disabled;
    """)
    assert results["postCount"] == 1, "second click during in-flight request must be ignored"
    assert results["btnEnabled"] is True, "button must be re-enabled for retry"


# ---------------------------------------------------------------------------
# M18: emergency-stop intent comes from server state; a failed toggle must
# never flip the visible state.
# ---------------------------------------------------------------------------

def test_emergency_stop_intent_from_server_state_and_failure_keeps_state():
    results = _run_dashboard(r"""
    // Intent: uiState says active even though the button text is misleading.
    uiState.emergencyStopActive = true;
    __el("btn-emergency-toggle").innerText = "⛔ توقف اضطراری (غیرفعال)";
    __state.queue.push(__respond(200, {}));
    __state.queue.push(__respond(200, __statusData(true)));
    await toggleEmergencyStop();
    results.intentUrl = __state.fetches[0].url;
    results.btnTextAfter = __el("btn-emergency-toggle").innerText;

    // Failure: activate fails with 500; visuals must stay at the server state.
    __state.fetches.length = 0; __state.alerts.length = 0;
    uiState.emergencyStopActive = false;
    __el("btn-emergency-toggle").innerText = "⛔ توقف اضطراری (غیرفعال)";
    __el("btn-emergency-toggle").className = "btn-emergency";
    __state.queue.push(__respond(500, { detail: "boom" }));
    __state.queue.push(__respond(200, __statusData(false)));
    await toggleEmergencyStop();
    results.failAlert = __state.alerts[0];
    results.btnText = __el("btn-emergency-toggle").innerText;
    results.btnClass = __el("btn-emergency-toggle").className;
    results.uiStateActive = uiState.emergencyStopActive;
    results.btnEnabled = !__el("btn-emergency-toggle").disabled;
    """)
    assert results["intentUrl"].endswith("/system/emergency-stop/deactivate"), (
        "intent must come from uiState, not button text")
    assert "رفع قفل" in results["btnTextAfter"], "button must re-render from server state"
    assert "boom" in results["failAlert"]
    assert results["btnText"] == "⛔ توقف اضطراری (غیرفعال)", "failed toggle must not flip visuals"
    assert results["btnClass"] == "btn-emergency"
    assert results["uiStateActive"] is False
    assert results["btnEnabled"] is True


# ---------------------------------------------------------------------------
# M19: naive-UTC backend timestamps are never rendered as local time; values
# that already carry Z/offset pass through; one formatter everywhere.
# ---------------------------------------------------------------------------

def test_utc_display_is_explicit_and_consistent():
    results = _run_dashboard(r"""
    results.naive = new Date(asUtcIso("2026-10-15T10:00:00")).getTime();
    results.fraction = new Date(asUtcIso("2026-10-15T10:00:00.500")).getTime();
    results.zulu = new Date(asUtcIso("2026-10-15T10:00:00Z")).getTime();
    results.offset = new Date(asUtcIso("2026-10-15T10:00:00+03:30")).getTime();
    results.spaced = new Date(asUtcIso("2026-10-09 23:53:00")).getTime();
    results.invalid = fmtUtcTime("not-a-date");
    results.fmt = fmtUtcTime("2026-10-15T10:00:00");

    // Render path: the models table formats trained_at through fmtUtcTime.
    __state.queue.push(__respond(200, [{ model_id: "m1", algorithm: "baseline",
        status: "candidate", test_metrics: { roc_auc: 0.5 },
        trained_at: "2026-10-15T10:00:00" }]));
    await loadModels();
    results.modelsHtml = __el("tbody-models").innerHTML;
    """)
    assert results["naive"] == _utc_seconds(2026, 10, 15, 10, 0, 0) * 1000
    assert results["fraction"] == _utc_seconds(2026, 10, 15, 10, 0, 0) * 1000 + 500
    assert results["zulu"] == _utc_seconds(2026, 10, 15, 10, 0, 0) * 1000, "Z values pass through"
    assert results["offset"] == _utc_seconds(2026, 10, 15, 6, 30, 0) * 1000, "offset preserved"
    assert results["spaced"] == _utc_seconds(2026, 10, 9, 23, 53, 0) * 1000, "space-separated form handled"
    assert results["invalid"] == "not-a-date"
    assert results["fmt"].endswith(" UTC")
    assert " UTC" in results["modelsHtml"], "rendered table must label times as UTC"

    src = _dashboard_script()
    assert src.count("new Date(") == 1, (
        "every timestamp parse must go through fmtUtcTime/asUtcIso — "
        "no ad-hoc new Date() left in handlers")


# ---------------------------------------------------------------------------
# M19: the worker telemetry clocks must render through fmtUtcTime like every
# other dashboard time sink.
# ---------------------------------------------------------------------------

def test_worker_telemetry_times_render_through_fmt_utctime():
    """M19: last_cycle_time / last_processed_bar_time must go through
    fmtUtcTime — an offset-carrying stamp must display its UTC wall time,
    never the raw broker wall time mislabeled as UTC (the old
    substring(11,19) path did exactly that)."""
    results = _run_dashboard(r"""
    const data = __statusData(false);
    data.worker.last_cycle_time = "2026-10-15T12:30:00+03:30";
    data.worker.last_processed_bar_time = "2026-10-15T12:30:00+03:30";
    __state.queue.push(__respond(200, data));
    await loadSystemStatus();
    results.cycleText = __el("val-worker-cycle-time").innerText;
    results.barText = __el("val-worker-bar").innerText;
    """)
    for key in ("cycleText", "barText"):
        assert " UTC" in results[key], f"{key} must be labeled UTC"
        assert "12:30" not in results[key] and "۱۲:۳۰" not in results[key], (
            f"{key} must not show the raw +03:30 wall time as UTC")


# ---------------------------------------------------------------------------
# Minor: loader failures must replace "loading..." with an error row.
# ---------------------------------------------------------------------------

def test_loader_failures_render_error_row_not_eternal_loading():
    results = _run_dashboard(r"""
    __state.queue.push(Promise.reject(new Error("network down")));
    await loadRules();
    results.rulesHtml = __el("tbody-rules").innerHTML;

    __state.queue.push(__respond(200, { detail: "not a list" }));  // non-array JSON
    await loadModels();
    results.modelsHtml = __el("tbody-models").innerHTML;

    __state.queue.push(__respond(500, { detail: "db down" }));
    await loadAuditLogs();
    results.auditHtml = __el("tbody-audit").innerHTML;
    """)
    for key in ("rulesHtml", "modelsHtml", "auditHtml"):
        assert "خطا" in results[key], f"{key} must show an error row"
        assert "در حال بارگذاری" not in results[key], f"{key} must not stay in loading state"


# ---------------------------------------------------------------------------
# Minor: invalid numeric inputs are rejected before any request is sent.
# ---------------------------------------------------------------------------

def test_invalid_numeric_inputs_rejected_before_any_request():
    results = _run_dashboard(r"""
    __el("ml-symbol").value = "EURUSD"; __el("ml-algo").value = "baseline";
    __el("ml-bars").value = "abc";
    await trainModel(__el("btn-train"));
    results.trainFetchCount = __state.fetches.length;
    results.trainAlert = __state.alerts[0];

    __el("bt-symbol").value = "EURUSD"; __el("bt-spread").value = "abc";
    __el("bt-slip").value = "3"; __el("bt-comm").value = "7";
    await runBacktest(__el("btn-bt"));
    results.btFetchCount = __state.fetches.length;
    results.btAlert = __state.alerts[1];
    """)
    assert results["trainFetchCount"] == 0, "invalid bars must not trigger a request"
    assert "نامعتبر" in results["trainAlert"]
    assert results["btFetchCount"] == 0, "invalid spread must not trigger a request"
    assert "نامعتبر" in results["btAlert"]