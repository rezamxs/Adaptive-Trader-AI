    const API_BASE = "/api/v1";

    // C1: state-changing endpoints require the X-API-TOKEN header. The token
    // lives ONLY in this page's memory: it never touches localStorage or
    // sessionStorage (XSS-resilient) and a page reload requires re-entry.
    // No server-side session exists, so a HttpOnly cookie swap is out of scope.
    let apiToken = "";

    function getApiToken() {
      return apiToken;
    }

    function setApiToken() {
      const value = prompt("توکن API را وارد کنید (پس از هر بارگذاری مجدد صفحه دوباره لازم است):", "");
      if (value === null) return;
      apiToken = value.trim();
      loadSystemStatus();
    }

    function authHeaders(extra) {
      const headers = Object.assign({}, extra || {});
      const token = getApiToken();
      if (token) headers["X-API-TOKEN"] = token;
      return headers;
    }

    // Mirrors the last server-known state so click handlers never infer intent
    // from rendered button text (M18).
    var uiState = { emergencyStopActive: false };

    // Throws a readable Error for any non-2xx response so no handler can ever
    // render a success message from a failed request (M17). FastAPI error
    // bodies are {"detail": <string | array>}.
    async function ensureOk(res) {
      if (res.ok) return res;
      let message = "HTTP " + res.status;
      try {
        const body = await res.json();
        if (body && body.detail) {
          message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
        }
      } catch (_) { /* non-JSON error body */ }
      const err = new Error(message);
      err.status = res.status;  // item 44: 401 is NOT "backend down"
      throw err;
    }

    // Numeric formatter that never throws on missing/invalid values (M17).
    function fmtFixed(value, digits) {
      return (typeof value === "number" && isFinite(value)) ? value.toFixed(digits) : "-";
    }

    // Backend timestamps are naive UTC (datetime.utcnow().isoformat(), no
    // offset). Offset-less strings are tagged as UTC before parsing so they
    // are never misread as local time; strings that already carry Z or an
    // explicit offset pass through untouched (M19).
    function asUtcIso(value) {
      if (typeof value !== "string") return value;
      if (/^[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9:.]+$/.test(value)) {
        return value.replace(" ", "T") + "Z";
      }
      return value;
    }

    // The single formatter for every dashboard clock: always explicit UTC.
    function fmtUtcTime(value) {
      const parsed = new Date(asUtcIso(value));
      if (isNaN(parsed.getTime())) return esc(String(value ?? "-"));
      return parsed.toLocaleTimeString("fa-IR", {timeZone: "UTC"}) + " UTC";
    }

    function switchTab(tabName) {
      document.querySelectorAll(".nav-tab").forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
      
      const btn = Array.from(document.querySelectorAll(".nav-tab")).find(b => b.innerText.includes(getTabTitle(tabName)));
      if (btn) btn.classList.add("active");
      
      const content = document.getElementById("tab-" + tabName);
      if (content) content.classList.add("active");

      // Auto-refresh data on tab change
      if (tabName === "system") loadSystemStatus();
      if (tabName === "knowledge") loadRules();
      if (tabName === "data-ml") loadModels();
      if (tabName === "signals") loadSignals();
      if (tabName === "settings") loadAuditLogs();
    }

    function getTabTitle(name) {
      if (name === "system") return "وضعیت";
      if (name === "knowledge") return "اسناد";
      if (name === "data-ml") return "داده";
      if (name === "backtest") return "بکتست";
      if (name === "signals") return "سیگنال";
      if (name === "settings") return "تنظیمات";
      return "";
    }

    async function loadSystemStatus() {
      try {
        const res = await fetch(`${API_BASE}/system/status`, { headers: authHeaders() });
        await ensureOk(res);
        const data = await res.json();

        // Backend Liveness
        const badgeBackend = document.getElementById("badge-backend");
        if (badgeBackend) {
          badgeBackend.innerText = `Backend: ${data.liveness || "LIVE"}`;
          badgeBackend.className = "badge badge-success";
        }

        // Worker Telemetry
        const worker = data.worker || {};
        const badgeWorker = document.getElementById("badge-worker");
        if (badgeWorker) {
          const isWorkerLive = worker.liveness === "LIVE";
          badgeWorker.innerText = `Worker: ${worker.liveness || "STOPPED"}`;
          badgeWorker.className = isWorkerLive ? "badge badge-success" : "badge badge-warning";
        }

        const elWorkerStatus = document.getElementById("val-worker-status");
        if (elWorkerStatus) {
          elWorkerStatus.innerText = `${worker.liveness || "STOPPED"} (${worker.readiness || "UNKNOWN"})`;
          elWorkerStatus.style.color = worker.readiness === "READY" ? "var(--success)" : "var(--warning)";
        }
        const elWorkerInfo = document.getElementById("val-worker-info");
        if (elWorkerInfo) {
          elWorkerInfo.innerText = `نماد: ${worker.symbol || "XAUUSD"} | ${worker.timeframe || "M1"} | بازه پایش: ${worker.poll_interval_seconds || 3}s`;
        }

        const elWorkerCycles = document.getElementById("val-worker-cycles-count");
        if (elWorkerCycles) {
          elWorkerCycles.innerText = `${worker.successful_cycles || 0} موفق / ${worker.total_cycles || 0} کل`;
        }
        const elWorkerCycleTime = document.getElementById("val-worker-cycle-time");
        if (elWorkerCycleTime) {
          elWorkerCycleTime.innerText = worker.last_cycle_time ? `آخرین چرخه: ${fmtUtcTime(worker.last_cycle_time)}` : "هنوز اجرا نشده";
        }

        const elWorkerBar = document.getElementById("val-worker-bar");
        if (elWorkerBar) {
          elWorkerBar.innerText = worker.last_processed_bar_time ? `کندل: ${fmtUtcTime(worker.last_processed_bar_time)}` : "در انتظار کندل جدید";
        }
        const elWorkerBarKey = document.getElementById("val-worker-bar-key");
        if (elWorkerBarKey) {
          elWorkerBarKey.innerText = worker.last_processed_bar_key ? `کلید: ${worker.last_processed_bar_key}` : `کد وضعیت: ${worker.last_status_code || "-"}`;
        }

        const elWorkerDecision = document.getElementById("val-worker-decision");
        if (elWorkerDecision) {
          elWorkerDecision.innerText = worker.last_decision || "HOLD";
          elWorkerDecision.style.color = (worker.last_decision === "BUY" ? "var(--success)" : (worker.last_decision === "SELL" ? "var(--danger)" : "var(--text-main)"));
        }
        const elWorkerReason = document.getElementById("val-worker-reason");
        if (elWorkerReason) {
          elWorkerReason.innerText = worker.last_reason || "در انتظار چرخه";
        }

        document.getElementById("badge-mode").innerText = data.execution_mode;
        document.getElementById("cfg-mode").innerText = data.execution_mode;
        document.getElementById("cfg-demo-gate").innerText = data.demo_trading_gate_enabled ? "فعال" : "غیرفعال (پیش‌فرض امن)";
        document.getElementById("cfg-mt5-path").innerText = data.mt5.terminal_path;

        // MT5 info
        const isMt5 = data.mt5.is_connected;
        document.getElementById("badge-mt5").innerText = isMt5 ? "MT5: متصل" : "MT5: قطع";
        document.getElementById("badge-mt5").className = isMt5 ? "badge badge-success" : "badge badge-danger";
        document.getElementById("val-mt5-status").innerText = isMt5 ? "متصل به سرور" : "قطع اتصال";
        document.getElementById("val-mt5-server").innerText = `سرور: ${data.mt5.server}`;

        // Account info
        document.getElementById("val-account-type").innerText = data.mt5.account_verified_demo ? "حساب دمو (Demo)" : "نامشخص / غیر دمو";
        document.getElementById("val-account-login").innerText = `شناسه کاربری: ${data.mt5.login_masked}`;
        document.getElementById("val-account-balance").innerText = `$${data.mt5.balance.toLocaleString()} ${data.mt5.currency}`;
        document.getElementById("val-account-equity").innerText = `اکوئیتی: $${data.mt5.equity.toLocaleString()}`;

        // Emergency Stop
        const isStop = data.emergency_stop.is_active;
        uiState.emergencyStopActive = isStop;
        const btnStop = document.getElementById("btn-emergency-toggle");
        if (isStop) {
          btnStop.innerText = "🚨 رفع قفل اضطراری";
          btnStop.className = "btn-emergency inactive-mode";
          document.getElementById("val-circuit-status").innerText = "فعال (مسدود)";
          document.getElementById("val-circuit-status").style.color = "var(--danger)";
        } else {
          btnStop.innerText = "⛔ توقف اضطراری (غیرفعال)";
          btnStop.className = "btn-emergency";
          document.getElementById("val-circuit-status").innerText = "آماده‌به‌کار (عادی)";
          document.getElementById("val-circuit-status").style.color = "var(--success)";
        }
        document.getElementById("val-circuit-reason").innerText = data.emergency_stop.reason;
      } catch (e) {
        console.error("Status load failed", e);
        const badgeBackend = document.getElementById("badge-backend");
        if (badgeBackend) {
          // Buglist item 44: a 401 means authentication failed (wrong/missing
          // token) — the backend is REACHABLE; showing "قطع ارتباط" sent the
          // operator debugging the wrong layer.
          if (e && e.status === 401) {
            badgeBackend.innerText = "Backend: دسترسی رد شد (توکن نامعتبر)";
            badgeBackend.className = "badge badge-warning";
          } else {
            badgeBackend.innerText = "Backend: قطع ارتباط";
            badgeBackend.className = "badge badge-danger";
          }
        }
        const badgeWorker = document.getElementById("badge-worker");
        if (badgeWorker) {
          badgeWorker.innerText = "Worker: متوقف";
          badgeWorker.className = "badge badge-danger";
        }
        const elWorkerStatus = document.getElementById("val-worker-status");
        if (elWorkerStatus) {
          elWorkerStatus.innerText = "عدم پاسخ سرور";
          elWorkerStatus.style.color = "var(--danger)";
        }
      }
    }

    async function toggleEmergencyStop() {
      const btn = document.getElementById("btn-emergency-toggle");
      if (!btn || btn.disabled) return;  // double-submit guard (M18)
      btn.disabled = true;
      try {
        // Intent comes from the last server state, never from button text (M18).
        const action = uiState.emergencyStopActive ? "deactivate" : "activate";
        const res = await fetch(`${API_BASE}/system/emergency-stop/${action}`, {
          method: "POST",
          headers: authHeaders({"Content-Type": "application/json"}),
          body: action === "deactivate"
            ? JSON.stringify({notes: "Resumed from dashboard by user"})
            : JSON.stringify({reason: "Manual emergency stop triggered via Web Dashboard", actor: "USER"})
        });
        await ensureOk(res);
      } catch (e) {
        alert("خطا در تغییر وضعیت توقف اضطراری: " + (e.message || e));
      } finally {
        // Re-render from the real server state on success AND failure, so the
        // visible emergency-stop state can never be flipped by a failed call.
        await loadSystemStatus();
        btn.disabled = false;
      }
    }

    async function uploadDocument(btn) {
      const fileInput = document.getElementById("input-doc-file");
      const descInput = document.getElementById("input-doc-desc");
      if (!fileInput.files[0]) {
        alert("لطفاً یک فایل انتخاب کنید.");
        return;
      }
      if (btn && btn.disabled) return;  // double-submit guard (M18)
      if (btn) btn.disabled = true;

      const fd = new FormData();
      fd.append("file", fileInput.files[0]);
      fd.append("description", descInput.value);

      try {
        const res = await fetch(`${API_BASE}/documents/upload`, {
          method: "POST",
          headers: authHeaders(),
          body: fd,
        });
        await ensureOk(res);
        const data = await res.json();
        const count = (data && typeof data.rules_extracted_count === "number") ? data.rules_extracted_count : 0;
        alert(`سند با موفقیت پردازش شد. تعداد ${count} قانون استخراج گردید.`);
        fileInput.value = "";
        descInput.value = "";
        loadRules();
      } catch (e) {
        // On failure the form keeps its values so the user can retry (M17).
        alert("خطا در بارگذاری سند: " + (e.message || e));
      } finally {
        if (btn) btn.disabled = false;
      }
    }

    // HTML-escapes server-controlled values before any innerHTML sink (XSS fix).
    // Aliased to `esc` for use inside template literals.
    function escapeHtml(value) {
      if (value === null || value === undefined) return "";
      return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;")
        .replace(/`/g, "&#96;");
    }
    const esc = escapeHtml;

    async function loadRules() {
      try {
        const res = await fetch(`${API_BASE}/rules/`, { headers: authHeaders() });
        await ensureOk(res);
        const rules = await res.json();
        if (!Array.isArray(rules)) throw new Error("پاسخ غیرمنتظره سرور");
        const tbody = document.getElementById("tbody-rules");
        if (rules.length === 0) {
          tbody.innerHTML = `<tr><td colspan="7" style="text-align: center;">قانونی یافت نشد. سندی بارگذاری کنید.</td></tr>`;
          return;
        }

        tbody.innerHTML = rules.map(r => {
          let badgeClass = "badge-warning";
          if (r.approval_status === "approved") badgeClass = "badge-success";
          if (r.approval_status === "rejected") badgeClass = "badge-danger";

          const ambigText = r.ambiguities && r.ambiguities.length > 0
            ? `<span style="color: #fbbf24;">⚠️ ${r.ambiguities.map(esc).join('<br>')}</span>`
            : "صریح و بدون ابهام";

          const safeRuleId = esc(r.rule_id);
          const actions = r.approval_status === "pending" ? `
            <button class="btn btn-primary btn-sm" data-action="approve-rule" data-rule-id="${safeRuleId}">تأیید</button>
            <button class="btn btn-emergency btn-sm" data-action="reject-rule" data-rule-id="${safeRuleId}" style="margin-right: 0.25rem;">رد</button>
          ` : `<span style="color: var(--text-muted); font-size: 0.8rem;">تکمیل‌شده</span>`;

          return `
            <tr>
              <td><code>${esc(r.rule_id)}</code></td>
              <td><strong>${esc(r.name)}</strong></td>
              <td>${esc(r.category)}</td>
              <td>صفحه ${esc((r.source_pages || []).join(', ') || '1')}</td>
              <td style="max-width: 250px; font-size: 0.8rem;">${ambigText}</td>
              <td><span class="badge ${badgeClass}">${esc(r.approval_status)}</span></td>
              <td>${actions}</td>
            </tr>
          `;
        }).join("");
      } catch (e) {
        console.error("Rules load error", e);
        const tbody = document.getElementById("tbody-rules");
        if (tbody) tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--danger);">خطا در بارگذاری قوانین — با تعویض تب دوباره تلاش کنید</td></tr>`;
      }
    }

    async function approveRule(ruleId, btn) {
      if (btn && btn.disabled) return;  // double-submit guard (M18)
      if (btn) btn.disabled = true;
      try {
        const res = await fetch(`${API_BASE}/rules/${ruleId}/approve`, {
          method: "POST",
          headers: authHeaders({"Content-Type": "application/json"}),
          body: JSON.stringify({actor: "USER"})
        });
        await ensureOk(res);
      } catch (e) {
        alert("خطا در تأیید قانون: " + (e.message || e));
      } finally {
        if (btn) btn.disabled = false;
        loadRules();
      }
    }

    async function rejectRule(ruleId, btn) {
      // Buglist item 44: an empty/whitespace rejection reason must never
      // reach the API (it is required evidence for the audit trail).
      const reason = (prompt("علت رد این قانون را وارد کنید:", "قانون مبهم است یا با سیستم ناسازگار است.") || "").trim();
      if (!reason) {
        alert("علت رد نمی‌تواند خالی باشد؛ یک دلیل بنویسید.");
        return;
      }
      if (btn && btn.disabled) return;  // double-submit guard (M18)
      if (btn) btn.disabled = true;
      try {
        const res = await fetch(`${API_BASE}/rules/${ruleId}/reject`, {
          method: "POST",
          headers: authHeaders({"Content-Type": "application/json"}),
          body: JSON.stringify({reason: reason, actor: "USER"})
        });
        await ensureOk(res);
      } catch (e) {
        alert("خطا در رد قانون: " + (e.message || e));
      } finally {
        if (btn) btn.disabled = false;
        loadRules();
      }
    }

    async function trainModel(btn) {
      const sym = document.getElementById("ml-symbol").value;
      const algo = document.getElementById("ml-algo").value;
      const barsNum = Number.parseInt(document.getElementById("ml-bars").value, 10);
      if (!Number.isFinite(barsNum) || barsNum <= 0) {
        alert("تعداد کندل‌های آموزش نامعتبر است؛ یک عدد مثبت وارد کنید.");
        return;
      }
      if (btn && btn.disabled) return;  // double-submit guard (M18)
      if (btn) btn.disabled = true;

      try {
        const res = await fetch(`${API_BASE}/ml/train`, {
          method: "POST",
          headers: authHeaders({"Content-Type": "application/json"}),
          body: JSON.stringify({symbol: sym, algorithm: algo, bars_count: barsNum})
        });
        await ensureOk(res);
        const data = await res.json();
        const modelId = (data && data.model_id) ? data.model_id : "-";
        const auc = (data && data.test_metrics && data.test_metrics.roc_auc != null)
          ? data.test_metrics.roc_auc : "N/A";
        alert(`آموزش مدل نامزد ${modelId} با موفقیت انجام شد. AUC تست: ${auc}`);
        loadModels();
      } catch (e) {
        alert("خطا در آموزش مدل: " + (e.message || e));
      } finally {
        if (btn) btn.disabled = false;
      }
    }

    async function loadModels() {
      try {
        const res = await fetch(`${API_BASE}/ml/models`, { headers: authHeaders() });
        await ensureOk(res);
        const models = await res.json();
        if (!Array.isArray(models)) throw new Error("پاسخ غیرمنتظره سرور");
        const tbody = document.getElementById("tbody-models");
        if (models.length === 0) {
          tbody.innerHTML = `<tr><td colspan="6" style="text-align: center;">هنوز مدلی آموزش داده نشده است.</td></tr>`;
          return;
        }

        tbody.innerHTML = models.map(m => {
          let badge = m.status === "active" ? "badge-success" : (m.status === "candidate" ? "badge-warning" : "badge-danger");
          const safeModelId = esc(m.model_id);
          const promoteBtn = m.status === "candidate" ? `
            <button class="btn btn-primary btn-sm" data-action="promote-model" data-model-id="${safeModelId}">ارتقا به فعال</button>
          ` : `<span style="color: var(--text-muted); font-size: 0.8rem;">${esc(m.status)}</span>`;

          const metrics = m.test_metrics || {};
          const mStr = `AUC: ${esc(metrics.roc_auc ?? '-')} | Brier: ${esc(metrics.brier_score ?? '-')} | F1: ${esc(metrics.f1 ?? '-')}`;

          return `
            <tr>
              <td><code>${esc(m.model_id)}</code></td>
              <td>${esc(m.algorithm)}</td>
              <td><span class="badge ${badge}">${esc(m.status)}</span></td>
              <td>${mStr}</td>
              <td>${fmtUtcTime(m.trained_at)}</td>
              <td>${promoteBtn}</td>
            </tr>
          `;
        }).join("");
      } catch (e) {
        console.error("Models load error", e);
        const tbody = document.getElementById("tbody-models");
        if (tbody) tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--danger);">خطا در بارگذاری مدل‌ها — با تعویض تب دوباره تلاش کنید</td></tr>`;
      }
    }

    async function promoteModel(modelId, btn) {
      if (btn && btn.disabled) return;  // double-submit guard (M18)
      if (btn) btn.disabled = true;
      try {
        const res = await fetch(`${API_BASE}/ml/models/${modelId}/promote`, {
          method: "POST",
          headers: authHeaders({"Content-Type": "application/json"}),
          body: JSON.stringify({notes: "Manual promotion via dashboard"})
        });
        await ensureOk(res);
      } catch (e) {
        alert("خطا در ارتقای مدل: " + (e.message || e));
      } finally {
        if (btn) btn.disabled = false;
        loadModels();
      }
    }

    async function runBacktest(btn) {
      const sym = document.getElementById("bt-symbol").value;
      const spread = Number.parseInt(document.getElementById("bt-spread").value, 10);
      const slip = Number.parseInt(document.getElementById("bt-slip").value, 10);
      const comm = Number.parseFloat(document.getElementById("bt-comm").value);
      if (![spread, slip, comm].every(Number.isFinite)) {
        alert("ورودی‌های عددی بکتست نامعتبر است؛ برای اسپرد/لغزش/کمیسیون عدد وارد کنید.");
        return;
      }
      if (btn && btn.disabled) return;  // double-submit guard (M18)
      if (btn) btn.disabled = true;

      try {
        const res = await fetch(`${API_BASE}/backtest/run`, {
          method: "POST",
          headers: authHeaders({"Content-Type": "application/json"}),
          body: JSON.stringify({
            symbol: sym,
            spread_points: spread,
            slippage_points: slip,
            commission_per_lot: comm
          })
        });
        await ensureOk(res);
        const data = await res.json();
        const m = data.metrics || {};

        document.getElementById("bt-net-profit").innerText =
          (typeof m.net_profit === "number" && isFinite(m.net_profit)) ? `$${m.net_profit.toFixed(2)}` : "-";
        document.getElementById("bt-max-dd").innerText =
          `${fmtFixed(m.max_drawdown_percent, 1)}% (${fmtFixed(m.max_drawdown_amount, 2)})`;
        document.getElementById("bt-profit-factor").innerText = fmtFixed(m.profit_factor, 2);
        document.getElementById("bt-win-rate").innerText =
          `${fmtFixed(m.win_rate, 1)}% (${m.winning_trades ?? "-"}/${m.total_trades ?? "-"})`;

        // Stress table
        const tbodyStress = document.getElementById("tbody-stress");
        if (data.stress_test && data.stress_test.scenarios) {
          tbodyStress.innerHTML = Object.entries(data.stress_test.scenarios).map(([scName, sc]) => {
            const v = sc || {};
            return `
            <tr>
              <td><strong>${esc(scName)}</strong></td>
              <td style="color: ${v.net_profit >= 0 ? '#34d399' : '#f87171'}">$${fmtFixed(v.net_profit, 2)}</td>
              <td>${esc(v.profit_factor)}</td>
              <td>${esc(v.max_drawdown_percent)}%</td>
              <td>${esc(v.win_rate)}%</td>
            </tr>
          `;}).join("");
        }

        document.getElementById("backtest-results").style.display = "block";
      } catch (e) {
        alert("خطا در بکتست: " + (e.message || e));
      } finally {
        if (btn) btn.disabled = false;
      }
    }

    async function generateSignal(btn) {
      if (btn && btn.disabled) return;  // double-submit guard (M18)
      if (btn) btn.disabled = true;
      try {
        const res = await fetch(`${API_BASE}/signals/generate`, {
          method: "POST",
          headers: authHeaders({"Content-Type": "application/json"}),
          body: JSON.stringify({symbol: "EURUSD"})
        });
        await ensureOk(res);
        const data = await res.json();
        if (data.status === "NO_SIGNAL") {
          alert("نتیجه بررسی: " + (data.message ?? "-"));
        } else {
          alert(`سیگنال صادر شد! جهت: ${data.direction ?? "-"} | نماد: ${data.symbol ?? "-"} | وضعیت ریسک: ${data.risk_status ?? "-"}`);
        }
        loadSignals();
      } catch (e) {
        alert("خطا در صدور سیگنال: " + (e.message || e));
      } finally {
        if (btn) btn.disabled = false;
      }
    }

    async function loadSignals() {
      try {
        const res = await fetch(`${API_BASE}/signals/`, { headers: authHeaders() });
        await ensureOk(res);
        const signals = await res.json();
        if (!Array.isArray(signals)) throw new Error("پاسخ غیرمنتظره سرور");
        const tbody = document.getElementById("tbody-signals");
        if (signals.length === 0) {
          tbody.innerHTML = `<tr><td colspan="9" style="text-align: center;">هنوز سیگنالی تولید نشده است.</td></tr>`;
          return;
        }

        tbody.innerHTML = signals.map(s => {
          const dirBadge = s.direction === 'BUY' ? 'badge-success' : 'badge-danger';
          const riskBadge = s.risk_status === 'APPROVED' ? 'badge-success' : 'badge-danger';
          return `
          <tr>
            <td><code>${esc(s.signal_id)}</code></td>
            <td><strong>${esc(s.symbol)}</strong></td>
            <td><span class="badge ${dirBadge}">${esc(s.direction)}</span></td>
            <td>${esc(s.entry_price)}</td>
            <td>${esc(s.stop_loss)} / ${esc(s.take_profit)}</td>
            <td><span class="badge ${riskBadge}">${esc(s.risk_status)}</span></td>
            <td>${esc(s.calculated_volume || '-')}</td>
            <td><span class="badge badge-info">${esc(s.execution_mode)}</span></td>
            <td>${fmtUtcTime(s.bar_time)}</td>
          </tr>
        `;}).join("");
      } catch (e) {
        console.error("Signals load error", e);
        const tbody = document.getElementById("tbody-signals");
        if (tbody) tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--danger);">خطا در بارگذاری سیگنال‌ها — با تعویض تب دوباره تلاش کنید</td></tr>`;
      }
    }

    async function loadAuditLogs() {
      try {
        const res = await fetch(`${API_BASE}/system/audit-logs`, { headers: authHeaders() });
        await ensureOk(res);
        const logs = await res.json();
        if (!Array.isArray(logs)) throw new Error("پاسخ غیرمنتظره سرور");
        const tbody = document.getElementById("tbody-audit");
        tbody.innerHTML = logs.map(l => `
          <tr>
            <td>#${esc(l.id)}</td>
            <td><strong>${esc(l.event_type)}</strong></td>
            <td><span class="badge badge-warning">${esc(l.severity)}</span></td>
            <td>${esc(l.actor)}</td>
            <td style="font-size: 0.8rem;"><code>${esc(JSON.stringify(l.details))}</code></td>
            <td>${fmtUtcTime(l.created_at)}</td>
          </tr>
        `).join("");
      } catch (e) {
        console.error("Audit load error", e);
        const tbody = document.getElementById("tbody-audit");
        if (tbody) tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--danger);">خطا در بارگذاری لاگ‌ها — با تعویض تب دوباره تلاش کنید</td></tr>`;
      }
    }

    // Initial load
    window.onload = () => {
      loadSystemStatus();
    };
  
    // Minor 18: ONE delegated listener replaces every inline onclick=
    // attribute, so the strict CSP (script-src 'self', no 'unsafe-inline')
    // can protect the whole dashboard without breaking any interaction.
    document.addEventListener("click", (event) => {
      const el = event.target.closest("[data-action]");
      if (!el) return;
      switch (el.dataset.action) {
        case "set-token": setApiToken(); break;
        case "toggle-emergency": toggleEmergencyStop(); break;
        case "switch-tab": switchTab(el.dataset.tab); break;
        case "upload-document": uploadDocument(el); break;
        case "train-model": trainModel(el); break;
        case "run-backtest": runBacktest(el); break;
        case "generate-signal": generateSignal(el); break;
        case "approve-rule": approveRule(el.dataset.ruleId, el); break;
        case "reject-rule": rejectRule(el.dataset.ruleId, el); break;
        case "promote-model": promoteModel(el.dataset.modelId, el); break;
      }
    });
