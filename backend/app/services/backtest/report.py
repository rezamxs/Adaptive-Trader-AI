"""Markdown Backtest Report Generator."""
from pathlib import Path
from typing import Any, Dict
from backend.app.core.config import settings


class BacktestReportGenerator:
    """Generates standardized markdown reports detailing backtest results and assumptions."""

    @staticmethod
    def generate_markdown(backtest_result: Dict[str, Any], stress_result: Dict[str, Any] = None) -> str:
        """Constructs markdown summary with Persian/English bilingual metrics."""
        m = backtest_result["metrics"]
        a = backtest_result["assumptions"]
        lims = backtest_result["limitations"]

        lines = [
            f"# گزارش تحلیلی بکتست استراتژی ({backtest_result['strategy_name']})",
            f"**شناسه اجرا:** `{backtest_result['run_id']}`  ",
            f"**نماد و تایم‌فریم:** {backtest_result['symbol']} | {backtest_result['timeframe']}  ",
            "",
            "## ۱. شاخص‌های آماری و عملکرد (Performance Summary)",
            "| شاخص | مقدار | توضیحات |",
            "| :--- | :--- | :--- |",
            f"| **موجودی اولیه (Initial Balance)** | ${m['initial_balance']:,.2f} | سرمایه پایه شبیه‌سازی |",
            f"| **موجودی نهایی (Final Balance)** | ${m['final_balance']:,.2f} | ارزش حساب پس از هزینه‌ها |",
            f"| **سود خالص (Net Profit)** | ${m['net_profit']:,.2f} | با کسر اسپرد، کمیسیون و لغزش |",
            f"| **حداکثر افت سرمایه (Max Drawdown)** | {m['max_drawdown_percent']}% (${m['max_drawdown_amount']:,.2f}) | افت از قله اکوئیتی |",
            f"| **فاکتور سود (Profit Factor)** | {m['profit_factor']} | نسبت سود ناخالص به زیان ناخالص |",
            f"| **نرخ برد (Win Rate)** | {m['win_rate']}% | درصد معاملات سودده |",
            f"| **امید ریاضی (Expectancy)** | ${m['expectancy']:,.2f} | سود مورد انتظار در هر معامله |",
            f"| **کل معاملات (Total Trades)** | {m['total_trades']} | {m['winning_trades']} برنده / {m['losing_trades']} بازنده |",
            f"| **میانگین زمان نگهداری** | {m['avg_holding_bars']} کندل | افق زمانی هر موقعیت |",
            "",
            "## ۲. فرض‌ها و پارامترهای هزینه (Assumptions & Cost Model)",
            f"- **اسپرد منظور شده:** {a['spread_points']} پوینت",
            f"- **لغزش قیمت (Slippage):** {a['slippage_points']} پوینت در ورود و خروج",
            f"- **کمیسیون کارگزار:** ${a['commission_per_lot']} به ازای هر لات استاندارد",
            f"- **سیاست برخورد همزمان (Collision Policy):** {a['same_bar_collision_policy']} (محافظه‌کارانه)",
            "",
            "## ۳. محدودیت‌ها و هشدار آماری (Limitations)",
        ]

        for lim in lims:
            lines.append(f"- ⚠️ {lim}")

        if stress_result and "scenarios" in stress_result:
            lines.append("")
            lines.append("## ۴. نتایج تحلیل استرس و حساسیت (Stress Testing Scenarios)")
            lines.append("| سناریو | سود خالص | فاکتور سود | حداکثر دراودان | نرخ برد |")
            lines.append("| :--- | :--- | :--- | :--- | :--- |")
            for sc_name, sc_data in stress_result["scenarios"].items():
                lines.append(f"| {sc_name} | ${sc_data['net_profit']:,.2f} | {sc_data['profit_factor']} | {sc_data['max_drawdown_percent']}% | {sc_data['win_rate']}% |")

        report_content = "\n".join(lines)

        # Save to artifacts/reports
        save_dir = settings.ARTIFACTS_REPORTS_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        report_path = save_dir / f"{backtest_result['run_id']}_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        return report_content
