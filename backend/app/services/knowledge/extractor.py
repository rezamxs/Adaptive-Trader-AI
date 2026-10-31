"""Rule Extractor for trading knowledge text with explicit ambiguity tracking.
Strictly separates objective algorithmic conditions from subjective trader terms.
"""
import re
import uuid
from typing import Any, Dict, List, Optional
from backend.app.schemas.rule_schema import RuleCategory, TradingRuleCreate

# Ambiguous keywords that must never be converted to arbitrary conditions without user verification
SUBJECTIVE_PATTERNS = {
    "strong trend": "اصطلاح 'روند قوی' (Strong Trend) مبهم است؛ نیاز به تعریف صریح بازه میانگین متحرک یا شیب ADX دارد.",
    "روند قوی": "اصطلاح 'روند قوی' مبهم است؛ نیاز به تعریف صریح اندیکاتور روند یا ADX دارد.",
    "valid support": "اصطلاح 'حمایت معتبر' (Valid Support) ذهنی است؛ نیازمند تعریف دقیق سووینگ قیمت و تعداد برخوردها است.",
    "حمایت معتبر": "اصطلاح 'حمایت معتبر' ذهنی است؛ نیازمند تعیین تعداد برخوردها یا پرایس اکشن است.",
    "good breakout": "اصطلاح 'شکست مناسب' (Good Breakout) مبهم است؛ میزان نفوذ قیمت یا کلوز کندل بالاتر از سطح مشخص نیست.",
    "شکست مناسب": "اصطلاح 'شکست مناسب' مبهم است؛ میزان نفوذ یا بسته شدن کندل خارج سطح تعیین نشده است.",
    "quality candle": "اصطلاح 'کندل باکیفیت' (Quality Candle) ذهنی است؛ حداقل نسبت بدنه به سایه باید به صورت عددی مشخص شود.",
    "کندل باکیفیت": "اصطلاح 'کندل باکیفیت' ذهنی است؛ نیاز به درصد اندازه بدنه به کل دامنه کندل دارد.",
    "high volume": "اصطلاح 'حجم بالا' مبهم است؛ نیاز به ضریب حجم نسبت به میانگین ۲۰ دوره گذشته دارد.",
    "حجم بالا": "اصطلاح 'حجم بالا' مبهم است؛ نسبت به میانگین حجم ۲۰ دوره گذشته نامشخص است.",
}


class RuleExtractor:
    """Extracts candidate trading rules from document page chunks with provenance."""

    @classmethod
    def extract_from_pages(
        cls,
        document_id: str,
        page_chunks: List[Dict[str, Any]]
    ) -> List[TradingRuleCreate]:
        """Extracts structured rules from document text chunks with source page tracing."""
        extracted_rules: List[TradingRuleCreate] = []

        for chunk in page_chunks:
            page_num = chunk.get("page", 1)
            text = chunk.get("text", "")
            if not text:
                continue

            lines = text.split("\n")
            current_rule_lines = []
            
            for line in lines:
                clean_line = line.strip()
                if not clean_line:
                    if current_rule_lines:
                        rule = cls._parse_block("\n".join(current_rule_lines), document_id, [page_num])
                        if rule:
                            extracted_rules.append(rule)
                        current_rule_lines = []
                    continue

                # Indicators of a rule boundary
                if any(k in clean_line.lower() for k in ["rule", "قانون", "strategy", "استراتژی", "buy", "sell", "خرید", "فروش", "ورود", "خروج", "entry", "exit"]):
                    if current_rule_lines:
                        rule = cls._parse_block("\n".join(current_rule_lines), document_id, [page_num])
                        if rule:
                            extracted_rules.append(rule)
                        current_rule_lines = []
                
                current_rule_lines.append(clean_line)

            if current_rule_lines:
                rule = cls._parse_block("\n".join(current_rule_lines), document_id, [page_num])
                if rule:
                    extracted_rules.append(rule)

        # If no granular rules were isolated by blocks, attempt holistic extraction
        if not extracted_rules:
            full_text = "\n".join([c.get("text", "") for c in page_chunks])
            pages_list = [c.get("page", 1) for c in page_chunks]
            fallback_rule = cls._parse_block(full_text[:2000], document_id, pages_list)
            if fallback_rule:
                extracted_rules.append(fallback_rule)

        return extracted_rules

    @classmethod
    def _parse_block(cls, block_text: str, document_id: str, pages: List[int]) -> Optional[TradingRuleCreate]:
        """Analyzes a block of text to synthesize a structured TradingRuleCreate object."""
        if len(block_text.strip()) < 15:
            return None

        lower_text = block_text.lower()

        # Categorize
        category = RuleCategory.FILTER
        if any(w in lower_text for w in ["خرید", "فروش", "ورود", "buy", "sell", "entry", "long", "short"]):
            category = RuleCategory.ENTRY
        elif any(w in lower_text for w in ["خروج", "exit", "close", "بستن", "take profit", "stop loss", "حد ضرر", "حد سود"]):
            category = RuleCategory.EXIT
        elif any(w in lower_text for w in ["ریسک", "risk", "حجم", "لات", "lot", "drawdown", "دراودان"]):
            category = RuleCategory.RISK

        # Check for ambiguous / subjective keywords
        detected_ambiguities = []
        for keyword, explanation in SUBJECTIVE_PATTERNS.items():
            if keyword in lower_text:
                detected_ambiguities.append(explanation)

        # Detect indicators & conditions
        conditions: List[Dict[str, Any]] = []
        parameters: Dict[str, Any] = {}

        # RSI detection
        rsi_match = re.search(r'rsi\s*(?:<|>|<=|>=|کمتر|بیشتر)?\s*(\d+)', lower_text)
        if "rsi" in lower_text:
            parameters["rsi_period"] = 14
            if rsi_match:
                thresh = float(rsi_match.group(1))
                op = "<" if any(w in lower_text for w in ["<", "below", "کمتر", "اشباع فروش"]) else ">"
                conditions.append({"indicator": "RSI", "operator": op, "threshold": thresh})
            else:
                conditions.append({"indicator": "RSI", "operator": "CROSSES", "threshold": 50})

        # Moving Average detection. Preserve type and period so the evaluator
        # never silently substitutes EMA(21) for a different moving average.
        ma_match = re.search(r'(ema|sma|ma|میانگین)\s*(\d+)', lower_text)
        if ma_match:
            average_type = ma_match.group(1)
            period = int(ma_match.group(2))
            parameters["ma_period"] = period
            if average_type == "sma":
                indicator = "SMA"
                expression_type = "SMA"
            elif average_type in {"ma", "میانگین"}:
                # Generic "moving average" does not identify SMA vs EMA.
                # Preserve the ambiguity so runtime evaluation fails closed.
                indicator = "MA"
                expression_type = "MA"
            else:
                indicator = "EMA"
                expression_type = "EMA"
            conditions.append({
                "indicator": indicator,
                "operator": ">",
                "threshold": f"Price > {expression_type}({period})",
            })

        # Symbols detection
        detected_symbols = []
        for sym in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD", "BTCUSD", "USDCAD"]:
            if sym.lower() in lower_text:
                detected_symbols.append(sym)

        # Timeframes detection — word-boundary match on the lowercased text: a
        # plain substring test makes "M15" also detect "M1" (and "H1" inside
        # other tokens), which would give a rule a wrong timeframe_scope.
        detected_timeframes = []
        for tf in ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]:
            if re.search(rf"\b{re.escape(tf.lower())}\b", lower_text):
                detected_timeframes.append(tf)

        # Invalidation conditions
        invalidations = []
        if "ابطال" in lower_text or "invalidation" in lower_text or "stop" in lower_text:
            invalidations.append("خروج در صورت نقض سطح حد ضرر یا شکست حمایت/مقاومت متضاد")

        assumptions = []
        if detected_ambiguities:
            assumptions.append("برخی پارامترها به صورت کیفی بیان شده‌اند و تا زمان تدقیق عددی توسط کاربر، به صورت فرض محافظه‌کارانه منظور خواهند شد.")

        name = block_text.split("\n")[0][:60].strip()
        if not name:
            name = f"قانون استخراج‌شده ({category.value})"

        confidence = 0.9 if not detected_ambiguities else 0.65

        return TradingRuleCreate(
            name=name,
            category=category,
            description=block_text.strip(),
            conditions=conditions,
            parameters=parameters,
            symbol_scope=detected_symbols or ["EURUSD"],
            timeframe_scope=detected_timeframes or ["M15"],
            invalidation_conditions=invalidations,
            source_document=document_id,
            source_pages=pages,
            assumptions=assumptions,
            ambiguities=detected_ambiguities,
            confidence=confidence,
        )
