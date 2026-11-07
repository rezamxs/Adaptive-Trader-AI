"""Knowledge chunk classification (buglist follow-up: "why did the glossary
produce 102 rules?").

Root cause of the 102-rule flood: the DOCX parser emits ONE chunk per table
row and the extractor turned every >=15-char block containing a trade keyword
("ورود", "buy", ...) into a rule candidate — fabricating structure along the
way (a synthetic "RSI CROSSES 50" condition, default EURUSD/M15 scopes). A
glossary is a DEFINITION list, not a rule book.

This module classifies each chunk into exactly one of:

- EDUCATIONAL_TERM   — a definition/glossary row: what a term MEANS. Stays a
                       knowledge chunk; never becomes a rule.
- ANALYTICAL_CONCEPT — explanatory text about market mechanics/analysis.
                       Stays a knowledge chunk; never becomes a rule.
- EXPLICIT_RULE      — an actionable, documented rule: an ACTION verb AND a
                       concrete condition (indicator+number, explicit
                       timeframe, or a quantified level). Enters the rule
                       registry as PENDING (human approval only — nothing is
                       ever auto-approved).
- AMBIGUOUS_RULE     — rule-INTENT with subjective/undefined parameters
                       ("روند قوی", "شکست مناسب"...). Kept OUT of the
                       registry as a standalone candidate; surfaced to the
                       human via the classification summary instead.

Deterministic heuristics only (no ML dependency) so classification is
auditable and reproducible. Persian digits are normalized before matching.
"""
import re
from enum import Enum
from typing import List, Tuple

PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

# Strong definition/glossary markers.
_DEFINITION_MARKERS = (
    "معنی", "یعنی", "تعریف", "اصطلاح", "به آن می‌گویند", "می‌گویند",
    "گفته می‌شود", "what is", "definition", "stands for", "refers to",
)
_TABLE_ROW_PREFIX = re.compile(r"^row\s*\d+\s*:", re.IGNORECASE)

# Action verbs: an imperative that tells the trader to DO something.
_ACTION_VERBS = (
    "وارد شو", "وارد معامله", "وارد شدن", "خرید کن", "فروش کن", "بخر",
    "بفروش", "ورود به معامله", "خروج از معامله", "حد ضرر را", "حد سود را",
    "بستن معامله", "entry", "entries", "enter", "exit", "buy", "sell", "go long",
    "go short", "open a position", "close the position",
    # Normative policy verbs: rule books state limits ("حداکثر یک درصد باشد"),
    # not only imperatives. Combined with a quantified condition below these
    # are EXPLICIT rules; alone they never are.
    "باید", "نباید", "باشد", "حداکثر", "حداقل",
)

# Concrete conditions: something MEASURABLE. Persian digits normalized first.
_CONDITION_PATTERNS = (
    re.compile(r"\b(rsi|ema|sma|atr|macd|adx)\s*[<>=≤≥]{0,2}\s*\d+", re.IGNORECASE),
    re.compile(r"\b(m1|m5|m15|m30|h1|h4|d1)\b", re.IGNORECASE),
    re.compile(r"\d+\s*(%|درصد|پیپ|پوینت|pip|point|dollar|دلار)", re.IGNORECASE),
    re.compile(r"[<>≤≥]\s*\d+"),
    re.compile(r"(زیر|بالا|بالای|کمتر از|بیشتر از|below|above|under|over)\s+\d+", re.IGNORECASE),
    # Persian word-numbers ("حداکثر یک درصد") count as quantified conditions.
    re.compile(r"(یک|دو|سه|چهار|پنج|شش|هفت|هشت|نه|ده|\d+)\s*(درصد|پیپ|پوینت|%)"),
)

# Rule-intent keywords strong enough to mark AMBIGUOUS (with subjective params).
_RULE_INTENT_KEYWORDS = ("قانون", "rule", "استراتژی", "strategy", "setup", "سیگنال ورود")


class KnowledgeClass(str, Enum):
    EDUCATIONAL_TERM = "educational_term"
    ANALYTICAL_CONCEPT = "analytical_concept"
    EXPLICIT_RULE = "explicit_rule"
    AMBIGUOUS_RULE = "ambiguous_rule"


def classify_chunk(text: str) -> Tuple[KnowledgeClass, List[str]]:
    """Classifies one chunk; returns (class, human-readable reasons)."""
    if not text or not text.strip():
        return KnowledgeClass.ANALYTICAL_CONCEPT, ["empty chunk"]
    normalized = text.translate(PERSIAN_DIGITS)
    lowered = normalized.lower()
    reasons: List[str] = []

    action_hit = any(verb in lowered for verb in _ACTION_VERBS)
    condition_hits = sum(1 for pat in _CONDITION_PATTERNS if pat.search(lowered))
    definition_hit = (
        bool(_TABLE_ROW_PREFIX.match(lowered))
        or any(marker in lowered for marker in _DEFINITION_MARKERS)
    )

    if action_hit:
        reasons.append("action verb present")
    if condition_hits:
        reasons.append(f"concrete condition matched ({condition_hits})")
    if definition_hit:
        reasons.append("definition/table-row marker present")

    # 1. An EXPLICIT rule needs BOTH an action and something measurable.
    if action_hit and condition_hits:
        reasons.insert(0, "actionable imperative with a measurable condition")
        return KnowledgeClass.EXPLICIT_RULE, reasons

    # 2. Definitions win over everything else that is not explicit — this is
    #    the exact path the 102-rule flood took (glossary rows containing the
    #    word "ورود" inside a MEANING explanation).
    if definition_hit:
        reasons.insert(0, "definition text: describes what a term MEANS, not what to DO")
        return KnowledgeClass.EDUCATIONAL_TERM, reasons

    # 3. Rule INTENT with subjective/undefined parameters → needs a human;
    #    NOT seeded into the registry as a standalone candidate.
    if any(keyword in lowered for keyword in _RULE_INTENT_KEYWORDS):
        reasons.insert(0, "rule-like intent without a measurable condition (subjective parameters)")
        return KnowledgeClass.AMBIGUOUS_RULE, reasons

    reasons.insert(0, "explanatory text without rule intent")
    return KnowledgeClass.ANALYTICAL_CONCEPT, reasons


def classify_chunks(page_chunks: List[dict]) -> List[Tuple[dict, KnowledgeClass, List[str]]]:
    """Classifies every chunk, preserving order and the chunk reference."""
    out = []
    for chunk in page_chunks:
        klass, reasons = classify_chunk(str(chunk.get("text", "")))
        out.append((chunk, klass, reasons))
    return out


def summarize(classes: List[KnowledgeClass]) -> dict:
    """Aggregates a classification run into a compact summary dict."""
    summary = {k.value: 0 for k in KnowledgeClass}
    for klass in classes:
        summary[klass.value] += 1
    return summary
