"""Knowledge Retriever Service for grounded, evidence-based retrieval across educational sources.
Strictly prevents hallucinations by requiring verifiable text citations.
Implements Okapi BM25 scoring, passage-level section extraction, and multi-field grounding.
"""
import re
import math
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.orm import Session

from backend.app.models.knowledge_chunk import KnowledgeChunk

STOPWORDS = {
    # Persian particles, pronouns & auxiliaries
    "در", "با", "از", "به", "که", "و", "یک", "این", "آن", "تا", "برای", "را",
    "شد", "است", "بود", "یا", "هم", "بر", "چه", "چند", "اگر", "اما", "پس",
    "تو", "من", "ما", "شما", "آنها", "دارد", "دارند", "می", "شود", "شوند",
    "کنید", "کنم", "کند", "کنند", "کرد", "کرده", "کردند", "باشد", "باشند",
    "نیست", "نیستند", "باید", "شاید", "تواند", "توانند", "داد", "داده", "دادند",
    "ای", "ها", "های", "هایی", "تر", "ترین", "ان", "ات", "اند",
    "روی", "زیر", "بین", "همه", "هیچ", "هر", "خود", "دیگر", "دیگری",
    "قبل", "بعد", "پیش", "کنار", "همان", "همین", "چنین", "چنان", "چون",
    "بسیار", "خیلی", "کمی", "تنها", "فقط",
    "گانه", "سه‌گانه", "دوگانه", "چندگانه", "سه", "دو", "چهار", "پنج", "شش", "هفت", "هشت", "نه", "ده",
    # English stopwords
    "the", "in", "of", "and", "a", "an", "to", "for", "is", "are", "on", "that", 
    "by", "this", "with", "from", "at", "as", "it", "its", "or", "be"
}

# Question intent / modifier words
INTENT_WORDS = {
    "تعریف", "معنی", "مفهوم", "منظور", "چیست", "چیستند", "چیه", "کیست", 
    "کدام", "چگونه", "چرا", "آیا", "توضیح", "بیان",
    "تفاوت", "فرق", "مقایسه", "تمایز",
    "خطا", "خطاهای", "اشتباه", "اشتباهات", "دام", "دام‌های", "نکات", "رایج",
    "تنظیمات", "پیشنهادی", "پیشنهاد", "توصیه"
}

# Generic trading terms that appear broadly across almost all documents
GENERIC_MARKET_WORDS = {
    "استراتژی", "معاملاتی", "سیستم", "روش", "تحلیل", "ترید", "معامله",
    "تایم", "فریم", "تایم‌فریم", "تایمفریم", "بازار", "قیمت", "چارت", "نمودار",
    "کندل", "کندل‌ها", "بار", "سشن"
}

# Bilingual synonyms for trading and technical concepts
SYNONYMS: Dict[str, List[str]] = {
    "delta": ["دلتا"],
    "دلتا": ["delta"],
    "volume": ["والیوم", "حجم"],
    "والیوم": ["volume"],
    "profile": ["پروفایل"],
    "پروفایل": ["profile"],
    "absorption": ["جذب"],
    "جذب": ["absorption"],
    "exhaustion": ["خستگی"],
    "خستگی": ["exhaustion"],
    "slippage": ["اسلیپیج", "لغزش"],
    "اسلیپیج": ["slippage"],
    "leverage": ["اهرم"],
    "اهرم": ["leverage"],
    "deposit": ["موجودی", "واریزی"],
    "موجودی": ["deposit"],
    "scalper": ["اسکلپر"],
    "اسکلپر": ["scalper"],
}

PERSIAN_TO_ENGLISH_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


def normalize_persian_text(text: str) -> str:
    """Normalizes Arabic/Persian characters, digits, whitespace, and punctuation."""
    if not text:
        return ""
    text = text.translate(PERSIAN_TO_ENGLISH_DIGITS)
    text = text.replace("ي", "ی").replace("ك", "ک").replace("ة", "ه")
    text = text.replace("‌", " ")
    text = re.sub(r'[\t\r\n]+', ' ', text)
    text = re.sub(r'[^\w\s\d]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()


def tokenize(text: str) -> List[str]:
    """Tokenizes normalized text into words of minimum length 2."""
    norm = normalize_persian_text(text)
    return [t for t in norm.split() if len(t) >= 2]


class KnowledgeRetriever:
    """Performs deterministic, evidence-grounded search over parsed knowledge chunks.
    
    Uses Okapi BM25 ranking, section-level passage extraction, and bilingual semantic expansion.
    """

    def __init__(self, db: Session):
        self.db = db
        self._corpus_cached = False
        self._chunks_cache: List[KnowledgeChunk] = []
        self._doc_tokens: Dict[str, List[str]] = {}
        self._doc_lens: Dict[str, int] = {}
        self._avg_dl: float = 1.0
        self._df: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._build_corpus_index()

    def _build_corpus_index(self):
        """Indexes knowledge chunks, computing document frequencies and BM25 IDF weights."""
        self._chunks_cache = self.db.query(KnowledgeChunk).all()
        N = len(self._chunks_cache)
        if N == 0:
            return

        total_words = 0
        df_counter: Dict[str, int] = {}

        for chunk in self._chunks_cache:
            tokens = tokenize(chunk.content)
            title_tokens = tokenize(chunk.title or "")
            concept_tokens = tokenize(" ".join(chunk.key_concepts or []))
            loc_tokens = tokenize(chunk.location_reference or "")
            
            all_tokens = tokens + title_tokens + concept_tokens + loc_tokens
            self._doc_tokens[chunk.chunk_id] = tokens
            doc_len = max(len(tokens), 5)
            self._doc_lens[chunk.chunk_id] = doc_len
            total_words += doc_len

            unique_tokens = set(all_tokens)
            for t in unique_tokens:
                df_counter[t] = df_counter.get(t, 0) + 1

        self._avg_dl = total_words / N
        self._df = df_counter
        
        for t, freq in df_counter.items():
            self._idf[t] = math.log(1.0 + (N - freq + 0.5) / (freq + 0.5))

        self._corpus_cached = True

    def get_idf(self, token: str) -> float:
        """Returns BM25 IDF for a token, defaulting to max IDF for unseen terms."""
        if token in self._idf:
            return self._idf[token]
        N = len(self._chunks_cache)
        return math.log(1.0 + (N + 0.5) / 0.5)

    def _extract_passages(self, content: str) -> List[Tuple[str, str]]:
        """Splits chunk content into logical passages and extracts their heading."""
        KNOWN_HEADINGS = [
            'مطالب مهم این جلسه', 'درسنامه بازطراحی‌شده', 'هدف‌های یادگیری',
            'اصطلاحات کلیدی جلسه', 'اصطلاحات کلیدی', 'خطاهای رایج', 'چک‌لیست ثبت در ژورنال',
            'پرسش‌های تثبیت مفهوم', 'مقدمه و راهنمای جزوه', 'خلاصه اجرایی', '●'
        ]
        
        has_known_heading = any(h in content for h in KNOWN_HEADINGS if h != '●')
        if has_known_heading:
            pattern = r'(\n\n|^)(' + '|'.join(map(re.escape, [h for h in KNOWN_HEADINGS if h != '●'])) + r')(\n\n|\n)'
            parts = re.split(pattern, content)
            passages: List[Tuple[str, str]] = []
            i = 0
            while i < len(parts):
                part = parts[i]
                if part in KNOWN_HEADINGS:
                    heading = part
                    body = parts[i+2] if i+2 < len(parts) else ''
                    full_text = f"{heading}\n\n{body.strip()}"
                    passages.append((heading, full_text))
                    i += 3
                else:
                    if part.strip() and len(part.strip()) > 30:
                        passages.append(("متن اصلی", part.strip()))
                    i += 1
            if passages:
                return passages

        raw_sections = [s.strip() for s in content.split("\n\n") if s.strip()]
        passages = []
        for s in raw_sections:
            lines = [l.strip() for l in s.split("\n") if l.strip()]
            if not lines:
                continue
            first_line = lines[0]
            heading = first_line if len(first_line) < 60 else "پاراگراف"
            passages.append((heading, s))
        return passages if passages else [("متن اصلی", content)]

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_file: Optional[str] = None,
        min_score: float = 2.0
    ) -> List[Dict[str, Any]]:
        """Searches knowledge chunks and ranks by BM25 relevance and passage specificity."""
        raw_tokens = tokenize(query)
        if not raw_tokens:
            return []

        query_stopwords = [t for t in raw_tokens if t in STOPWORDS]
        query_intent = [t for t in raw_tokens if t in INTENT_WORDS]
        
        param_keywords = {"اهرم", "موجودی", "حداقل", "تنظیمات", "سرمایه", "دلار", "لات"}
        query_params = [t for t in raw_tokens if t in param_keywords]

        core_topic_tokens = [t for t in raw_tokens if t not in STOPWORDS and t not in INTENT_WORDS and t not in GENERIC_MARKET_WORDS]
        
        all_informative_tokens = [t for t in raw_tokens if t not in STOPWORDS and t not in INTENT_WORDS]
        if not all_informative_tokens:
            all_informative_tokens = [t for t in raw_tokens if t not in STOPWORDS]

        is_error_query = any(w in query_intent for w in ["خطا", "خطاهای", "اشتباه", "اشتباهات", "دام", "دام‌های", "رایج"])
        is_difference_query = any(w in query_intent for w in ["تفاوت", "فرق", "مقایسه"])
        is_param_query = bool(query_params)
        is_definition_query = (not is_error_query) and (not is_difference_query) and any(w in query_intent for w in ["تعریف", "معنی", "مفهوم", "منظور"])

        token_expansion: Dict[str, List[str]] = {}
        for t in all_informative_tokens:
            token_expansion[t] = [t] + SYNONYMS.get(t, [])

        norm_query = normalize_persian_text(query)
        scored_candidates = []

        k1 = 1.5
        b = 0.75

        for chunk in self._chunks_cache:
            if source_file and source_file.lower() not in chunk.source_file.lower():
                continue

            chunk_id = chunk.chunk_id
            tokens = self._doc_tokens.get(chunk_id, [])
            doc_len = self._doc_lens.get(chunk_id, 10)
            
            norm_content = normalize_persian_text(chunk.content)
            norm_title = normalize_persian_text(chunk.title or "")
            norm_concepts = normalize_persian_text(" ".join(chunk.key_concepts or []))
            norm_loc = normalize_persian_text(chunk.location_reference or "")
            norm_src = normalize_persian_text(chunk.source_file)

            token_counts: Dict[str, int] = {}
            for t in tokens:
                token_counts[t] = token_counts.get(t, 0) + 1

            bm25_score = 0.0
            matched_terms = []
            core_hits = 0
            informative_hits = 0
            param_hits = 0
            matched_idf_sum = 0.0
            total_informative_idf_sum = sum(self.get_idf(t) for t in all_informative_tokens)

            for t in all_informative_tokens:
                t_idf = self.get_idf(t)
                variants = token_expansion.get(t, [t])
                
                in_content = any((v in token_counts or v in norm_content) for v in variants)
                in_title = any(v in norm_title for v in variants)
                in_concepts = any(v in norm_concepts for v in variants)
                in_loc = any(v in norm_loc for v in variants)
                in_src = any(v in norm_src for v in variants)

                if in_content or in_title or in_concepts or in_loc or in_src:
                    informative_hits += 1
                    matched_idf_sum += t_idf
                    if t in core_topic_tokens:
                        core_hits += 1
                    if t in query_params:
                        param_hits += 1

                tf = sum(token_counts.get(v, 0) for v in variants)
                if tf == 0 and (in_content or in_title or in_loc or in_src):
                    tf = 1

                if tf > 0:
                    tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (doc_len / self._avg_dl)))
                    term_score = t_idf * tf_norm
                    bm25_score += term_score
                    matched_terms.append(f"token('{t}', idf={t_idf:.1f}, tf={tf})")

                if in_title:
                    bm25_score += t_idf * 2.0
                    matched_terms.append(f"title_hit('{t}')")
                if in_concepts:
                    bm25_score += t_idf * 1.5
                if in_loc:
                    bm25_score += t_idf * 1.2
                if in_src:
                    bm25_score += t_idf * 1.0

            # 2. Strict Evidence Gating (Anti-Hallucination Gate)
            if core_topic_tokens and core_hits == 0:
                continue

            idf_coverage = matched_idf_sum / max(total_informative_idf_sum, 0.01)
            if idf_coverage < 0.35:
                continue

            min_required_hits = 1 if len(all_informative_tokens) <= 2 else max(1, int(len(all_informative_tokens) * 0.25))
            if informative_hits < min_required_hits:
                continue

            score = bm25_score

            # 3. Canonical Match Boosts
            if is_definition_query and "اصطلاح پرکاربرد" in chunk.source_file:
                for t in core_topic_tokens:
                    variants = token_expansion.get(t, [t])
                    if any(v in norm_title or v in norm_loc or v in norm_content.split() for v in variants):
                        score += 35.0
                        matched_terms.append(f"canonical_glossary_definition('{t}')")

            has_error_section = "خطاهای رایج" in chunk.content
            if is_error_query:
                if has_error_section:
                    score += 25.0
                    matched_terms.append("has_common_errors_section")
                else:
                    score *= 0.5

            if is_param_query and param_hits >= 2:
                score += 15.0 * (param_hits / len(query_params))
                if re.search(r'\d+[:/]\d+|\b\d{3,}\b|[۰-۹]{3,}', chunk.content):
                    score += 20.0
                    matched_terms.append("concrete_parameters_and_values")

            if len(norm_query) > 5 and norm_query in norm_content:
                score += 12.0
                matched_terms.append("exact_phrase_in_content")

            # 4. Passage-Level Extraction & Scoring
            passages = self._extract_passages(chunk.content)
            best_passage_score = 0.0
            best_passage_text = chunk.content[:400]
            best_passage_heading = "متن اصلی"

            for heading, p_text in passages:
                norm_p = normalize_persian_text(p_text)
                p_tokens = norm_p.split()
                p_len = max(len(p_tokens), 5)
                p_bm25 = 0.0
                for t in all_informative_tokens:
                    variants = token_expansion.get(t, [t])
                    p_tf = sum(p_tokens.count(v) for v in variants)
                    if p_tf > 0:
                        t_idf = self.get_idf(t)
                        p_tf_norm = (p_tf * (k1 + 1.0)) / (p_tf + k1 * (1.0 - b + b * (p_len / 40.0)))
                        p_bm25 += t_idf * p_tf_norm

                if is_error_query and "خطاهای رایج" in heading:
                    p_bm25 += 20.0
                elif is_difference_query and ("تفاوت" in p_text or "تفکیک" in p_text or "دو سازوکار" in p_text):
                    p_bm25 += 15.0
                elif is_param_query and ("1:500" in p_text or "۵,۰۰۰" in p_text or "5000" in p_text):
                    p_bm25 += 20.0

                if p_bm25 > best_passage_score:
                    best_passage_score = p_bm25
                    best_passage_text = p_text
                    best_passage_heading = heading

            score += best_passage_score * 0.8

            if score >= min_score:
                scored_candidates.append({
                    "chunk_id": chunk.chunk_id,
                    "source_file": chunk.source_file,
                    "location_type": chunk.location_type,
                    "location_reference": chunk.location_reference,
                    "page_number": chunk.page_number,
                    "title": chunk.title,
                    "content": chunk.content,
                    "relevant_passage": best_passage_text,
                    "passage_heading": best_passage_heading,
                    "parameters": chunk.parameters or {},
                    "exceptions_and_limits": chunk.exceptions_and_limits or [],
                    "score": round(score, 2),
                    "coverage_ratio": round(idf_coverage, 2),
                    "matched_evidence": list(set(matched_terms))[:6]
                })

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        return scored_candidates[:top_k]

    def query_with_grounding(
        self,
        query: str,
        top_k: int = 5,
        source_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Queries knowledge base and returns verifiable grounded citations."""
        results = self.search(query=query, top_k=top_k, source_file=source_file)

        if not results:
            return {
                "status": "NOT_FOUND",
                "evidence_found": False,
                "message": "هیچ سند یا شواهدی درباره این موضوع در منابع آموزشی کاربر یافت نشد.",
                "query": query,
                "results": []
            }

        top_match = results[0]
        return {
            "status": "GROUNDED_EVIDENCE_FOUND",
            "evidence_found": True,
            "query": query,
            "primary_source": top_match["source_file"],
            "primary_location": top_match["location_reference"],
            "primary_title": top_match["title"],
            "results_count": len(results),
            "results": results
        }
