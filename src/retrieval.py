"""Local hybrid retrieval: E5 multilingual dense embeddings plus BM25 lexical ranking."""
import re
import unicodedata
import numpy as np
from src.kb import load_sections

def normalize_text(text):
    # Normalize Arabic letter variants, diacritics and tatweel for BM25.
    text = unicodedata.normalize("NFKC", text.lower())
    text = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", text)
    return text.translate(str.maketrans({"أ":"ا", "إ":"ا", "آ":"ا", "ى":"ي"}))

# Compact bilingual synonyms connect informal Arabic tickets with English KB text.
# Only used for lexical retrieval; dense retrieval still embeds original text.
ALIASES = {
    "refund": ["استرجاع", "استرداد", "ارجاع", "فلوسي", "مبلغ", "refund", "refunded"],
    "timeline": ["تأخر", "تاخر", "اسبوع", "متى", "وصلني", "الان", "pending", "delay", "late", "week"],
    "track": ["تتبع", "حاله", "حالة", "status", "track", "tracking"],
    "shipping": ["شحن", "توصيل", "delivery", "shipping"],
    "payment": ["دفع", "خصم", "بطاقه", "بطاقة", "payment", "charged"],
    "account": ["حساب", "كلمة المرور", "دخول", "account", "password"],
}

def expanded_tokens(text):
    normalized = normalize_text(text)
    tokens = re.findall(r"\w+", normalized)
    for english, variants in ALIASES.items():
        if any(normalize_text(term) in normalized for term in variants):
            tokens.append(english)
            if english == "timeline":
                tokens += ["refund", "timelines", "tracking"]
    return tokens

class HybridRetriever:
    def __init__(self, model_name="intfloat/multilingual-e5-small", sections=None):
        self.sections = sections if sections is not None else load_sections()
        if not self.sections:
            raise ValueError("Knowledge base is empty")
        try:
            from sentence_transformers import SentenceTransformer
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError("Install optional dependencies: pip install -e '.[rag]'") from exc
        self.model = SentenceTransformer(model_name)
        self.reranker = None  # Lazy initialization: no extra model download unless enabled.
        self.bm25 = BM25Okapi([expanded_tokens(s.section_id.replace('-', ' ') + ' ' + s.title + ' ' + s.text) for s in self.sections])
        # E5 requires distinct passage/query prefixes; normalize for cosine similarity.
        self.vectors = self.model.encode(["passage: " + s.title + "\n" + s.text for s in self.sections], normalize_embeddings=True, show_progress_bar=False)

    @staticmethod
    def tokenize(text):
        return expanded_tokens(text)

    def _rerank(self, query, candidates):
        """Rescore the top hybrid candidates with a cross-encoder.

        Enabled with RAG_RERANK=1. Only the shortlist is scored, not the whole KB.
        """
        import os
        if self.reranker is None:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(os.getenv(
                "RAG_RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"))
        pairs = [(query, self.sections[i].title + "\n" + self.sections[i].text) for i in candidates]
        scores = self.reranker.predict(pairs)
        return sorted(zip(candidates, scores), key=lambda item: float(item[1]), reverse=True)

    def search(self, query, k=3):
        q = self.model.encode(["query: " + query], normalize_embeddings=True, show_progress_bar=False)[0]
        dense = np.asarray(self.vectors) @ q
        sparse = self.bm25.get_scores(expanded_tokens(query))
        # Reciprocal Rank Fusion; lexical weight helps cross-language policy terms.
        fused = np.zeros(len(self.sections))
        for scores, weight in ((dense, 1.0), (sparse, 1.4)):
            for rank, idx in enumerate(np.argsort(-np.asarray(scores))):
                fused[idx] += weight / (60 + rank + 1)
        import os
        if os.getenv("RAG_RERANK", "0") == "1":
            candidate_count = max(k, min(len(self.sections), int(os.getenv("RAG_CANDIDATES", "20"))))
            ids = np.argsort(-fused)[:candidate_count]
            ranked = self._rerank(query, [int(i) for i in ids])[:k]
            return [(self.sections[i], float(score)) for i, score in ranked]
        ids = np.argsort(-fused)[:k]
        return [(self.sections[int(i)], float(fused[i])) for i in ids]

    def debug_search(self, query, k=5):
        """Inspect selected section IDs and fused scores during retrieval analysis."""
        return [{"section_id": s.section_id, "score": round(score, 6)} for s, score in self.search(query, k)]
