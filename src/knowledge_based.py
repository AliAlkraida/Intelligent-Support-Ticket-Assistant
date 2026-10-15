import re
import time
from src.kb import load_sections
from src.schemas import TicketResult

KEYWORDS = {
    "refunds": ["refund", "return", "money back", "استرجاع", "استرداد", "ارجاع", "إرجاع", "فلوسي", "مبلغ"],
    "cancellations": ["cancel", "cancellation", "الغاء", "إلغاء", "الغي", "كنسل"],
    "shipping": ["shipping", "delivery", "courier", "شحن", "توصيل", "التوصيل", "مندوب", "عنوان"],
    "payments": ["payment", "charged", "card", "pay", "دفع", "دفعت", "بطاقة", "خصم", "مدى"],
    "invoices": ["invoice", "receipt", "فاتورة", "الفاتورة", "ايصال"],
    "account": ["account", "password", "login", "حساب", "كلمة المرور", "تسجيل", "الدخول"],
    "orders": ["order", "tracking", "track", "طلب", "طلبي", "تتبع", "الطلب"],
    "contact": ["human", "agent", "support", "موظف", "خدمة العملاء", "تواصل"],
    "feedback": ["complaint", "feedback", "review", "شكوى", "تقييم"],
    "newsletter": ["newsletter", "subscribe", "نشرة", "اشتراك"],
}
URGENT = ["fraud", "stolen", "unauthorized", "اختراق", "احتيال", "غير مصرح"]
HIGH = ["charged twice", "double charged", "twice", "مرتين", "مكرر", "urgent", "عاجل"]

def normalize(text):
    text = text.lower().replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي")
    return re.sub(r"[\u064b-\u065f]", "", text)

def classify_ticket(ticket):
    """Rule-based category and priority classifier.

    RAG reuses this function, so the baseline and RAG share classification
    labels; report that dependency when comparing scores.
    """
    t = normalize(ticket)
    scores = {cat: sum(1 for kw in kws if normalize(kw) in t)
              for cat, kws in KEYWORDS.items()}
    category = max(scores, key=scores.get) if scores else "unknown"
    if not scores or scores[category] == 0:
        category = "unknown"
    # Specific delivery intent overrides the generic word "order".
    delivery_phrases = (
        "did not receive", "didn't receive", "not received", "never arrived",
        "not delivered", "where is my package", "missing package",
        "لم استلم", "ما استلمت", "لم يصل", "ما وصل", "ما وصلني",
        "لم يصلني", "الطلب متاخر", "تاخر التوصيل",
    )
    if any(normalize(phrase) in t for phrase in delivery_phrases) and category != "refunds":
        category = "shipping"
    priority = ("urgent" if any(normalize(x) in t for x in URGENT)
                else "high" if any(normalize(x) in t for x in HIGH)
                else "normal")
    return category, priority


class KnowledgeBasedAssistant:
    def __init__(self, sections=None):
        self.sections = sections if sections is not None else load_sections()

    def answer(self, ticket):
        start = time.perf_counter()
        t = normalize(ticket)
        category, priority = classify_ticket(ticket)
        relevant = [s for s in self.sections if s.section_id.startswith(category + "#")]
        tokens = set(re.findall(r"\w+", t))
        def overlap(s):
            return len(tokens & set(re.findall(r"\w+", normalize(s.title))))
        # Intent-aware matching: refund status and timelines differ from eligibility.
        section_hints = {
            "payments": [
                (["charged twice", "double charged", "twice", "مرتين", "مكرر"], "charged-twice"),
                (["declined", "failed", "rejected", "فشل", "رفض", "مرفوض"], "payment-failed"),
            ],
            "shipping": [
                (["did not receive", "didn't receive", "not received", "never arrived", "ما وصل", "لم يصل", "ما استلمت", "لم استلم"], "tracking-a-shipment"),
            ],
            "refunds": [
                (["وصل", "تاخر", "تأخر", "اسبوع", "week", "late", "delay", "pending", "status", "حالة", "متى"], "tracking-a-refund"),
                (["كم يوم", "مدة", "time", "timeline", "days", "يوم"], "refund-timelines"),
                (["eligible", "eligibility", "شروط", "مسموح"], "refund-eligibility"),
            ],
        }
        selected = None
        for terms, slug in section_hints.get(category, []):
            if any(normalize(term) in t for term in terms):
                selected = next((s for s in relevant if s.section_id.endswith("#" + slug)), None)
                if selected:
                    break
        if selected is None:
            selected = max(relevant, key=overlap) if relevant else None
        escalate = category == "unknown" or priority == "urgent" or not selected
        if escalate:
            reply = "يرجى التواصل مع فريق الدعم للتحقق من التفاصيل. / Please contact support to verify the details."
        else:
            reply = selected.text.split("\n")[0][:700]
        return TicketResult("knowledge_based", category, priority, reply, escalate,
                            [selected.section_id] if selected else [],
                            round((time.perf_counter()-start)*1000, 2))
