"""Local RAG: hybrid retrieval followed by grounded generation with Qwen.

Python produces category, priority and evidence; the model only drafts a
short plain-text reply.
"""
import os
import time

from src.knowledge_based import classify_ticket
from src.local_llm import LocalGenerator
from src.retrieval import HybridRetriever
from src.schemas import TicketResult

# Plain-text output avoids fragile JSON parsing with a small local model.
RAG_SYSTEM = """You are a DemoMart customer-support reply writer.
Use ONLY the policy excerpts supplied in the latest user message.
Answer the customer's actual question in the same language as the customer.
Be polite, concise (1-3 sentences), and include relevant policy timelines if present.
Do not invent policies, promise actions, claim to check accounts, or ask for full card numbers.
If the excerpts cannot answer the question, write exactly: NEEDS_HUMAN
Return only the customer-facing reply: no JSON, no preamble, no source labels."""

RAG_EXAMPLE = [
    {"role": "user", "content": "POLICY [1]: Wallet refunds arrive within 24 hours.\nTICKET: How long does a wallet refund take?"},
    {"role": "assistant", "content": "DemoMart Wallet refunds usually arrive within 24 hours."},
]


def fallback(ticket):
    """Language-aware handoff when there is no usable answer."""
    if any("\u0600" <= c <= "\u06ff" for c in ticket):
        return "يرجى التواصل مع فريق الدعم للحصول على مساعدة إضافية."
    return "Please contact our support team for further assistance."


def build_messages(ticket, hits):
    """Create a compact grounded prompt from top-ranked KB sections."""
    context = "\n\n".join(
        f"POLICY [{i}] ({section.section_id}): {section.text}"
        for i, (section, _) in enumerate(hits, start=1)
    )
    return [
        {"role": "system", "content": RAG_SYSTEM},
        *RAG_EXAMPLE,
        {"role": "user", "content": f"{context}\n\nTICKET: {ticket}"},
    ]


class RAGAssistant:
    """Hybrid retrieval + local Qwen plain-text answer generation."""

    def __init__(self, retriever=None, generator=None):
        self.retriever = retriever if retriever is not None else HybridRetriever()
        self.generator = generator if generator is not None else LocalGenerator()
        self.last_diagnostics = {}

    def answer(self, ticket):
        start = time.perf_counter()
        category, priority = classify_ticket(ticket)
        top_k = max(1, min(5, int(os.getenv("RAG_TOP_K", "2"))))
        hits = self.retriever.search(ticket, k=top_k)
        self.last_diagnostics = {"retrieved": [s.section_id for s, _ in hits]}

        if not hits:
            self.last_diagnostics["status"] = "no_documents"
            return TicketResult("rag", category, priority, fallback(ticket), True, [],
                                round((time.perf_counter() - start) * 1000, 2))

        raw = self.generator.chat(
            build_messages(ticket, hits),
            max_new_tokens=int(os.getenv("LOCAL_RAG_MAX_TOKENS", "160")),
        )
        reply = raw.strip() if isinstance(raw, str) else ""
        needs_human = not reply or reply.upper().startswith("NEEDS_HUMAN")
        self.last_diagnostics["status"] = "handoff" if needs_human else "answered"
        if needs_human:
            reply = fallback(ticket)

        # Evidence denotes retrieved context, NOT independently verified citations.
        evidence = [] if needs_human else [s.section_id for s, _ in hits]
        return TicketResult("rag", category, priority, reply, needs_human, evidence,
                            round((time.perf_counter() - start) * 1000, 2), 0.0)
