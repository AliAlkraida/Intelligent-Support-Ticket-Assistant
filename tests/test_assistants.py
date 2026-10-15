"""Fast offline tests; no model downloads or API requests."""
from types import SimpleNamespace
from src.knowledge_based import KnowledgeBasedAssistant, classify_ticket
from src.rag import RAGAssistant, build_messages

class FakeRetriever:
    def __init__(self, hits): self.hits = hits
    def search(self, ticket, k=2): return self.hits[:k]

class FakeGenerator:
    def __init__(self, response): self.response = response; self.messages = None
    def chat(self, messages, max_new_tokens=160):
        self.messages = messages
        return self.response

SECTION = SimpleNamespace(section_id="payments#charged-twice", title="Charged twice", text="One charge may be pending for 3 to 5 business days.")

def test_baseline_duplicate_payment():
    result = KnowledgeBasedAssistant().answer("I was charged twice for my order")
    assert result.category == "payments"
    assert "payments#charged-twice" in result.evidence

def test_delivery_intent():
    assert classify_ticket("I did not receive my order")[0] == "shipping"
    assert classify_ticket("ما وصلني الطلب")[0] == "shipping"

def test_rag_generates_plain_text_without_json():
    gen = FakeGenerator("One charge may be pending for 3 to 5 business days.")
    result = RAGAssistant(FakeRetriever([(SECTION, 0.04)]), gen).answer("I was charged twice")
    assert result.category == "payments" and result.priority == "high"
    assert result.suggested_reply.startswith("One charge")
    assert result.evidence == ["payments#charged-twice"]
    assert result.escalate is False
    assert "POLICY [1]" in gen.messages[-1]["content"]

def test_rag_abstains_when_no_evidence():
    result = RAGAssistant(FakeRetriever([]), FakeGenerator("unused")).answer("Unknown request")
    assert result.escalate and not result.evidence

def test_rag_abstains_when_model_says_no():
    result = RAGAssistant(FakeRetriever([(SECTION, 0.04)]), FakeGenerator("NEEDS_HUMAN")).answer("I was charged twice")
    assert result.escalate and not result.evidence

def test_rag_prompt_has_grounding_and_few_shot():
    messages = build_messages("I was charged twice", [(SECTION, 0.04)])
    assert "Use ONLY" in messages[0]["content"]
    assert len(messages) == 4
