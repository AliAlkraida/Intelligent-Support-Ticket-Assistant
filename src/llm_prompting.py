import time
from src.cloud import CloudGenerator, estimate_cost
from src.schemas import TicketResult

CATEGORIES = {"orders","shipping","refunds","cancellations","payments","invoices","account","contact","feedback","newsletter","unknown"}
PRIORITIES = {"normal","high","urgent"}

def validated(payload):
    category = str(payload.get("category", "unknown"))
    priority = str(payload.get("priority", "normal"))
    if category not in CATEGORIES or priority not in PRIORITIES:
        raise ValueError("Invalid category or priority in LLM response")
    return category, priority, str(payload.get("suggested_reply", "")), bool(payload.get("escalate", True))

class PromptingAssistant:
    def __init__(self, generator=None):
        self.generator = generator or CloudGenerator()

    def answer(self, ticket):
        start = time.perf_counter()
        payload, usage = self.generator.generate(ticket, context=None)
        cat, pri, reply, escalate = validated(payload)
        return TicketResult("llm_prompting", cat, pri, reply, escalate, [],
                            round((time.perf_counter()-start)*1000, 2), estimate_cost(usage))
