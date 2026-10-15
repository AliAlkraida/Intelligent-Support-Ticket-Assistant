from dataclasses import dataclass, asdict, field
from typing import List

@dataclass
class TicketResult:
    model: str
    category: str
    priority: str
    suggested_reply: str
    escalate: bool
    evidence: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    error: str = ""

    def to_dict(self):
        return asdict(self)
