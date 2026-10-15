# Blind answer-quality evaluation

For every held-out ticket, review the three responses in random order without model names.

- `correctness`: 0 incorrect, 1 partly correct, 2 correct and actionable against the KB.
- `unsupported_claim`: 1 if any company-specific fact is not supported by the KB, else 0.
- `escalation_correct`: 1 if the escalation decision matches the gold label, else 0.
- `failure_type`: one of `wrong_category`, `wrong_priority`, `wrong_policy`, `hallucination`, `missed_escalation`, `over_escalation`, `language`, `retrieval_miss`, `incomplete`, `other`.

Annotate at least 10 concrete failures. Use two reviewers where feasible and adjudicate disagreements. Do not tune using the held-out test set. Calculate mean correctness, unsupported-claim rate, and paired bootstrap confidence intervals after annotation. Store human scores in a separate CSV keyed by `ticket_id` and `model`.
