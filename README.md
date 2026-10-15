# Intelligent Support Ticket Assistant

A bilingual (Arabic/English) e-commerce ticket assistant comparing three approaches: Knowledge-based, RAG, and LLM prompting.

**SDAIA Academy:** https://github.com/SDAIAAcademy

## Three approaches

| Approach | Implementation | Generation |
|---|---|---|
| Knowledge-based baseline | Deterministic keyword rules + stored DemoMart KB passages | None |
| RAG | Local multilingual E5 embeddings + BM25 + RRF; top-2 KB sections | Local Qwen2.5 plain-text response |
| LLM prompting | Role instructions, few-shot demonstrations, JSON output, safety constraints | OpenRouter cloud LLM |

The orange Gradio interface includes Quick Examples, side-by-side results, and Advanced Details.

## Prompt engineering techniques

- **Role prompting:** the model acts as a DemoMart support agent.
- **Grounding:** RAG is told to use only retrieved policy excerpts.
- **Few-shot prompting:** RAG includes a short policy-answer example; prompt-only includes English and Arabic demonstrations.
- **Output constraints:** RAG produces 1–3 plain-text sentences; prompt-only produces structured JSON.
- **Safety / uncertainty instructions:** do not claim real account access, invent policies, or promise refunds. RAG may output `NEEDS_HUMAN`.

RAG intentionally does **not** ask the small local Qwen model for JSON, category labels, or citations. Python handles category/priority and records retrieved sections. The evidence IDs identify retrieved passages, not proof of factual support. RAG and the baseline share the same rule-based classifier; report this dependency when comparing classification scores.

## Windows PowerShell setup

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[rag,local,test]"
```

To enable **LLM Prompting**, set your OpenRouter key in the same PowerShell session:

```powershell
$env:OPENROUTER_API_KEY="YOUR_KEY"
$env:OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
$env:OPENROUTER_MODEL="openrouter/free"
```

Use a **fixed model ID** rather than `openrouter/free` for final reproducible evaluation. Never commit API keys or personal customer data.

```powershell
python -m pytest -q
python app.py
```

Open http://127.0.0.1:7860. First RAG run downloads local embedding/Qwen weights. CPU generation can be slow. No API key is required for RAG or the knowledge-based baseline.

## Evaluation (required for capstone)

Original SDAIA KB and ticket files are under `data/`. Additional capstone splits are in `data/development.csv`, `data/validation.csv`, and `data/test.csv`; these are small illustrative datasets, not a fully verified real-world benchmark. Avoid treating template-derived tickets as independent real customer cases.

```powershell
python -m evaluation.retrieval_eval
python -m evaluation.run --models knowledge_based
# With OpenRouter configured and local Qwen available:
python -m evaluation.run --models knowledge_based rag llm_prompting
```

Review `evaluation/ANNOTATION.md` for manual answer-quality annotations. Report category macro-F1 by language, retrieval Hit@1/Hit@3/MRR, answer correctness, latency, and at least 10 categorized failures. Run evaluation on a held-out set; do not claim final accuracy until all models are tested. Fine-tuning is excluded by project scope and requires a brief rubric justification.
