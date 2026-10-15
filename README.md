# Intelligent Support Ticket Assistant

A bilingual (Arabic/English) support assistant for an e-commerce store. It reads a customer ticket, assigns a category and priority, drafts a reply, and decides whether a human should review it. The project implements three approaches side by side so they can be compared on the same tickets.

| Approach | How it works | Generation |
|---|---|---|
| **Knowledge-based** | Keyword rules classify the ticket and pick a stored policy section | None (returns the policy text) |
| **RAG** | Multilingual E5 embeddings + BM25, fused with Reciprocal Rank Fusion, retrieve the top policy sections | Local Qwen2.5 writes a short plain-text reply |
| **LLM prompting** | Role instructions, few-shot examples and a JSON output contract, with no knowledge base | Cloud LLM through OpenRouter |

A Gradio interface shows the three results next to each other, with category, priority, escalation decision, evidence and response time.

## How each approach works

**Knowledge-based** (`src/knowledge_based.py`). Arabic text is normalized, keywords are counted per category, and rules handle delivery intent and priority (`urgent`, `high`, `normal`). A section of the matching knowledge-base file is chosen by intent hints or title overlap. Unknown or urgent tickets are escalated.

**RAG** (`src/rag.py`, `src/retrieval.py`, `src/local_llm.py`).
- Dense retrieval uses `intfloat/multilingual-e5-small` (384 dimensions, `query:` / `passage:` prefixes, normalized vectors).
- BM25 runs alongside it, with a small bilingual alias list so informal Arabic terms match the English policy text.
- Both rankings are merged with Reciprocal Rank Fusion. An optional cross-encoder reranker can be enabled with `RAG_RERANK=1`.
- Qwen2.5 (0.5B on CPU, 1.5B on GPU) writes a 1–3 sentence reply using only the retrieved sections, or answers `NEEDS_HUMAN` when they don't cover the question.
- Category and priority come from the same rule-based classifier as the baseline, so the two approaches share classification labels. Keep this in mind when comparing classification scores.
- Evidence IDs list the retrieved sections. They show what the model was given, not proof that each claim is supported.

**LLM prompting** (`src/llm_prompting.py`, `src/cloud.py`). The prompt sets a support-agent role, forbids claiming account access or inventing policy, treats ticket text as data, and includes English and Arabic few-shot examples. The model returns JSON with `category`, `priority`, `suggested_reply` and `escalate`, which is parsed and validated against the allowed values.

## Project structure

```
app.py                  Gradio interface
src/
  pipeline.py           builds an assistant by name
  knowledge_based.py    rule-based classifier and baseline
  retrieval.py          hybrid E5 + BM25 retriever
  rag.py                retrieval-augmented assistant
  local_llm.py          local Qwen generator
  cloud.py              OpenRouter client
  llm_prompting.py      prompt-only assistant
  kb.py                 knowledge-base loader
  schemas.py            shared result object
data/
  kb/                   policy sections (Markdown)
  development.csv, validation.csv, test.csv
  source/               original ticket and retrieval-question files
  make_splits.py        regenerates the splits
evaluation/             retrieval and classification evaluation
tests/                  offline unit tests
```

## Setup

Requires Python 3.10 or newer.

**Windows (PowerShell)**

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[rag,local,test]"
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[rag,local,test]"
```

The knowledge-based approach needs only the base install (`pip install -e .`). RAG needs the `rag` and `local` extras. The first RAG run downloads the embedding and Qwen weights, and CPU generation can be slow.

## Configuration

Set these in the same terminal session before running the app. Never commit API keys.

| Variable | Purpose | Default |
|---|---|---|
| `OPENROUTER_API_KEY` | Key for the LLM prompting approach | required for that approach |
| `OPENROUTER_BASE_URL` | API endpoint | `https://openrouter.ai/api/v1` |
| `OPENROUTER_MODEL` | Model ID | `openrouter/free` |
| `RAG_TOP_K` | Sections passed to the generator (1–5) | `2` |
| `RAG_RERANK` | Set to `1` to enable the cross-encoder reranker | `0` |
| `RAG_CANDIDATES` | Candidates scored by the reranker | `20` |
| `LOCAL_RAG_MODEL` | Override the local Qwen model | auto (0.5B CPU / 1.5B GPU) |
| `LOCAL_RAG_MAX_TOKENS` | Reply length limit | `160` |
| `INPUT_USD_PER_MILLION`, `OUTPUT_USD_PER_MILLION` | Prices used for the cost estimate | `0` |

PowerShell example:

```powershell
$env:OPENROUTER_API_KEY="YOUR_KEY"
$env:OPENROUTER_MODEL="your-chosen-model-id"
```

`openrouter/free` is a router that may serve a different model between runs. Use a fixed model ID for reproducible results.

## Run

```powershell
python app.py
```

Open http://127.0.0.1:7860. RAG and the baseline work without an API key; the LLM prompting card shows an error until a key is set.

Run the tests (no model downloads or API calls):

```powershell
python -m pytest -q
```

## Evaluation

```powershell
python -m evaluation.retrieval_eval
python -m evaluation.run --models knowledge_based
python -m evaluation.run --models knowledge_based rag llm_prompting
```

- `evaluation.retrieval_eval` reports Recall@3 and MRR for the retriever, split by language, using `data/source/rag_eval.csv`.
- `evaluation.run` runs the chosen approaches on `data/test.csv`, saves `results/predictions.csv`, and prints accuracy and macro-F1 overall and per language.
- `evaluation/ANNOTATION.md` describes a blind manual review of reply quality (correctness, unsupported claims, escalation, failure types).

## Data

- `data/kb/`: ten Markdown policy files for DemoMart, a fictional store. The policies are invented.
- `data/source/arabic_tickets.csv`: Arabic tickets in Modern Standard Arabic and Gulf dialect, each with an intent label.
- `data/source/rag_eval.csv`: questions paired with the correct policy section.
- `development.csv`, `validation.csv`, `test.csv`: stratified 60/20/20 splits by language and category, generated by `data/make_splits.py` with a fixed seed. The English tickets were written for this project.

## Limitations

- The datasets are small and partly template-based, so scores have wide uncertainty and are not a real-world benchmark.
- Reply quality is not scored automatically. It needs the manual review described above.
- Classification rows where an approach returns an error are excluded from the scores, so report the error count alongside accuracy.
- Customer tickets sent to a cloud model leave your machine. Mask personal data before using real tickets.
- Fine-tuning is not part of this project.

## Acknowledgments

The Arabic ticket set and the knowledge-base content originate from SDAIA Academy's Applied NLP course materials.
**SDAIA Academy:** https://github.com/SDAIAAcademy

