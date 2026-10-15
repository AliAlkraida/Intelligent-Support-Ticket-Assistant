import csv
import json
from pathlib import Path
from src.retrieval import HybridRetriever
from evaluation.metrics import recall_at_k, mrr

if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "data/source/rag_eval.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    retriever = HybridRetriever()
    retrieved = [[s.section_id for s, _ in retriever.search(r["question"], k=3)] for r in rows]
    for lang in ("ar", "en"):
        pairs = [(r["gold_section"], ids) for r, ids in zip(rows, retrieved) if r["lang"] == lang]
        print(lang, len(pairs), "Recall@3", recall_at_k([g for g, _ in pairs], [x for _, x in pairs]),
              "MRR", mrr([g for g, _ in pairs], [x for _, x in pairs]))
