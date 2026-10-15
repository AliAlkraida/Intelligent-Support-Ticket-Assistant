import argparse
import csv
import json
from pathlib import Path
from src.pipeline import build_assistant
from evaluation.metrics import classification_scores

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "data" / "test.csv"))
    parser.add_argument("--models", nargs="+", default=["knowledge_based", "rag", "llm_prompting"])
    parser.add_argument("--output", default=str(ROOT / "results" / "predictions.csv"))
    args = parser.parse_args()
    rows = list(csv.DictReader(open(args.dataset, encoding="utf-8")))
    output = []
    for model in args.models:
        assistant = build_assistant(model)
        for r in rows:
            try:
                pred = assistant.answer(r["text"]).to_dict()
            except Exception as exc:
                pred = {"model": model, "error": str(exc)}
            output.append({**r, **pred, "evidence": json.dumps(pred.get("evidence", []), ensure_ascii=False)})
    path = Path(args.output);path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(r.keys() for r in output)))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields);writer.writeheader();writer.writerows(output)
    for model in args.models:
        group = [r for r in output if r.get("model") == model and "category" in r]
        print(model, json.dumps(classification_scores(group), ensure_ascii=False))
        for lang in ("ar", "en"):
            sub = [r for r in group if r.get("lang") == lang]
            print(" ", lang, len(sub), json.dumps(classification_scores(sub), ensure_ascii=False))
    print("Saved", path)
    print("Answer correctness and groundedness require independent human annotations; see evaluation/ANNOTATION.md")

if __name__ == "__main__":
    main()
