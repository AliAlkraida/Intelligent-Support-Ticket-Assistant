from collections import defaultdict

def classification_scores(rows, truth="gold_category", pred="category"):
    labels = sorted({r[truth] for r in rows} | {r[pred] for r in rows})
    per_class = {}
    for label in labels:
        tp = sum(r[truth] == label and r[pred] == label for r in rows)
        fp = sum(r[truth] != label and r[pred] == label for r in rows)
        fn = sum(r[truth] == label and r[pred] != label for r in rows)
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        per_class[label] = {"precision": round(precision, 4), "recall": round(recall, 4),
                            "f1": round(2*precision*recall/(precision+recall), 4) if precision+recall else 0}
    return {"accuracy": sum(r[truth] == r[pred] for r in rows)/len(rows) if rows else 0,
            "macro_f1": sum(x["f1"] for x in per_class.values())/len(per_class) if per_class else 0,
            "per_class": per_class}

def recall_at_k(gold, retrieved, k=3):
    return sum(g in ids[:k] for g, ids in zip(gold, retrieved))/len(gold) if gold else 0

def mrr(gold, retrieved):
    return sum(next((1/(i+1) for i, v in enumerate(ids) if v == g), 0) for g, ids in zip(gold, retrieved))/len(gold) if gold else 0
