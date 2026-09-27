"""Macro F0.5 evaluator matching the challenge's singleton convention."""

from __future__ import annotations

from collections import defaultdict


def macro_metrics(predicted: dict[str, set[str]], truth: dict[str, set[str]], source_ids: list[str]) -> dict[str, float]:
    beta2 = 0.25
    scores, precisions, recalls = [], [], []
    singleton_correct = 0
    fp = predicted_count = 0
    for source_id in source_ids:
        actual = truth.get(source_id, set())
        pred = predicted.get(source_id, set())
        if not actual and not pred:
            score = precision = recall = 1.0
            singleton_correct += 1
        elif not actual:
            score = precision = recall = 0.0
        else:
            tp = len(actual & pred)
            precision = tp / len(pred) if pred else 0.0
            recall = tp / len(actual)
            score = (1 + beta2) * precision * recall / (beta2 * precision + recall) if (precision + recall) else 0.0
        fp += len(pred - actual)
        predicted_count += len(pred)
        scores.append(score)
        precisions.append(precision)
        recalls.append(recall)
    return {
        "macro_f0_5": sum(scores) / len(scores), "macro_precision": sum(precisions) / len(precisions),
        "macro_recall": sum(recalls) / len(recalls), "singleton_accuracy": singleton_correct / len(source_ids),
        "false_merge_rate": fp / predicted_count if predicted_count else 0.0,
    }


def truth_from_rows(rows):
    out = defaultdict(set)
    for source1_id, target_id in rows:
        out[source1_id].add(target_id)
    return dict(out)
