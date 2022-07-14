import numpy as np
from datasets import load_metric

metric = load_metric("seqeval")

label_list = [
    "B-PS",
    "I-PS",
    "B-LC",
    "I-LC",
    "B-OG",
    "I-OG",
    "B-DT",
    "I-DT",
    "B-TI",
    "I-TI",
    "B-QT",
    "I-QT",
    "O",
]


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
