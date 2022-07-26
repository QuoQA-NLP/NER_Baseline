
from datasets import load_metric

class Metric :

    def __init__(self, ) :
        self.metric = load_metric("seqeval")

    def compute_metrics(self, p):
        char_predictions = p["prediction"]
        char_labels = p["labels"]

        results = self.metric.compute(predictions=char_predictions, references=char_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
