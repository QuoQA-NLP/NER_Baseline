import numpy as np
from datasets import load_metric
from tqdm import tqdm

class Metric :

    def __init__(self, ) :
        self.metric = load_metric("seqeval")
        self.label_list = [
            "B-PS",
            "B-LC",
            "B-OG",
            "B-DT",
            "B-TI",
            "B-QT",
            "O",
            "I-PS",
            "I-LC",
            "I-OG",
            "I-DT",
            "I-TI",
            "I-QT",
        ]

    def recover(self, p) :
        batch_size, seq_size = p.shape

        for i in tqdm(range(batch_size)) :
            for j in range(1, seq_size) :

                if (p[i,j] == 0 and p[i,j-1] == 0) or (p[i,j] == 0 and p[i,j-1] == 7) : # B-PS or I-PS
                    p[i,j] = 7 # I-PS

                if (p[i,j] == 1 and p[i,j-1] == 1)  or (p[i,j] == 1 and p[i,j-1] == 8): # B-LC or I-LC
                    p[i,j] = 8 # I-LC

                if (p[i,j] == 2 and p[i,j-1] == 2)  or (p[i,j] == 2 and p[i,j-1] == 9) : # B-OG or I-OG
                    p[i,j] = 9 # I-OG

                if (p[i,j] == 3 and p[i,j-1] == 3) or (p[i,j] == 3 and p[i,j-1] == 10) : # B-DT or I-DT
                    p[i,j] = 10 # I-DT

                if (p[i,j] == 4 and p[i,j-1] == 4)  or (p[i,j] == 4 and p[i,j-1] == 11) : # B-IT or I-IT
                    p[i,j] = 11 # I-IT

                if (p[i,j] == 5 and p[i,j-1] == 5)  or (p[i,j] == 5 and p[i,j-1] == 12) : # B-QT or I-QT
                    p[i,j] = 12 # I-QT
        return p

    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        print("Recovering Labels & Predictions")
        labels = self.recover(labels)
        predictions = self.recover(predictions)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
