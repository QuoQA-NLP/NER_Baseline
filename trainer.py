
import math
import time
import copy
import numpy as np
from transformers import Trainer
from torch.utils.data import Dataset
from typing import Optional, List, Dict

class NERTrainer(Trainer) :

    def __init__(self, *args, postprocessor=None, max_token_length, **kwargs):
        super().__init__(*args, **kwargs)
        self.postprocessor = postprocessor
        self.max_token_length = max_token_length
        self.id_convertor = {
            0: 0,       # B-PS
            1: 7,       # I-PS
            2: 1,       # B-LC
            3: 8,       # I-LC
            4: 2,       # B-OG
            5: 9,       # I-OG
            6: 3,       # B-DT
            7: 10,      # I-DT
            8: 4,       # B-IT
            9: 11,      # I-IT
            10: 5,      # B-QT
            11: 12,     # I-QT
            12: 6,      # O
        }

    def subword_to_char(self, pred, token_list) :
        sentence = "".join(token_list)
        tokenized_output = self.tokenizer(
            sentence,
            return_token_type_ids=False,
            return_offsets_mapping=True,
            max_length=self.args.max_length,
            truncation=True,
        )

        offset_mapping = tokenized_output["offset_mapping"]
        length = len(offset_mapping)
        
        mapping = {}
        for i in range(length) :
            offset = offset_mapping[i]
            start_pos, end_pos = offset

            if start_pos == 0 and end_pos == 0 :
                continue

            for j in range(start_pos, end_pos) :
                mapping[j] = i

        char_pred = []
        for i in range(len(token_list)) :
            p = pred[mapping[i]] if i in mapping else 6
            char_pred.append(p)
        return char_pred


    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset

        tokens = copy.deepcopy(eval_dataset["tokens"])
        labels = copy.deepcopy(eval_dataset["ner_tags"])
        eval_dataset = eval_dataset.remove_columns(column_names=["ner_tags", "tokens"])

        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        prediction = output.predictions
        pred_ids = np.argmax(prediction, axis=-1)
        
        tag_predictions, tag_labels = [], []
        for i in range(len(pred_ids)) :
            pred = pred_ids[i]
            token = tokens[i]

            char_pred = self.subword_to_char(pred, token)
            tag_predictions.append(char_pred)

            ner_tag = [self.id_convertor[t] for t in labels[i]]
            tag_labels.append(ner_tag)

        tag_predictions, tag_labels = self.postprocessor(tag_predictions, labels)
        output.metrics.update(
            self.compute_metrics({"prediction" : tag_predictions, 
            "labels" : tag_labels}
            )
        )

        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics