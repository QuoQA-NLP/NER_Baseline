
import math
import time
import copy
import numpy as np
from transformers import Trainer
from torch.utils.data import Dataset
from typing import Optional, List, Dict

class NERTrainer(Trainer) :

    def __init__(self, *args, postprocess_fn, max_token_length, **kwargs):
        super().__init__(*args, **kwargs)
        self.postprocess_fn = postprocess_fn
        self.max_token_length = max_token_length
        self.mapping = {
            0:  "B-DT",
            1:  "I-DT",
            2:  "B-LC",
            3:  "I-LC",
            4:  "B-OG",
            5:  "I-OG",
            6:  "B-PS",
            7:  "I-PS",
            8:  "B-QT",
            9:  "I-QT",
            10:  "B-TI",
            11:  "I-TI",
            12:  "O",
        }

    def subword_to_char(self, pred_id, token_list, offset_mapping) :
        sentence = "".join(token_list)
        length = len(offset_mapping)
        
        # key : char_id, value : token_id
        mapping = {}
        for i in range(length) :
            offset = offset_mapping[i]
            start_pos, end_pos = offset
            if start_pos == 0 and end_pos == 0 :
                continue

            for j in range(start_pos, end_pos) :
                mapping[j] = pred_id[i]
        
        for i in range(len(token_list)) :
            ch = token_list[i]
            if i not in mapping :
                if i > 0 and i < len(token_list) - 1 :
                    if i < len(token_list) - 1 and mapping[i-1] == mapping[i+1] :
                        mapping[i] = mapping[i-1]
                    else :
                        mapping[i] = 6
                else :
                    mapping[i] = 6

        char_pred = [mapping[i] for i in range(len(token_list))]
        return char_pred


    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset

        tokens = eval_dataset["tokens"]
        labels = eval_dataset["ner_tags"]
        offset_mappings = eval_dataset["offset_mapping"]

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
        
        predictions = []
        for i in range(len(pred_ids)) :
            pred_id = pred_ids[i]
            token_list = tokens[i]
            offset_mapping = offset_mappings[i]

            char_pred = self.subword_to_char(pred_id, token_list, offset_mapping)
            char_pred = self.postprocess_fn(char_pred)
            predictions.append(char_pred)
        
        predictions = [[self.mapping[i] for i in p]for p in predictions]
        labels = [[self.mapping[i] for i in l]for l in labels]

        metrics = self.compute_metrics({"prediction" : predictions, "labels" : labels})
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        output.metrics.update(metrics)

        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics