
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
        self.post_process_fn = postprocess_fn
        self.max_token_length = max_token_length
        self.mapping = {
            0:  "B-DT",
            1:  "B-LC",
            2:  "B-OG",
            3:  "B-PS",
            4:  "B-QT",
            5:  "B-TI",
            6:  "O",
        }

    # need to change this function
    def subword_to_char(self, pred_id, token_list, offset_mapping) :
        sentence = "".join(token_list)
        length = len(offset_mapping)

        breakpoint()
        
        mapping = {}
        for i in range(length) :
            offset = offset_mapping[i]
            start_pos, end_pos = offset

            if start_pos == 0 and end_pos == 0 :
                continue

            for j in range(start_pos, end_pos) :
                mapping[j] = pred_id[i]
        
        for i in range(len(sentence)) :
            ch = sentence[i]
            if ch == " " :
                if i not in mapping :
                    if mapping[i-1] == mapping[i+1] :
                        mapping[i] = mapping[i-1]
                    else :
                        mapping[i] = 12

        char_pred = []
        for i in range(len(token_list)) :
            p = pred_id[mapping[i]] if i in mapping else 12
            char_pred.append(p)
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
        
        char_predictions, char_labels = [], []
        for i in range(len(pred_ids)) :
            pred_id = pred_ids[i]
            token_list = tokens[i]
            offset_mapping = offset_mappings[i]

            char_pred = self.subword_to_char(pred_id, token_list, offset_mapping)

        breakpoint()

        metrics = self.compute_metrics({"prediction" : char_predictions, "labels" : char_labels})
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        output.metrics.update(metrics)

        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics