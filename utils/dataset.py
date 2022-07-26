import yaml
import numpy as np
from functools import partial
from transformers import AutoTokenizer
from datasets import load_dataset
from easydict import EasyDict

class Loader :

    def __init__(self, path, max_length) :
        self.path = path
        self.max_length = max_length
        self.mapping = {
            0:  0,  # B-DT
            1:  0,  # I-DT -> B-DT
            2:  1,  # B-LC
            3:  1,  # I-LC -> B-LC
            4:  2,  # B-OG
            5:  2,  # I-OG -> B-OG
            6:  3,  # B-PS
            7:  3,  # I-PS -> B-PS
            8:  4,  # B-QT
            9:  4,  # I-QT -> B-QT
            10: 5,  # B-TI
            11: 5,  # I-TI -> B-TI
            12: 6
        }
        
    def load(self) :
        # Read config.yaml file
        with open(self.path) as infile:
            SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
            CFG = EasyDict(SAVED_CFG["CFG"])

        # Loading Datasets
        dataset = load_dataset(CFG.dset_name, CFG.task)
        train_dataset, eval_dataset = dataset["train"], dataset["validation"]

        # Preprocessed Training Datasets
        train_dataset = train_dataset.map(self.preprocess, batched=True)

        tokenizer = AutoTokenizer.from_pretrained(CFG.PLM)
        encode_fn = partial(self.label_tokens_ner, tokenizer=tokenizer)

        # Encoding Datasets
        train_dataset = train_dataset.map(encode_fn, batched=False)
        eval_dataset = eval_dataset.map(encode_fn, batched=False)

        # Remove unnecessary columns
        train_dataset = train_dataset.remove_columns(column_names=["sentence", "tokens", "ner_tags", "offset_mapping"])
        eval_dataset = eval_dataset.remove_columns(column_names=["sentence", "labels"])
        return train_dataset, eval_dataset

    def preprocess(self, examples) :

        batch_size = len(examples["ner_tags"])

        preprocessed_tags = []
        for i in range(batch_size) :
            ner_tag = examples["ner_tags"][i]
            ner_tag = [self.mapping[t] for t in ner_tag]
            preprocessed_tags.append(ner_tag)

        examples["ner_tags"] = preprocessed_tags
        return examples

    def label_tokens_ner(self, examples, tokenizer):
        sentence = "".join(examples["tokens"])
        tokenized_output = tokenizer(
            sentence,
            return_token_type_ids=False,
            return_offsets_mapping=True,
            max_length=self.max_length,
            truncation=True,
        )

        label_token_map = []

        list_label = examples["ner_tags"]
        list_label = [-100] + list_label + [-100]
        for token_idx, offset_map in enumerate(tokenized_output["offset_mapping"]):
            begin_letter_idx, end_letter_idx = offset_map
            label_begin = list_label[begin_letter_idx]
            label_end = list_label[end_letter_idx]
            token_label = np.array([label_begin, label_end])
            if label_begin == 6 and label_end == 6:
                token_label = 6
            elif label_begin == -100 and label_end == -100:
                token_label = -100
            else:
                token_label = label_begin if label_begin != 6 else 6
                token_label = label_end if label_end != 6 else 6

            label_token_map.append(token_label)

        tokenized_output["labels"] = label_token_map
        return tokenized_output

    # if CFG.DEBUG:
    #     sample_idx = 0
    #     data_seg = "train"
    #     print(dataset[data_seg][sample_idx]["sentence"])
    #     print("------Labeled output------")
    #     output = label_tokens_ner(dataset[data_seg][sample_idx])
    #     for idx, id in enumerate(output["input_ids"]):
    #         print(idx, tokenizer.convert_ids_to_tokens(output["input_ids"])[idx], output["labels"][idx])

