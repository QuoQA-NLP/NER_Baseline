import argparse
import csv
import os
import tarfile
from typing import List

import torch

from dataset import NerDataset
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          PreTrainedTokenizer)
from utils import read_data

KLUE_NER_OUTPUT = "output.csv"  # the name of the output file should be output.csv


def load_model(model_dir, model_tar_file):
    tarpath = os.path.join(model_dir, model_tar_file)
    tar = tarfile.open(tarpath, "r:gz")
    tar.extractall(path=model_dir)

    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    return model


class OutputConvertor(object):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        label_list: List[str],
        strip_char: str,
        max_length: int,
    ):
        self.in_unk_token = "[+UNK]"

        self.tokenizer = tokenizer
        self.label_list = label_list
        self.strip_char = strip_char
        self.max_length = max_length

    def tokenizer_out_aligner(self, t_in, t_out, strip_char="##"):
        t_out_new = []
        i, j = 0, 0
        UNK_flag = False

        while True:
            if i == len(t_in) and j == len(t_out) - 1:
                break
            step_t_out = (
                len(t_out[j].replace(strip_char, ""))
                if t_out[j] != self.tokenizer.unk_token
                else 1
            )
            if UNK_flag:
                t_out_new.append(self.in_unk_token)
            else:
                t_out_new.append(t_out[j])
            if (
                j < len(t_out) - 1
                and t_out[j] == self.tokenizer.unk_token
                and t_out[j + 1] != self.tokenizer.unk_token
            ):
                i += step_t_out
                UNK_flag = True
                if t_in[i] == t_out[j + 1][0]:
                    j += 1
                    UNK_flag = False
            else:
                i += step_t_out
                j += 1
                UNK_flag = False
            if j == len(t_out):
                UNK_flag = True
                j -= 1
        return t_out_new

    def convert_into_character_pred(self, data, subword_preds):
        text = data["text_a"]

        original_sentence = text  # 안녕 하세요 ^^
        subword_preds = [int(x) for x in subword_preds]
        character_preds = [subword_preds[0]]  # [CLS]
        character_preds_idx = 1

        for word in original_sentence.split(" "):  # 안녕 하세요
            if character_preds_idx >= self.max_length - 1:
                break
            subwords = self.tokenizer.tokenize(word)
            if self.tokenizer.unk_token in subwords:  # 뻥튀기가 필요한 case!
                # case1: ..찝찝..찝찝해 --> [".", ".", "[UNK]", ".", ".", "[UNK]"]
                # case2: 미나藤井美菜27가 --> ['미나', '[UNK]', '[UNK]', '美', '[UNK]', '27', '##가']
                unk_aligned_subwords = self.tokenizer_out_aligner(
                    word, subwords, self.strip_char
                )  # 복원 함수
                # case1: [".", ".", "[UNK]", "[+UNK]", ".", ".", "[UNK]", "[+UNK]", "[+UNK]"]
                # case2: ['미나', '[UNK]', '[UNK]', '美', '[UNK]', '27', '##가']
                unk_flag = False
                for subword in unk_aligned_subwords:
                    if character_preds_idx >= self.max_length - 1:
                        break
                    subword_pred = subword_preds[character_preds_idx]
                    subword_pred_label = self.label_list[subword_pred]
                    if subword == self.tokenizer.unk_token:
                        unk_flag = True
                        character_preds.append(subword_pred)
                        continue
                    elif subword == self.in_unk_token:
                        if subword_pred_label == "O":
                            character_preds.append(subword_pred)
                        else:
                            _, entity_category = subword_pred_label.split("-")
                            character_pred_label = "I-" + entity_category
                            character_pred = self.label_list.index(character_pred_label)
                            character_preds.append(character_pred)
                        continue
                    else:
                        if unk_flag:
                            character_preds_idx += 1
                            subword_pred = subword_preds[character_preds_idx]
                            character_preds.append(subword_pred)
                            unk_flag = False
                        else:
                            character_preds.append(subword_pred)
                            character_preds_idx += (
                                1  # TODO +unk가 끝나는 시점에서도 += 1 을 해줘야 다음 label로 넘어감
                            )
            else:
                for subword in subwords:  # 안녕 -> 안, ##녕하, ##세요
                    if character_preds_idx >= self.max_length - 1:
                        break
                    subword = subword.replace(
                        self.strip_char, ""
                    )  # xlm roberta: "▁" / others "##"
                    subword_pred = subword_preds[character_preds_idx]
                    subword_pred_label = self.label_list[subword_pred]
                    for i in range(0, len(subword)):  # 안, 녕
                        if i == 0:
                            character_preds.append(subword_pred)
                        else:
                            if subword_pred_label == "O":
                                character_preds.append(subword_pred)
                            else:
                                _, entity_category = subword_pred_label.split("-")
                                character_pred_label = "I-" + entity_category
                                character_pred = self.label_list.index(
                                    character_pred_label
                                )
                                character_preds.append(character_pred)
                    character_preds_idx += 1

        character_preds.append(subword_preds[-1])  # [SEP] label
        return character_preds

    def return_char_pred_output(self, data_list, preds):
        list_of_character_preds = []
        for data, pred in zip(data_list, preds):
            character_preds = self.convert_into_character_pred(data, pred)
            list_of_character_preds.append(character_preds)

        return list_of_character_preds


@torch.no_grad()
def inference(args):
    # Set GPU
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(args.model_dir, args.model_tar_file).to(device)
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    # Load tokenzier
    kwargs = (
        {"num_workers": num_gpus, "pin_memory": True}
        if torch.cuda.is_available()
        else {}
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # Build dataloader
    test_data, label_list, strip_char = read_data(
        os.path.join(args.data_dir, args.test_filename), tokenizer
    )
    dataset = NerDataset(
        tokenizer,
        test_data,
        label_list,
        args.max_length,
        args.batch_size,
        shuffle=False,
        **kwargs
    )
    dataloader = dataset.loader

    # Run Inference
    preds = []
    for input_ids, token_type_ids, attention_mask, labels in dataloader:
        input_ids, token_type_ids, attention_mask, labels = (
            input_ids.to(device),
            token_type_ids.to(device),
            attention_mask.to(device),
            labels.to(device),
        )

        output = model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels,
        )[1]

        pred = output.argmax(dim=2)
        pred = pred.detach().cpu().numpy()

        for p in pred:
            preds.append(p.tolist())

    # Convert sub-word preds into char preds
    output_convertor = OutputConvertor(
        tokenizer, label_list, strip_char, args.max_length
    )
    list_of_character_preds = output_convertor.return_char_pred_output(test_data, preds)

    with open(os.path.join(args.output_dir, KLUE_NER_OUTPUT), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(list_of_character_preds)
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for inference (default: 64)",
    )
    parser.add_argument(
        "--data_dir", type=str, default=os.environ.get("SM_CHANNEL_EVAL", "/data")
    )
    parser.add_argument(
        "--model_dir", type=str, default='./model'
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/output"),
    )
    parser.add_argument(
        "--model_tar_file",
        type=str,
        default="klue_ner_model.tar.gz",
        help="it needs to include all things for loading baseline model & tokenizer, \
             only supporting transformers.AutoModelForSequenceClassification as a model \
             transformers.XLMRobertaTokenizer or transformers.BertTokenizer as a tokenizer",
    )
    parser.add_argument(
        "--test_filename",
        default="klue-ner-v1.1_test.tsv",
        type=str,
        help="Name of the test file (default: klue-ner-v1.1_test.tsv)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=510,
        help="maximum sequence length (default: 510)",
    )
    args = parser.parse_args()

    inference(args)
