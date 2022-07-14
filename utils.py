import logging
import re
from pathlib import Path

import transformers
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def return_tokenizer_type_and_strip_char(tokenizer: PreTrainedTokenizer):
    """
    Check tokenizer type.
    In KLUE paper, we only support wordpiece (BERT, KLUE-RoBERTa, ELECTRA) & sentencepiece (XLM-R).
    Will give error if you use other tokenization. (e.g. bbpe)
    """
    if isinstance(tokenizer, transformers.XLMRobertaTokenizer):
        logger.info(f"Using {type(tokenizer).__name__} for fixing tokenization result")
        return "xlm-sp", "_"  # Sentencepiece
    elif isinstance(tokenizer, transformers.BertTokenizer):
        logger.info(f"Using {type(tokenizer).__name__} for fixing tokenization result")
        return "bert-wp", "##"  # Wordpiece (including BertTokenizer & ElectraTokenizer)
    else:
        raise ValueError(
            "This code only supports XLMRobertaTokenizer & BertWordpieceTokenizer"
        )


def read_data(file_path: str, tokenizer: PreTrainedTokenizer = None):
    if tokenizer:
        _, strip_char = return_tokenizer_type_and_strip_char(tokenizer)

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
    file_path = Path(file_path)
    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r"\n\t?\n", raw_text)

    data_list = []
    for doc in raw_docs:
        original_clean_labels = []  # clean labels (bert clean func)
        sentence = ""
        for line in doc.split("\n"):
            if line.startswith("##"):  # skip comment
                continue
            token, tag = line.split("\t")
            sentence += token
            if token != " ":
                original_clean_labels.append(tag)

        if tokenizer:
            # sentence: "안녕 하세요.."
            # original_clean_labels: [안, 녕, 하, 세, 요, ., .]
            # sent_words: [안녕, 하세요..]
            sent_words = sentence.split(" ")
            modi_labels = []
            char_idx = 0
            for word in sent_words:
                # 안녕, 하세요
                correct_syllable_num = len(word)
                tokenized_word = tokenizer.tokenize(word)
                # case1: 음절 tokenizer --> [안, ##녕]
                # case2: wp tokenizer --> [안녕]
                # case3: 음절, wp tokenizer에서 unk --> [unk]
                # unk규칙 --> 어절이 통채로 unk로 변환, 단, 기호는 분리
                contain_unk = True if tokenizer.unk_token in tokenized_word else False
                for i, token in enumerate(tokenized_word):
                    token = token.replace(strip_char, "")
                    if not token:
                        continue
                    modi_labels.append(original_clean_labels[char_idx])
                    if not contain_unk:
                        char_idx += len(token)
                if contain_unk:
                    char_idx += correct_syllable_num
        else:
            modi_labels = []
            strip_char = None

        text_a = sentence  # original sentence
        instance = {
            "text_a": text_a,
            "label": modi_labels,
            "original_clean_labels": original_clean_labels,
        }
        data_list.append(instance)

    return data_list, label_list, strip_char
