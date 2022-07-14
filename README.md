# KLUE NER Baseline

Named Entity Recognition Baseline for KLUE Benchmark dataset

### Baseline Metric

Training config available at [config.yaml](./config.yaml)

|     Evaluation     | Loss  | Accuracy | Precision | Recall | F1 Score |
| :----------------: | :---: | :------: | :-------: | :----: | :------: |
| klue/roberta-large | 0.061 |  0.980   |   0.876   | 0.906  |  0.891   |
| klue/roberta-base  | 0.183 |  0.952   |   0.782   | 0.834  |  0.807   |
|      kpfbert       |       |          |           |        |          |

### How to run

1. Change configurations in `config.yaml`
2. Run `python3 main.py`

### Reference

- [Huggingface Token Classification](https://github.com/huggingface/notebooks/blob/main/examples/token_classification.ipynb)
- [KLUE Dataset on Huggingface](https://huggingface.co/datasets/klue)