# KLUE NER Baseline

Named Entity Recognition Baseline for KLUE Benchmark dataset

### Baseline Metric

Training config available at [config.yaml](./config.yaml)

|                Evaluation                | Loss  | Accuracy | Precision | Recall | Character F1 Score |
| :--------------------------------------: | :---: | :------: | :-------: | :----: | :----------------: |
|            klue/roberta-large            | 0.061 |  0.981   |   0.893   | 0.909  |       0.899        |
| monologg/koelectra-base-v3-discriminator | 0.085 |  0.976   |   0.853   | 0.891  |       0.872        |
|                 kpfbert                  | 0.086 |  0.972   |   0.846   | 0.877  |       0.861        |
|            klue/roberta-base             | 0.183 |  0.952   |   0.782   | 0.834  |       0.807        |

### How to run

1. Change configurations in `config.yaml`
2. Run `python3 main.py`

### Reference

- [Huggingface Token Classification](https://github.com/huggingface/notebooks/blob/main/examples/token_classification.ipynb)
- [KLUE Dataset on Huggingface](https://huggingface.co/datasets/klue)
