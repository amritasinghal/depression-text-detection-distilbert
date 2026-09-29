# Depression Text Detection with DistilBERT

Fine-tunes a DistilBERT model (`distilbert-base-uncased-finetuned-sst-2-english`) to classify text as **Normal (1)** or **Depressed / Abnormal (0)**, using cross-dataset training and testing.

> **Disclaimer:** This is a student ML project for learning purposes. It is **not** a diagnostic tool and must not be used to assess anyone's mental health.

https://colab.research.google.com/drive/1svaVoaQeclYNA85YMSnUCHdRyOUQoObV?usp=sharing

## Problem Statement
Detect signs of depression in short text (social media posts and statements) and check how well a model trained on one source generalises to another.

## Datasets
| Dataset | Columns | Role |
|---|---|---|
| Dataset 1 (Sentiment Analysis for Mental Health) | `statement`, `status` | "Normal" rows used for training; non-Normal rows used for testing |
| Dataset 3 (Reddit depression posts) | `title`, `content`, `label` | All label 0 (depressed); used for training |

Dataset 3 has no "Normal" class, so Normal rows from Dataset 1 were added to the training set.

| Split | Size | Label 0 | Label 1 |
|---|---|---|---|
| Train | 27,892 | 11,549 | 16,343 |
| Test | 36,338 | 36,338 | 0 |

The datasets are **not included** in this repo. Download them from their original sources and place them in `data/`.

## Approach
1. Map Dataset 1 `status` to binary (Normal = 1, everything else = 0).
2. Combine Dataset 3 (label 0) with Dataset 1 Normal rows for training.
3. Tokenize with the DistilBERT tokenizer (max length 128).
4. Fine-tune with Hugging Face `Trainer` for 3 epochs (batch size 16, warmup 500 steps, weight decay 0.01).
5. Evaluate on held-out Dataset 1 rows and save the model.

## Results
| Metric | Value |
|---|---|
| Eval loss | 0.372 |
| Accuracy | 90.6% |
| Recall (Depressed / Abnormal) | 0.91 |

**Important:** the test set contains only label-0 samples, so accuracy here equals recall on the depressed class. Precision, F1 and performance on Normal text are not measured yet (see Limitations).

## Limitations and Future Work
- Add Normal samples to the test set (e.g. a stratified split of Dataset 1) to get real precision, F1 and a confusion matrix.
- Use a separate validation set for `load_best_model_at_end`, so the test set is not used to pick the checkpoint.
- Training data for the two classes comes from different sources, so the model may partly learn the source style rather than depression cues. Check with source-balanced data.
- Try class weights, other backbones (BERT, RoBERTa, MentalBERT) and multi-class classification.

## Tech Stack
Python, PyTorch, Hugging Face Transformers, scikit-learn, pandas, NumPy

## How to Run
```bash
git clone https://github.com/<your-username>/depression-text-detection-distilbert.git
cd depression-text-detection-distilbert
pip install -r requirements.txt
# place Dataset 1.csv and dataset3.csv in data/, update paths in the notebook
jupyter notebook notebooks/depression_detection_distilbert.ipynb
```
A GPU is recommended (Colab free GPU works).

