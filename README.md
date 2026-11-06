# Fake News Classification: DistilBERT vs a Simple Baseline

I wanted to see how much a fine-tuned language model improves on a classic text classifier for spotting fake news. This repo has the code for that, plus two smaller side projects.

## Results (3,000-article sample of WELFake, 2 epochs, max length 256)

| Model | Accuracy | F1 |
|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 88.2% | 0.868 |
| Fine-tuned DistilBERT | 97.8% | 0.976 |

Source names such as Reuters or CNN are removed from the text first, so the model learns from the writing itself and not from the outlet name. This was a CPU run on a small sample, so treat the numbers as a student experiment, not a benchmark.

## What is in here

- `1_finetune_distilbert.py`: cleans the data, trains the baseline, fine-tunes DistilBERT with Hugging Face Transformers, and writes the scores to `results_distilbert.json`.
- `2_llm_labeler.py`: sends articles to an LLM through the Anthropic API and scores its real/fake labels against the true ones (in progress, no results yet).
- `3_mnist_tf_serving.py`: trains a small TensorFlow digit classifier and serves it with TensorFlow Serving.

## How to run

1. Download `WELFake_Dataset.csv` from Kaggle and put it in this folder (it is too big for GitHub).
2. `pip install -r requirements.txt`
3. `python 1_finetune_distilbert.py --csv WELFake_Dataset.csv --n 3000 --epochs 2`

For the LLM script, set the `ANTHROPIC_API_KEY` environment variable first. Never put a key in the code.

Dataset: WELFake (Verma et al.), about 72,000 news articles labeled real or fake.
