"""Fine-tune DistilBERT on the WELFake fake-news dataset and compare with TF-IDF + Logistic Regression.

Run:  python 1_finetune_distilbert.py --csv WELFake_Dataset.csv --n 12000 --epochs 2
Outputs: results_distilbert.json (real metrics from YOUR run).
"""
import argparse, json, re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

MARKERS = ["reuters", "new york times", "breitbart", "cnn", "fox news"]  # source-marker leakage

def clean(t):
    t = str(t).lower()
    for m in MARKERS:
        t = t.replace(m, " ")
    return re.sub(r"\s+", " ", t).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n", type=int, default=12000, help="rows to sample (keeps training fast on a laptop)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=256)
    a = ap.parse_args()

    df = pd.read_csv(a.csv).dropna(subset=["title", "text"]).drop_duplicates(subset=["title", "text"])
    df = df.sample(n=min(a.n, len(df)), random_state=42)
    df["content"] = (df["title"] + ". " + df["text"]).map(clean)
    X_tr, X_te, y_tr, y_te = train_test_split(df["content"], df["label"], test_size=0.2, stratify=df["label"], random_state=42)

    # Baseline: TF-IDF + logistic regression
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    lr = LogisticRegression(max_iter=1000).fit(vec.fit_transform(X_tr), y_tr)
    pred = lr.predict(vec.transform(X_te))
    res = {"baseline_tfidf_lr": {"accuracy": accuracy_score(y_te, pred), "f1": f1_score(y_te, pred)}}
    print("Baseline:", res["baseline_tfidf_lr"])

    # DistilBERT fine-tuning with Hugging Face
    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer)
    name = "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(name)
    def enc(texts, labels):
        ds = Dataset.from_dict({"text": list(texts), "label": list(map(int, labels))})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=a.max_len, padding="max_length"), batched=True)
    train_ds, test_ds = enc(X_tr, y_tr), enc(X_te, y_te)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)
    args = TrainingArguments(output_dir="out_distilbert", num_train_epochs=a.epochs, per_device_train_batch_size=16,
                             per_device_eval_batch_size=32, learning_rate=2e-5, eval_strategy="epoch",
                             save_strategy="no", logging_steps=50, fp16=torch.cuda.is_available(), report_to="none")
    def metrics(p):
        preds = p.predictions.argmax(-1)
        return {"accuracy": accuracy_score(p.label_ids, preds), "f1": f1_score(p.label_ids, preds)}
    tr = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds, compute_metrics=metrics)
    tr.train()
    out = tr.evaluate()
    res["distilbert"] = {"accuracy": out["eval_accuracy"], "f1": out["eval_f1"]}
    res["setup"] = {"rows_sampled": len(df), "epochs": a.epochs, "max_len": a.max_len}
    json.dump(res, open("results_distilbert.json", "w"), indent=2)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
