"""Ask an LLM (via API) to label news articles real/fake and compare with ground truth.

Setup:  export ANTHROPIC_API_KEY=your_key      (Windows PowerShell: $env:ANTHROPIC_API_KEY="your_key")
Run:    python 2_llm_labeler.py --csv WELFake_Dataset.csv --n 100 --model claude-haiku-5-5
Outputs: results_llm.json and llm_predictions.csv (real numbers from YOUR run).
Note: check which label value means "real" in YOUR copy of the dataset (use --real_label).
"""
import argparse, json, os, re, time
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import anthropic

PROMPT = """You are checking news articles. Read the article and decide if it is REAL news or FAKE news.
Reply with only a JSON object: {{"label": "real" or "fake", "reason": "one short sentence"}}

Title: {title}

Article: {text}"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--real_label", type=int, default=0, help="which label value means REAL in your dataset copy")
    a = ap.parse_args()

    df = pd.read_csv(a.csv).dropna(subset=["title", "text"]).sample(n=a.n, random_state=1)
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    rows = []
    for _, r in df.iterrows():
        msg = client.messages.create(
            model=a.model, max_tokens=150,
            messages=[{"role": "user", "content": PROMPT.format(title=r["title"], text=str(r["text"])[:3000])}])
        raw = msg.content[0].text
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            obj = json.loads(m.group(0)); pred = obj["label"].strip().lower()
        except Exception:
            obj, pred = {"reason": "parse error"}, "unparsed"
        truth = "real" if int(r["label"]) == a.real_label else "fake"
        rows.append({"title": r["title"], "truth": truth, "llm": pred, "reason": obj.get("reason", "")})
        time.sleep(0.2)
    out = pd.DataFrame(rows); out.to_csv("llm_predictions.csv", index=False)
    ok = out[out["llm"].isin(["real", "fake"])]
    res = {"model": a.model, "articles": len(out), "unparsed": int((out["llm"] == "unparsed").sum()),
           "accuracy": accuracy_score(ok["truth"], ok["llm"]),
           "f1_fake": f1_score(ok["truth"], ok["llm"], pos_label="fake")}
    json.dump(res, open("results_llm.json", "w"), indent=2)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
