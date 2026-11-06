HOW TO RUN (on your laptop; I could not run these here because this workspace cannot download models or datasets)

0. pip install -r requirements.txt
   Get the WELFake CSV (Kaggle: "WELFake dataset") and put WELFake_Dataset.csv in this folder.
   Check which label means REAL in your copy (your team found 0 = real, 1 = fake).

1. DistilBERT:  python 1_finetune_distilbert.py --csv WELFake_Dataset.csv --n 12000 --epochs 2
   (use --n 3000 if it is slow). Creates results_distilbert.json.

2. LLM labeler: set ANTHROPIC_API_KEY, then  python 2_llm_labeler.py --csv WELFake_Dataset.csv --n 100 --real_label 0
   (check the model name is valid for your account). Creates results_llm.json.

3. TF-Serving: follow the steps at the top of 3_mnist_tf_serving.py (needs Docker running).

AFTER YOU RUN THEM: put each project on GitHub, then tell me the real numbers and I will add the bullets.
Only describe what you actually ran and measured.
