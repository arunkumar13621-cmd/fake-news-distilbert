@echo off
cd /d "%~dp0"
echo Installing packages...
pip install -r requirements.txt
echo.
echo === 1/3 MNIST (TensorFlow) ===
python 3_mnist_tf_serving.py train
echo.
if exist WELFake_Dataset.csv (
  echo === 2/3 LLM labeler ===
  if "%ANTHROPIC_API_KEY%"=="" set /p ANTHROPIC_API_KEY=Paste your API key here: 
  python 2_llm_labeler.py --csv WELFake_Dataset.csv --n 50 --real_label 0
  echo.
  echo === 3/3 DistilBERT ===
  python 1_finetune_distilbert.py --csv WELFake_Dataset.csv --n 3000 --epochs 2
) else (
  echo WELFake_Dataset.csv not found in this folder. Download it from Kaggle, put it here, and run this file again.
)
echo.
echo Done. Send me the numbers printed above, or the results_*.json files.
pause
