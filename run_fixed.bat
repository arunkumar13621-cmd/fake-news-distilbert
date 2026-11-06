@echo off
cd /d "%~dp0"
set PY=py -3.12
echo Using:
%PY% -c "import sys; print(sys.version); print(sys.executable)"
echo.
echo Installing packages into Python 3.12 (takes several minutes)...
%PY% -m pip install --upgrade pip
%PY% -m pip install -r requirements.txt
echo.
echo === 1/3 MNIST (TensorFlow) ===
%PY% 3_mnist_tf_serving.py train
echo.
if not exist WELFake_Dataset.csv (
  echo WELFake_Dataset.csv not found in this folder.
  goto end
)
echo === 2/3 LLM labeler ===
if "%ANTHROPIC_API_KEY%"=="" set /p ANTHROPIC_API_KEY=Paste your API key here, or just press Enter to skip: 
if "%ANTHROPIC_API_KEY%"=="" (
  echo No key, skipping the LLM step.
) else (
  %PY% 2_llm_labeler.py --csv WELFake_Dataset.csv --n 50 --real_label 0
)
echo.
echo === 3/3 DistilBERT ===
%PY% 1_finetune_distilbert.py --csv WELFake_Dataset.csv --n 3000 --epochs 2
:end
echo.
echo Done. Copy the numbers printed above and send them to me.
pause
