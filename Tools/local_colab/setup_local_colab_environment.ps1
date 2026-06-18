$ErrorActionPreference = 'Stop'

$PythonExe = 'C:\Users\hanna\anaconda3\envs\google-colab\python.exe'

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python environment not found: $PythonExe"
}

& $PythonExe -m pip install `
    'torch==2.5.1+cu121' `
    'torchvision==0.20.1+cu121' `
    --index-url 'https://download.pytorch.org/whl/cu121' `
    --retries 10 `
    --timeout 120

& $PythonExe -m pip install `
    'transformers==4.48.3' `
    'accelerate==1.3.0' `
    'bitsandbytes==0.49.2' `
    'peft==0.14.0' `
    'datasets==3.2.0' `
    'sentencepiece==0.2.1' `
    'pandas==2.2.3' `
    'numpy==1.26.4' `
    'sympy==1.13.1' `
    --retries 10 `
    --timeout 120

& $PythonExe -c @'
import torch
import transformers
import bitsandbytes
import peft

print("Environment is ready.")
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Transformers:", transformers.__version__)
print("bitsandbytes:", bitsandbytes.__version__)
print("PEFT:", peft.__version__)
'@
