import subprocess
import sys

subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "--upgrade",

    "torch==2.11.0+cu126",
    "torchvision==0.26.0+cu126",
    "torchao==0.17.0",
    "unsloth==2026.11.7",
    "unsloth_zoo==2026.11.7",
    "datasets",
    "--extra-index-url", "https://download.pytorch.org/whl/cu126",
])

subprocess.check_call([
    sys.executable, "-m", "pip", "check",
])

print("Installation finished. Start a new Python process before testing imports.")