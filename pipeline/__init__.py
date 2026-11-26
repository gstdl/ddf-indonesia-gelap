import os
from pathlib import Path

# Hugging Face model files stay inside the repository's disposable tmp/ folder.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "tmp" / "huggingface"))
