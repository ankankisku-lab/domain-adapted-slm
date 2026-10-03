"""Keep every local cache on the D: drive (project folder), never under C:\\Users\\...\\.cache.

Import this module before any Hugging Face / torch / sentence-transformers import in local scripts.
On Colab the variables are left alone (the runtime is ephemeral and has its own disk).
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / ".cache"

if "COLAB_RELEASE_TAG" not in os.environ:
    for var, sub in {
        "HF_HOME": "huggingface",
        "TORCH_HOME": "torch",
        "SENTENCE_TRANSFORMERS_HOME": "sentence_transformers",
        "PIP_CACHE_DIR": "pip",
    }.items():
        os.environ.setdefault(var, str(CACHE_DIR / sub))
