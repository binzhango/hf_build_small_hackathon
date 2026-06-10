from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi


SPACE_ID = os.getenv("HF_SPACE_ID", "binzhango/hf-build-small-hackathon")
HF_TOKEN = os.getenv("HF_TOKEN")
ROOT = Path(__file__).resolve().parents[1]

IGNORE_PATTERNS = [
    ".git/*",
    ".github/*",
    ".venv/*",
    "__pycache__/*",
    "*.pyc",
    ".pytest_cache/*",
    ".ruff_cache/*",
    ".mypy_cache/*",
    "uv.lock",
]


def main() -> None:
    if not HF_TOKEN:
        raise SystemExit("HF_TOKEN is required to deploy to Hugging Face Spaces.")

    api = HfApi(token=HF_TOKEN)
    api.create_repo(
        repo_id=SPACE_ID,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=ROOT,
        repo_id=SPACE_ID,
        repo_type="space",
        token=HF_TOKEN,
        commit_message="Deploy from GitHub Actions",
        ignore_patterns=IGNORE_PATTERNS,
    )


if __name__ == "__main__":
    main()
