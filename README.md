---
title: MiniCPM-V 4.6 Hackathon Demo
sdk: gradio
python_version: "3.12"
suggested_hardware: t4-small
app_file: app.py
---

# MiniCPM-V 4.6 Hackathon Demo

This repository contains a minimal Hugging Face Space for inference with
[`openbmb/MiniCPM-V-4.6`](https://huggingface.co/openbmb/MiniCPM-V-4.6).

The app is intentionally inference-only. It does not include fine-tuning,
dataset preparation, trainer scripts, adapters, checkpoints, or training
artifacts.

## Local Development

```bash
uv sync --python 3.12
uv run python app.py
```

The model ID can be overridden without changing code:

```bash
MODEL_ID=openbmb/MiniCPM-V-4.6 uv run python app.py
```

## Space Deployment

The Space reads dependencies from `requirements.txt` and starts `app.py` with
Gradio. Python 3.12 is requested in this README's Space metadata.
