# slm/download_model.py
# ─────────────────────────────────────────────────────────────────────────────
# Downloads google/flan-t5-large to slm/models/flan-t5-large/
# Run ONCE before using the pipeline.
#
# Usage:
#   python slm/download_model.py
# ─────────────────────────────────────────────────────────────────────────────

import os
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_NAME = "google/flan-t5-large"
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models", "flan-t5-large")


def download():
    if os.path.isdir(MODEL_DIR) and len(os.listdir(MODEL_DIR)) > 3:
        print(f"[download] flan-t5-large already present at:\n  {MODEL_DIR}")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"[download] Downloading {MODEL_NAME} (~780MB) ...")
    print("[download] This takes a few minutes on first run.\n")

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)

    print(f"\n[download] Done. Saved to:\n  {MODEL_DIR}")


if __name__ == "__main__":
    download()
