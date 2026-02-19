# slm/src/sentence_model.py
# ─────────────────────────────────────────────────────────────────────────────
# Loads google/flan-t5-large for instruction-following sentence generation.
#
# Why flan-t5-large:
#   - Instruction-tuned by Google — understands "fill missing grammar words"
#   - ~780MB — best quality/size tradeoff for this task
#   - Fast on NVIDIA GPU with float16
#   - T5-small / T5-base produce nonsense for sign-language grammar filling
# ─────────────────────────────────────────────────────────────────────────────

import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_NAME = "google/flan-t5-large"
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models", "flan-t5-large")
MODEL_DIR  = os.path.normpath(MODEL_DIR)

_model     = None
_tokenizer = None


def _get_device() -> str:
    if torch.cuda.is_available():
        print(f"[sentence_model] GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("[sentence_model] No GPU found — using CPU.")
    return "cpu"


def load_model():
    """Load flan-t5-large once, cache for all subsequent calls."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    source = MODEL_DIR if os.path.isdir(MODEL_DIR) else MODEL_NAME
    print(f"[sentence_model] Loading flan-t5-large from:\n  {source}")

    _tokenizer = T5Tokenizer.from_pretrained(source)
    _model = T5ForConditionalGeneration.from_pretrained(
        source,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        _model = _model.to("cpu")

    _model.eval()
    print("[sentence_model] flan-t5-large ready.\n")
    return _model, _tokenizer


def generate(prompt: str, max_length: int = 128) -> str:
    """Run flan-t5-large inference on a prompt string."""
    model, tokenizer = load_model()
    device = next(model.parameters()).device

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
