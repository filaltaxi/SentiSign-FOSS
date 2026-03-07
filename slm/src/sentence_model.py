# slm/src/sentence_model.py
# ─────────────────────────────────────────────────────────────────────────────
# Sentence generation backend.
#
# Default path:
#   - Ollama + qwen3.5:0.8b over the local HTTP API
#
# Fallback path:
#   - google/flan-t5-large via Transformers
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

try:
    from dotenv import load_dotenv

    _ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    load_dotenv(os.path.join(_ROOT, ".env"))
except Exception:
    pass

DEFAULT_SENTENCE_PROVIDER = "ollama"
SUPPORTED_SENTENCE_PROVIDERS = {"hf", "ollama"}

OLLAMA_MODEL_NAME = "qwen3.5:0.8b"
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_KEEP_ALIVE = "10m"

HF_MODEL_NAME = "google/flan-t5-large"
HF_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "flan-t5-large")
HF_MODEL_DIR = os.path.normpath(HF_MODEL_DIR)

_model = None
_tokenizer = None
_ollama_ready = False


def resolve_provider(provider: str | None = None) -> str:
    raw = provider or os.environ.get("SENTISIGN_SENTENCE_PROVIDER", DEFAULT_SENTENCE_PROVIDER)
    key = (raw or DEFAULT_SENTENCE_PROVIDER).strip().lower()
    if key not in SUPPORTED_SENTENCE_PROVIDERS:
        raise ValueError(
            f"[sentence_model] Unsupported provider '{raw}'. "
            f"Use one of: {', '.join(sorted(SUPPORTED_SENTENCE_PROVIDERS))}."
        )
    return key


def resolve_ollama_model() -> str:
    raw = os.environ.get("SENTISIGN_OLLAMA_MODEL", OLLAMA_MODEL_NAME)
    return (raw or OLLAMA_MODEL_NAME).strip() or OLLAMA_MODEL_NAME


def resolve_ollama_host() -> str:
    raw = os.environ.get("SENTISIGN_OLLAMA_HOST", OLLAMA_HOST)
    return (raw or OLLAMA_HOST).strip().rstrip("/") or OLLAMA_HOST


def resolve_hf_model_source() -> str:
    return HF_MODEL_DIR if os.path.isdir(HF_MODEL_DIR) else HF_MODEL_NAME


def get_backend_label(provider: str | None = None) -> str:
    key = resolve_provider(provider)
    if key == "ollama":
        return f"Ollama ({resolve_ollama_model()})"
    return "flan-t5-large"


def _get_device() -> str:
    if torch.cuda.is_available():
        print(f"[sentence_model] GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("[sentence_model] No GPU found — using CPU.")
    return "cpu"


def _ollama_request(method: str, path: str, payload: dict | None = None) -> dict:
    host = resolve_ollama_host()
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{host}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"[sentence_model] Ollama request failed ({e.code}) for {path}: {detail or e.reason}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"[sentence_model] Could not reach Ollama at {host}. "
            f"Make sure `ollama serve` is running."
        ) from e

    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"[sentence_model] Ollama returned invalid JSON for {path}: {raw[:200]}"
        ) from e


def _load_ollama():
    global _ollama_ready
    if _ollama_ready:
        return {"provider": "ollama", "model": resolve_ollama_model()}

    version = _ollama_request("GET", "/api/version")
    model_name = resolve_ollama_model()
    tags = _ollama_request("GET", "/api/tags")
    installed = {item.get("name") for item in tags.get("models", []) if item.get("name")}
    if model_name not in installed:
        available = ", ".join(sorted(installed)) if installed else "none"
        raise RuntimeError(
            f"[sentence_model] Ollama model '{model_name}' is not installed. "
            f"Available models: {available}. Run `ollama pull {model_name}` if needed."
        )

    _ollama_ready = True
    print(
        "[sentence_model] Ollama ready.\n"
        f"  Host : {resolve_ollama_host()}\n"
        f"  Model: {model_name}\n"
        f"  API  : {version.get('version', 'unknown')}\n"
    )
    return {"provider": "ollama", "model": model_name}


def _load_hf():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    source = resolve_hf_model_source()
    print(f"[sentence_model] Loading flan-t5-large from:\n  {source}")

    # `legacy=True` keeps the existing tokenization behavior and suppresses the
    # Transformers warning about the default legacy behavior.
    _tokenizer = T5Tokenizer.from_pretrained(source, legacy=True)
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


def load_model(provider: str | None = None):
    """Load the configured sentence backend once and cache it."""
    key = resolve_provider(provider)
    if key == "ollama":
        return _load_ollama()
    return _load_hf()


def _generate_with_ollama(prompt: str, max_length: int, system_prompt: str | None = None) -> str:
    load_model("ollama")
    payload = {
        "model": resolve_ollama_model(),
        "prompt": prompt,
        "system": system_prompt or "",
        "stream": False,
        "think": False,
        "stop": ["\n", "Signs:", "English:"],
        "keep_alive": os.environ.get("SENTISIGN_OLLAMA_KEEP_ALIVE", OLLAMA_KEEP_ALIVE),
        "options": {
            "temperature": 0,
            "top_p": 0.8,
            "repeat_penalty": 1.1,
            "num_predict": max_length,
        },
    }
    response = _ollama_request("POST", "/api/generate", payload)
    text = (response.get("response") or "").strip()
    if not text:
        raise RuntimeError("[sentence_model] Ollama returned an empty response.")
    return text


def _generate_with_hf(prompt: str, max_length: int, system_prompt: str | None = None) -> str:
    model, tokenizer = load_model("hf")
    device = next(model.parameters()).device

    full_prompt = prompt
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{prompt}"

    inputs = tokenizer(
        full_prompt,
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


def generate(prompt: str, max_length: int = 128, system_prompt: str | None = None) -> str:
    """Run sentence generation using the configured backend."""
    key = resolve_provider()
    if key == "ollama":
        return _generate_with_ollama(prompt, max_length=max_length, system_prompt=system_prompt)
    return _generate_with_hf(prompt, max_length=max_length, system_prompt=system_prompt)
