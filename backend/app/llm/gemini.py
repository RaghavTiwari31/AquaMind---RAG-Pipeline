# backend/app/llm/gemini.py
from __future__ import annotations
import logging
from google import genai
from google.genai import types
from ..config import GOOGLE_API_KEY, GENERATION_MODEL, MAX_OUTPUT_TOKENS

_client: genai.Client | None = None
log = logging.getLogger(__name__)

FALLBACKS = [
    "gemini-2.0-flash-001",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GOOGLE_API_KEY)
    return _client

def _first_supported(model_candidates: list[str]) -> str:
    """Return the first candidate present in ListModels that supports generate_content."""
    try:
        models = list(get_client().models.list())  # iterable -> list for reuse
        ids = {m.name for m in models if getattr(m, "name", None)}
        for m in model_candidates:
            # SDK model.name is often 'models/<id>'
            if m in ids or f"models/{m}" in ids:
                return m
    except Exception as e:
        log.warning("ListModels failed; proceeding with hardcoded fallbacks: %s", e)
    # If listing failed, just return the first candidate
    return model_candidates[0]

def _pick_model(initial: str) -> str:
    # Try initial first; if it 404s we’ll catch and retry in generate_text
    return initial or FALLBACKS[0]

def generate_text(contents: str, temperature: float = 0.0, max_tokens: int | None = None) -> str:
    model_id = _pick_model(GENERATION_MODEL)
    cfg = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens or MAX_OUTPUT_TOKENS,
    )
    try:
        resp = get_client().models.generate_content(
            model=model_id,
            contents=contents,
            config=cfg,
        )
        return (resp.text or "").strip()
    except Exception as e:
        # If “not found / not supported”, try fallbacks discovered from ListModels
        msg = str(e).lower()
        if "not found" in msg or "is not supported" in msg:
            candidates = [m for m in [model_id, *FALLBACKS] if m is not None]
            chosen = _first_supported(candidates)
            if chosen != model_id:
                log.warning("Model '%s' failed; retrying with '%s'", model_id, chosen)
                resp = get_client().models.generate_content(
                    model=chosen,
                    contents=contents,
                    config=cfg,
                )
                return (resp.text or "").strip()
        # Otherwise, bubble up
        raise
