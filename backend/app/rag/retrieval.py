# =============================
# backend/app/rag/retrieval.py
# =============================
from __future__ import annotations
from typing import Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from ..db import get_cursor

_model: SentenceTransformer | None = None


def _model_inst() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    m = _model_inst()
    vecs = m.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


def retrieve_topk(question: str, top_k: int = 5, year: Optional[int] = None) -> list[dict]:
    vec = embed_texts([question])[0]
    params = {"q": list(map(float, vec)), "k": max(1, int(top_k))}
    sql = (
        "SELECT id, source_table, source_id, year, item_type, summary_text, metadata,\n"
        "       1 - (embedding <=> %(q)s::vector) AS score\n"
        "FROM argo_embeddings\n{where}\n"
        "ORDER BY embedding <=> %(q)s::vector\nLIMIT %(k)s;"
    ).format(where=("WHERE year = %(yr)s" if year else ""))
    with get_cursor() as cur:
        if year:
            params["yr"] = int(year)
        cur.execute(sql, params)
        rows = cur.fetchall()
        return rows