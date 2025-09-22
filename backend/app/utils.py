# backend/app/utils.py
from .db import get_cursor
from .embeddings import embed_texts, format_vector_for_pg
import math

def retrieve_topk_by_embedding(query_embedding_vec, top_k=5):
    """Return top_k rows from summaries ordered by cosine distance (embedding <=> vec)."""
    vec_str = format_vector_for_pg(query_embedding_vec)
    sql = """
    SELECT id, source_table, source_id, summary_text, 1 - (embedding <=> %s::vector) AS score
    FROM summaries
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
    """
    with get_cursor() as cur:
        cur.execute(sql, (vec_str, vec_str, int(top_k)))
        rows = cur.fetchall()
    return rows
