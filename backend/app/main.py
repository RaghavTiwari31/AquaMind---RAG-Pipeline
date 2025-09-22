# backend/app/main.py
import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from .embeddings import embed_texts, format_vector_for_pg, EMBEDDING_DIM
from .utils import retrieve_topk_by_embedding
from .sanitizer import is_safe_select, extract_first_select
from .db import get_cursor
from transformers import pipeline
import numpy as np
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
GEN_MODEL = os.environ.get("GENERATION_MODEL", "google/flan-t5-small")

app = FastAPI(title="ARGO_RAG", version="0.1")

# Load the generation pipeline (seq2seq for flan-t5)
_gen_pipe = None
def get_gen_pipe():
    global _gen_pipe
    if _gen_pipe is None:
        # Use text2text-generation for FLAN-T5 models
        _gen_pipe = pipeline("text2text-generation", model=GEN_MODEL)
    return _gen_pipe

executor = ThreadPoolExecutor(max_workers=2)

class QueryPayload(BaseModel):
    question: str
    top_k: int = 5

@app.post("/query")
def query(req: QueryPayload):
    question = req.question
    top_k = int(req.top_k)
    # 1) embed the question
    q_vec = embed_texts([question])[0]  # numpy array
    # 2) retrieve top-k summaries
    retrieved = retrieve_topk_by_embedding(q_vec, top_k=top_k)
    context_text = "\n\n".join([f"TABLE:{r['source_table']} | ID:{r['source_id']} | {r['summary_text']}" for r in retrieved])

    # 3) build prompt instructing LLM to respond with a READ-ONLY SELECT if possible
    prompt = f"""
You are an assistant that transforms a natural-language question about the ARGO_D database into a single READ-ONLY SQL SELECT query (or a short natural-language answer when a SELECT is inappropriate).
Rules:
- Only output a single SQL SELECT statement when possible. Do NOT output INSERT/UPDATE/DELETE/CREATE/DROP/ALTER/TRUNCATE.
- Use only tables/column names you see in the provided context below. If the information required is not present, reply with a short answer (no SQL) describing that you cannot run a SQL query.
- Use explicit column names if present in the context. Otherwise select source_table, source_id, summary_text (safe fallback).
- Output only the SQL (no explanation) when returning a SQL query. If returning text, make it brief.

Context (most relevant summaries first):
{context_text}

User question:
{question}
"""

    # 4) ask the generator to produce SQL (or answer)
    gen = get_gen_pipe()
    out = gen(prompt, max_length=256, do_sample=False)[0]["generated_text"].strip()

    # 5) try to extract SELECT and sanitize
    sql_candidate = extract_first_select(out)
    safe = False
    if sql_candidate:
        safe = is_safe_select(sql_candidate)
    if safe:
        # Execute the safe SQL and return rows (limit enforced)
        with get_cursor() as cur:
            try:
                cur.execute(sql_candidate)
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description] if cur.description else []
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"SQL execution error: {e}")
        return {
            "type": "sql",
            "sql": sql_candidate,
            "results": rows,
            "columns": cols,
            "retrieved_context": retrieved
        }
    else:
        # Not safe or not SQL: return LLM's natural language answer (ask LLM explicitly for answer)
        # Build prompt to answer directly using context
        answer_prompt = f"""
Using only the context below, answer the question succinctly. If you cannot answer from the context, say you cannot answer.

Context:
{context_text}

Question:
{question}

Answer:
"""
        ans = gen(answer_prompt, max_length=256, do_sample=False)[0]["generated_text"].strip()
        return {
            "type": "text",
            "answer": ans,
            "retrieved_context": retrieved,
            "generated_raw": out
        }
