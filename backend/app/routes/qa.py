# backend/app/routes/qa.py
from __future__ import annotations

import logging
import re
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..rag.retrieval import retrieve_topk
from ..rag.context import build_truncated_context, get_schema_text
from ..rag.sql_guard import is_safe_select, validate_sql_syntax
from ..llm.gemini import generate_text
from ..db import get_cursor
from ..config import MAX_OUTPUT_TOKENS

logger = logging.getLogger(__name__)
router = APIRouter(prefix="")

# ---------- Request model ----------
class QueryPayload(BaseModel):
    question: str
    top_k: int = 6
    year: Optional[int] = None
    session_id: Optional[str] = None

# ---------- Helpers ----------
# _YEAR_RE = re.compile(r"\b2025\b")
# _YEAR_RANGE_RE = re.compile(r"\b2025\b", re.IGNORECASE)

DATA_DICTIONARY = """
Columns used across argo_details_YYYY tables:
- latitude (float8): Latitude in decimal degrees.
- longitude (float8): Longitude in decimal degrees.
- time(timestamp without time zone): Time and Date of data measurement.
- pressure (float8): pressure in atm.
- temperature (float8): seawater temperature in °C.
- salinity (float8): practical salinity units (PSU).
"""

# def _years_from_text(text: str) -> List[int]:
#     years = [int(y) for y in _YEAR_RE.findall(text or "")]
#     m = _YEAR_RANGE_RE.search(text or "")
#     if m:
#         a, b = int(m.group(1)), int(m.group(2))
#         lo, hi = (a, b) if a <= b else (b, a)
#         years = list(range(lo, hi + 1))
#     years = [y for y in years if 2001 <= y <= 2017]
#     return sorted(set(years))
_YEAR_RE = re.compile(r"\b2025\b")
_YEAR_RANGE_RE = re.compile(r"(2025)\s*(?:to|-)\s*(2025)", re.IGNORECASE)

def _years_from_text(text: str) -> List[int]:
    years = [int(y) for y in _YEAR_RE.findall(text or "")]
    m = _YEAR_RANGE_RE.search(text or "")
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        lo, hi = (a, b) if a <= b else (b, a)
        years = list(range(lo, hi + 1))
    return sorted(set(years))


def _union_all_subquery(col_expr: str, years: List[int]) -> str:
    if not years: return ""
    parts = [f"SELECT {col_expr} FROM argo_details_{y}" for y in years]
    return "(\n  " + "\n  UNION ALL ".join(parts) + "\n) t"

def _dynamic_examples(years: List[int]) -> str:
    # One-shot example based on detected years (single or range)
    if len(years) == 1:
        y = years[0]
        return f"""
Q: average salinity in {y}
SQL:
SELECT AVG(salinity) AS avg_salinity
FROM argo_details_{y}
"""
    if len(years) >= 2:
        sub = _union_all_subquery("salinity", years[:5])  # cap example length
        return f"""
Q: average salinity from {years[0]} to {years[-1]}
SQL:
SELECT AVG(salinity) AS avg_salinity
FROM {sub}
"""
    # generic single-year + range examples if nothing detected
    return """
Q: average salinity in 2005
SQL:
SELECT AVG(salinity) AS avg_salinity FROM argo_details_2005

Q: average salinity from 2001 to 2005
SQL:
SELECT AVG(salinity) AS avg_salinity
FROM (
  SELECT salinity FROM argo_details_2001
  UNION ALL SELECT salinity FROM argo_details_2002
  UNION ALL SELECT salinity FROM argo_details_2003
  UNION ALL SELECT salinity FROM argo_details_2004
  UNION ALL SELECT salinity FROM argo_details_2005
) t
"""

_CODE_FENCE = re.compile(r"```(?:sql)?\s*|\s*```", re.IGNORECASE)

def _extract_sql(text: str) -> str:
    """
    Robustly extract a SELECT/WITH query from model output.
    - strips ```sql fences and 'SQL:' prefixes
    - returns the first SELECT/WITH block
    """
    if not text: return ""
    s = _CODE_FENCE.sub("", text).strip()
    # kill leading 'SQL:' label lines
    s = re.sub(r"^\s*SQL\s*:\s*", "", s, flags=re.IGNORECASE)
    # find first select/with
    m = re.search(r"(?is)(with\b[\s\S]*?select\b|select\b)[\s\S]*", s)
    if not m: return ""
    sql = s[m.start():].strip()
    # drop trailing extra paragraphs
    sql = sql.split("\n\n")[0].strip()
    # forbid stray trailing semicolon (optional)
    if sql.endswith(";"):
        sql = sql[:-1].strip()
    return sql

# ---------- Route ----------
@router.post("/query")
async def query(req: QueryPayload, request: Request):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # 1) Retrieve context (pgvector) + schema
    try:
        retrieved = retrieve_topk(req.question, top_k=req.top_k, year=req.year)
    except Exception:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail="Retrieval failed")

    try:
        context_text = build_truncated_context(retrieved)
        schema_text = get_schema_text()
    except Exception:
        logger.exception("Context/schema assembly failed")
        raise HTTPException(status_code=500, detail="Context preparation failed")

    mentioned_years = _years_from_text(req.question)
    examples = _dynamic_examples(mentioned_years)

    # 2) Primary SQL prompt (strict)
    prompt = f"""
You are an assistant that transforms a natural-language question about the ARGO_D database
into a single READ-ONLY SQL SELECT query (or a short natural-language answer when a SELECT is inappropriate).

Hard requirements:
- Return ONLY ONE SQL SELECT statement when possible (no prose).
- NO INSERT/UPDATE/DELETE/CREATE/DROP/ALTER/TRUNCATE.
- Use these tables: argo_details_YYYY where YYYY is the year.
- If a YEAR RANGE is mentioned (e.g., 2001..2005 or "2001 to 2005"), UNION ALL across those year tables, then aggregate.
- Use columns from the data dictionary; ignore *_flag columns.
- You MAY generate SQL using the schema and mapping even if the retrieval context is empty.
- For a location specific query, produce results according to the latitude/longitude columns in the schema.
- Interpret the lat/long columns for location-based queries.
Mapping rule:
- Year Y ⇒ table argo_details_Y

Data dictionary:
{DATA_DICTIONARY}

Examples:
{examples}

Database schema (truncated):
{schema_text}

Context (most relevant summaries first):
{context_text}

User question:
{req.question}

Years detected from question (if any): {mentioned_years or "none"}
"""

    try:
        out = generate_text(prompt, temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS)
    except Exception:
        logger.exception("Gemini generate_text failed")
        raise HTTPException(status_code=500, detail="Text generation failed")

    sql_candidate = _extract_sql(out)
    if sql_candidate and is_safe_select(sql_candidate):
        valid, msg = validate_sql_syntax(sql_candidate)
        if not valid:
            logger.warning("SQL failed validation: %s | SQL: %s", msg, sql_candidate)
            sql_candidate = None
    else:
        sql_candidate = None

    # 3) If no SQL, try a minimal, SQL-only retry
    if not sql_candidate:
        retry_prompt = f"""
Return ONLY a single SQL SELECT statement that answers the question.
Rules:
- Use argo_details_YYYY tables (YYYY = year).
- If a year RANGE is asked, UNION ALL those year tables then aggregate.
- Use columns: temperature, salinity, depth, density (ignore *_flag).
- Do not include explanations or code fences.

Question: {req.question}
Detected years: {mentioned_years or "none"}
"""
        try:
            out2 = generate_text(retry_prompt, temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS)
            sql_candidate = _extract_sql(out2)
            if sql_candidate and is_safe_select(sql_candidate):
                valid, msg = validate_sql_syntax(sql_candidate)
                if not valid:
                    logger.warning("Retry SQL failed validation: %s | SQL: %s", msg, sql_candidate)
                    sql_candidate = None
        except Exception:
            logger.exception("Gemini retry failed")
            sql_candidate = None

    # 4) Execute SQL if we have it; else fallback to text
    if sql_candidate:
        try:
            with get_cursor() as cur:
                cur.execute(sql_candidate)
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description] if cur.description else []
            return {
                "type": "sql",
                "sql": sql_candidate,
                "results": rows,
                "columns": cols,
                "retrieved_context": retrieved,
            }
        except Exception:
            logger.exception("SQL execution failed")
            # fall through to text answer

    # 5) Text fallback (still schema-aware)
    answer_prompt = f"""
Using only the context and database schema below, answer the question succinctly.
If you cannot answer from the context or schema, say so.

Guidance:
- Year Y ⇒ table argo_details_Y
- Year ranges (a..b) ⇒ UNION ALL across those tables.
- Use columns: temperature, salinity, depth, density. Ignore *_flag columns.

Data dictionary:
{DATA_DICTIONARY}

Database schema (truncated):
{schema_text}

Context:
{context_text}

Question:
{req.question}

Years detected from question (if any): {mentioned_years or "none"}

Answer:
"""
    try:
        ans = generate_text(answer_prompt, temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS).strip()
    except Exception:
        logger.exception("Gemini fallback answer failed")
        raise HTTPException(status_code=500, detail="Text generation failed")

    return {
        "type": "text",
        "answer": ans or "I cannot answer from the context or schema.",
        "retrieved_context": retrieved,
        "generated_raw": out,
    }
