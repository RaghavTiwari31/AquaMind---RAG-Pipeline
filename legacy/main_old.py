# backend/app/main.py
import os
import json
import logging
import traceback
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from .embeddings import embed_texts
from .utils import retrieve_topk_by_embedding
from .sanitizer import is_safe_select, extract_first_select
from .db import get_cursor
from concurrent.futures import ThreadPoolExecutor

# NEW imports for Google Gen AI
from google import genai
from google.genai import types

load_dotenv()

# Read generation config from env
GEN_MODEL = os.environ.get("GENERATION_MODEL", "gemini-2.0-flash-001")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", 8000))
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", 6000))

if not GOOGLE_API_KEY:
    # fail fast so the developer notices
    raise RuntimeError("Please set GOOGLE_API_KEY (or GEMINI_API_KEY) in .env")

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="ARGO_RAG", version="0.1")

# ---- GenAI client ----
_gen_client = None

def get_gen_client():
    """Lazy init of Google GenAI client (gemini)."""
    global _gen_client
    if _gen_client is None:
        _gen_client = genai.Client(api_key=GOOGLE_API_KEY)
    return _gen_client

# ---- Utility: build/safely truncate context ----
def build_truncated_context(retrieved, max_chars=MAX_PROMPT_CHARS, per_summary_cap=800):
    """
    Build a compact context text from retrieved rows; enforce an overall character budget.
    Each summary is truncated to `per_summary_cap` characters.
    """
    pieces = []
    total = 0
    for r in retrieved:
        s = f"TABLE:{r['source_table']} | ID:{r['source_id']} | {r['summary_text']}"
        if len(s) > per_summary_cap:
            s = s[:per_summary_cap-3] + "..."
        if total + len(s) > max_chars:
            break
        pieces.append(s)
        total += len(s)
    return "\n\n".join(pieces)

# (Keep your existing validate_sql_syntax function)
def validate_sql_syntax(sql: str) -> tuple[bool, str]:
    sql_upper = sql.upper().strip()
    if not sql_upper.startswith('SELECT'):
        return False, "Only SELECT statements are allowed"
    if ' FROM ' not in sql_upper:
        return False, "SQL must include a FROM clause"
    dangerous_ops = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE']
    for op in dangerous_ops:
        if op in sql_upper:
            return False, f"Operation {op} is not allowed"
    import re
    alias_pattern = r'\b[T]\d+\.'
    if re.search(alias_pattern, sql.upper()):
        from_match = re.search(r'FROM\s+(\w+)(?:\s+(?:AS\s+)?([T]\d+))?', sql.upper())
        if not from_match:
            return False, "Table aliases found but FROM clause is missing or malformed"
        if from_match.group(2) is None:
            return False, "Table alias used but not properly defined in FROM clause"
    return True, "Valid"

executor = ThreadPoolExecutor(max_workers=2)

class QueryPayload(BaseModel):
    question: str
    top_k: int = 5

@app.post("/query")
async def query(req: QueryPayload, request: Request):
    try:
        body = await request.body()
        logger.debug(f"Received request body: {body.decode()}")
        logger.debug(f"Parsed payload: question='{req.question}', top_k={req.top_k}")

        question = req.question
        top_k = int(req.top_k)

        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # 1) embed the question
        try:
            q_vec = embed_texts([question])[0]
            logger.debug(f"Successfully embedded question, vector shape: {q_vec.shape}")
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to embed question: {str(e)}")

        # 2) retrieve top-k summaries
        try:
            retrieved = retrieve_topk_by_embedding(q_vec, top_k=top_k)
            logger.debug(f"Retrieved {len(retrieved)} contexts")
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve context: {str(e)}")

        # 3) build a bounded context (avoid long prompts)
        context_text = build_truncated_context(retrieved, max_chars=MAX_PROMPT_CHARS, per_summary_cap=800)
        logger.debug(f"Context length (chars): {len(context_text)}")

        # 4) construct the SQL-generation prompt (same rules as before)
        prompt = f"""
You are an assistant that transforms a natural-language question about the ARGO_D database
into a single READ-ONLY SQL SELECT query (or a short natural-language answer when a SELECT is inappropriate).

Rules:
- Only output a single SQL SELECT statement when possible.
- Do NOT output INSERT/UPDATE/DELETE/CREATE/DROP/ALTER/TRUNCATE.
- Use only tables/column names you see in the provided context below.
- If the info is not present, reply briefly in natural language (no SQL).
- Use explicit column names if present in the context.
- Otherwise select source_table, source_id, summary_text as a fallback.
- Output only the SQL (no explanation) when returning a SQL query.

Context (most relevant summaries first):
{context_text}

User question:
{question}
"""

        # 5) call Gemini / GenAI to generate SQL
        try:
            client = get_gen_client()
            cfg = types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=MAX_OUTPUT_TOKENS
            )
            response = client.models.generate_content(
                model=GEN_MODEL,
                contents=prompt,
                config=cfg
            )
            out = response.text.strip()
            logger.debug(f"Generated response: {out[:1000]}")  # avoid logging huge text
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")

        # 6) extraction & sanitization
        sql_candidate = extract_first_select(out)
        safe = False
        validation_error = ""

        if sql_candidate:
            safe = is_safe_select(sql_candidate)
            if safe:
                is_valid, error_msg = validate_sql_syntax(sql_candidate)
                if not is_valid:
                    safe = False
                    validation_error = error_msg
                    logger.warning(f"SQL failed validation: {error_msg}")
            logger.debug(f"SQL candidate: {sql_candidate}, safe: {safe}")
            if validation_error:
                logger.debug(f"Validation error: {validation_error}")

        if safe:
            # Execute SQL and return results
            try:
                with get_cursor() as cur:
                    cur.execute(sql_candidate)
                    rows = cur.fetchall()
                    cols = [d[0] for d in cur.description] if cur.description else []
                    logger.debug(f"SQL executed successfully, {len(rows)} rows returned")
            except Exception as e:
                logger.error(f"SQL execution error: {e}")
                # Fall back to NL response
                logger.info("Falling back to natural language response due to SQL error")
                sql_candidate = None
                safe = False

            return {
                "type": "sql",
                "sql": sql_candidate,
                "results": rows,
                "columns": cols,
                "retrieved_context": retrieved,
            }
        else:
            # Not safe or no SQL → fallback to direct answer
            answer_prompt = f"""
Using only the context below, answer the question succinctly.
If you cannot answer from the context, say so.

Context:
{context_text}

Question:
{question}

Answer:
"""
            try:
                client = get_gen_client()
                cfg = types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=MAX_OUTPUT_TOKENS
                )
                answer_resp = client.models.generate_content(
                    model=GEN_MODEL,
                    contents=answer_prompt,
                    config=cfg
                )
                ans = answer_resp.text.strip()
                logger.debug(f"Generated answer: {ans[:1000]}")
            except Exception as e:
                logger.error(f"Answer generation failed: {e}")
                raise HTTPException(status_code=500, detail=f"Answer generation failed: {str(e)}")

            return {
                "type": "text",
                "answer": ans,
                "retrieved_context": retrieved,
                "generated_raw": out,
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in query endpoint: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "Backend is running"}
