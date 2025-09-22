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
from transformers import pipeline, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

# Default generation model (can override with env var)
GEN_MODEL = os.environ.get("GENERATION_MODEL", "Salesforce/codet5p-770m-py")

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="ARGO_RAG", version="0.1")

# ---- Global caches ----
_gen_pipe = None
_tokenizer = None


def get_gen_pipe():
    """Lazy-load the text2text pipeline (flan-t5)."""
    global _gen_pipe
    if _gen_pipe is None:
        _gen_pipe = pipeline("text2text-generation", model=GEN_MODEL)
    return _gen_pipe


def get_tokenizer():
    """Lazy-load tokenizer for safe truncation."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    return _tokenizer


def truncate_prompt(prompt: str, max_tokens: int = 512) -> str:
    """
    Ensure prompt does not exceed model max input length.
    Uses the model's tokenizer for safe truncation.
    """
    tokenizer = get_tokenizer()
    tokens = tokenizer(prompt, truncation=True, max_length=max_tokens, return_tensors="pt")
    return tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)

def validate_sql_syntax(sql: str) -> tuple[bool, str]:
    """
    Additional SQL validation to catch common issues.
    Returns (is_valid, error_message)
    """
    sql_upper = sql.upper().strip()
    
    # Check if it's a SELECT statement
    if not sql_upper.startswith('SELECT'):
        return False, "Only SELECT statements are allowed"
    
    # Check for FROM clause
    if ' FROM ' not in sql_upper:
        return False, "SQL must include a FROM clause"
    
    # Check for dangerous operations
    dangerous_ops = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE']
    for op in dangerous_ops:
        if op in sql_upper:
            return False, f"Operation {op} is not allowed"
    
    # Basic alias validation - if using aliases, make sure FROM clause exists
    lines = sql.replace('\n', ' ').strip()
    
    # Look for potential table aliases (T1, T2, etc.)
    import re
    alias_pattern = r'\b[T]\d+\.'
    if re.search(alias_pattern, sql.upper()):
        # Check if FROM clause properly defines the alias
        from_match = re.search(r'FROM\s+(\w+)(?:\s+(?:AS\s+)?([T]\d+))?', sql.upper())
        if not from_match:
            return False, "Table aliases found but FROM clause is missing or malformed"
        if from_match.group(2) is None:  # No alias defined after table name
            return False, "Table alias used but not properly defined in FROM clause"
    
    return True, "Valid"


executor = ThreadPoolExecutor(max_workers=2)


class QueryPayload(BaseModel):
    question: str
    top_k: int = 5


@app.post("/query")
async def query(req: QueryPayload, request: Request):
    try:
        # Log the incoming request
        body = await request.body()
        logger.debug(f"Received request body: {body.decode()}")
        logger.debug(f"Parsed payload: question='{req.question}', top_k={req.top_k}")
        
        question = req.question
        top_k = int(req.top_k)

        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # 1) embed the question
        try:
            q_vec = embed_texts([question])[0]  # numpy array
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

        context_text = "\n\n".join(
            [
                f"TABLE:{r['source_table']} | ID:{r['source_id']} | {r['summary_text']}"
                for r in retrieved
            ]
        )

        # 3) build prompt for SQL generation
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

        # Truncate safely to fit model limits
        try:
            prompt = truncate_prompt(prompt, max_tokens=512)
            logger.debug(f"Prompt truncated to length: {len(prompt)}")
        except Exception as e:
            logger.error(f"Prompt truncation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process prompt: {str(e)}")

        # 4) ask the generator
        try:
            gen = get_gen_pipe()
            out = gen(prompt, max_new_tokens=128, do_sample=False)[0]["generated_text"].strip()
            logger.debug(f"Generated response: {out}")
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")

                # 5) try to extract SQL and sanitize
        sql_candidate = extract_first_select(out)
        safe = False
        validation_error = ""
        
        if sql_candidate:
            safe = is_safe_select(sql_candidate)
            if safe:
                # Additional syntax validation
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
                # If SQL execution fails, fall back to natural language response
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
                answer_prompt = truncate_prompt(answer_prompt, max_tokens=512)
                ans = gen(answer_prompt, max_new_tokens=128, do_sample=False)[0]["generated_text"].strip()
                logger.debug(f"Generated answer: {ans}")
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
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error in query endpoint: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "Backend is running"}





