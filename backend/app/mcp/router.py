# =============================
# backend/app/mcp/router.py
# =============================
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from .models import (
    McpOpenSessionRequest, McpOpenSessionResponse,
    McpAppendMessageRequest, McpResourceListResponse,
    ToolCallRequest, ToolCallResponse,
)
from .session import store
from ..rag.retrieval import retrieve_topk
from ..rag.context import get_schema_text
from ..rag.sql_guard import is_safe_select, validate_sql_syntax
from ..db import get_cursor

router = APIRouter(prefix="/mcp", tags=["mcp"])


@router.post("/session.open", response_model=McpOpenSessionResponse)
def open_session(req: McpOpenSessionRequest):
    s, created = store.open(req.session_id)
    if req.labels:
        s.meta.update(req.labels)
    return McpOpenSessionResponse(session_id=s.id, created=created)


@router.post("/session.append")
def append_message(req: McpAppendMessageRequest):
    if not store.get(req.session_id):
        raise HTTPException(404, "session not found")
    store.append(req.session_id, req.role, req.content, req.meta)
    return {"ok": True}


@router.get("/resources.list", response_model=McpResourceListResponse)
def resources_list():
    return McpResourceListResponse(resources=[
        {"name": "db.describe_schema", "args": {}},
        {"name": "db.search_embeddings", "args": {"query": "str", "top_k": "int", "year": "int?"}},
        {"name": "db.exec_sql_readonly", "args": {"sql": "SELECT ..."}},
    ])


@router.post("/tools.call", response_model=ToolCallResponse)
def tools_call(req: ToolCallRequest):
    if not store.get(req.session_id):
        return ToolCallResponse(ok=False, error="session not found")

    try:
        if req.tool == "db.describe_schema":
            return ToolCallResponse(ok=True, result={"schema": get_schema_text()})

        if req.tool == "db.search_embeddings":
            q = req.args.get("query")
            k = int(req.args.get("top_k", 5))
            year = req.args.get("year")
            hits = retrieve_topk(q, top_k=k, year=year)
            return ToolCallResponse(ok=True, result=hits)

        if req.tool == "db.exec_sql_readonly":
            sql = str(req.args.get("sql") or "")
            safe = is_safe_select(sql)  # <-- just call the boolean function
            if not safe:
                return ToolCallResponse(ok=False, error="unsafe SQL")
            valid, msg = validate_sql_syntax(sql)
            if not valid:
                return ToolCallResponse(ok=False, error=msg)
            with get_cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description] if cur.description else []
            return ToolCallResponse(ok=True, result={"columns": cols, "rows": rows})


        return ToolCallResponse(ok=False, error=f"unknown tool {req.tool}")
    except Exception as e:
        return ToolCallResponse(ok=False, error=str(e))