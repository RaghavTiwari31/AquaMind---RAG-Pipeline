# =============================
# backend/app/mcp/models.py
# =============================
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class McpOpenSessionRequest(BaseModel):
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    labels: Dict[str, Any] = Field(default_factory=dict)


class McpOpenSessionResponse(BaseModel):
    session_id: str
    created: bool


class McpAppendMessageRequest(BaseModel):
    session_id: str
    role: str  # user|assistant|tool
    content: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class McpResourceListResponse(BaseModel):
    resources: List[Dict[str, Any]]


class ToolCallRequest(BaseModel):
    session_id: str
    tool: str  # "db.search_embeddings" | "db.describe_schema" | "db.exec_sql_readonly"
    args: Dict[str, Any] = Field(default_factory=dict)


class ToolCallResponse(BaseModel):
    ok: bool
    result: Any | None = None
    error: str | None = None