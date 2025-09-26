# =============================
# backend/app/main.py (app factory)
# =============================
from fastapi import FastAPI
from .routes.health import router as health_router
from .routes.qa import router as qa_router
from .mcp.router import router as mcp_router

app = FastAPI(title="ARGO_RAG", version="0.2")
app.include_router(health_router)
app.include_router(qa_router)
app.include_router(mcp_router)