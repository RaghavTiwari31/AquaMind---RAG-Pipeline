# =============================
# backend/app/config.py
# =============================
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

# Core
DATABASE_URL: str | None = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Please set DATABASE_URL in the environment or .env")

# LLM
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "gemini-2.0-flash-001")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Please set GOOGLE_API_KEY (or GEMINI_API_KEY) in .env")
MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", 8000))
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", 6000))

# RAG
SCHEMA_DB_SCHEMA = os.environ.get("SCHEMA_DB_SCHEMA", "public")
SCHEMA_MAX_CHARS = int(os.environ.get("SCHEMA_MAX_CHARS", 6000))
SCHEMA_REFRESH_SEC = int(os.environ.get("SCHEMA_REFRESH_SEC", 0))  # 0 => no TTL