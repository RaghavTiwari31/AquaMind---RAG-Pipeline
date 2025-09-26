# =============================
# backend/app/rag/context.py
# =============================
from __future__ import annotations
import time
from ..config import MAX_PROMPT_CHARS, SCHEMA_DB_SCHEMA, SCHEMA_MAX_CHARS, SCHEMA_REFRESH_SEC
from ..db import get_cursor

_schema_cache = {"text": "", "loaded_at": 0.0}


def build_truncated_context(rows: list[dict], max_chars: int = MAX_PROMPT_CHARS, per_summary_cap: int = 800) -> str:
    pieces: list[str] = []
    total = 0
    for r in rows:
        s = f"TABLE:{r['source_table']} | ID:{r['source_id']} | {r['summary_text']}"
        if len(s) > per_summary_cap:
            s = s[: per_summary_cap - 3] + "..."
        if total + len(s) > max_chars:
            break
        pieces.append(s)
        total += len(s)
    return "\n\n".join(pieces)


def load_schema_text(schema: str = SCHEMA_DB_SCHEMA, max_chars: int = SCHEMA_MAX_CHARS) -> str:
    sql = (
        "SELECT table_name, column_name, data_type\n"
        "FROM information_schema.columns\n"
        "WHERE table_schema=%s ORDER BY table_name, ordinal_position;"
    )
    by_table: dict[str, list[tuple[str, str]]] = {}
    with get_cursor() as cur:
        cur.execute(sql, (schema,))
        for t, c, d in cur.fetchall():
            by_table.setdefault(t, []).append((c, d))
    parts: list[str] = []
    total = 0
    for t, cols in by_table.items():
        block = "TABLE " + t + "\n" + "\n".join(f"  - {c} ({d})" for c, d in cols) + "\n\n"
        if total + len(block) > max_chars:
            parts.append(block[: max_chars - total])
            break
        parts.append(block)
        total += len(block)
    return "".join(parts).rstrip()


def get_schema_text(force: bool = False) -> str:
    now = time.time()
    if force or _schema_cache["text"] == "" or (SCHEMA_REFRESH_SEC > 0 and now - _schema_cache["loaded_at"] > SCHEMA_REFRESH_SEC):
        text = load_schema_text()
        _schema_cache.update({"text": text, "loaded_at": now})
    return _schema_cache["text"]