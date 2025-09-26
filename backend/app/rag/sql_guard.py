# =============================
# backend/app/rag/sql_guard.py
# =============================
from __future__ import annotations
import re

FORBIDDEN = [
    r"\binsert\b", r"\bupdate\b", r"\bdelete\b", r"\bdrop\b", r"\balter\b",
    r"\bcreate\b", r"\btruncate\b", r"\brevoke\b", r"\bgrant\b", r";", r"--", r"/\*",
]
SELECT_RE = re.compile(r'^\s*(with\b[\s\S]+?select\b|select\b)', re.IGNORECASE)


def is_safe_select(sql_text: str) -> bool:
    txt = (sql_text or "").strip()
    if not SELECT_RE.search(txt):
        return False
    ltxt = txt.lower()
    for f in FORBIDDEN:
        if re.search(f, ltxt):
            return False
    if ";" in txt:
        return False
    return True


def extract_first_select(sql_text: str) -> str:
    txt = (sql_text or "").strip()
    m = re.search(r'(?si)(with\b[\s\S]*?select\b|select\b)', txt)
    if not m:
        return ""
    start = m.start()
    candidate = txt[start:].strip()
    return candidate.split("\n\n")[0].strip()


def validate_sql_syntax(sql: str) -> tuple[bool, str]:
    up = (sql or "").upper().strip()
    if not up.startswith("SELECT"):
        return False, "Only SELECT statements are allowed"

    # Block DML/DDL and multi-statement tricks
    for op in ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"]:
        if op in up:
            return False, f"Operation {op} is not allowed"

    # If there's a FROM, we can do a couple of extra sanity checks.
    # If there's no FROM (e.g. SELECT 1), that's fine — skip alias checks that rely on FROM.
    if " FROM " in up:
        # Simple alias sanity (optional)
        alias_pattern = r"\b[T]\d+\."
        if re.search(alias_pattern, up):
            from_match = re.search(r"FROM\s+(\w+)(?:\s+(?:AS\s+)?([T]\d+))?", up)
            if not from_match:
                return False, "Table aliases found but FROM clause is missing or malformed"
            if from_match.group(2) is None:
                return False, "Table alias used but not properly defined in FROM clause"

    return True, "Valid"
