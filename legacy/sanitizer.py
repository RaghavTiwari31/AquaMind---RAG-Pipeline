# backend/app/sanitizer.py
import re

FORBIDDEN = [
    r"\binsert\b", r"\bupdate\b", r"\bdelete\b", r"\bdrop\b", r"\balter\b",
    r"\bcreate\b", r"\btruncate\b", r"\brevoke\b", r"\bgrant\b", r";", r"--", r"/\*"
]

SELECT_RE = re.compile(r'^\s*(with\b[\s\S]+?select\b|select\b)', re.IGNORECASE)

def is_safe_select(sql_text: str) -> bool:
    txt = sql_text.strip()
    # quick check: must start with SELECT or WITH ... SELECT
    if not SELECT_RE.search(txt):
        return False
    ltxt = txt.lower()
    for f in FORBIDDEN:
        if re.search(f, ltxt):
            return False
    # basic sanity: no multiple statements by semicolon
    if ";" in txt:
        return False
    return True

def extract_first_select(sql_text: str) -> str:
    """Try to extract the first SELECT / WITH ... SELECT block."""
    txt = sql_text.strip()
    # find first 'select' or 'with' then return until end (we will reject ';' later)
    m = re.search(r'(?si)(with\b[\s\S]*?select\b|select\b)', txt)
    if not m:
        return ""
    start = m.start()
    candidate = txt[start:].strip()
    # if model produced extra commentary after SQL, we can try to stop at common terminators
    # but we purposefully reject semicolons to avoid multi-statement outputs.
    candidate = candidate.split("\n\n")[0].strip()
    return candidate
