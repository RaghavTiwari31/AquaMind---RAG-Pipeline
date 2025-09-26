# backend/app/db.py
from __future__ import annotations
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from .config import DATABASE_URL

def _get_conn():
    return psycopg2.connect(DATABASE_URL)

@contextmanager
def get_cursor(commit: bool = False):
    """Context manager that yields a RealDictCursor and optionally commits."""
    conn = _get_conn()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        yield cur
        if commit:
            conn.commit()
    finally:
        try:
            cur.close()
        finally:    
            conn.close()
