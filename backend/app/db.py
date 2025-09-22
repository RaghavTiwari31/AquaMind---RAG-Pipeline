# backend/app/db.py
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from urllib.parse import urlparse
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Please set DATABASE_URL in the environment or .env")

def _get_conn():
    return psycopg2.connect(DATABASE_URL)

@contextmanager
def get_cursor(commit=False):
    conn = _get_conn()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        yield cur
        if commit:
            conn.commit()
    finally:
        cur.close()
        conn.close()
