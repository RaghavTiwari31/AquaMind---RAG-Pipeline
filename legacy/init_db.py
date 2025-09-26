# scripts/init_db.py
"""
Helper to create extension and summaries table from the command line.
Run: python scripts/init_db.py
"""
from backend.app.db import get_cursor
from dotenv import load_dotenv
load_dotenv()

def init():
    with get_cursor(commit=True) as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
              id BIGSERIAL PRIMARY KEY,
              source_table TEXT NOT NULL,
              source_id TEXT NOT NULL,
              summary_text TEXT NOT NULL,
              embedding VECTOR(384),
              created_at TIMESTAMPTZ DEFAULT now(),
              UNIQUE(source_table, source_id)
            );
        """)
        # Do not create IVFFlat index until vectors exist; user can run later.
        print("Created summaries table and enabled vector extension (if available).")

if __name__ == "__main__":
    init()
