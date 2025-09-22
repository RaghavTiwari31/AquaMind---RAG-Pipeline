# backend/app/summarizer.py
"""
Summarizer job:
- discover ARGO_D tables (the DB already has tables for 2001..2017)
- for each row, build a textual summary (compact)
- upsert into `summaries` table (source_table, source_id, summary_text)
"""
from .db import get_cursor
from .embeddings import embed_texts, format_vector_for_pg, EMBEDDING_DIM
import re
from tqdm import tqdm

def list_argo_tables():
    with get_cursor() as cur:
        cur.execute("""
          SELECT table_name
          FROM information_schema.tables
          WHERE table_schema='public'
            AND (table_name ~ '^[0-9]{4}$' OR table_name ILIKE 'argo%')
          ORDER BY table_name;
        """)
        rows = cur.fetchall()
    return [r['table_name'] for r in rows]

def get_primary_key_column(table_name):
    with get_cursor() as cur:
        cur.execute("""
            SELECT a.attname as col
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = %s::regclass AND i.indisprimary;
        """, (table_name,))
        r = cur.fetchone()
        return r['col'] if r else None

def table_columns(table_name):
    with get_cursor() as cur:
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
        """, (table_name,))
        return [r['column_name'] for r in cur.fetchall()]

def make_summary_from_row(table_name, row, pk_col=None, max_len=800):
    # Choose sensible fields: date/time, lat/lon, and up to 6 other columns.
    cols = list(row.keys())
    parts = [f"table={table_name}"]
    if pk_col and pk_col in row:
        parts.append(f"id={row.get(pk_col)}")
    # include date/time-like columns
    for c in cols:
        if re.search(r'date|time|day|julian', c, re.IGNORECASE):
            parts.append(f"{c}={row.get(c)}")
    # lat/lon
    lat = None
    lon = None
    for c in cols:
        if re.search(r'lat', c, re.IGNORECASE):
            lat = row.get(c)
        if re.search(r'lon|long', c, re.IGNORECASE):
            lon = row.get(c)
    if lat is not None and lon is not None:
        parts.append(f"location=({lat},{lon})")

    # sample some measurement columns
    extras = []
    for c in cols:
        if c not in (pk_col, ) and not re.search(r'date|time|lat|lon|long|id', c, re.IGNORECASE):
            extras.append(f"{c}={row.get(c)}")
        if len(extras) >= 6:
            break
    if extras:
        parts.append("measurements: " + ", ".join(extras))
    summary = "; ".join(parts)
    if len(summary) > max_len:
        summary = summary[:max_len-3] + "..."
    return summary

def upsert_summary(table_name, source_id, summary_text):
    with get_cursor(commit=True) as cur:
        cur.execute("""
            INSERT INTO summaries (source_table, source_id, summary_text)
            VALUES (%s, %s, %s)
            ON CONFLICT (source_table, source_id)
            DO UPDATE SET summary_text = EXCLUDED.summary_text
            RETURNING id;
        """, (table_name, str(source_id), summary_text))
        r = cur.fetchone()
        return r['id']

def process_table(table_name, batch=1000, limit_rows=None):
    pk = get_primary_key_column(table_name)
    # read rows
    with get_cursor() as cur:
        q = f"SELECT * FROM {table_name}"
        if limit_rows:
            q += f" LIMIT {int(limit_rows)}"
        cur.execute(q)
        rows = cur.fetchall()
    for row in tqdm(rows, desc=f"Summarizing {table_name}"):
        source_id = row.get(pk) if pk else str(hash(str(row)))
        summary = make_summary_from_row(table_name, row, pk)
        upsert_summary(table_name, source_id, summary)

def run_full_summary(limit_per_table=None):
    tables = list_argo_tables()
    for t in tables:
        process_table(t, limit_rows=limit_per_table)
