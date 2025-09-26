"""
Build embeddings & summaries for ARGO_D using pgvector (RESUMABLE).

What’s new vs previous version
------------------------------
- **Resumable ingestion**: skips row-groups already stored in `argo_embeddings`.
- **Idempotent upsert**: unique key on `(source_table,item_type,source_id)`; re-runs update existing rows.
- **No duplicates**: schema items now use `source_id='__schema__'` so uniqueness works.

Usage
-----
set PG_DSN=postgresql://user:pass@localhost:5432/ARGO_D
python -m scripts.build_pgvector_embeddings --ingest schema rows --max-rows 100000

# resume later (continues where it left off):
python -m scripts.build_pgvector_embeddings --ingest rows --max-rows 200000

# full refresh for a couple of tables (optional):
python -m scripts.build_pgvector_embeddings --rebuild argo_details_2005 argo_details_2006 --ingest schema rows
"""
import os
import re
import json
import argparse
from typing import List, Dict, Any, Iterable, Tuple, Optional

import psycopg2
import psycopg2.extras as pgx
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np

# ---------------------------
# Config
# ---------------------------
PG_DSN = os.environ.get("PG_DSN", "postgresql://postgres:root@localhost:5432/ARGO_D")
MODEL_NAME = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMB_DIM = int(os.environ.get("EMBED_DIM", "384"))  # all-MiniLM-L6-v2
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "512"))
IVFFLAT_LISTS = int(os.environ.get("IVFFLAT_LISTS", "100"))
MAX_SCHEMA_CHARS = int(os.environ.get("MAX_SCHEMA_CHARS", "6000"))

TABLE_PREFIX = "argo_details_"
TABLE_START = 2001
TABLE_END = 2017

# ---------------------------
# DDL (unique key + ivfflat index)
# ---------------------------
DDL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS argo_embeddings (
    id           BIGSERIAL PRIMARY KEY,
    source_table TEXT NOT NULL,
    source_id    TEXT NOT NULL,          -- set to '__schema__' for schema items
    year         INT,
    item_type    TEXT NOT NULL,          -- 'schema' | 'row'
    summary_text TEXT NOT NULL,
    metadata     JSONB NOT NULL DEFAULT '{{}}',
    embedding    VECTOR({EMB_DIM}) NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Uniqueness across table/type/id enables idempotent upsert & resume
CREATE UNIQUE INDEX IF NOT EXISTS argo_embeddings_unique
  ON argo_embeddings (source_table, item_type, source_id);

-- ANN index for cosine similarity
CREATE INDEX IF NOT EXISTS argo_embeddings_embedding_idx
  ON argo_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = {IVFFLAT_LISTS});

CREATE INDEX IF NOT EXISTS argo_embeddings_table_year_idx
  ON argo_embeddings (source_table, year);
"""

# ---------------------------
# DB helpers
# ---------------------------

def connect():
    return psycopg2.connect(PG_DSN)


def ensure_schema():
    with connect() as con, con.cursor() as cur:
        cur.execute(DDL)
        con.commit()


def list_argo_tables(con) -> List[str]:
    with con.cursor(cursor_factory=pgx.DictCursor) as cur:
        cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema='public' AND table_name ~ %s
            ORDER BY table_name;
            """,
            (rf"^{TABLE_PREFIX}(?:{TABLE_START}|{TABLE_START+1}|{TABLE_START+2}|{TABLE_START+3}|{TABLE_START+4}|{TABLE_START+5}|{TABLE_START+6}|{TABLE_START+7}|{TABLE_START+8}|{TABLE_START+9}|{TABLE_START+10}|{TABLE_START+11}|{TABLE_START+12}|{TABLE_START+13}|{TABLE_START+14}|{TABLE_START+15}|{TABLE_END})$",),
        )
        return [r[0] for r in cur.fetchall()]


def infer_year_from_table(t: str) -> Optional[int]:
    m = re.search(r"(\d{4})$", t)
    return int(m.group(1)) if m else None

# ---------------------------
# Summaries
# ---------------------------

def summarize_schema(con, table: str) -> Tuple[str, Dict[str, Any]]:
    with con.cursor(cursor_factory=pgx.DictCursor) as cur:
        cur.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name=%s
            ORDER BY ordinal_position;
            """,
            (table,),
        )
        cols = cur.fetchall()
    col_lines = [f"- {c['column_name']} ({c['data_type']})" for c in cols]
    summary = f"TABLE {table}\n" + "\n".join(col_lines)
    if len(summary) > MAX_SCHEMA_CHARS:
        summary = summary[: MAX_SCHEMA_CHARS - 3] + "..."
    metadata = {
        "type": "schema",
        "columns": [{"name": c["column_name"], "type": c["data_type"]} for c in cols],
        "column_count": len(cols),
    }
    return summary, metadata


def summarize_rows_grouped(con, table: str, skip_existing: bool, limit: Optional[int] = None) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Yield (summary_text, metadata) per ad_observation_id group.
    Uses anti-join to **skip rows already in argo_embeddings** when skip_existing=True.
    """
    where_not_exists = (
        "WHERE NOT EXISTS (SELECT 1 FROM argo_embeddings e\n"
        "                    WHERE e.source_table = %s AND e.item_type='row'\n"
        "                      AND e.source_id = agg.ad_observation_id)"
        if skip_existing else ""
    )
    sql = f"""
        WITH agg AS (
            SELECT ad_observation_id,
                   COUNT(*) AS n,
                   MIN(depth) AS min_depth,
                   MAX(depth) AS max_depth,
                   AVG(temperature) AS mean_temp,
                   MIN(temperature) AS min_temp,
                   MAX(temperature) AS max_temp,
                   AVG(salinity) AS mean_sal,
                   MIN(salinity) AS min_sal,
                   MAX(salinity) AS max_sal,
                   AVG(density) AS mean_rho,
                   MIN(density) AS min_rho,
                   MAX(density) AS max_rho
            FROM {table}
            GROUP BY ad_observation_id
        )
        SELECT * FROM agg
        {where_not_exists}
        ORDER BY ad_observation_id
        {f'LIMIT {int(limit)}' if limit else ''}
    """
    params = ([table] if skip_existing else [])
    with con.cursor(cursor_factory=pgx.DictCursor) as cur:
        cur.execute(sql, params)
        while True:
            rows = cur.fetchmany(2000)
            if not rows:
                break
            for r in rows:
                text = (
                    f"Observation {r['ad_observation_id']} in {table}: "
                    f"n={r['n']}, depth {r['min_depth']:.1f}–{r['max_depth']:.1f} m; "
                    f"temperature {r['min_temp']:.3f}–{r['max_temp']:.3f} (mean {r['mean_temp']:.3f}); "
                    f"salinity {r['min_sal']:.3f}–{r['max_sal']:.3f} (mean {r['mean_sal']:.3f}); "
                    f"density {r['min_rho']:.3f}–{r['max_rho']:.3f} (mean {r['mean_rho']:.3f})."
                )
                meta = {
                    "type": "row",
                    "ad_observation_id": r["ad_observation_id"],
                    "n": int(r["n"]),
                    "depth_range": [float(r["min_depth"]), float(r["max_depth"])],
                    "temperature": {
                        "mean": float(r["mean_temp"]),
                        "min": float(r["min_temp"]),
                        "max": float(r["max_temp"]),
                    },
                    "salinity": {
                        "mean": float(r["mean_sal"]),
                        "min": float(r["min_sal"]),
                        "max": float(r["max_sal"]),
                    },
                    "density": {
                        "mean": float(r["mean_rho"]),
                        "min": float(r["min_rho"]),
                        "max": float(r["max_rho"]),
                    },
                }
                yield text, meta

# ---------------------------
# Embeddings
# ---------------------------
_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_model()
    vecs = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)

# ---------------------------
# Upsert (idempotent)
# ---------------------------
UPSERT_SQL = f"""
INSERT INTO argo_embeddings (source_table, source_id, year, item_type, summary_text, metadata, embedding)
VALUES (%(source_table)s, %(source_id)s, %(year)s, %(item_type)s, %(summary_text)s, %(metadata)s, %(embedding)s)
ON CONFLICT (source_table, item_type, source_id)
DO UPDATE SET
  year = EXCLUDED.year,
  summary_text = EXCLUDED.summary_text,
  metadata = EXCLUDED.metadata,
  embedding = EXCLUDED.embedding,
  created_at = now();
"""


def insert_embeddings(con, rows: List[Dict[str, Any]]):
    if not rows:
        return
    with con.cursor() as cur:
        pgx.register_default_jsonb(cur)
        psycopg2.extras.execute_batch(cur, UPSERT_SQL, rows, page_size=1000)
    con.commit()

# ---------------------------
# Ingest pipelines (resumable)
# ---------------------------

def ingest_schema(con, tables: List[str]) -> int:
    texts, payloads = [], []
    for t in tables:
        year = infer_year_from_table(t)
        summary, meta = summarize_schema(con, t)
        texts.append(summary)
        payloads.append({
            "source_table": t,
            "source_id": "__schema__",     # make unique deterministic id
            "year": year,
            "item_type": "schema",
            "summary_text": summary,
            "metadata": json.dumps(meta),
        })
    if not payloads:
        return 0
    vecs = embed_texts(texts)
    for i, v in enumerate(vecs):
        payloads[i]["embedding"] = list(map(float, v))
    insert_embeddings(con, payloads)
    return len(payloads)


def ingest_rows(con, tables: List[str], max_rows: Optional[int] = None, skip_existing: bool = True) -> int:
    total = 0
    remaining = max_rows if max_rows else None

    for t in tables:
        year = infer_year_from_table(t)
        # Stream groups, optionally skipping already-ingested ones
        stream = summarize_rows_grouped(con, t, skip_existing=skip_existing, limit=None)

        batch_texts, batch_payloads = [], []
        for text, meta in tqdm(stream, desc=f"{t}"):
            if remaining is not None and remaining <= 0:
                # stop across all tables
                if batch_texts:
                    vecs = embed_texts(batch_texts)
                    for i, v in enumerate(vecs):
                        batch_payloads[i]["embedding"] = list(map(float, v))
                    insert_embeddings(con, batch_payloads)
                    total += len(batch_texts)
                return total

            batch_texts.append(text)
            batch_payloads.append({
                "source_table": t,
                "source_id": meta.get("ad_observation_id"),
                "year": year,
                "item_type": "row",
                "summary_text": text,
                "metadata": json.dumps(meta),
            })

            if len(batch_texts) >= BATCH_SIZE:
                vecs = embed_texts(batch_texts)
                for i, v in enumerate(vecs):
                    batch_payloads[i]["embedding"] = list(map(float, v))
                insert_embeddings(con, batch_payloads)
                total += len(batch_texts)
                if remaining is not None:
                    remaining -= len(batch_texts)
                batch_texts, batch_payloads = [], []

        if batch_texts:
            vecs = embed_texts(batch_texts)
            for i, v in enumerate(vecs):
                batch_payloads[i]["embedding"] = list(map(float, v))
            insert_embeddings(con, batch_payloads)
            total += len(batch_texts)
            if remaining is not None:
                remaining -= len(batch_texts)

    return total

# ---------------------------
# Retrieval helper
# ---------------------------

def retrieve_topk(question: str, top_k: int = 5, year: Optional[int] = None) -> List[Dict[str, Any]]:
    vec = embed_texts([question])[0]
    params = {"q": list(map(float, vec)), "k": max(1, int(top_k))}
    sql = """
        SELECT id, source_table, source_id, year, item_type, summary_text, metadata,
               1 - (embedding <=> %(q)s::vector) AS score
        FROM argo_embeddings
        {where}
        ORDER BY embedding <=> %(q)s::vector
        LIMIT %(k)s;
    """.format(where=("WHERE year = %(yr)s" if year else ""))

    with connect() as con, con.cursor(cursor_factory=pgx.DictCursor) as cur:
        if year:
            params["yr"] = int(year)
        cur.execute(sql, params)
        rows = cur.fetchall()
        return [dict(r) for r in rows]

# ---------------------------
# Rebuild (optional)
# ---------------------------

def rebuild_tables(con, tables: List[str]):
    if not tables:
        return 0
    with con.cursor() as cur:
        cur.execute("DELETE FROM argo_embeddings WHERE source_table = ANY(%s);", (tables,))
    con.commit()
    return len(tables)

# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Build pgvector embeddings for ARGO_D (resumable)")
    p.add_argument("--dsn", default=PG_DSN, help="Postgres DSN (env PG_DSN)")
    p.add_argument("--ingest", nargs='+', choices=["schema", "rows"], default=["schema", "rows"], help="What to ingest")
    p.add_argument("--tables", nargs='*', help="Explicit tables to ingest (default: argo_details_2001..2017)")
    p.add_argument("--max-rows", type=int, default=None, help="Global cap for row-group items across all tables")
    p.add_argument("--no-skip-existing", action="store_true", help="Do not skip existing row-groups (will upsert/refresh)")
    p.add_argument("--rebuild", nargs='*', help="Delete existing embeddings for these tables before ingest (e.g., argo_details_2005 argo_details_2006)")
    return p.parse_args()


def main():
    args = parse_args()

    global PG_DSN
    PG_DSN = args.dsn

    ensure_schema()
    with connect() as con:
        tables = args.tables or list_argo_tables(con)
        if not tables:
            print("No ARGO tables found. Expected names like argo_details_2001..2017.")
            return

        if args.rebuild:
            n = rebuild_tables(con, args.rebuild)
            print(f"Purged embeddings for {n} tables: {', '.join(args.rebuild)}")

        print(f"Using model: {MODEL_NAME} (dim={EMB_DIM})")
        if "schema" in args.ingest:
            n = ingest_schema(con, tables)
            print(f"Upserted {n} schema summaries")
        if "rows" in args.ingest:
            n = ingest_rows(con, tables, max_rows=args.max_rows, skip_existing=not args.no_skip_existing)
            print(f"Ingested/updated {n} row-group summaries")


if __name__ == "__main__":
    main()
