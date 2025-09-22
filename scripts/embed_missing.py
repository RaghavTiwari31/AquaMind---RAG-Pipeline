# scripts/embed_missing.py
from backend.app.db import get_cursor
from backend.app.embeddings import embed_texts, format_vector_for_pg
from tqdm import tqdm
BATCH = 128

def fetch_missing(batch=BATCH):
    with get_cursor() as cur:
        cur.execute("SELECT id, summary_text FROM summaries WHERE embedding IS NULL LIMIT %s", (batch,))
        return cur.fetchall()

def update_embedding(row_id, vec_str):
    with get_cursor(commit=True) as cur:
        cur.execute("UPDATE summaries SET embedding = %s::vector WHERE id = %s", (vec_str, row_id))

def run_once():
    while True:
        chunk = fetch_missing()
        if not chunk:
            print("All embeddings present.")
            break
        texts = [r['summary_text'] for r in chunk]
        ids = [r['id'] for r in chunk]
        embs = embed_texts(texts, show_progress_bar=False)
        for i, e in enumerate(embs):
            vec = format_vector_for_pg(e)
            update_embedding(ids[i], vec)
        print(f"Updated {len(chunk)} embeddings.")

if __name__ == "__main__":
    run_once()
