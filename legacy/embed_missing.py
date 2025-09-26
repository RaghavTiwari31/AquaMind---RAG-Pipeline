# scripts/embed_missing.py

from backend.app.db import get_cursor
from backend.app.embeddings import embed_texts, format_vector_for_pg
from tqdm import tqdm

BATCH = 1000  # rows per batch for processing


def ensure_label_column():
    """Create the embedding_label column if it doesn't already exist."""
    with get_cursor(commit=True) as cur:
        cur.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_name='summaries'
                      AND column_name='embedding_label'
                ) THEN
                    ALTER TABLE summaries
                    ADD COLUMN embedding_label TEXT;
                END IF;
            END;
            $$;
            """
        )


def fetch_missing(batch=BATCH):
    """
    Fetch a batch of rows with no embedding.
    Returns list of dicts: [{'id': ..., 'summary_text': ...}, ...]
    """
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT id, summary_text
            FROM summaries
            WHERE embedding IS NULL
            ORDER BY id
            LIMIT %s
            """,
            (batch,),
        )
        return cur.fetchall()


def update_embedding(row_id: int, vec_str: str, label_index: int) -> None:
    """
    Update a single row with:
      - numeric embedding (pgvector)
      - string label "<label_index>: [v1, v2, ...]" in embedding_label
    """
    label = f"{label_index}: {vec_str}"
    with get_cursor(commit=True) as cur:
        cur.execute(
            """
            UPDATE summaries
            SET embedding = %s::vector,
                embedding_label = %s
            WHERE id = %s
            """,
            (vec_str, label, row_id),
        )


def run_once() -> None:
    """
    Process all rows missing embeddings.
    Maintains a running global counter so labels are consecutive
    across multiple batches.
    """
    # determine current max label index so we keep counting correctly
    with get_cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) AS cnt FROM summaries WHERE embedding_label IS NOT NULL"
        )
        counter = cur.fetchone()["cnt"]

    while True:
        chunk = fetch_missing()
        if not chunk:
            print("All embeddings present.")
            break

        texts = [r["summary_text"] for r in chunk]
        ids = [r["id"] for r in chunk]

        # generate embeddings for the batch
        embs = embed_texts(texts, show_progress_bar=False)

        for i, e in enumerate(tqdm(embs, desc="Updating embeddings")):
            counter += 1
            vec = format_vector_for_pg(e)
            update_embedding(ids[i], vec, counter)

        print(f"Updated {len(chunk)} embeddings. Total so far: {counter}")


if __name__ == "__main__":
    ensure_label_column()
    run_once()
