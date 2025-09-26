# backend/app/embeddings.py
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

load_dotenv()
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", 384))

# load model once
_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def embed_texts(texts, show_progress_bar=False):
    model = get_model()
    # returns numpy array shape (n, dim)
    embeddings = model.encode(texts, show_progress_bar=show_progress_bar, convert_to_numpy=True)
    return embeddings

def format_vector_for_pg(vec: np.ndarray) -> str:
    # Format like: [0.123,0.234,...]
    return "[" + ",".join(f"{float(x):.6f}" for x in vec.tolist()) + "]"
