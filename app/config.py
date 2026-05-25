import os
from dotenv import load_dotenv

load_dotenv()

def get_secret(key):
    """Lit depuis .env (local) ou Streamlit secrets (cloud)."""
    val = os.getenv(key)
    if not val:
        try:
            import streamlit as st
            val = st.secrets.get(key)
        except Exception:
            pass
    return val

QDRANT_URL = get_secret("QDRANT_URL")
QDRANT_API_KEY = get_secret("QDRANT_API_KEY")
COLLECTION_NAME = "gots_documents"

OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "meta-llama/llama-3.3-70b-instruct"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024

CHUNK_SIZE = 100
CHUNK_OVERLAP = 20
TOP_K = 5

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
EMBEDDINGS_DATA_PATH = os.path.join(BASE_DIR, "data", "embeddings")
PDF_FILE = os.path.join(BASE_DIR, "data", "raw", "GOTS_7.0__FR__signed.pdf")