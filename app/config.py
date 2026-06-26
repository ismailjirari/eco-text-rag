# config.py :
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
COLLECTION_NAME = "eco_textile_documents"

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

# Les 3 PDFs éco-textile
PDF_FILES = [
    {
        "path": os.path.join(RAW_DATA_PATH, "GOTS_7.0__FR__signed.pdf"),
        "label": "GOTS 7.0 (FR)",
        "short": "GOTS"
    },
    {
        "path": os.path.join(RAW_DATA_PATH, "A New Textiles Economy - Summary of findings.pdf"),
        "label": "A New Textiles Economy",
        "short": "NTE"
    },
    {
        "path": os.path.join(RAW_DATA_PATH, "eco-conception-des-produits-textiles-habillement.pdf"),
        "label": "Éco-conception Textiles & Habillement",
        "short": "ECO"
    },
]
