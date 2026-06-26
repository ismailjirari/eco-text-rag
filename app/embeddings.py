# embeddings.py :

import os
import json
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, EMBEDDINGS_DATA_PATH

_model = None


def get_model():
    """Charge le modèle une seule fois (lazy loading)."""
    global _model
    if _model is None:
        print(f"⏳ Chargement du modèle d'embeddings : {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✅ Modèle chargé.")
    return _model


def embed_text(text: str) -> list[float]:
    """Génère un vecteur d'embedding pour un texte unique."""
    vector = get_model().encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_documents(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Génère des embeddings pour une liste de textes."""
    print(f"⏳ Génération embeddings pour {len(texts)} textes...")
    vectors = get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    print(f"✅ {len(vectors)} embeddings générés.")
    return vectors.tolist()


def save_embeddings(vectors: list[list[float]], filename: str = "embeddings.json"):
    """Sauvegarde les embeddings localement."""
    os.makedirs(EMBEDDINGS_DATA_PATH, exist_ok=True)
    filepath = os.path.join(EMBEDDINGS_DATA_PATH, filename)
    with open(filepath, 'w') as f:
        json.dump(vectors, f)
    print(f"✅ Embeddings sauvegardés dans {filepath}")


def load_embeddings(filename: str = "embeddings.json") -> list[list[float]]:
    """Charge les embeddings sauvegardés localement."""
    filepath = os.path.join(EMBEDDINGS_DATA_PATH, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Embeddings non trouvés : {filepath}")
    with open(filepath, 'r') as f:
        vectors = json.load(f)
    print(f"✅ {len(vectors)} embeddings chargés depuis {filepath}")
    return vectors
