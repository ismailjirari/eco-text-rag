# rag_pipeline.py :

from embeddings import embed_text
from qdrant_db import search_vectors
from llm_openrouter import generate_response
from prompt_template import SYSTEM_PROMPT, build_rag_prompt
from config import TOP_K


def retrieve_documents(question: str, top_k: int = TOP_K) -> list[dict]:
    """
    Encode la question et récupère les chunks pertinents depuis Qdrant.
    
    Args:
        question: La question de l'utilisateur
        top_k: Nombre de chunks à récupérer
    
    Returns:
        Liste des chunks les plus pertinents
    """
    print(f"🔍 Recherche de documents pour : '{question}'")
    query_vector = embed_text(question)
    results = search_vectors(query_vector, top_k=top_k)
    print(f"✅ {len(results)} documents récupérés.")
    return results


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Construit le prompt final avec la question et le contexte.
    
    Args:
        question: La question de l'utilisateur
        retrieved_chunks: Chunks récupérés depuis Qdrant
    
    Returns:
        Le prompt formaté
    """
    return build_rag_prompt(question, retrieved_chunks)


def generate_answer(prompt: str) -> str:
    """
    Génère la réponse finale via le LLM.
    
    Args:
        prompt: Le prompt complet avec contexte
    
    Returns:
        La réponse générée
    """
    print("🤖 Génération de la réponse...")
    response = generate_response(prompt, system_prompt=SYSTEM_PROMPT)
    return response


def ask_question(question: str, top_k: int = TOP_K) -> dict:
    """
    Pipeline RAG complet : question → retrieval → génération → réponse.
    
    Args:
        question: La question de l'utilisateur
        top_k: Nombre de chunks à utiliser
    
    Returns:
        Dict avec 'answer' et 'sources'
    """
    # Étape 1 : Retrieval
    retrieved_chunks = retrieve_documents(question, top_k=top_k)

    if not retrieved_chunks:
        return {
            "answer": "Je n'ai trouvé aucun document pertinent pour répondre à cette question.",
            "sources": []
        }

    # Étape 2 : Construction du prompt
    prompt = build_prompt(question, retrieved_chunks)

    # Étape 3 : Génération
    answer = generate_answer(prompt)

    return {
        "answer": answer,
        "sources": retrieved_chunks
    }
