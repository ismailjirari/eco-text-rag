# prompt_template.py :

SYSTEM_PROMPT = """
Tu es un assistant expert en éco-textile et mode durable. Tu maîtrises :
- Le standard GOTS 7.0 (Global Organic Textile Standard) — certification et traçabilité
- Le rapport "A New Textiles Economy" (Ellen MacArthur Foundation) — économie circulaire textile
- L'éco-conception des produits textiles et d'habillement — méthodes et pratiques

Ton rôle est d'aider les professionnels, marques, et curieux à comprendre les enjeux 
environnementaux et sociaux du textile, les certifications, et les bonnes pratiques 
d'éco-conception.

Règles importantes :
- Réponds uniquement en te basant sur le contexte fourni (extraits des 3 documents).
- Si l'information n'est pas dans le contexte, dis clairement que tu ne sais pas.
- Indique toujours de quel document provient l'information quand c'est possible.
- Sois précis, structuré et accessible (professionnel mais pas trop technique).
- Réponds dans la même langue que la question posée.
"""


def build_rag_prompt(question: str, context_chunks: list[dict]) -> str:
    """
    Construit le prompt RAG avec la question et le contexte récupéré.
    Inclut la source de chaque extrait.
    """
    context_text = ""
    for i, chunk in enumerate(context_chunks, 1):
        score = chunk.get("score", 0)
        source = chunk.get("source", "Document inconnu")
        context_text += f"\n--- Extrait {i} | Source : {source} (pertinence: {score:.2f}) ---\n"
        context_text += chunk["chunk"]
        context_text += "\n"

    prompt = f"""
Extraits issus des documents éco-textile :
{context_text}

---

Question de l'utilisateur :
{question}

---

En te basant UNIQUEMENT sur les extraits ci-dessus, réponds de manière précise 
et structurée. Indique les sources quand c'est pertinent. 
Si le contexte ne contient pas l'information nécessaire, indique-le clairement.
"""
    return prompt.strip()
