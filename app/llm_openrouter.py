# llm_openrouter.py :

import requests
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL


def generate_response(prompt: str, system_prompt: str = None) -> str:
    """
    Appelle l'API OpenRouter avec Llama 3.3 70B
    et retourne la réponse générée.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/eco-textile-rag",
        "X-Title": "Éco Textile RAG"
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 1024,
    }

    try:
        response = requests.post(
            url=f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        return answer.strip()
    except requests.exceptions.HTTPError as e:
        print(f"❌ Erreur HTTP OpenRouter : {e}")
        print(f"   Réponse : {response.text}")
        raise
    except Exception as e:
        print(f"❌ Erreur OpenRouter : {e}")
        raise
