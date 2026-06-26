# streamlit_app/streamlit_app.py :

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

import streamlit as st

# ─── Configuration Page ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Éco Textile RAG",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Cache modèle et pipeline ─────────────────────────────────────────────────
@st.cache_resource
def load_rag():
    """Charge le modèle d'embeddings une seule fois pour toute la session."""
    from embeddings import get_model
    get_model()
    from rag_pipeline import ask_question
    return ask_question

ask_question = load_rag()

from config import PDF_FILES

# ─── CSS Custom ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --green-dark:  #2d6a4f;
        --green-mid:   #40916c;
        --green-light: #74c69d;
        --green-pale:  #d8f3dc;
        --cream:       #f8f9f0;
        --text-dark:   #1b4332;
    }
    .stApp { background-color: var(--cream); }
    .hero-header {
        background: linear-gradient(135deg, #2d6a4f 0%, #40916c 60%, #74c69d 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(45,106,79,0.25);
    }
    .hero-header h1 { margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: -0.5px; }
    .hero-header p  { margin: 0.4rem 0 0; opacity: 0.88; font-size: 1rem; }
    .source-badge {
        display: inline-block;
        background: var(--green-dark);
        color: white;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 20px;
        margin-right: 6px;
        letter-spacing: 0.5px;
    }
    .source-card {
        background: white;
        border-left: 4px solid var(--green-mid);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    }
    .source-score { color: var(--green-dark); font-weight: 700; font-size: 0.85rem; }
    .source-text  { color: #444; font-size: 0.82rem; margin-top: 4px; line-height: 1.5; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1b4332 0%, #2d6a4f 100%);
    }
    [data-testid="stSidebar"] * { color: #d8f3dc !important; }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #b7e4c7 !important; font-weight: 700; }
    .doc-pill {
        background: rgba(116,198,157,0.15);
        border: 1px solid rgba(116,198,157,0.4);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.78rem;
        margin-bottom: 6px;
        display: block;
    }
    [data-testid="stChatMessage"] { border-radius: 12px !important; margin-bottom: 0.5rem; }
    [data-testid="stChatInput"] textarea { border: 2px solid var(--green-light) !important; border-radius: 12px !important; }
    [data-testid="stChatInput"] textarea:focus { border-color: var(--green-dark) !important; }
    hr { border-color: var(--green-pale) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌱 Éco Textile RAG")
    st.markdown("---")

    st.markdown("### 📚 Sources indexées")
    for pdf in PDF_FILES:
        st.markdown(f'<span class="doc-pill">📄 {pdf["label"]}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Paramètres de recherche")
    top_k = st.slider("Nombre de chunks récupérés", min_value=1, max_value=10, value=5,
                       help="Plus de chunks = contexte plus riche, mais plus lent")
    show_sources = st.toggle("Afficher les sources", value=True)

    st.markdown("---")
    st.markdown("### 🛠️ Stack technique")
    st.markdown("""
    🔍 **Qdrant** — Vector DB  
    🧠 **BAAI/bge-m3** — Embeddings  
    🤖 **Llama 3.3 70B** — LLM  
    🎨 **Streamlit** — UI  
    """)
    st.markdown("---")

    if st.button("🗑️ Vider l'historique", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    with st.expander("🔄 Ré-ingestion des données"):
        st.warning("Ceci supprimera et recréera la collection Qdrant avec les 3 PDFs.", icon="⚠️")
        if st.button("▶️ Lancer l'ingestion", use_container_width=True, type="primary"):
            with st.spinner("Ingestion en cours…"):
                try:
                    from utils import extract_texts_from_pdfs, build_chunks_from_pdfs, save_chunks
                    from embeddings import embed_documents, save_embeddings
                    from qdrant_db import reset_collection, upload_vectors, get_collection_info
                    from config import PDF_FILES as PDFS

                    pdf_texts = extract_texts_from_pdfs(PDFS)
                    chunks = build_chunks_from_pdfs(pdf_texts)
                    save_chunks(chunks)

                    texts = [c["chunk"] for c in chunks]
                    vectors = embed_documents(texts)
                    save_embeddings(vectors)

                    reset_collection()
                    upload_vectors(chunks, vectors)
                    info = get_collection_info()

                    st.success(f"✅ Ingestion terminée ! {info.points_count} vecteurs chargés.")
                except Exception as e:
                    st.error(f"❌ Erreur : {e}")

# ─── Header Hero ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🌱 Éco Textile RAG</h1>
    <p>Votre assistant IA sur le textile durable — GOTS 7.0 · New Textiles Economy · Éco-conception</p>
</div>
""", unsafe_allow_html=True)

# ─── Initialisation session state ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─── Message de bienvenue ─────────────────────────────────────────────────────
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("""
Bonjour ! 👋 Je suis votre assistant spécialisé en **éco-textile et mode durable**.

Je m'appuie sur **3 documents de référence** :
- 📗 **GOTS 7.0** — Standard biologique pour les textiles
- 📘 **A New Textiles Economy** — Économie circulaire et circularité
- 📙 **Éco-conception Textiles** — Méthodes et pratiques

Quelques questions pour commencer :
> 🌱 *Quelles fibres sont certifiables GOTS ?*  
> ♻️ *Comment fonctionne l'économie circulaire dans la mode ?*  
> 🧪 *Quels produits chimiques sont interdits dans le textile bio ?*  
> 👗 *Comment éco-concevoir un vêtement dès le départ ?*  
> 🏷️ *Quelles sont les exigences d'étiquetage GOTS ?*
        """)

# ─── Historique des messages ──────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} source(s) utilisée(s)"):
                for src in msg["sources"]:
                    badge = src.get("source_short", "?")
                    label = src.get("source", "Document")
                    score = src.get("score", 0)
                    text_preview = src["chunk"][:280] + "…"
                    st.markdown(f"""
<div class="source-card">
    <span class="source-badge">{badge}</span>
    <span class="source-score">Score {score:.3f}</span> — <em>{label}</em>
    <div class="source-text">{text_preview}</div>
</div>
""", unsafe_allow_html=True)

# ─── Input utilisateur ────────────────────────────────────────────────────────
question = st.chat_input("💬 Posez votre question sur l'éco-textile…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Recherche dans les documents en cours…"):
            try:
                result = ask_question(question, top_k=top_k)
                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                if show_sources and sources:
                    with st.expander(f"📚 {len(sources)} source(s) utilisée(s)"):
                        for src in sources:
                            badge = src.get("source_short", "?")
                            label = src.get("source", "Document")
                            score = src.get("score", 0)
                            text_preview = src["chunk"][:280] + "…"
                            st.markdown(f"""
<div class="source-card">
    <span class="source-badge">{badge}</span>
    <span class="source-score">Score {score:.3f}</span> — <em>{label}</em>
    <div class="source-text">{text_preview}</div>
</div>
""", unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                error_msg = f"❌ Erreur : {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })

