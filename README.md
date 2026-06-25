# 🌱 Éco Textile RAG

An AI-powered question-answering assistant specialized in sustainable textiles and eco-fashion. Ask questions in natural language and get precise, sourced answers drawn from three reference documents.

---

## 📌 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that lets you query a curated knowledge base of eco-textile documents. Instead of relying on general AI knowledge, every answer is grounded in the actual content of the indexed PDFs, with sources cited.

**Knowledge base — 3 reference documents:**

| Label | Content |
|---|---|
| 📗 GOTS 7.0 (FR) | Global Organic Textile Standard — certification, traceability, prohibited substances |
| 📘 A New Textiles Economy | Ellen MacArthur Foundation report on circular economy in fashion |
| 📙 Éco-conception Textiles & Habillement | Eco-design methods and practices for textile products |

---

## 🏗️ Architecture

```
User question
     │
     ▼
Embedding (BAAI/bge-m3)
     │
     ▼
Vector search (Qdrant)
     │  top-k relevant chunks
     ▼
Prompt construction
     │  question + context
     ▼
LLM generation (Llama 3.3 70B via OpenRouter)
     │
     ▼
Answer + cited sources
```

**Stack:**

- **Embeddings** — `BAAI/bge-m3` via `sentence-transformers` (1024-dim vectors)
- **Vector database** — Qdrant Cloud (cosine similarity)
- **LLM** — Llama 3.3 70B Instruct via OpenRouter API
- **UI** — Streamlit

---

## 📁 Project Structure

```
eco-textile-rag/
└── rag-tex/
    ├── app/
    │   ├── config.py           # API keys, model names, paths
    │   ├── embeddings.py       # Embedding model (lazy-loaded)
    │   ├── llm_openrouter.py   # OpenRouter API calls
    │   ├── main.py             # CLI: ingest or chat mode
    │   ├── prompt_template.py  # System prompt and RAG prompt builder
    │   ├── qdrant_db.py        # Qdrant client: create, upload, search
    │   ├── rag_pipeline.py     # Full RAG pipeline: retrieve → build → generate
    │   └── utils.py            # PDF extraction, chunking, save/load
    ├── data/
    │   ├── raw/                # Source PDF files
    │   ├── processed/          # chunks.json (text chunks)
    │   └── embeddings/         # embeddings.json (cached vectors)
    ├── streamlit_app/
    │   └── streamlit_app.py    # Web UI
    ├── .env                    # API keys (not committed)
    ├── requirements.txt
    └── README.md
```

---

## ⚙️ Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd rag-tex
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file at the root of `rag-tex/`:

```env
QDRANT_URL=https://<your-cluster>.qdrant.io
QDRANT_API_KEY=<your-qdrant-api-key>
OPENROUTER_API_KEY=<your-openrouter-api-key>
```

- **Qdrant** — create a free cluster at [cloud.qdrant.io](https://cloud.qdrant.io)
- **OpenRouter** — get an API key at [openrouter.ai](https://openrouter.ai)

### 4. Add your PDF files

Place the three PDFs in `data/raw/`:

```
data/raw/
├── GOTS_7.0__FR__signed.pdf
├── A New Textiles Economy - Summary of findings.pdf
└── eco-conception-des-produits-textiles-habillement.pdf
```

---

## 🚀 Usage

### Step 1 — Ingest documents (run once)

This extracts text, chunks it, generates embeddings, and uploads vectors to Qdrant.

```bash
cd rag-tex
python app/main.py ingest
```

To re-ingest without resetting the Qdrant collection:

```bash
python app/main.py ingest --no-reset
```

### Step 2 — Launch the web UI

```bash
streamlit run streamlit_app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Alternative — CLI chat mode

```bash
python app/main.py
```

---

## 🖥️ Streamlit UI Features

- **Chat interface** — conversational Q&A with history
- **Source panel** — expandable per-answer source cards showing the exact chunk, document label, and relevance score
- **Top-k slider** — control how many chunks are retrieved (1–10)
- **Re-ingestion panel** — trigger a full pipeline re-run from the sidebar without leaving the app
- **Clear history** — reset the conversation in one click

---

## 🔧 Configuration

All parameters are in `app/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Sentence-transformers model |
| `EMBEDDING_DIMENSION` | `1024` | Vector size |
| `LLM_MODEL` | `meta-llama/llama-3.3-70b-instruct` | Model via OpenRouter |
| `CHUNK_SIZE` | `100` | Words per chunk |
| `CHUNK_OVERLAP` | `20` | Overlap between consecutive chunks |
| `TOP_K` | `5` | Chunks retrieved per query |
| `COLLECTION_NAME` | `eco_textile_documents` | Qdrant collection name |

---

## 🌐 Deployment (Streamlit Cloud)

1. Push the repo to GitHub (make sure `.env` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set the main file path to `rag-tex/streamlit_app/streamlit_app.py`
4. Add your secrets under **Settings → Secrets**:

```toml
QDRANT_URL = "https://..."
QDRANT_API_KEY = "..."
OPENROUTER_API_KEY = "..."
```

The app reads secrets from both `.env` (local) and `st.secrets` (cloud) automatically.

---

## 📦 Dependencies

```
streamlit
sentence-transformers
qdrant-client
pypdf
python-dotenv
requests
```

Install all with:

```bash
pip install -r requirements.txt
```

---

## 📄 License

MIT
