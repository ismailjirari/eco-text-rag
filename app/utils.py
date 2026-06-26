# utils.py :

import re
import json
import os
from config import CHUNK_SIZE, CHUNK_OVERLAP, PROCESSED_DATA_PATH


def clean_text(text: str) -> str:
    """Nettoie le texte brut extrait du PDF en préservant la structure."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'[^\w\s\.,;:!?()\-–/\'\\"àâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ\n]', '', text)
    text = text.strip()
    return text


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Découpe le texte en chunks avec chevauchement (basé sur les mots)."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def save_chunks(chunks: list[dict], filename: str = "chunks.json"):
    """Sauvegarde les chunks dans le dossier processed."""
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    filepath = os.path.join(PROCESSED_DATA_PATH, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"✅ {len(chunks)} chunks sauvegardés dans {filepath}")


def load_chunks(filename: str = "chunks.json") -> list[dict]:
    """Charge les chunks depuis le dossier processed."""
    filepath = os.path.join(PROCESSED_DATA_PATH, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier non trouvé : {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"✅ {len(chunks)} chunks chargés depuis {filepath}")
    return chunks


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrait le texte d'un fichier PDF page par page."""
    try:
        import pypdf
        text = ""
        with open(pdf_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            if reader.is_encrypted:
                reader.decrypt("")
            total_pages = len(reader.pages)
            print(f"   → {total_pages} pages détectées")
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text += f"\n\n--- Page {page_num + 1} ---\n\n"
                    text += page_text.strip()
        print(f"✅ Texte extrait : {len(text)} caractères, {len(text.split())} mots")
        return text
    except Exception as e:
        print(f"❌ Erreur extraction PDF : {e}")
        raise


def extract_texts_from_pdfs(pdf_files: list[dict]) -> list[dict]:
    """
    Extrait le texte de plusieurs PDFs et retourne une liste de dicts
    avec 'text', 'label' et 'short' pour chaque PDF.
    """
    results = []
    for pdf_info in pdf_files:
        path = pdf_info["path"]
        label = pdf_info.get("label", os.path.basename(path))
        short = pdf_info.get("short", label[:10])

        if not os.path.exists(path):
            print(f"⚠️  PDF introuvable : {path}")
            continue

        print(f"\n📄 Lecture de : {label}")
        text = extract_text_from_pdf(path)
        results.append({
            "text": text,
            "label": label,
            "short": short,
            "path": path
        })
    return results


def build_chunks_from_pdfs(pdf_texts: list[dict]) -> list[dict]:
    """
    Découpe plusieurs PDFs en chunks, chaque chunk mémorise sa source.
    """
    all_chunks = []
    chunk_id = 0
    for pdf in pdf_texts:
        clean = clean_text(pdf["text"])
        raw_chunks = split_text(clean)
        for c in raw_chunks:
            all_chunks.append({
                "chunk": c,
                "id": chunk_id,
                "source": pdf["label"],
                "source_short": pdf["short"]
            })
            chunk_id += 1
        print(f"   → {len(raw_chunks)} chunks depuis '{pdf['label']}'")
    print(f"✅ Total : {len(all_chunks)} chunks créés")
    return all_chunks
