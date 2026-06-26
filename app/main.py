# main.py :

import os
import sys
from config import PDF_FILES, COLLECTION_NAME
from utils import extract_texts_from_pdfs, build_chunks_from_pdfs, save_chunks, load_chunks
from embeddings import embed_documents, save_embeddings, load_embeddings
from qdrant_db import reset_collection, upload_vectors, get_collection_info
from rag_pipeline import ask_question


def ingest_pipeline(force_reset: bool = True):
    """
    Pipeline complet d'ingestion : 3 PDFs → chunks → embeddings → Qdrant.
    Si force_reset=True, supprime et recrée la collection Qdrant.
    """
    print("\n" + "="*60)
    print("🚀 DÉMARRAGE PIPELINE D'INGESTION — ÉCO TEXTILE RAG")
    print("="*60)

    # 1. Extraction texte des 3 PDFs
    print(f"\n📄 Étape 1 : Extraction du texte des {len(PDF_FILES)} PDFs...")
    pdf_texts = extract_texts_from_pdfs(PDF_FILES)
    if not pdf_texts:
        print("❌ Aucun PDF chargé. Vérifiez les chemins dans config.py.")
        return

    # 2. Chunking multi-source
    print("\n✂️  Étape 2 : Découpage en chunks (avec métadonnée source)...")
    chunks = build_chunks_from_pdfs(pdf_texts)

    # 3. Sauvegarde chunks
    save_chunks(chunks)

    # 4. Génération embeddings
    print("\n🔢 Étape 3 : Génération des embeddings...")
    texts = [c["chunk"] for c in chunks]
    vectors = embed_documents(texts)
    save_embeddings(vectors)

    # 5. Reset collection Qdrant (supprime l'ancienne si elle a un problème)
    print(f"\n🗄️  Étape 4 : Reset collection Qdrant '{COLLECTION_NAME}'...")
    if force_reset:
        reset_collection()
    else:
        from qdrant_db import create_collection
        create_collection()

    # 6. Upload vers Qdrant
    print("\n⬆️  Étape 5 : Upload vers Qdrant...")
    upload_vectors(chunks, vectors)

    # 7. Vérification
    print("\n📊 Étape 6 : Vérification...")
    get_collection_info()

    print("\n" + "="*60)
    print("✅ INGESTION TERMINÉE AVEC SUCCÈS !")
    print("="*60)


def chat_mode():
    """Mode chat interactif en ligne de commande."""
    print("\n" + "="*60)
    print("💬 MODE CHAT — ÉCO TEXTILE RAG")
    print("   Tapez 'quit' ou 'exit' pour quitter")
    print("="*60 + "\n")

    while True:
        try:
            question = input("❓ Question : ").strip()
            if not question:
                continue
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Au revoir !")
                break

            print("\n⏳ Traitement en cours...\n")
            result = ask_question(question)

            print("📝 RÉPONSE :")
            print("-" * 40)
            print(result["answer"])
            print("\n📚 SOURCES UTILISÉES :")
            for i, src in enumerate(result["sources"], 1):
                print(f"  [{i}] Score: {src['score']:.3f} | [{src.get('source_short','?')}] {src['chunk'][:100]}...")
            print("\n" + "="*60 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Au revoir !")
            break
        except Exception as e:
            print(f"❌ Erreur : {e}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        # Passe --no-reset pour ne pas supprimer la collection existante
        force = "--no-reset" not in sys.argv
        ingest_pipeline(force_reset=force)
    else:
        chat_mode()
