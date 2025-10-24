"""Script para inicializar la base vectorial Chroma una sola vez.

Este script generará un número pequeño de casos sintéticos (configurable)
usando `MedicalKnowledgeLoader` y persistirá el vectorstore en
`medical_vectorstore/`. Consume llamadas al LLM — reduce `n_casos` si
quieres minimizar consumo.
"""
import os
from medical.knowledge import MedicalKnowledgeLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
try:
    from langchain.vectorstores import Chroma
except Exception:
    try:
        from langchain_community.vectorstores import Chroma
    except Exception:
        Chroma = None


def init_db(n_casos: int = 50, persist_dir: str = "medical_vectorstore"):
    print(f"Inicializando DB Chroma con {n_casos} casos sintéticos (persist: {persist_dir})")

    # Verificar que la clave de OpenAI está configurada para evitar fallos al llamar al LLM
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY no encontrada en el entorno.")
        print("Crea un archivo .env en la raíz del proyecto con: OPENAI_API_KEY=tu_api_key")
        print("O exporta la variable en tu shell: export OPENAI_API_KEY=tu_api_key")
        return

    if Chroma is None:
        print("⚠️ Chroma no está disponible en esta instalación. No se creó la DB.")
        return

    loader = MedicalKnowledgeLoader()
    documentos = loader.construir_base_conocimiento(n_casos=n_casos)

    # Convertir a Document
    docs = [Document(page_content=d["content"], metadata=d.get("metadata", {})) for d in documentos]

    embeddings = OpenAIEmbeddings()

    # Crear/llenar Chroma persistente
    try:
        chroma_vs = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
        try:
            chroma_vs.persist()
        except Exception:
            pass
        print("✅ DB inicializada y persistida en:", persist_dir)
    except Exception as e:
        print("⚠️ Er ror creando Chroma desde documentos:", e)


if __name__ == "__main__":
    # Agregar soporte a CLI para pasar n_casos desde la línea de comandos
    import argparse    

    parser = argparse.ArgumentParser(description="Inicializar DB Chroma con casos sintéticos")
    parser.add_argument("--n-casos", type=int, default=50, help="Número total de casos sintéticos a generar (por defecto 50)")
    parser.add_argument("--persist-dir", type=str, default="medical_vectorstore", help="Directorio donde persistir la DB Chroma")
    args = parser.parse_args()

    init_db(n_casos=args.n_casos, persist_dir=args.persist_dir)
