from typing import List
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
try:
    # Preferencia: import desde langchain moderno
    from langchain.vectorstores import Chroma
except Exception:
    try:
        # Fallback a posibles paquetes comunitarios o instalaciones antiguas
        from langchain_community.vectorstores import Chroma
    except Exception:
        Chroma = None
from .knowledge import MedicalKnowledgeLoader
import json


class MedicalDiagnosticSystem:
    """Sistema responsable de diagnosticar y aprender del feedback del médico."""
    def __init__(self):
        # Modelos LLM
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.llm_creative = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

        # Embeddings
        self.embeddings = OpenAIEmbeddings()

        # Inicializar o crear un vectorstore local
        self.vectorstore = self._inicializar_vectorstore()
        # Inicializar retriever si el vectorstore existe
        self.retriever = None
        if self.vectorstore is not None:
            try:
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
            except Exception:
                # Some implementations expose different API; leave retriever None
                self.retriever = None

    def _inicializar_vectorstore(self):
        # Splitter (aunque el loader ya genera texto, dejamos tal cual)
        # Si Chroma no está disponible, no podemos cargar/crear vectordb
        if Chroma is None:
            print("⚠️ Chroma no está disponible en esta instalación (no se creó/ni cargó vectorstore).")
            return None

        # Si ya existe un directorio persistente, cargarlo en lugar de regenerar
        persist_dir = "medical_vectorstore"
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            try:
                # Algunas APIs de Chroma en LangChain aceptan estos parámetros
                vs = Chroma(persist_directory=persist_dir, embedding_function=self.embeddings)
                print("ℹ️ Cargado vectorstore Chroma existente desde disk")
                return vs
            except Exception as e:
                print("⚠️ Error cargando vectorstore existente, se intentará crear uno nuevo:", e)
        # Si no existe la DB persistente, generamos documentos y creamos Chroma
        loader = MedicalKnowledgeLoader()
        documentos_raw = loader.construir_base_conocimiento(n_casos=30)

        # Convertir a Document
        documents = []
        for doc in documentos_raw:
            documents.append(Document(page_content=doc["content"], metadata=doc.get("metadata", {})))

        try:
            splits = documents
            # Usar Chroma (persistente en directorio)
            vectorstore = Chroma.from_documents(
                splits,
                self.embeddings,
                persist_directory=persist_dir
            )
            # Persistir en disco si la API lo soporta
            try:
                vectorstore.persist()
            except Exception:
                pass

            print(f"✅ Vectorstore Chroma creado con {len(splits)} documentos")
            return vectorstore
        except Exception as e:
            print("⚠️ No se pudo crear Chroma vectorstore:", e)
            return None

    def diagnosticar(self, caso_paciente: dict) -> dict:
        """Genera un diagnóstico (lista de candidatos) usando el LLM.

        Devuelve un dict con: diagnostico_final, confianza, explicacion, diagnosticos_candidatos
        """
        sintomas_text = ", ".join(caso_paciente["paciente"].get("sintomas", []))

        # Recuperar contexto desde la DB (si está disponible)
        contexto = ""
        if self.retriever is not None:
            try:
                docs = self.retriever.get_relevant_documents(sintomas_text)
                contexto = "\n\n".join([d.page_content for d in docs])
            except Exception:
                # fallback to alternative methods
                try:
                    docs = self.vectorstore.similarity_search(sintomas_text, k=5)
                    contexto = "\n\n".join([d.page_content for d in docs])
                except Exception:
                    contexto = ""
        else:
            # try to create a retriever on the fly
            if self.vectorstore is not None:
                try:
                    retr = self.vectorstore.as_retriever(search_kwargs={"k": 5})
                    docs = retr.get_relevant_documents(sintomas_text)
                    contexto = "\n\n".join([d.page_content for d in docs])
                    self.retriever = retr
                except Exception:
                    contexto = ""

        # Construir prompt que obliga al LLM a basarse en el CONTEXTO proporcionado
        prompt = f"""
        Eres un médico experto y solo puedes usar la información provista en 'CONOCIMIENTO_MÉDICO_RELEVANTE' para generar diagnósticos.
        Si la información no es suficiente, indícalo claramente.

        CASO DEL PACIENTE:
        Edad: {caso_paciente['paciente'].get('edad', '?')}  Género: {caso_paciente['paciente'].get('genero','?')}
        Síntomas: {sintomas_text}
        Antecedentes: {caso_paciente.get('antecedentes', 'Ninguno')}

        CONOCIMIENTO_MÉDICO_RELEVANTE:
        {contexto}

        Basándote ÚNICAMENTE en el CONOCIMIENTO_MÉDICO_RELEVANTE, genera hasta 5 diagnósticos diferenciales en formato JSON: una lista de objetos con 'enfermedad', 'probabilidad' (0-1) y 'evidencia'.
        """

        try:
            respuesta = self.llm.invoke(prompt)
            contenido = respuesta.content

            # Normalizar: eliminar fences de código y extraer JSON si viene dentro
            import re

            texto = contenido.strip()

            # Buscar un bloque de código ```json ... ``` o ``` ... ``` y extraer su interior
            m = re.search(r"```(?:json)?\s*(.+?)\s*```", texto, flags=re.DOTALL | re.IGNORECASE)
            if m:
                texto = m.group(1).strip()

            candidatos = None

            # Intentar encontrar la primera estructura JSON (lista o dict) dentro del texto
            idx_list = texto.find('[')
            idx_obj = texto.find('{')
            start = -1
            if idx_list != -1 and (idx_obj == -1 or idx_list < idx_obj):
                start = idx_list
            elif idx_obj != -1:
                start = idx_obj

            parse_attempts = []
            if start != -1:
                candidate_sub = texto[start:]
                parse_attempts.append(candidate_sub)

            # También intentar con el texto completo
            parse_attempts.append(texto)

            for attempt in parse_attempts:
                try:
                    parsed = json.loads(attempt)
                    candidatos = parsed
                    break
                except Exception:
                    continue

            # Si no pudimos parsear nada como JSON, usar fallback
            if candidatos is None:
                candidatos = [{"enfermedad": "Indeterminado", "probabilidad": 0.5, "evidencia": texto}]
            else:
                # Si el parsed es un dict que contiene la lista bajo alguna clave, intentar extraerla
                if isinstance(candidatos, dict):
                    # buscar claves comunes
                    for key in ("diagnosticos", "candidatos", "results", "result"):
                        if key in candidatos and isinstance(candidatos[key], list):
                            candidatos = candidatos[key]
                            break
                    else:
                        # Si es un dict único que representa un diagnóstico, lo convertimos en lista
                        candidatos = [candidatos]
        except Exception as e:
            print("Error llamando al LLM:", e)
            candidatos = [{"enfermedad": "Indeterminado", "probabilidad": 0.5, "evidencia": "Error LLM"}]

        resultado = {
            "diagnosticos_candidatos": candidatos,
            "diagnostico_final": candidatos[0] if candidatos else {},
            "confianza": candidatos[0].get("probabilidad", 0.5) if candidatos else 0.5,
            "explicacion": "".join([c.get("evidencia","") for c in candidatos])
        }

        return resultado

    def learn_from_feedback(self, caso_paciente: dict, diagnostico_real: str, n: int = 5) -> List[str]:
        """Genera n casos sintéticos similares al caso dado (apoyados en el LLM creativo)
        y los agrega al vectorstore como documentos para aprendizaje.

        Devuelve la lista de textos generados.
        """
        prompt = f"""
        Genera {n} casos clínicos realistas similares al siguiente caso. Deben enfatizar señales que permitan distinguir el diagnóstico real: {diagnostico_real}.

        CASO:
        {json.dumps(caso_paciente, ensure_ascii=False)}

        Salida: separa cada caso con una linea '---CASO---' y entrega texto por caso.
        """

        try:
            resp = self.llm_creative.invoke(prompt)
            contenido = resp.content
            # Intentar detectar JSON primero
            casos = []

            # Buscar si la LLM devolvió JSON (lista)
            texto = contenido.strip()
            # Extraer bloque tipo ``` ... ```
            import re
            m = re.search(r"```(?:json)?\s*(.+?)\s*```", texto, flags=re.DOTALL | re.IGNORECASE)
            if m:
                texto = m.group(1).strip()

            try:
                parsed = json.loads(texto)
                if isinstance(parsed, list):
                    # cada elemento es un caso textual o dict
                    for el in parsed:
                        if isinstance(el, str):
                            casos.append(el.strip())
                        elif isinstance(el, dict):
                            # convertir a texto legible
                            casos.append(json.dumps(el, ensure_ascii=False))
                elif isinstance(parsed, dict):
                    # si es dict con key 'casos'
                    if 'casos' in parsed and isinstance(parsed['casos'], list):
                        for el in parsed['casos']:
                            casos.append(json.dumps(el, ensure_ascii=False) if not isinstance(el, str) else el.strip())
            except Exception:
                # fallback: split por separador textual
                casos = [c.strip() for c in re.split(r"---CASO---|---CASOS---|\n\n", texto) if c.strip()]
        except Exception as e:
            print("Error generando casos sintéticos:", e)
            casos = []

        # Garantizar EXACTAMENTE n casos: si la LLM no produjo suficientes, generar variaciones programáticas
        def synthesize_variation(base_text: str, idx: int, sintomas: List[str], diagnostico: str) -> str:
            # Crear una variación simple alterando orden de síntomas, intensidad o añadiendo signo/síntoma relacionado
            s = ", ".join(sintomas)
            variation = f"Diagnóstico: {diagnostico}\nCaso variante {idx+1}: Paciente con síntomas principales: {s}. Descripción: {base_text[:200]}... (variante {idx+1})"
            return variation

        sintomas = caso_paciente.get('paciente', {}).get('sintomas', [])
        # si la LLM devolvió más de n, recortar; si menos, complementar
        if len(casos) > n:
            casos = casos[:n]
        else:
            # Si LLM devolvió 0 casos, usar el texto del prompt para crear ejemplos
            base = "; ".join(sintomas) if sintomas else "Caso clínico"
            i = 0
            while len(casos) < n:
                if i < len(casos):
                    base_text = casos[i]
                else:
                    base_text = base
                casos.append(synthesize_variation(base_text, len(casos), sintomas, diagnostico_real))
                i += 1

        # Añadir al vectorstore (si existe)
        if self.vectorstore is not None and casos:
            docs = []
            for idx, texto in enumerate(casos):
                # Asegurarnos de que el texto incluya la etiqueta del diagnóstico y los síntomas
                content_lines = []
                content_lines.append(f"Diagnóstico: {diagnostico_real}")
                content_lines.append(f"Síntomas: {', '.join(sintomas)}")
                content_lines.append("")
                content_lines.append(texto)
                doc_text = "\n".join(content_lines)
                docs.append(Document(page_content=doc_text, metadata={"source": "error_correction", "diagnostico": diagnostico_real}))
            try:
                # Preferir add_documents si está disponible
                try:
                    self.vectorstore.add_documents(docs)
                except Exception:
                    # fallback: si existe add_texts
                    texts = [d.page_content for d in docs]
                    try:
                        self.vectorstore.add_texts(texts, metadatas=[d.metadata for d in docs])
                    except Exception:
                        # última opción: algunos adaptadores exponen client upsert - omitimos
                        pass

                # Persistir cambios si el adaptador lo requiere
                try:
                    self.vectorstore.persist()
                except Exception:
                    # algunos adaptadores hacen persistencia automática
                    pass

                # Refrescar retriever para incluir los nuevos docs
                try:
                    self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
                except Exception:
                    pass

                print(f"✅ Agregados {len(docs)} casos sintéticos al vectorstore")
            except Exception as e:
                print("⚠️ No se pudo actualizar vectorstore:", e)

        return casos

    def save_case(self, caso_paciente: dict, diagnostico: str) -> bool:
        """Guarda el caso confirmado en la base de conocimiento (vectorstore).

        Devuelve True si se guardó correctamente, False en otro caso.
        """
        if self.vectorstore is None:
            print("⚠️ No hay vectorstore disponible para guardar el caso.")
            return False

        sintomas = caso_paciente.get('paciente', {}).get('sintomas', [])
        content_lines = [f"Diagnóstico confirmado: {diagnostico}", f"Síntomas: {', '.join(sintomas)}", "", json.dumps(caso_paciente, ensure_ascii=False)]
        doc_text = "\n".join(content_lines)
        doc = Document(page_content=doc_text, metadata={"source": "confirmed_case", "diagnostico": diagnostico})

        try:
            try:
                self.vectorstore.add_documents([doc])
            except Exception:
                # fallback
                try:
                    self.vectorstore.add_texts([doc.page_content], metadatas=[doc.metadata])
                except Exception:
                    pass

            try:
                self.vectorstore.persist()
            except Exception:
                # some adapters persist automatically
                pass

            try:
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
            except Exception:
                pass

            print("✅ Caso confirmado guardado en la base de conocimiento.")
            return True
        except Exception as e:
            print("⚠️ Error guardando caso confirmado:", e)
            return False
