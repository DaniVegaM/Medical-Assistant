from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json


class MedicalKnowledgeLoader:
    """Carga y genera conocimiento médico (casos sintéticos)"""
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )

    def cargar_casos_sinteticos(self, n_casos=50):
        """Genera casos sintéticos con el LLM creativo."""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

        enfermedades = [
            "Diabetes tipo 2", "Hipertensión", "Neumonía",
            "COVID-19", "Influenza", "Dengue", "Apendicitis",
            "Gastroenteritis", "Migraña", "Asma"
        ]

        # Interpretar n_casos como número TOTAL de casos a generar.
        # Repartimos los casos entre las enfermedades para evitar generar
        # n_casos por cada enfermedad (lo que produciría demasiados documentos).
        casos = []
        m = len(enfermedades)
        base = max(1, n_casos // m)
        resto = max(0, n_casos - base * m)

        # Crear una lista con el número de casos por enfermedad (distribuyendo el resto)
        casos_por_enfermedad_list = [base + (1 if i < resto else 0) for i in range(m)]

        for enfermedad, casos_por_enfermedad in zip(enfermedades, casos_por_enfermedad_list):
            for _ in range(casos_por_enfermedad):
                prompt = f"""
                Genera un caso clínico realista de {enfermedad}.

                Incluye: presentación del paciente, síntomas, progresión, exámenes y diagnóstico.
                Formato: Texto descriptivo (máximo 200 palabras).
                """
                try:
                    resp = llm.invoke(prompt)
                    casos.append(resp.content)
                except Exception:
                    # Si falla una llamada, seguimos con las demás
                    continue

        # Devolver como lista de dicts compatibles con el resto del sistema
        documentos = [{"content": c, "metadata": {"source": "synthetic"}} for c in casos]
        return documentos

    def construir_base_conocimiento(self, n_casos=50):
        """Construye y devuelve una lista de documentos (dicts) con conocimiento."""
        print("📚 Construyendo base de conocimiento (casos sintéticos)...")
        documentos = self.cargar_casos_sinteticos(n_casos=n_casos)
        print(f"✅ Generados {len(documentos)} documentos sintéticos")
        return documentos
