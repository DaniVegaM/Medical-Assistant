

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import TypedDict, List, Annotated
import operator
import json

# Configuración
from dotenv import load_dotenv
load_dotenv()

class MedicalKnowledgeLoader:
    """
    Carga conocimiento médico de múltiples fuentes
    """
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.documentos = []
    
    def cargar_casos_sinteticos(self, n_casos=1000):
        """
        Genera casos sintéticos usando GPT-4
        """
        from langchain_openai import ChatOpenAI
        
        print(f"  - Generando {n_casos} casos sintéticos...")
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        
        enfermedades = [
            "Diabetes tipo 2", "Hipertensión", "Neumonía", 
            "COVID-19", "Influenza", "Dengue", "Apendicitis",
            "Gastroenteritis", "Migraña", "Asma"
        ]
        
        casos = []
        casos_por_enfermedad = n_casos // len(enfermedades)
        
        for enfermedad in enfermedades:
            print(f"    Generando casos de {enfermedad}...")
            
            for i in range(casos_por_enfermedad):
                prompt = f"""
                Genera un caso clínico realista de {enfermedad}.
                
                Incluye:
                1. Presentación del paciente (edad, género)
                2. Síntomas principales y duración
                3. Progresión de los síntomas
                4. Resultados de exámenes físicos
                5. Diagnóstico: {enfermedad}
                
                Formato: Texto descriptivo claro y conciso (máximo 200 palabras).
                """
                
                try:
                    caso = llm.invoke(prompt)
                    casos.append(caso.content)
                except Exception as e:
                    print(f"    Error generando caso: {e}")
                    continue
        
        return casos
    
    def construir_base_conocimiento(self):
        """
        Construye la base de conocimiento completa
        """
        print("📚 Cargando conocimiento médico...")
        
        # Por ahora solo generamos casos sintéticos
        # Puedes agregar más fuentes después
        print("  - Generando casos sintéticos...")
        casos = self.cargar_casos_sinteticos(n_casos=50)  # Empezar con pocos para probar
        
        self.documentos.extend([
            {"content": caso, "metadata": {"source": "synthetic"}}
            for caso in casos
        ])
        
        print(f"✅ Total documentos cargados: {len(self.documentos)}")
        
        return self.documentos
    
class MedicalDiagnosticState(TypedDict):
    """
    Estado que se pasa entre nodos del grafo
    """
    caso_paciente: dict
    sintomas_analizados: dict
    diagnosticos_candidatos: List[dict]
    incertidumbres: List[dict]
    diagnostico_final: dict
    confianza: float
    necesita_revision: bool
    explicacion: str
    casos_similares: List[dict]
    iteracion: Annotated[int, operator.add]


class MedicalDiagnosticSystem:
    """
    Sistema completo implementado con LangChain y LangGraph
    """
    def __init__(self):
        # LLM
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.llm_creative = ChatOpenAI(model="gpt-4o", temperature=0.7)
        
        # Embeddings
        self.embeddings = OpenAIEmbeddings()
        
        # Cargar conocimiento
        self.vectorstore = self._inicializar_vectorstore()
        self.retriever = self._crear_ensemble_retriever()
        
        # Grafo de LangGraph
        self.graph = self._construir_grafo()
        
        # Historial
        self.casos_historicos = []
        self.uncertainty_graph = {}
    
    def _inicializar_vectorstore(self):
        """
        Inicializa el vectorstore con conocimiento médico
        """
        print("🔧 Inicializando vectorstore...")
        
        # Cargar documentos
        loader = MedicalKnowledgeLoader()
        documentos_raw = loader.construir_base_conocimiento()
        
        # Convertir a Documents de LangChain
        documents = []
        for doc in documentos_raw:
            if isinstance(doc, dict):
                documents.append(Document(
                    page_content=doc["content"],
                    metadata=doc.get("metadata", {})
                ))
            else:
                documents.append(doc)
        
        # Dividir en chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Crear vectorstore
        vectorstore = FAISS.from_documents(splits, self.embeddings)
        
        # Guardar localmente
        vectorstore.save_local("medical_vectorstore")
        
        print(f"✅ Vectorstore creado con {len(splits)} chunks")
        
        return vectorstore
    
    def _crear_ensemble_retriever(self):
        """
        Crea EnsembleRetriever con BM25 + Vector Search
        """
        # BM25 Retriever (búsqueda léxica)
        docs = [doc.page_content for doc in self.vectorstore.docstore._dict.values()]
        bm25_retriever = BM25Retriever.from_texts(docs)
        bm25_retriever.k = 5
        
        # Vector Retriever (búsqueda semántica)
        vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )
        
        # Ensemble
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.3, 0.7]  # Más peso a búsqueda semántica
        )
        
        return ensemble_retriever
    
    def _construir_grafo(self):
        """
        Construye el grafo de estados con LangGraph
        """
        workflow = StateGraph(MedicalDiagnosticState)
        
        # Nodos
        workflow.add_node("analizar_sintomas", self.analizar_sintomas)
        workflow.add_node("buscar_conocimiento", self.buscar_conocimiento)
        workflow.add_node("generar_diagnosticos", self.generar_diagnosticos)
        workflow.add_node("detectar_incertidumbres", self.detectar_incertidumbres)
        workflow.add_node("resolver_incertidumbres", self.resolver_incertidumbres)
        workflow.add_node("critica_medica", self.critica_medica)
        workflow.add_node("decision_final", self.decision_final)
        
        # Entry point
        workflow.set_entry_point("analizar_sintomas")
        
        # Flujo
        workflow.add_edge("analizar_sintomas", "buscar_conocimiento")
        workflow.add_edge("buscar_conocimiento", "generar_diagnosticos")
        workflow.add_edge("generar_diagnosticos", "detectar_incertidumbres")
        
        # Decisión condicional
        workflow.add_conditional_edges(
            "detectar_incertidumbres",
            self.debe_resolver_incertidumbres,
            {
                "resolver": "resolver_incertidumbres",
                "continuar": "critica_medica"
            }
        )
        
        workflow.add_edge("resolver_incertidumbres", "critica_medica")
        workflow.add_edge("critica_medica", "decision_final")
        workflow.add_edge("decision_final", END)
        
        return workflow.compile()
    
    def analizar_sintomas(self, state: MedicalDiagnosticState):
        """
        Nodo 1: Analiza los síntomas del paciente
        """
        caso = state["caso_paciente"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un médico experto analizando síntomas.
            Analiza los síntomas y clasifícalos por sistema corporal y urgencia."""),
            ("user", """
            Paciente: {edad} años, {genero}
            Síntomas: {sintomas}
            Duración: {duracion}
            Progresión: {progresion}
            
            Proporciona:
            1. Clasificación por sistemas (respiratorio, cardiovascular, etc)
            2. Nivel de urgencia (1-10)
            3. Síntomas principales vs secundarios
            4. Banderas rojas (red flags)
            
            Formato JSON.
            """)
        ])
        
        chain = prompt | self.llm
        
        resultado = chain.invoke({
            "edad": caso["paciente"]["edad"],
            "genero": caso["paciente"]["genero"],
            "sintomas": ", ".join(caso["paciente"]["sintomas"]),
            "duracion": caso["paciente"]["duracion"],
            "progresion": caso["paciente"]["progresion"]
        })
        
        # Parsear respuesta
        try:
            analisis = json.loads(resultado.content)
        except:
            analisis = {"error": "No se pudo parsear", "raw": resultado.content}
        
        return {
            **state,
            "sintomas_analizados": analisis,
            "iteracion": 1
        }
    
    def buscar_conocimiento(self, state: MedicalDiagnosticState):
        """
        Nodo 2: Busca en la base de conocimiento
        """
        sintomas = state["sintomas_analizados"]
        
        # Construir query para el retriever
        query = f"""
        Síntomas principales: {sintomas.get('principales', [])}
        Sistema afectado: {sintomas.get('sistema', 'desconocido')}
        Urgencia: {sintomas.get('urgencia', 5)}/10
        """
        
        # Buscar documentos relevantes
        docs = self.retriever.invoke(query)
        
        return {
            **state,
            "conocimiento_recuperado": docs
        }
    
    def generar_diagnosticos(self, state: MedicalDiagnosticState):
        """
        Nodo 3: Genera diagnósticos candidatos
        """
        caso = state["caso_paciente"]
        sintomas = state["sintomas_analizados"]
        docs = state.get("conocimiento_recuperado", [])
        
        # Preparar contexto
        contexto = "\n\n".join([doc.page_content for doc in docs[:5]])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un médico diagnosticador experto.
            Basándote en el conocimiento médico proporcionado y los síntomas,
            genera los 5 diagnósticos más probables."""),
            ("user", """
            SÍNTOMAS ANALIZADOS:
            {sintomas}
            
            CONOCIMIENTO MÉDICO RELEVANTE:
            {contexto}
            
            DATOS DEL PACIENTE:
            - Edad: {edad}
            - Género: {genero}
            - Síntomas: {sintomas_raw}
            - Duración: {duracion}
            - Resultados de laboratorio: {labs}
            
            Genera 5 diagnósticos diferenciales con:
            1. Nombre de la enfermedad
            2. Probabilidad (0-1)
            3. Evidencia que lo soporta
            4. Evidencia en contra
            5. Pruebas adicionales recomendadas
            
            Formato JSON: Lista de diagnósticos ordenados por probabilidad.
            """)
        ])
        
        chain = prompt | self.llm
        
        resultado = chain.invoke({
            "sintomas": json.dumps(sintomas, indent=2),
            "contexto": contexto,
            "edad": caso["paciente"]["edad"],
            "genero": caso["paciente"]["genero"],
            "sintomas_raw": ", ".join(caso["paciente"]["sintomas"]),
            "duracion": caso["paciente"]["duracion"],
            "labs": json.dumps(caso.get("resultados_lab", {}), indent=2)
        })
        
        try:
            diagnosticos = json.loads(resultado.content)
            if isinstance(diagnosticos, dict):
                diagnosticos = diagnosticos.get("diagnosticos", [])
        except:
            diagnosticos = []
        
        return {
            **state,
            "diagnosticos_candidatos": diagnosticos
        }
    
    def detectar_incertidumbres(self, state: MedicalDiagnosticState):
        """
        Nodo 4: Detecta puntos de incertidumbre
        """
        diagnosticos = state["diagnosticos_candidatos"]
        
        incertidumbres = []
        
        # 1. Diagnósticos con probabilidades similares
        if len(diagnosticos) >= 2:
            prob1 = diagnosticos[0].get("probabilidad", 0)
            prob2 = diagnosticos[1].get("probabilidad", 0)
            
            if abs(prob1 - prob2) < 0.15:
                incertidumbres.append({
                    "tipo": "diagnosticos_competitivos",
                    "descripcion": "Dos diagnósticos tienen probabilidades muy similares",
                    "diagnosticos": [diagnosticos[0], diagnosticos[1]],
                    "severidad": "alta"
                })
        
        # 2. Confianza baja en el principal
        if diagnosticos and diagnosticos[0].get("probabilidad", 0) < 0.6:
            incertidumbres.append({
                "tipo": "confianza_baja",
                "descripcion": "El diagnóstico principal tiene baja confianza",
                "diagnostico": diagnosticos[0],
                "severidad": "media"
            })
        
        # 3. Síntomas sin explicar
        sintomas = set(state["caso_paciente"]["paciente"]["sintomas"])
        sintomas_explicados = set()
        
        for diag in diagnosticos[:2]:
            evidencia = diag.get("evidencia_a_favor", "")
            for sintoma in sintomas:
                if sintoma.lower() in evidencia.lower():
                    sintomas_explicados.add(sintoma)
        
        sintomas_huerfanos = sintomas - sintomas_explicados
        if sintomas_huerfanos:
            incertidumbres.append({
                "tipo": "sintomas_inexplicados",
                "descripcion": "Algunos síntomas no se explican por el diagnóstico",
                "sintomas": list(sintomas_huerfanos),
                "severidad": "media"
            })
        
        return {
            **state,
            "incertidumbres": incertidumbres
        }
    
    def debe_resolver_incertidumbres(self, state: MedicalDiagnosticState):
        """
        Decisión condicional: ¿Hay incertidumbres que resolver?
        """
        incertidumbres = state.get("incertidumbres", [])
        
        # Si hay incertidumbres de severidad alta
        incertidumbres_altas = [
            i for i in incertidumbres 
            if i.get("severidad") == "alta"
        ]
        
        if incertidumbres_altas and state["iteracion"] < 3:
            return "resolver"
        return "continuar"
    
    def resolver_incertidumbres(self, state: MedicalDiagnosticState):
        """
        Nodo 5: Intenta resolver incertidumbres buscando casos similares
        """
        incertidumbres = state["incertidumbres"]
        caso = state["caso_paciente"]
        
        # Buscar casos históricos similares
        casos_similares = self.buscar_casos_similares(caso)
        
        if casos_similares:
            # Usar los casos históricos para resolver
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Analiza estos casos históricos similares
                para resolver las incertidumbres del diagnóstico actual."""),
                ("user", """
                CASO ACTUAL:
                {caso_actual}
                
                INCERTIDUMBRES:
                {incertidumbres}
                
                CASOS HISTÓRICOS SIMILARES:
                {casos_historicos}
                
                ¿Cómo se resolvieron casos similares?
                ¿Qué patrón común existe?
                ¿Cuál es el diagnóstico más probable basándose en estos casos?
                
                Formato JSON.
                """)
            ])
            
            chain = prompt | self.llm
            
            resolucion = chain.invoke({
                "caso_actual": json.dumps(caso, indent=2),
                "incertidumbres": json.dumps(incertidumbres, indent=2),
                "casos_historicos": json.dumps(casos_similares, indent=2)
            })
            
            try:
                resolucion_parsed = json.loads(resolucion.content)
            except:
                resolucion_parsed = {"resolucion": resolucion.content}
            
            return {
                **state,
                "resolucion_incertidumbres": resolucion_parsed,
                "casos_similares": casos_similares
            }
        
        return {**state, "casos_similares": []}
    
    def critica_medica(self, state: MedicalDiagnosticState):
        """
        Nodo 6: Agente crítico evalúa el diagnóstico
        """
        diagnosticos = state["diagnosticos_candidatos"]
        caso = state["caso_paciente"]
        
        if not diagnosticos:
            return {**state, "critica": {"error": "No hay diagnósticos"}}
        
        diagnostico_principal = diagnosticos[0]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un médico crítico experto.
            Tu trabajo es cuestionar y verificar diagnósticos.
            Sé riguroso pero constructivo."""),
            ("user", """
            DIAGNÓSTICO PROPUESTO:
            {diagnostico}
            
            CASO COMPLETO:
            {caso}
            
            DIAGNÓSTICOS ALTERNATIVOS:
            {alternativos}
            
            Evalúa críticamente:
            1. ¿La evidencia es suficiente?
            2. ¿Se consideraron los diferenciales adecuadamente?
            3. ¿Hay banderas rojas ignoradas?
            4. ¿Qué pruebas adicionales son críticas?
            5. Score de confianza (0-1)
            6. ¿Recomiendas cambiar el diagnóstico?
            
            Formato JSON con score y recomendaciones.
            """)
        ])
        
        chain = prompt | self.llm
        
        critica = chain.invoke({
            "diagnostico": json.dumps(diagnostico_principal, indent=2),
            "caso": json.dumps(caso, indent=2),
            "alternativos": json.dumps(diagnosticos[1:3], indent=2)
        })
        
        try:
            critica_parsed = json.loads(critica.content)
        except:
            critica_parsed = {"score": 0.5, "comentarios": critica.content}
        
        return {
            **state,
            "critica": critica_parsed
        }
    
    def decision_final(self, state: MedicalDiagnosticState):
        """
        Nodo 7: Decisión final y generación de explicación
        """
        diagnosticos = state["diagnosticos_candidatos"]
        critica = state.get("critica", {})
        
        # Ajustar confianza basándose en la crítica
        score_critica = critica.get("score", 0.5)
        prob_diagnostico = diagnosticos[0].get("probabilidad", 0.5) if diagnosticos else 0.5
        
        confianza_final = (score_critica + prob_diagnostico) / 2
        
        # Decidir si necesita revisión humana
        necesita_revision = confianza_final < 0.7
        
        # Generar explicación
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Genera una explicación clara para el médico."),
            ("user", """
            Genera un reporte clínico conciso:
            
            Diagnóstico: {diagnostico}
            Confianza: {confianza}
            Crítica: {critica}
            
            Incluye:
            - Diagnóstico principal y alternativas
            - Evidencia clave
            - Recomendaciones de pruebas
            - Precauciones
            """)
        ])
        
        chain = prompt | self.llm
        
        explicacion = chain.invoke({
            "diagnostico": diagnosticos[0]["enfermedad"] if diagnosticos else "Indeterminado",
            "confianza": confianza_final,
            "critica": json.dumps(critica, indent=2)
        })
        
        return {
            **state,
            "diagnostico_final": diagnosticos[0] if diagnosticos else {},
            "confianza": confianza_final,
            "necesita_revision": necesita_revision,
            "explicacion": explicacion.content
        }
    
    def buscar_casos_similares(self, caso, k=3):
        """
        Busca casos históricos similares
        """
        if not self.casos_historicos:
            return []
        
        # Crear embedding del caso actual
        caso_str = json.dumps(caso["paciente"])
        
        # Buscar en casos históricos (simplificado)
        # En producción: usar otro vectorstore para casos
        similares = self.casos_historicos[:k]
        
        return similares
    
    def diagnosticar(self, caso_paciente):
        """
        Método principal: ejecuta el grafo completo
        """
        initial_state = MedicalDiagnosticState(
            caso_paciente=caso_paciente,
            sintomas_analizados={},
            diagnosticos_candidatos=[],
            incertidumbres=[],
            diagnostico_final={},
            confianza=0.0,
            necesita_revision=False,
            explicacion="",
            casos_similares=[],
            iteracion=0
        )
        
        # Ejecutar el grafo
        resultado = self.graph.invoke(initial_state)
        
        return resultado
    
    def confirmar_diagnostico(self, caso_id, diagnostico_real, feedback):
        """
        Aprendizaje: confirma el diagnóstico real y aprende
        """
        # Buscar el caso
        caso = next((c for c in self.casos_historicos if c.get("id") == caso_id), None)
        
        if not caso:
            print(f"Caso {caso_id} no encontrado")
            return
        
        # Comparar con diagnóstico dado
        diagnostico_dado = caso["resultado"]["diagnostico_final"].get("enfermedad")
        acierto = (diagnostico_dado == diagnostico_real)
        
        if not acierto:
            # APRENDER DEL ERROR
            self.aprender_de_error(caso, diagnostico_real, feedback)
        else:
            # REFORZAR PATRÓN EXITOSO
            self.reforzar_exito(caso)
        
        # Guardar caso verificado
        caso["verificado"] = True
        caso["diagnostico_real"] = diagnostico_real
        caso["feedback"] = feedback
        caso["acierto"] = acierto

    def generar_casos_sinteticos_correccion(self, diagnostico_real, analisis_error, n=5):
        """
        Genera casos sintéticos para evitar el error en el futuro
        """
        print(f"📚 Aprendiendo del error: {caso['resultado']['diagnostico_final'].get('enfermedad')} → {diagnostico_real}")
        
        # Analizar por qué falló
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un experto en análisis de errores médicos.
            Analiza por qué el sistema falló y cómo evitarlo."""),
            ("user", """
            DIAGNÓSTICO DADO POR EL SISTEMA:
            {diagnostico_dado}
            
            DIAGNÓSTICO REAL:
            {diagnostico_real}
            
            CASO COMPLETO:
            {caso}
            
            FEEDBACK DEL MÉDICO:
            {feedback}
            
            Analiza:
            1. ¿Qué señales fueron malinterpretadas?
            2. ¿Qué información crítica faltó considerar?
            3. ¿Qué patrón no fue reconocido?
            4. ¿Cómo identificar casos similares en el futuro?
            5. Lista 3-5 reglas específicas para evitar este error
            
            Formato JSON con análisis detallado.
            """)
        ])
        
        chain = prompt | self.llm
        
        analisis = chain.invoke({
            "diagnostico_dado": caso['resultado']['diagnostico_final'].get('enfermedad', 'Desconocido'),
            "diagnostico_real": diagnostico_real,
            "caso": json.dumps(caso['caso_paciente'], indent=2),
            "feedback": feedback
        })
        
        try:
            analisis_parsed = json.loads(analisis.content)
        except:
            analisis_parsed = {"analisis": analisis.content}
        
        # Generar casos sintéticos para reforzar el aprendizaje
        casos_sinteticos = self.generar_casos_sinteticos_correccion(
            diagnostico_real,
            analisis_parsed,
            n=5
        )
        
        # Agregar al vectorstore
        docs_nuevos = []
        for caso_sint in casos_sinteticos:
            doc = Document(
                page_content=caso_sint,
                metadata={
                    "source": "error_correction",
                    "diagnostico": diagnostico_real,
                    "fecha_aprendizaje": json.dumps({"year": 2025, "month": 10, "day": 23})
                }
            )
            docs_nuevos.append(doc)
        
        # Actualizar vectorstore
        self.vectorstore.add_documents(docs_nuevos)
        self.vectorstore.save_local("medical_vectorstore")
        
        print(f"✅ Agregados {len(casos_sinteticos)} casos sintéticos al conocimiento")
        
        # Guardar regla de corrección
        regla = {
            "patron_error": analisis_parsed.get("patron_no_reconocido"),
            "diagnostico_correcto": diagnostico_real,
            "reglas_prevencion": analisis_parsed.get("reglas", []),
            "peso": 1.0
        }
        
        # Actualizar grafo de incertidumbres
        caso_id = caso.get("id", "unknown")
        self.uncertainty_graph[caso_id] = {
            "tipo": "error_corregido",
            "regla": regla,
            "fecha": "2025-10-23"
        }
    
    def generar_casos_sinteticos_correccion(self, diagnostico_real, analisis_error, n=5):
        """
        Genera casos sintéticos específicos para corregir el error
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Genera casos clínicos realistas que ayuden al sistema
            a aprender el patrón correcto."""),
            ("user", """
            DIAGNÓSTICO: {diagnostico}
            
            ANÁLISIS DEL ERROR PREVIO:
            {analisis}
            
            Genera {n} casos clínicos que:
            1. Presenten {diagnostico} claramente
            2. Incluyan las señales que el sistema ignoró anteriormente
            3. Enfaticen los patrones que no fueron reconocidos
            4. Tengan variaciones realistas
            
            Cada caso debe incluir:
            - Presentación del paciente
            - Síntomas con timing y progresión
            - Resultados de exámenes
            - Puntos clave de diagnóstico
            - Diagnóstico confirmado: {diagnostico}
            
            Formato: Lista de textos descriptivos (no JSON), uno por caso.
            Separa cada caso con "---CASO---"
            """)
        ])
        
        chain = prompt | self.llm_creative
        
        resultado = chain.invoke({
            "diagnostico": diagnostico_real,
            "analisis": json.dumps(analisis_error, indent=2),
            "n": n
        })
        
        # Separar casos
        casos = resultado.content.split("---CASO---")
        casos = [c.strip() for c in casos if c.strip()]
        
        return casos
    
    def reforzar_exito(self, caso):
        """
        Refuerza patrones exitosos
        """
        print(f"✅ Reforzando patrón exitoso: {caso['resultado']['diagnostico_final'].get('enfermedad')}")
        
        # Aumentar peso de los documentos que contribuyeron
        # En una implementación completa, ajustar embeddings o pesos
        pass


# ============================================
# 4. SISTEMA DE EVALUACIÓN Y MÉTRICAS
# ============================================

class MedicalSystemEvaluator:
    """
    Evalúa el desempeño del sistema y genera reportes
    """
    def __init__(self, sistema):
        self.sistema = sistema
        self.metricas = {
            "total_casos": 0,
            "aciertos": 0,
            "errores": 0,
            "precision": 0.0,
            "confianza_promedio": 0.0,
            "casos_revision": 0
        }
    
    def evaluar_caso(self, caso_real, diagnostico_real):
        """
        Evalúa un caso individual
        """
        # Diagnosticar
        resultado = self.sistema.diagnosticar(caso_real)
        
        # Comparar
        diagnostico_sistema = resultado["diagnostico_final"].get("enfermedad", "")
        acierto = (diagnostico_sistema == diagnostico_real)
        
        # Actualizar métricas
        self.metricas["total_casos"] += 1
        if acierto:
            self.metricas["aciertos"] += 1
        else:
            self.metricas["errores"] += 1
        
        self.metricas["precision"] = self.metricas["aciertos"] / self.metricas["total_casos"]
        
        confianza_actual = resultado["confianza"]
        self.metricas["confianza_promedio"] = (
            (self.metricas["confianza_promedio"] * (self.metricas["total_casos"] - 1) + confianza_actual)
            / self.metricas["total_casos"]
        )
        
        if resultado["necesita_revision"]:
            self.metricas["casos_rev    ision"] += 1
        
        return {
            "acierto": acierto,
            "diagnostico_sistema": diagnostico_sistema,
            "diagnopipstico_real": diagnostico_real,
            "confianza": confianza_actual,
            "necesita_revision": resultado["necesita_revision"]
        }
    
    def evaluar_dataset(self, dataset):
        """
        Evalúa el sistema contra un dataset de casos con diagnósticos conocidos
        """
        resultados = []
        
        for caso in dataset:
            eval_resultado = self.evaluar_caso(
                caso["caso_paciente"],
                caso["diagnostico_real"]
            )
            resultados.append(eval_resultado)
        
        return resultados
    
    def generar_reporte(self):
        """
        Genera reporte de métricas
        """
        reporte = f"""
        ╔══════════════════════════════════════════╗
        ║   REPORTE DE EVALUACIÓN DEL SISTEMA     ║
        ╚══════════════════════════════════════════╝
        
        Total de casos evaluados: {self.metricas['total_casos']}
        Aciertos: {self.metricas['aciertos']}
        Errores: {self.metricas['errores']}
        
        Precisión: {self.metricas['precision']:.2%}
        Confianza promedio: {self.metricas['confianza_promedio']:.2f}
        Casos que requirieron revisión: {self.metricas['casos_revision']} ({self.metricas['casos_revision']/self.metricas['total_casos']:.1%})
        """
        
        print(reporte)
        return self.metricas


# ============================================
# 5. INTERFAZ Y USO DEL SISTEMA
# ============================================

def main():
    """
    Ejemplo de uso completo del sistema
    """
    print("🏥 Iniciando Sistema Médico Auto-Mejorable...")
    print("=" * 60)
    
    # 1. Inicializar sistema
    sistema = MedicalDiagnosticSystem()
    
    # 2. Caso de ejemplo
    caso_ejemplo = {
        "id": "PAC-2025-001",
        "paciente": {
            "edad": 45,
            "genero": "F",
            "sintomas": [
                "fiebre persistente",
                "tos seca",
                "fatiga extrema",
                "dolor de cabeza",
                "pérdida del olfato"
            ],
            "duracion": "7 días",
            "progresion": "empeorando gradualmente"
        },
        "resultados_lab": {
            "leucocitos": 12000,
            "linfocitos": "bajo",
            "PCR": 45,
            "d-dimero": 850
        },
        "ubicacion": "Monterrey, NL",
        "fecha": "2025-10-23"
    }
    
    print("\n📋 CASO DEL PACIENTE:")
    print(json.dumps(caso_ejemplo, indent=2, ensure_ascii=False))
    
    # 3. Diagnosticar
    print("\n🔍 Procesando diagnóstico...")
    print("-" * 60)
    
    resultado = sistema.diagnosticar(caso_ejemplo)
    
    # 4. Mostrar resultado
    print("\n📊 RESULTADO DEL DIAGNÓSTICO:")
    print("=" * 60)
    print(f"Diagnóstico: {resultado['diagnostico_final'].get('enfermedad', 'No determinado')}")
    print(f"Confianza: {resultado['confianza']:.2%}")
    print(f"Necesita revisión: {'Sí ⚠️' if resultado['necesita_revision'] else 'No ✓'}")
    
    print(f"\n📝 Explicación:\n{resultado['explicacion']}")
    
    if resultado['incertidumbres']:
        print(f"\n⚠️ Incertidumbres detectadas:")
        for inc in resultado['incertidumbres']:
            print(f"  - {inc['tipo']}: {inc['descripcion']}")
    
    if resultado['casos_similares']:
        print(f"\n📚 Casos similares encontrados: {len(resultado['casos_similares'])}")
    
    # 5. Guardar caso
    sistema.casos_historicos.append({
        "id": caso_ejemplo["id"],
        "caso_paciente": caso_ejemplo,
        "resultado": resultado,
        "fecha": "2025-10-23",
        "verificado": False
    })
    
    print("\n" + "=" * 60)
    
    # 6. Simular confirmación del médico (después de unos días)
    print("\n👨‍⚕️ SIMULANDO CONFIRMACIÓN DEL MÉDICO...")
    print("-" * 60)
    
    diagnostico_real = "COVID-19 variante JN.1"
    feedback_medico = """
    El diagnóstico fue correcto. El paciente respondió bien al tratamiento.
    Los síntomas de pérdida de olfato y la elevación del d-dímero fueron
    indicadores clave que el sistema identificó correctamente.
    """
    
    sistema.confirmar_diagnostico(
        caso_id="PAC-2025-001",
        diagnostico_real=diagnostico_real,
        feedback=feedback_medico
    )
    
    print(f"✅ Sistema actualizado con el diagnóstico real: {diagnostico_real}")
    
    # 7. Evaluar sistema
    print("\n📈 EVALUACIÓN DEL SISTEMA")
    print("=" * 60)
    
    evaluador = MedicalSystemEvaluator(sistema)
    
    # Dataset de prueba (en producción, cargar desde archivo)
    dataset_prueba = [
        {
            "caso_paciente": caso_ejemplo,
            "diagnostico_real": "COVID-19 variante JN.1"
        },
        # Agregar más casos...
    ]
    
    evaluador.evaluar_dataset(dataset_prueba)
    evaluador.generar_reporte()


# ============================================
# 6. DETECTOR DE PATRONES EMERGENTES
# ============================================

class PatternEmergenceDetector:
    """
    Detecta brotes o patrones emergentes en tiempo real
    """
    def __init__(self):
        self.casos_recientes = []
        self.ventana_temporal_dias = 30
        self.embeddings = OpenAIEmbeddings()
    
    def agregar_caso(self, caso, diagnostico):
        """
        Agrega un caso nuevo al detector
        """
        from datetime import datetime
        
        self.casos_recientes.append({
            "caso": caso,
            "diagnostico": diagnostico,
            "fecha": datetime.now(),
            "ubicacion": caso.get("ubicacion", "unknown")
        })
        
        # Limpiar casos antiguos
        self.limpiar_casos_antiguos()
    
    def limpiar_casos_antiguos(self):
        """
        Elimina casos fuera de la ventana temporal
        """
        from datetime import datetime, timedelta
        
        fecha_limite = datetime.now() - timedelta(days=self.ventana_temporal_dias)
        self.casos_recientes = [
            c for c in self.casos_recientes
            if c["fecha"] > fecha_limite
        ]
    
    def detectar_clusters(self):
        """
        Detecta clusters de casos similares usando embeddings
        """
        if len(self.casos_recientes) < 5:
            return []
        
        from sklearn.cluster import DBSCAN
        import numpy as np
        
        # Crear embeddings de los casos
        textos = []
        for caso in self.casos_recientes:
            texto = f"{caso['diagnostico']} {' '.join(caso['caso']['paciente'].get('sintomas', []))}"
            textos.append(texto)
        
        embeddings = self.embeddings.embed_documents(textos)
        embeddings_array = np.array(embeddings)
        
        # Clustering
        clustering = DBSCAN(eps=0.3, min_samples=3)
        labels = clustering.fit_predict(embeddings_array)
        
        # Analizar clusters
        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:  # Ruido
                continue
            
            if label not in clusters:
                clusters[label] = []
            
            clusters[label].append(self.casos_recientes[i])
        
        return clusters
    
    def analizar_patron_emergente(self, cluster):
        """
        Analiza si un cluster representa un patrón emergente
        """
        if len(cluster) < 3:
            return None
        
        # Verificar si está creciendo
        cluster_ordenado = sorted(cluster, key=lambda x: x["fecha"])
        
        # Dividir en dos mitades temporales
        mitad = len(cluster_ordenado) // 2
        primera_mitad = cluster_ordenado[:mitad]
        segunda_mitad = cluster_ordenado[mitad:]
        
        # Si la segunda mitad tiene más casos, está creciendo
        esta_creciendo = len(segunda_mitad) > len(primera_mitad)
        
        if not esta_creciendo:
            return None
        
        # Extraer características comunes
        diagnosticos = [c["diagnostico"] for c in cluster]
        diagnostico_comun = max(set(diagnosticos), key=diagnosticos.count)
        
        ubicaciones = [c["ubicacion"] for c in cluster]
        ubicacion_comun = max(set(ubicaciones), key=ubicaciones.count)
        
        # Sintomas comunes
        todos_sintomas = []
        for c in cluster:
            todos_sintomas.extend(c["caso"]["paciente"].get("sintomas", []))
        
        from collections import Counter
        sintomas_frecuentes = Counter(todos_sintomas).most_common(5)
        
        return {
            "tipo": "patron_emergente",
            "diagnostico_base": diagnostico_comun,
            "num_casos": len(cluster),
            "ubicacion_principal": ubicacion_comun,
            "sintomas_frecuentes": sintomas_frecuentes,
            "tasa_crecimiento": len(segunda_mitad) / len(primera_mitad),
            "urgencia": "alta" if len(cluster) >= 10 else "media"
        }
    
    def generar_alerta(self, patron_emergente):
        """
        Genera una alerta epidemiológica
        """
        alerta = f"""
        ⚠️ ALERTA EPIDEMIOLÓGICA
        ════════════════════════════════════════════
        
        Se ha detectado un patrón emergente:
        
        Diagnóstico: {patron_emergente['diagnostico_base']}
        Número de casos: {patron_emergente['num_casos']}
        Ubicación: {patron_emergente['ubicacion_principal']}
        Tasa de crecimiento: {patron_emergente['tasa_crecimiento']:.1f}x
        
        Síntomas más frecuentes:
        {chr(10).join([f"  - {sintoma}: {freq} casos" for sintoma, freq in patron_emergente['sintomas_frecuentes']])}
        
        Urgencia: {patron_emergente['urgencia'].upper()}
        
        Recomendación: Intensificar vigilancia epidemiológica
        ════════════════════════════════════════════
        """
        
        print(alerta)
        return alerta


# ============================================
# 7. UTILIDADES PARA CARGAR DATASETS REALES
# ============================================

class MedicalDatasetManager:
    """
    Gestor para cargar y preparar datasets médicos
    """
    
    @staticmethod
    def cargar_disease_symptom_dataset():
        """
        Carga el dataset de Kaggle: Disease Symptom Description
        Descarga manual desde: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
        """
        import pandas as pd
        
        try:
            # Leer los archivos
            df_symptoms = pd.read_csv("dataset/dataset.csv")
            df_description = pd.read_csv("dataset/symptom_Description.csv")
            df_precaution = pd.read_csv("dataset/symptom_precaution.csv")
            
            # Combinar información
            documentos = []
            
            for _, row in df_symptoms.iterrows():
                # Extraer síntomas (columnas Symptom_1 a Symptom_17)
                sintomas = [
                    row[f'Symptom_{i}'] 
                    for i in range(1, 18) 
                    if pd.notna(row.get(f'Symptom_{i}'))
                ]
                
                # Buscar descripción
                desc_row = df_description[df_description['Disease'] == row['Disease']]
                descripcion = desc_row['Description'].values[0] if not desc_row.empty else ""
                
                # Buscar precauciones
                prec_row = df_precaution[df_precaution['Disease'] == row['Disease']]
                precauciones = []
                if not prec_row.empty:
                    precauciones = [
                        prec_row[f'Precaution_{i}'].values[0]
                        for i in range(1, 5)
                        if pd.notna(prec_row.get(f'Precaution_{i}').values[0] if not prec_row.get(f'Precaution_{i}').empty else None)
                    ]
                
                # Crear documento estructurado
                doc_texto = f"""
                Enfermedad: {row['Disease']}
                
                Síntomas principales:
                {chr(10).join([f"- {s}" for s in sintomas])}
                
                Descripción:
                {descripcion}
                
                Precauciones:
                {chr(10).join([f"- {p}" for p in precauciones])}
                """
                
                documentos.append({
                    "content": doc_texto,
                    "metadata": {
                        "disease": row['Disease'],
                        "symptoms": sintomas,
                        "source": "kaggle_disease_symptom"
                    }
                })
            
            print(f"✅ Cargadas {len(documentos)} enfermedades del dataset")
            return documentos
            
        except FileNotFoundError:
            print("⚠️ Dataset no encontrado. Descárgalo de Kaggle:")
            print("   https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset")
            return []
    
    @staticmethod
    def crear_casos_prueba_realistas():
        """
        Crea casos de prueba realistas para evaluar el sistema
        """
        casos = [
            {
                "caso_paciente": {
                    "id": "TEST-001",
                    "paciente": {
                        "edad": 35,
                        "genero": "M",
                        "sintomas": ["fiebre alta", "escalofríos", "dolor de cabeza severo", "dolor muscular", "náuseas"],
                        "duracion": "3 días",
                        "progresion": "empeorando rápidamente"
                    },
                    "resultados_lab": {
                        "plaquetas": 95000,
                        "hematocrito": 48,
                        "leucocitos": 3500
                    },
                    "ubicacion": "Monterrey, NL",
                    "fecha": "2025-10-23"
                },
                "diagnostico_real": "Dengue"
            },
            {
                "caso_paciente": {
                    "id": "TEST-002",
                    "paciente": {
                        "edad": 65,
                        "genero": "F",
                        "sintomas": ["dolor de pecho", "dificultad para respirar", "sudoración", "náuseas"],
                        "duracion": "2 horas",
                        "progresion": "súbito"
                    },
                    "resultados_lab": {
                        "troponina": "elevada",
                        "ECG": "elevación del segmento ST"
                    },
                    "ubicacion": "Monterrey, NL",
                    "fecha": "2025-10-23"
                },
                "diagnostico_real": "Infarto agudo de miocardio"
            },
            {
                "caso_paciente": {
                    "id": "TEST-003",
                    "paciente": {
                        "edad": 28,
                        "genero": "F",
                        "sintomas": ["tos persistente", "fiebre", "pérdida de peso", "sudoración nocturna"],
                        "duracion": "6 semanas",
                        "progresion": "gradual"
                    },
                    "resultados_lab": {
                        "baciloscopía": "BAAR positivo",
                        "radiografía": "infiltrados en lóbulo superior"
                    },
                    "ubicacion": "Monterrey, NL",
                    "fecha": "2025-10-23"
                },
                "diagnostico_real": "Tuberculosis pulmonar"
            }
        ]
        
        return casos


# ============================================
# 8. SCRIPT PRINCIPAL CON TODO INTEGRADO
# ============================================

if __name__ == "__main__":
    """Entrypoint mínimo: carga .env, verifica clave y ejecuta el CLI modular."""
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY no encontrada en .env")
        print("Crea un archivo .env con: OPENAI_API_KEY=tu_api_key")
        exit(1)

    from medical.cli import run_cli
    run_cli()