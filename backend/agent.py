"""
Agente IA para SGD·IA con LangGraph
Orquestador que decide qué herramientas usar
"""

import os
from typing import TypedDict, Annotated, List, Dict, Any
from datetime import datetime

# LangChain y LangGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver  # ← CORREGIDO

# Importar herramientas
from backend.tools import (
    buscar_en_documentos,
    resumir_documento,
    extraer_datos_numericos,
    generar_word,
    generar_pdf,
    generar_grafica,
    comparar_documentos,
    obtener_info_sistema,
    HERRAMIENTAS
)

# Configuración del LLM
def get_llm():
    """Obtiene el modelo Gemini configurado"""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY no está configurada")
    
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.3,
        convert_system_message_to_human=True
    )


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DEL AGENTE CON LANGGRAPH
# ──────────────────────────────────────────────────────────────────────────────

class AgenteState(TypedDict):
    """Estado del agente durante la conversación"""
    pregunta: str
    user_id: int
    session_id: str
    herramientas_usadas: List[str]
    respuestas_parciales: List[str]
    respuesta_final: str
    fuentes: List[Dict]


def crear_agente():
    """Crea y configura el agente con LangGraph"""
    
    # Convertir herramientas al formato de LangChain
    herramientas = []
    for tool_func in HERRAMIENTAS:
        herramientas.append(Tool(
            name=tool_func.name,
            description=tool_func.description,
            func=lambda q, t=tool_func: t.invoke({"query": q}) if hasattr(t, 'invoke') else t(q)
        ))
    
    # Prompt del agente - CORREGIDO (sin llaves)
    prompt = PromptTemplate.from_template("""
Eres un asistente experto en análisis de documentos llamado SGD·IA.

**INSTRUCCIONES:**
1. Analiza la pregunta del usuario y decide qué herramienta usar
2. Si la pregunta es simple (ej: "¿qué dice el documento X?"), usa buscar_en_documentos
3. Si pide un resumen, usa resumir_documento
4. Si pide exportar a Word/PDF, usa generar_word o generar_pdf
5. Si pide una gráfica, primero extrae datos con extraer_datos_numericos, luego generar_grafica
6. Si pide comparar documentos, usa comparar_documentos
7. Si pregunta por el estado del sistema, usa obtener_info_sistema

**HERRAMIENTAS DISPONIBLES:**
- buscar_en_documentos: Busca información en documentos indexados
- resumir_documento: Resume un documento específico por nombre
- extraer_datos_numericos: Extrae números de Excel o CSV
- generar_word: Crea archivo Word descargable
- generar_pdf: Crea archivo PDF descargable
- generar_grafica: Genera gráfica con datos numéricos
- comparar_documentos: Compara dos documentos entre sí
- obtener_info_sistema: Muestra estado del sistema

**PREGUNTA DEL USUARIO:**
{input}

**HISTORIAL:**
{chat_history}

**RESPUESTA:**
""")
    
    # Crear agente
    llm = get_llm()
    agent = create_react_agent(llm, herramientas, prompt)
    executor = AgentExecutor(agent=agent, tools=herramientas, verbose=True, handle_parsing_errors=True)  # ← CORREGIDO (tools no herramientas)
    
    return executor


# ──────────────────────────────────────────────────────────────────────────────
# AGENTE CON LANGGRAPH (VERSIÓN MÁS AVANZADA)
# ──────────────────────────────────────────────────────────────────────────────

class AgenteGraph:
    """Agente orquestador con LangGraph para flujos más complejos"""
    
    def __init__(self):
        self.llm = get_llm()
        self.graph = self._build_graph()
        self.memory = MemorySaver()
    
    def _build_graph(self):
        """Construye el grafo de decisión del agente"""
        
        # Definir nodos
        def clasificar_pregunta(state: AgenteState) -> AgenteState:
            """Clasifica la pregunta para decidir qué herramienta usar"""
            pregunta = state["pregunta"].lower()
            
            if "word" in pregunta or "documento word" in pregunta:
                state["herramientas_usadas"].append("generar_word")
            elif "pdf" in pregunta:
                state["herramientas_usadas"].append("generar_pdf")
            elif "gráfica" in pregunta or "grafica" in pregunta:
                state["herramientas_usadas"].append("extraer_datos_numericos")
                state["herramientas_usadas"].append("generar_grafica")
            elif "resume" in pregunta or "resumen" in pregunta:
                state["herramientas_usadas"].append("resumir_documento")
            elif "compara" in pregunta:
                state["herramientas_usadas"].append("comparar_documentos")
            elif "cuántos" in pregunta or "cuantos" in pregunta:
                state["herramientas_usadas"].append("obtener_info_sistema")
            else:
                state["herramientas_usadas"].append("buscar_en_documentos")
            
            return state
        
        def ejecutar_herramientas(state: AgenteState) -> AgenteState:
            """Ejecuta las herramientas según la clasificación"""
            respuestas = []
            
            for herramienta in state["herramientas_usadas"]:
                try:
                    if herramienta == "buscar_en_documentos":
                        from backend.tools import buscar_en_documentos
                        resultado = buscar_en_documentos.invoke({"query": state["pregunta"]})
                        respuestas.append(resultado)
                    
                    elif herramienta == "resumir_documento":
                        from backend.tools import resumir_documento
                        palabras = state["pregunta"].split()
                        for palabra in palabras:
                            if '.' in palabra:
                                resultado = resumir_documento.invoke({"nombre_documento": palabra})
                                respuestas.append(resultado)
                                break
                        else:
                            respuestas.append("No especificaste qué documento resumir.")
                    
                    elif herramienta == "extraer_datos_numericos":
                        from backend.tools import extraer_datos_numericos
                        palabras = state["pregunta"].split()
                        for palabra in palabras:
                            if '.xlsx' in palabra or '.xls' in palabra or '.csv' in palabra:
                                resultado = extraer_datos_numericos.invoke({"nombre_documento": palabra})
                                respuestas.append(resultado)
                                break
                    
                    elif herramienta == "generar_grafica":
                        from backend.tools import generar_grafica
                        if respuestas and "Datos extraídos" in respuestas[-1]:
                            import re
                            numeros = re.findall(r'\d+\.?\d*', respuestas[-1])
                            if numeros:
                                datos = ','.join(numeros[:10])
                                resultado = generar_grafica.invoke({"datos": datos, "tipo": "auto", "titulo": "Gráfica SGD·IA"})
                                respuestas.append(resultado)
                    
                    elif herramienta == "comparar_documentos":
                        from backend.tools import comparar_documentos
                        import re
                        docs = re.findall(r'[\w\s]+\.(pdf|docx|xlsx|txt)', state["pregunta"])
                        if len(docs) >= 2:
                            resultado = comparar_documentos.invoke({"doc1": docs[0], "doc2": docs[1]})
                            respuestas.append(resultado)
                        else:
                            respuestas.append("Necesito dos documentos para comparar. Ejemplo: 'compara contrato.pdf con factura.xlsx'")
                    
                    elif herramienta == "obtener_info_sistema":
                        from backend.tools import obtener_info_sistema
                        resultado = obtener_info_sistema.invoke({})
                        respuestas.append(resultado)
                    
                    elif herramienta == "generar_word":
                        from backend.tools import generar_word
                        contenido = "\n\n".join(respuestas) if respuestas else state["pregunta"]
                        resultado = generar_word.invoke({"contenido": contenido, "titulo": "Documento SGD·IA"})
                        respuestas.append(resultado)
                    
                    elif herramienta == "generar_pdf":
                        from backend.tools import generar_pdf
                        contenido = "\n\n".join(respuestas) if respuestas else state["pregunta"]
                        resultado = generar_pdf.invoke({"contenido": contenido, "titulo": "Documento SGD·IA"})
                        respuestas.append(resultado)
                        
                except Exception as e:
                    respuestas.append(f"Error ejecutando {herramienta}: {str(e)}")
                    print(f"[AGENT] Error: {e}")
            
            state["respuestas_parciales"] = respuestas
            
            # Extraer fuentes reales de las respuestas
            fuentes_encontradas = []
            for resp in respuestas:
                if "Fuentes:" in resp:
                    linea_fuentes = resp.split("Fuentes:")[-1].strip()
                    archivos = [f.strip() for f in linea_fuentes.split(",")]
                    for archivo in archivos:
                        if archivo and "." in archivo:
                            fuentes_encontradas.append({
                                "filename": archivo,
                                "fragment": 1,
                                "page": 0,
                                "preview": ""
                            })
            state["fuentes"] = fuentes_encontradas
            
            return state
        
        def generar_respuesta_final(state: AgenteState) -> AgenteState:
            """Genera una respuesta final coherente"""
            if not state["respuesta_final"] and state["respuestas_parciales"]:
                state["respuesta_final"] = "\n\n---\n\n".join(state["respuestas_parciales"])
            elif not state["respuesta_final"]:
                state["respuesta_final"] = "No pude procesar tu solicitud. Intenta ser más específico."
            
            if state["user_id"] and state["session_id"]:
                try:
                    from backend.rag import save_to_memory
                    save_to_memory(state["user_id"], state["session_id"], "user", state["pregunta"])
                    save_to_memory(state["user_id"], state["session_id"], "assistant", state["respuesta_final"][:2000])
                except:
                    pass
            
            return state
        
        # Construir grafo
        graph = StateGraph(AgenteState)
        
        graph.add_node("clasificar", clasificar_pregunta)
        graph.add_node("ejecutar", ejecutar_herramientas)
        graph.add_node("responder", generar_respuesta_final)
        
        graph.set_entry_point("clasificar")
        graph.add_edge("clasificar", "ejecutar")
        graph.add_edge("ejecutar", "responder")
        graph.add_edge("responder", END)
        
        return graph.compile(checkpointer=self.memory)
    
    def invoke(self, pregunta: str, user_id: int = None, session_id: str = None) -> Dict[str, Any]:
        """Ejecuta el agente con la pregunta del usuario"""
        
        estado_inicial: AgenteState = {
            "pregunta": pregunta,
            "user_id": user_id,
            "session_id": session_id,
            "herramientas_usadas": [],
            "respuestas_parciales": [],
            "respuesta_final": "",
            "fuentes": []
        }
        
        try:
            resultado = self.graph.invoke(estado_inicial)
            
            respuesta_final = resultado["respuesta_final"]
            
            return {
                "answer": respuesta_final,
                "sources": resultado["fuentes"],
                "cached": False,
                "confidence": {"score": 0.95, "level": "alta", "reason": "Análisis por agente"}
            }
        except Exception as e:
            print(f"[AGENT] Error: {e}")
            return {
                "answer": f"⚠️ Error en el agente: {str(e)}\n\nPor favor, intenta con una pregunta más específica.",
                "sources": [],
                "cached": False,
                "confidence": {"score": 0.5, "level": "media", "reason": "Error en procesamiento"}
            }


# Instancia global del agente
_agente_graph = None

def get_agente():
    """Obtiene la instancia del agente (singleton)"""
    global _agente_graph
    if _agente_graph is None:
        _agente_graph = AgenteGraph()
    return _agente_graph


def agente_responder(pregunta: str, user_id: int = None, session_id: str = None) -> Dict[str, Any]:
    """
    Función principal para que el agente responda.
    Compatible con el formato de query_rag()
    """
    print(f"[AGENT] Procesando: {pregunta[:100]}...")
    
    agente = get_agente()
    return agente.invoke(pregunta, user_id, session_id)