# backend/rag.py
import os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader, UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai

# ─── Configuración ────────────────────────────────────────────────────────────

EMBED_MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

chroma_client = chromadb.PersistentClient(path="./vectorstore")
collection = chroma_client.get_or_create_collection(
    name="documentos",
    metadata={"heuristic": "cosine"}
)

SYSTEM_PROMPT = """Eres un asistente técnico especializado en análisis de documentos.

REGLAS ESTRICTAS:
1. Responde ÚNICAMENTE basándote en los fragmentos de documentos proporcionados.
2. Si la información no está en los fragmentos, di explícitamente: "Esta información no se encuentra en los documentos cargados."
3. Usa terminología técnica precisa y apropiada al dominio del documento.
4. Cita el nombre del archivo fuente cuando uses información específica (ej: "Según [nombre_archivo]...").
5. Estructura tus respuestas con claridad: usa listas, numeración o secciones cuando aplique.
6. Si hay contradicciones entre fragmentos, señálalas explícitamente.
7. No inventes datos, fórmulas, cifras ni procedimientos que no estén en el contexto.
8. Responde siempre en el mismo idioma en que se formula la pregunta."""

SALUDOS = {
    "hola", "hi", "hello", "hey", "buenas", "buen día", "buen dia",
    "buenos días", "buenos dias", "buenas tardes", "buenas noches",
    "qué tal", "que tal", "cómo estás", "como estas"
}

def _get_gemini_model():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY no está configurada.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT
    )

# ─── Loaders ──────────────────────────────────────────────────────────────────

LOADERS = {
    "pdf":  PyPDFLoader,
    "docx": Docx2txtLoader,
    "doc":  Docx2txtLoader,
    "txt":  lambda p: TextLoader(p, encoding="utf-8"),
    "csv":  lambda p: CSVLoader(p, encoding="utf-8"),
    "xlsx": UnstructuredExcelLoader,
    "xls":  UnstructuredExcelLoader,
}

def load_document(file_path: str, file_type: str) -> list:
    ext = file_type.lower().lstrip(".")
    loader_fn = LOADERS.get(ext)
    if not loader_fn:
        print(f"[RAG] Tipo de archivo no soportado: {ext}")
        return []
    try:
        loader = loader_fn(file_path)
        docs = loader.load()
        print(f"[RAG] Cargado '{file_path}': {len(docs)} sección(es)")
        return docs
    except Exception as e:
        print(f"[RAG] Error cargando '{file_path}': {e}")
        return []

# ─── Indexación ───────────────────────────────────────────────────────────────

def index_document(doc_id: int, file_path: str, file_type: str, filename: str) -> bool:
    docs = load_document(file_path, file_type)
    if not docs:
        return False

    delete_document_from_index(doc_id)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        print(f"[RAG] El documento '{filename}' no generó chunks.")
        return False

    texts      = [c.page_content.strip() for c in chunks if c.page_content.strip()]
    embeddings = EMBED_MODEL.encode(texts, show_progress_bar=False).tolist()
    ids        = [f"doc{doc_id}_chunk{i}" for i in range(len(texts))]
    metadatas  = [
        {
            "doc_id":   str(doc_id),
            "filename": filename,
            "chunk":    i,
            "page":     chunks[i].metadata.get("page", 0),
        }
        for i in range(len(texts))
    ]

    collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)
    print(f"[RAG] Indexado '{filename}': {len(texts)} chunks")
    return True


def delete_document_from_index(doc_id: int) -> None:
    try:
        results = collection.get(where={"doc_id": str(doc_id)})
        if results["ids"]:
            collection.delete(ids=results["ids"])
            print(f"[RAG] Eliminados {len(results['ids'])} chunks del doc_id={doc_id}")
    except Exception as e:
        print(f"[RAG] Error eliminando doc_id={doc_id}: {e}")


def list_indexed_documents() -> list:
    try:
        all_meta = collection.get()["metadatas"]
        seen = {}
        for m in all_meta:
            doc_id = m.get("doc_id")
            if doc_id not in seen:
                seen[doc_id] = m.get("filename", "desconocido")
        return [{"doc_id": k, "filename": v} for k, v in seen.items()]
    except Exception as e:
        print(f"[RAG] Error listando documentos: {e}")
        return []

# ─── Consulta RAG ─────────────────────────────────────────────────────────────

def retrieve_chunks(question: str, doc_ids: list = None, n_results: int = 6) -> tuple:
    question_embedding = EMBED_MODEL.encode([question]).tolist()[0]

    where_filter = None
    if doc_ids:
        where_filter = (
            {"doc_id": str(doc_ids[0])} if len(doc_ids) == 1
            else {"doc_id": {"$in": [str(d) for d in doc_ids]}}
        )

    try:
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=min(n_results, collection.count()),
            where=where_filter,
        )
    except Exception:
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=min(n_results, collection.count()),
        )

    docs  = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return docs, metas


def build_context(docs: list, metas: list) -> tuple:
    context_parts = []
    sources = []

    for i, (doc, meta) in enumerate(zip(docs, metas)):
        filename  = meta.get("filename", "desconocido")
        page_info = f", p.{meta['page']+1}" if meta.get("page") else ""
        header    = f"[Fragmento {i+1} — {filename}{page_info}]"
        context_parts.append(f"{header}\n{doc}")
        sources.append({
            "filename": filename,
            "fragment": i + 1,
            "page":     meta.get("page", 0),
            "preview":  doc[:220] + ("..." if len(doc) > 220 else ""),
        })

    return "\n\n".join(context_parts), sources


def query_rag(question: str, doc_ids: list = None, n_results: int = 6) -> dict:

    # ── Respuesta a saludos sin consumir API ──────────────────────────────────
    if question.strip().lower() in SALUDOS:
        return {
            "answer": (
                "¡Hola! 👋 Soy tu asistente de documentos.\n\n"
                "Puedes hacerme preguntas sobre los archivos que has cargado "
                "y te responderé basándome exclusivamente en su contenido.\n\n"
                "Por ejemplo:\n"
                "- *¿De qué trata el documento?*\n"
                "- *¿Cuáles son los puntos principales?*\n"
                "- *¿Qué dice sobre [tema específico]?*"
            ),
            "sources": [],
        }

    # ── Retrieval ─────────────────────────────────────────────────────────────
    docs, metas = retrieve_chunks(question, doc_ids, n_results)

    if not docs:
        return {
            "answer": (
                "⚠️ No encontré información relevante en los documentos cargados. "
                "Verifica que los archivos estén correctamente indexados."
            ),
            "sources": [],
        }

    # ── Contexto ──────────────────────────────────────────────────────────────
    context, sources = build_context(docs, metas)

    # ── Generación con Gemini ─────────────────────────────────────────────────
    user_message = (
        f"A continuación tienes fragmentos extraídos de documentos técnicos:\n\n"
        f"{context}\n\n"
        f"---\n"
        f"PREGUNTA: {question}\n\n"
        f"Responde de forma técnica y precisa basándote SOLO en los fragmentos anteriores."
    )

    try:
        model    = _get_gemini_model()
        response = model.generate_content(user_message)
        answer   = response.text

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg:
            answer = (
                "⏳ Se superó el límite de solicitudes de la API gratuita de Gemini.\n\n"
                "Espera unos segundos e intenta de nuevo.\n\n"
                "**Fragmentos encontrados en los documentos:**\n\n" + context
            )
        elif "GEMINI_API_KEY" in error_msg:
            answer = (
                "❌ API key no configurada.\n"
                "Ejecuta: `$env:GEMINI_API_KEY = 'AIza...'` y reinicia el servidor."
            )
        else:
            answer = (
                f"❌ Error al generar respuesta: {e}\n\n"
                f"**Fragmentos encontrados en los documentos:**\n\n{context}"
            )

    return {"answer": answer, "sources": sources}