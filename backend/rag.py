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

# ─── Loaders individuales ─────────────────────────────────────────────────────

def _load_pdf(file_path: str) -> list:
    """PDF: extrae texto página por página."""
    return PyPDFLoader(file_path).load()


def _load_docx(file_path: str) -> list:
    """Word .docx moderno."""
    return Docx2txtLoader(file_path).load()


def _load_doc_old(file_path: str) -> list:
    """Word .doc antiguo — usa docx2txt como fallback."""
    try:
        import docx2txt
        from langchain_core.documents import Document
        text = docx2txt.process(file_path)
        return [Document(page_content=text, metadata={"source": file_path})]
    except Exception as e:
        print(f"[RAG] Fallback .doc: {e}")
        return Docx2txtLoader(file_path).load()


def _load_excel(file_path: str) -> list:
    """
    Excel .xlsx / .xls — lee cada hoja como texto tabular.
    Usa openpyxl para xlsx y xlrd para xls.
    """
    from langchain_core.documents import Document
    docs = []
    ext  = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".xlsx":
            import openpyxl
            wb = openpyxl.load_workbook(file_path, data_only=True)
            for sheet_name in wb.sheetnames:
                ws   = wb[sheet_name]
                rows = []
                for row in ws.iter_rows(values_only=True):
                    row_str = "\t".join(str(c) if c is not None else "" for c in row)
                    if row_str.strip():
                        rows.append(row_str)
                if rows:
                    docs.append(Document(
                        page_content=f"[Hoja: {sheet_name}]\n" + "\n".join(rows),
                        metadata={"source": file_path, "sheet": sheet_name}
                    ))
        else:  # .xls
            import xlrd
            wb = xlrd.open_workbook(file_path)
            for sheet in wb.sheets():
                rows = []
                for r in range(sheet.nrows):
                    row_str = "\t".join(str(sheet.cell_value(r, c)) for c in range(sheet.ncols))
                    if row_str.strip():
                        rows.append(row_str)
                if rows:
                    docs.append(Document(
                        page_content=f"[Hoja: {sheet.name}]\n" + "\n".join(rows),
                        metadata={"source": file_path, "sheet": sheet.name}
                    ))
    except Exception as e:
        print(f"[RAG] Error leyendo Excel, usando fallback UnstructuredExcelLoader: {e}")
        try:
            docs = UnstructuredExcelLoader(file_path).load()
        except Exception as e2:
            print(f"[RAG] Fallback Excel falló: {e2}")
    return docs


def _load_pptx(file_path: str) -> list:
    """
    PowerPoint .pptx / .ppt — extrae texto de cada diapositiva
    y sus notas del presentador.
    """
    from langchain_core.documents import Document
    docs = []
    try:
        from pptx import Presentation
        prs = Presentation(file_path)
        for i, slide in enumerate(prs.slides, start=1):
            texts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        line = " ".join(run.text for run in para.runs).strip()
                        if line:
                            texts.append(line)
            # Notas del presentador
            if slide.has_notes_slide:
                notes = slide.notes_slide.notes_text_frame
                if notes:
                    note_text = notes.text.strip()
                    if note_text:
                        texts.append(f"[Notas presentador]: {note_text}")
            if texts:
                docs.append(Document(
                    page_content=f"[Diapositiva {i}]\n" + "\n".join(texts),
                    metadata={"source": file_path, "slide": i}
                ))
    except ImportError:
        print("[RAG] python-pptx no instalado. Ejecuta: pip install python-pptx")
    except Exception as e:
        print(f"[RAG] Error cargando PPTX '{file_path}': {e}")
    return docs


def _load_image(file_path: str) -> list:
    """
    Imágenes — extrae texto mediante OCR (Tesseract).
    Soporta: PNG, JPG, JPEG, BMP, TIFF, WEBP.

    Requisitos:
        pip install pytesseract pillow
        Instalar Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
        (Windows: marcar idioma español en el instalador)
    """
    from langchain_core.documents import Document
    try:
        import pytesseract
        from PIL import Image

        img  = Image.open(file_path)
        text = pytesseract.image_to_string(img, lang="spa+eng").strip()

        if not text:
            text = "[La imagen no contiene texto legible detectado por OCR]"

        return [Document(
            page_content=text,
            metadata={"source": file_path, "type": "image_ocr"}
        )]

    except ImportError:
        print(
            "[RAG] Faltan dependencias para imágenes.\n"
            "Ejecuta: pip install pytesseract pillow\n"
            "Descarga Tesseract: https://github.com/UB-Mannheim/tesseract/wiki"
        )
        return []
    except Exception as e:
        print(f"[RAG] Error procesando imagen '{file_path}': {e}")
        return []


def _load_txt(file_path: str) -> list:
    return TextLoader(file_path, encoding="utf-8").load()


def _load_csv(file_path: str) -> list:
    return CSVLoader(file_path, encoding="utf-8").load()


# ─── Mapa de extensiones ──────────────────────────────────────────────────────

LOADER_MAP = {
    # PDF
    "pdf":  _load_pdf,
    # Word
    "docx": _load_docx,
    "doc":  _load_doc_old,
    # Excel
    "xlsx": _load_excel,
    "xls":  _load_excel,
    # PowerPoint
    "pptx": _load_pptx,
    "ppt":  _load_pptx,
    # Texto plano
    "txt":  _load_txt,
    # CSV
    "csv":  _load_csv,
    # Imágenes (OCR)
    "png":  _load_image,
    "jpg":  _load_image,
    "jpeg": _load_image,
    "bmp":  _load_image,
    "tiff": _load_image,
    "tif":  _load_image,
    "webp": _load_image,
}

EXTENSIONES_SOPORTADAS = list(LOADER_MAP.keys())


def load_document(file_path: str, file_type: str) -> list:
    ext = file_type.lower().lstrip(".")
    loader_fn = LOADER_MAP.get(ext)
    if not loader_fn:
        print(f"[RAG] Tipo de archivo no soportado: {ext}")
        return []
    try:
        docs = loader_fn(file_path)
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

    total = collection.count()
    if total == 0:
        return [], []

    # ── Sin filtro: busca en TODOS los documentos ──
    try:
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=min(n_results, total),
        )
    except Exception as e:
        print(f"[RAG] Error en query: {e}")
        return [], []

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

    # ── Saludos ───────────────────────────────────────────────────────────────
    if question.strip().lower() in SALUDOS:
        return {
            "answer": (
                "¡Hola! 👋 Soy tu asistente de documentos.\n\n"
                "Puedes hacerme preguntas sobre los archivos cargados.\n\n"
                "**Formatos soportados:**\n"
                "- 📄 PDF\n"
                "- 📝 Word (.docx, .doc)\n"
                "- 📊 Excel (.xlsx, .xls)\n"
                "- 📊 PowerPoint (.pptx, .ppt)\n"
                "- 🖼️ Imágenes con texto (.png, .jpg, .jpeg, .bmp, .tiff)\n"
                "- 📋 CSV y TXT\n\n"
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
                "⚠️ No encontré información relevante en los documentos cargados.\n"
                "Verifica que los archivos estén correctamente indexados."
            ),
            "sources": [],
        }

    # ── Contexto y generación ─────────────────────────────────────────────────
    context, sources = build_context(docs, metas)

    user_message = (
        f"A continuación tienes fragmentos extraídos de documentos:\n\n"
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
                "⏳ Se superó el límite de la API de Gemini. Espera unos segundos.\n\n"
                "**Fragmentos encontrados:**\n\n" + context
            )
        elif "GEMINI_API_KEY" in error_msg:
            answer = (
                "❌ API key no configurada.\n"
                "Ejecuta en PowerShell:\n"
                "`$env:GEMINI_API_KEY = 'AIza...'`\n"
                "y reinicia el servidor."
            )
        else:
            answer = (
                f"❌ Error al generar respuesta: {e}\n\n"
                f"**Fragmentos encontrados:**\n\n{context}"
            )

    return {"answer": answer, "sources": sources}


# ─── Escaneo de carpeta local ─────────────────────────────────────────────────

def scan_folder(folder_path: str) -> list:
    files = []
    try:
        for root, dirs, filenames in os.walk(folder_path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower().lstrip(".")
                if ext in EXTENSIONES_SOPORTADAS:
                    full_path = os.path.join(root, fname)
                    size = os.path.getsize(full_path)
                    files.append({
                        "name": fname,
                        "path": full_path,
                        "ext":  ext,
                        "size": round(size / 1024, 1)
                    })
    except Exception as e:
        print(f"[RAG] Error escaneando carpeta: {e}")
    return files