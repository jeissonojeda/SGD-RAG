import os
import uuid
import shutil
import time
import threading
import tempfile
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import get_db, create_tables, User, Document, SessionLocal
from backend.auth import (
    verify_password, get_password_hash,
    create_access_token, get_current_user
)
from backend.rag import (
    index_document, delete_document_from_index, query_rag, 
    clear_cache, scan_folder, EXTENSIONES_SOPORTADAS,
    collection, get_documents_info, get_conversation_context,
    limpiar_documentos_huertanos, list_indexed_documents  # ← AGREGADO
)

app = FastAPI(title="SGD-IA CESMAG", version="3.0")

# ─── CORS ─────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── DIRECTORIOS ──────────────────────────────────
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─── MODELOS ──────────────────────────────────────
class UserRegister(BaseModel):
    username: str
    email: str
    password: str
    role: Optional[str] = "Estándar"

class ChatRequest(BaseModel):
    question: str
    doc_ids: Optional[list] = None

class UserUpdate(BaseModel):
    username: str
    email: str
    role: str
    password: Optional[str] = None

class FolderScanRequest(BaseModel):
    folder_path: str

class FolderImportRequest(BaseModel):
    file_paths: list
    area: str = "General"

# ─── STARTUP ──────────────────────────────────────
@app.on_event("startup")
def startup():
    create_tables()
    limpiar_documentos_huertanos()  # ← AGREGADO: Limpia documentos huérfanos
    db = next(get_db())
    if not db.query(User).filter(User.username == "admin").first():
        admin = User(
            username="admin",
            email="admin@cesmag.edu.co",
            hashed_password=get_password_hash("admin123"),
            role="Administrador"
        )
        db.add(admin)
        db.commit()
        print("[MAIN] Usuario administrador creado: admin / admin123")

# ─── AUTH ─────────────────────────────────────────
@app.post("/api/auth/register")
def register(user: UserRegister, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(400, "El usuario ya existe")
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(400, "El email ya está registrado")
    new_user = User(
        username=user.username,
        email=user.email,
        hashed_password=get_password_hash(user.password),
        role=user.role
    )
    db.add(new_user)
    db.commit()
    return {"message": "Usuario creado exitosamente"}


@app.post("/api/auth/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(401, "Credenciales incorrectas")
    token = create_access_token({"sub": user.username})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "username": user.username,
            "role": user.role,
            "email": user.email
        }
    }


@app.get("/api/auth/me")
def me(current_user: User = Depends(get_current_user)):
    return {
        "username": current_user.username,
        "role": current_user.role,
        "email": current_user.email
    }

# ─── DOCUMENTOS ───────────────────────────────────
@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    area: str = Form("General"),
    description: str = Form(""),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    allowed = ["pdf", "docx", "doc", "txt", "csv", "xlsx", "xls", "pptx", "ppt", "png", "jpg", "jpeg", "bmp", "tiff", "webp"]
    ext = file.filename.split(".")[-1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Tipo de archivo no permitido. Permitidos: {', '.join(allowed)}")

    unique_name = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    doc = Document(
        filename=unique_name,
        original_name=file.filename,
        file_type=ext,
        file_path=file_path,
        area=area,
        description=description,
        owner_id=current_user.id,
        status="Procesando"
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    index_document(doc.id, file_path, ext, file.filename, SessionLocal)
    clear_cache()

    return {"message": "Documento subido, indexando en segundo plano", "doc_id": doc.id}


@app.get("/api/documents")
def list_documents(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.role == "Administrador":
        docs = db.query(Document).all()
    else:
        docs = db.query(Document).filter(Document.owner_id == current_user.id).all()
    return [
        {
            "id": d.id,
            "original_name": d.original_name,
            "file_type": d.file_type,
            "status": d.status,
            "area": d.area,
            "description": d.description,
            "summary": d.summary,
            "uploaded_at": d.uploaded_at.strftime("%Y-%m-%d %H:%M"),
            "owner": d.owner.username if d.owner else "—"
        }
        for d in docs
    ]


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Documento no encontrado")
    if current_user.role != "Administrador" and doc.owner_id != current_user.id:
        raise HTTPException(403, "Sin permiso")
    
    delete_document_from_index(doc_id)
    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)
    db.delete(doc)
    db.commit()
    clear_cache()
    
    return {"message": "Documento eliminado"}

# ─── CHAT IA (MEJORADO CON MEMORIA) ───────────────
@app.post("/api/chat")
def chat(request: ChatRequest, current_user: User = Depends(get_current_user)):
    session_id = str(request.doc_ids[0]) if request.doc_ids and len(request.doc_ids) > 0 else str(current_user.id)
    
    return query_rag(
        question=request.question,
        doc_ids=request.doc_ids,
        user_id=current_user.id,
        session_id=session_id
    )

# ─── ESCANEAR CARPETA LOCAL ───────────────────────
@app.post("/api/folder/scan")
async def folder_scan(
    data: FolderScanRequest,
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "Administrador":
        raise HTTPException(403, "Solo administradores pueden escanear carpetas")
    
    folder_path = data.folder_path.strip()
    if not folder_path:
        raise HTTPException(400, "Ruta de carpeta requerida")
    
    if not os.path.exists(folder_path):
        raise HTTPException(404, "La ruta especificada no existe")
    
    if not os.path.isdir(folder_path):
        raise HTTPException(400, "La ruta debe ser una carpeta")
    
    files = scan_folder(folder_path)
    
    return {
        "total": len(files),
        "files": files,
        "folder": folder_path
    }


@app.post("/api/folder/import")
async def folder_import(
    data: FolderImportRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "Administrador":
        raise HTTPException(403, "Solo administradores pueden importar carpetas")
    
    file_paths = data.file_paths
    area = data.area
    
    if not file_paths:
        raise HTTPException(400, "No se proporcionaron archivos")
    
    results = []
    for file_path in file_paths:
        try:
            if not os.path.exists(file_path):
                results.append({"path": file_path, "status": "error", "error": "Archivo no encontrado"})
                continue
            
            ext = os.path.splitext(file_path)[1].lower().lstrip(".")
            if ext not in EXTENSIONES_SOPORTADAS:
                results.append({"path": file_path, "status": "error", "error": f"Tipo no soportado: {ext}"})
                continue
            
            unique_name = f"{uuid.uuid4()}_{os.path.basename(file_path)}"
            dest_path = os.path.join(UPLOAD_DIR, unique_name)
            
            shutil.copy2(file_path, dest_path)
            
            doc = Document(
                filename=unique_name,
                original_name=os.path.basename(file_path),
                file_type=ext,
                file_path=dest_path,
                area=area,
                description=f"Importado desde: {file_path}",
                owner_id=current_user.id,
                status="Procesando"
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)
            
            index_document(doc.id, dest_path, ext, doc.original_name, SessionLocal)
            
            results.append({"path": file_path, "status": "ok", "doc_id": doc.id})
                
        except Exception as e:
            results.append({"path": file_path, "status": "error", "error": str(e)})
    
    clear_cache()
    
    return {"results": results, "total": len(results)}

# ─── USUARIOS ─────────────────────────────────────
@app.get("/api/users")
def list_users(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.role != "Administrador":
        raise HTTPException(403, "Solo administradores")
    users = db.query(User).all()
    return [
        {"id": u.id, "username": u.username, "email": u.email, "role": u.role}
        for u in users
    ]


@app.put("/api/users/{user_id}")
def update_user(
    user_id: int,
    data: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "Administrador":
        raise HTTPException(403, "Solo administradores")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "Usuario no encontrado")

    if db.query(User).filter(User.username == data.username, User.id != user_id).first():
        raise HTTPException(400, "Ese nombre de usuario ya existe")
    if db.query(User).filter(User.email == data.email, User.id != user_id).first():
        raise HTTPException(400, "Ese email ya está registrado")

    user.username = data.username
    user.email    = data.email
    user.role     = data.role
    if data.password:
        user.hashed_password = get_password_hash(data.password)

    db.commit()
    return {"message": "Usuario actualizado"}

# ─── RAG: Limpiar caché manualmente ───────────────
@app.post("/api/rag/clear-cache")
def clear_cache_endpoint(current_user: User = Depends(get_current_user)):
    if current_user.role != "Administrador":
        raise HTTPException(403, "Solo administradores")
    clear_cache()
    return {"message": "Caché limpiado exitosamente"}


@app.get("/api/rag/stats")
def rag_stats(current_user: User = Depends(get_current_user)):
    if current_user.role != "Administrador":
        raise HTTPException(403, "Solo administradores")
    
    total_chunks = collection.count()
    indexed_docs = list_indexed_documents()
    
    return {
        "total_chunks": total_chunks,
        "indexed_documents": len(indexed_docs),
        "documents": indexed_docs
    }


# ─── NUEVOS ENDPOINTS PARA ESTADÍSTICAS E HISTORIAL ───────────────────────────

@app.get("/api/stats")
async def system_stats(current_user: User = Depends(get_current_user)):
    """Estadísticas del sistema"""
    if current_user.role != "Administrador":
        raise HTTPException(403, "Solo administradores")
    
    docs_info = get_documents_info()
    total_chunks = collection.count()
    
    return {
        "total_documents": docs_info["total"],
        "document_list": docs_info["documentos"],
        "total_chunks": total_chunks,
        "vectorstore_size": "ChromaDB",
        "user": current_user.username
    }


@app.get("/api/history/{user_id}")
async def user_history(user_id: int, current_user: User = Depends(get_current_user)):
    """Historial de consultas de un usuario"""
    if current_user.role != "Administrador" and current_user.id != user_id:
        raise HTTPException(403, "No tienes permiso para ver este historial")
    
    from backend.database import SessionLocal, ConversationMemory
    
    db = SessionLocal()
    try:
        memories = db.query(ConversationMemory).filter(
            ConversationMemory.user_id == user_id
        ).order_by(ConversationMemory.timestamp.desc()).limit(50).all()
        
        historial = []
        for m in memories:
            historial.append({
                "session_id": m.session_id,
                "role": m.role,
                "content": m.content[:500],
                "timestamp": m.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        return {"user_id": user_id, "total": len(historial), "history": historial}
    finally:
        db.close()


# ─── DESCARGA DE ARCHIVOS TEMPORALES ──────────────────────────────────────────

@app.get("/api/download/{filename}")
async def download_file(filename: str, current_user: User = Depends(get_current_user)):
    """Descarga un archivo temporal generado por el asistente (compatible Windows/Linux)"""
    # Buscar en tempfile.gettempdir() primero (funciona en Windows)
    temp_path = os.path.join(tempfile.gettempdir(), filename)
    
    # Si no está en temp, buscar en /tmp (Linux/Mac)
    if not os.path.exists(temp_path):
        alt_path = os.path.join("/tmp", filename)
        if os.path.exists(alt_path):
            temp_path = alt_path
    
    if not os.path.exists(temp_path):
        raise HTTPException(404, "El archivo no existe o expiró")
    
    def eliminar_archivo():
        time.sleep(300)  # 5 minutos
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"[MAIN] Archivo eliminado: {filename}")
    
    threading.Thread(target=eliminar_archivo, daemon=True).start()
    
    return FileResponse(
        temp_path,
        media_type="application/octet-stream",
        filename=filename
    )


# ─── FRONTEND ─────────────────────────────────────
@app.get("/")
def root():
    return FileResponse("frontend/landing.html")

@app.get("/landing.html")
def landing():
    return FileResponse("frontend/landing.html")

@app.get("/index.html")
def index():
    return FileResponse("frontend/index.html")

@app.get("/app")
def app_page():
    return FileResponse("frontend/index.html")

# Archivos estáticos del frontend
app.mount("/", StaticFiles(directory="frontend"), name="frontend")