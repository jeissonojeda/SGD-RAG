import os
import uuid
import shutil
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import get_db, create_tables, User, Document
from backend.auth import (
    verify_password, get_password_hash,
    create_access_token, get_current_user
)
from backend.rag import index_document, delete_document_from_index, query_rag

app = FastAPI(title="SGD-IA CESMAG", version="1.0")

# ─── CORS ─────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── ARCHIVOS ─────────────────────────────────────
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 🔥 SOLUCIÓN CLAVE: SERVIR CSS Y JS
app.mount("/css", StaticFiles(directory="frontend/css"), name="css")
app.mount("/js", StaticFiles(directory="frontend/js"), name="js")

# ─── MODELOS ──────────────────────────────────────
class UserRegister(BaseModel):
    username: str
    email: str
    password: str
    role: Optional[str] = "Estándar"

class ChatRequest(BaseModel):
    question: str
    doc_ids: Optional[list] = None

# ─── STARTUP ──────────────────────────────────────
@app.on_event("startup")
def startup():
    create_tables()
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
    allowed = ["pdf", "docx", "doc", "txt", "csv", "xlsx", "xls"]

    ext = file.filename.split(".")[-1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Tipo de archivo no permitido")

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
        owner_id=current_user.id
    )

    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Indexar IA
    success = index_document(doc.id, file_path, ext, file.filename)
    if success:
        doc.status = "Indexado"
        db.commit()

    return {"message": "Documento subido", "doc_id": doc.id}


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

    return {"message": "Documento eliminado"}

# ─── CHAT IA ──────────────────────────────────────
@app.post("/api/chat")
def chat(request: ChatRequest, current_user: User = Depends(get_current_user)):
    return query_rag(request.question, request.doc_ids)

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

# ─── FRONTEND ─────────────────────────────────────
@app.get("/")
def root():
    return FileResponse("frontend/index.html")