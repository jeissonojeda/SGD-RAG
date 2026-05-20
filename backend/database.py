from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

DATABASE_URL = "sqlite:///./sgd.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="Estándar")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relaciones
    documents = relationship("Document", back_populates="owner")
    conversations = relationship("ConversationMemory", back_populates="user")


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    original_name = Column(String)
    file_type = Column(String)
    file_path = Column(String)
    status = Column(String, default="Activo")
    area = Column(String, default="General")
    description = Column(Text, default="")
    summary = Column(Text, nullable=True)  # 📝 Mejora 7: Resumen automático del documento
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    owner_id = Column(Integer, ForeignKey("users.id"))
    
    # Relaciones
    owner = relationship("User", back_populates="documents")


# 🆕 Mejora 6: Memoria de conversación
class ConversationMemory(Base):
    __tablename__ = "conversation_memory"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    session_id = Column(String(100))  # Identificador de la conversación (para múltiples chats)
    role = Column(String(20))  # 'user' o 'assistant'
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relaciones
    user = relationship("User", back_populates="conversations")


# 🆕 Mejora 9: Feedback de confianza (opcional, para mejorar el sistema)
class ConfidenceFeedback(Base):
    __tablename__ = "confidence_feedback"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    question = Column(Text)
    confidence_score = Column(Float)
    was_helpful = Column(Integer, default=0)  # 0=no feedback, 1=sí útil, 2=no útil
    timestamp = Column(DateTime, default=datetime.utcnow)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Crea todas las tablas en la base de datos"""
    Base.metadata.create_all(bind=engine)
    print("[DB] Tablas creadas/verificadas correctamente")