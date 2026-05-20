🤖 SGD RAG · Sistema de Gestión Documental con IA
🚀 Plataforma inteligente que permite gestionar documentos y realizar consultas usando Inteligencia Artificial (RAG) con capacidades avanzadas de análisis, exportación y orquestación.

🧠 ¿Qué hace este sistema?
SGD RAG combina gestión documental con IA para:

📂 Subir y organizar documentos (PDF, Word, Excel, PowerPoint, TXT, CSV, imágenes)

🔎 Buscar información semántica dentro de archivos (no solo palabras clave)

🤖 Responder preguntas con IA usando el contenido real de tus documentos

📄 Generar resúmenes automáticos de documentos

📊 Analizar Excel con estadísticas (min, max, promedio, suma)

📈 Crear gráficas con datos reales de Excel/CSV

📑 Exportar respuestas a Word y PDF con formato profesional

🗂️ Comparar documentos entre sí con análisis de IA

🧠 Memoria de conversación para mantener el contexto

🌐 Detección automática de idioma (respuesta en español/inglés/portugués)

🎨 Indicador visual de confianza en las respuestas

👥 Gestionar usuarios con roles (administrador / estándar)

🧩 Módulos principales
💬 Asistente IA
Responde preguntas sobre documentos usando RAG

Mantiene memoria de la conversación

Recuerda sobre qué documento estás hablando

Detecta errores de escritura (fuzzy matching)

Muestra indicador de confianza 🟢🟡🟠🔴

Puede generar gráficas con datos reales

📁 Gestión de documentos
Subida de archivos (PDF, DOCX, XLSX, PPTX, TXT, CSV, PNG, JPG)

Indexación automática en segundo plano

Escaneo e importación de carpetas completas

Resumen automático de documentos

Soporte para múltiples hojas de Excel

Extracción de texto de imágenes (Gemini Vision / OCR)

👥 Gestión de usuarios
Roles: Administrador y Usuario Estándar

Cada usuario ve solo sus documentos

Historial de consultas por usuario

📊 Exportación y análisis
Exportar respuestas a Word (.docx) con formato profesional

Exportar respuestas a PDF (.pdf)

Generar informes completos de conversación

Crear gráficas con datos reales de Excel/CSV

Comparar documentos entre sí con análisis IA

🔎 Búsqueda inteligente
Búsqueda semántica con embeddings vectoriales

Filtrado por documentos específicos

n_results dinámico según longitud de pregunta

Caché de respuestas (5 minutos)

⚙️ Tecnologías utilizadas
Capa	Tecnología
Backend	Python + FastAPI
Base de datos	SQLite con SQLAlchemy
Vector store	ChromaDB
Embeddings	Sentence Transformers (paraphrase-multilingual-MiniLM-L12-v2)
IA / LLM	Gemini 2.5 Flash / Gemini Vision
Orquestación	LangChain + LangGraph
Exportación Word	python-docx
Exportación PDF	reportlab
Gráficas	matplotlib
Procesamiento Excel	pandas, openpyxl, xlrd
OCR imágenes	pytesseract (fallback)
Frontend	HTML + CSS + JavaScript