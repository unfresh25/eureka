# Eureka Backend

Agente legal colombiano con RAG, generación de documentos y memoria de conversación.

## Stack
- FastAPI · Python 3.12 · Pydantic v2
- Qwen2.5-VL 7B (Ollama) — chat e intent routing
- BGE-M3 (Ollama) — embeddings densos
- BGE-ReRanker-V2-M3 (HuggingFace) — reranking
- Qdrant — búsqueda vectorial híbrida
- Supabase — historial, memoria y metadata

## Setup

### 1. Requisitos previos
```bash
# Ollama corriendo con los modelos
ollama pull qwen2.5vl:7b
ollama pull bge-m3

# Qdrant (Docker)
docker run -p 6333:6333 qdrant/qdrant

# Supabase: crear proyecto y correr migrations/001_initial.sql
```

### 2. Instalar dependencias
```bash
cd backend/
pip install -e ".[dev]"
```

### 3. Configurar variables de entorno
```bash
cp .env.example .env
# Editar .env con tus credenciales de Supabase
```

### 4. Correr el servidor
```bash
uvicorn src.api.main:app --reload --port 8000
```

### 5. Ingestar documentos legales
```bash
# Copiar PDFs/DOCX en src/ingestion/docs/
python -m src.ingestion.ingest --docs-path src/ingestion/docs/
```

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/chat` | Chat (RAG o generación) |
| GET | `/api/v1/conversations` | Listar conversaciones |
| GET | `/api/v1/conversations/{id}/messages` | Historial |
| DELETE | `/api/v1/conversations/{id}` | Eliminar conversación |
| GET | `/api/v1/documents/{id}/download` | Descargar DOCX |

API docs: http://localhost:8000/docs

## Tests
```bash
pytest tests/ -v
```
