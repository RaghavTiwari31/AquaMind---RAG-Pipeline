# 🌊 FLOATCHAT — Retrieval-Augmented Generation with FastAPI, pgvector, and Dash

This project is a **Retrieval-Augmented Generation (RAG) system** built on top of the ARGO oceanographic database.  

It combines:
- **Ingestion Layer** ingests netcdf files into postgre database
- **FastAPI** as the backend
- **PostgreSQL + pgvector** for embeddings storage and similarity search
- **Sentence-Transformers** to embed text and summaries
- **Google Gemini API** for generating SQL queries and natural language answers
- **Dash** as the frontend for an interactive chat-style interface
- **MCP (Model Context Protocol)** for session and tool management

### What it does
- Lets users ask natural language questions about ARGO_D (e.g. “average salinity in 2025”).
- Retrieves relevant rows and schema context using embeddings from pgvector.
- Uses Gemini to generate safe, read-only SQL queries or fallback text answers.
- Executes those queries against the ARGO_D database.
- Displays results in a Dash UI with context, SQL, tables, charts, and map visualizations.

---

## 📺 Demo Video

https://github.com/user-attachments/assets/825df63e-3e77-45a4-bc24-6cc3b27b6be2

---

## 🚀 Deployment
👉 [Live Deployment Link]

---

## ▶️ How to Run

### Backend (FastAPI)
uvicorn backend.app.main:app --host 0.0.0.0 --port 8080 --reload

### Frontend 
python frontend/dash_app.py



