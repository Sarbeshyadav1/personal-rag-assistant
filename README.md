# Personal RAG Assistant (MVP)

A minimal Retrieval-Augmented Generation (RAG) system that lets you upload documents, builds a local FAISS index, and answers questions using OpenAI's models.

## 🚀 Quickstart

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows (CMD)
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

1. Run the backend server:

```bash
uvicorn backend.main:app --reload --port 8000
```

Or use Docker Compose:

```bash
docker-compose up --build -d
```

1. Open `frontend/index.html` in your browser and upload a file (pdf/txt/docx/md).

## 📁 Project Structure

```
personal-rag-assistant/
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── ingest.py
│   └── utils.py
├── frontend/
│   └── index.html
├── uploads/
├── faiss_index/
├── .env.example
├── .env
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
└── app.py
```

## 🧪 Running tests

- Create and activate a venv, then install dev deps:

```bash
pip install -r requirements-dev.txt
```

- Run tests:

```bash
python -m pytest -q
```

## ⚠️ Notes & Limitations

- Single FAISS index (ingest overwrites index).
- No authentication or multi-user support.
- Local storage only.

## 🚀 Next steps

- Add CI tests (added GitHub Actions workflow to run pytest on push/PR).
- Consider adding per-user indices and persistent storage (pgvector).

---

If you want, I can set up the virtual environment in this workspace and run the tests for you.
