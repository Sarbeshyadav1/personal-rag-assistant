#!/usr/bin/env python3
"""
Single-file Personal RAG Assistant (MVP)

- Serves a minimal web UI at GET /
- POST /upload to upload and ingest files (txt, md, pdf, docx)
- POST /chat to ask questions (uses ConversationalRetrievalChain)
- Stores local FAISS index under ./faiss_index (overwrites on new ingest)

Usage:
  1) Create .env with OPENAI_API_KEY (see .env.example)
  2) pip install -r requirements.txt
  3) python app.py
  4) Open http://localhost:8000
"""
import os
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Any

# FAISS import: try LangChain locations first, then langchain_community, else set FAISS=None
try:
from langchain_community.vectorstores import FAISS
except Exception:
    try:
        from langchain_community.vectorstores import FAISS

    except Exception:
        try:
            from langchain_community.vectorstores import FAISS
        except Exception:
            FAISS = None

# LangChain imports (use fallbacks when optional packages aren't installed)
try:
    from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader  # type: ignore
except Exception:
    try:
        from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader  # type: ignore
    except Exception:
        TextLoader = PyPDFLoader = UnstructuredWordDocumentLoader = None

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
except Exception:
    # Simple fallback splitter
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=200):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_documents(self, docs):
            out = []
            for d in docs:
                text = getattr(d, "page_content", str(d))
                step = max(1, self.chunk_size - self.chunk_overlap)
                for i in range(0, max(1, len(text)), step):
                    out.append(type("Doc", (), {"page_content": text[i:i + self.chunk_size]}))
            return out

# Embeddings import (LangChain moved OpenAIEmbeddings under embeddings.openai in newer versions)
try:
    from langchain.embeddings.openai import OpenAIEmbeddings  # preferred
except Exception:
    try:
        from langchain.embeddings import OpenAIEmbeddings  # older compat
    except Exception:
        OpenAIEmbeddings = None


from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

# Config
PORT = int(os.getenv("PORT", 8000))
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", None)  # let LangChain choose default if None
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
FAISS_DIR = os.path.join(os.path.dirname(__file__), "faiss_index")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Minimal frontend (single page)
FRONTEND_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Personal RAG Assistant (MVP)</title>
  <style>
    body{font-family: Arial; max-width:900px;margin:20px auto}
    #chat{border:1px solid #ddd;padding:10px;height:400px;overflow:auto}
    .msg{margin:6px 0}
    .user{color:blue}
    .bot{color:green}
    #uploadStatus{margin-left:10px;color:#444}
  </style>
</head>
<body>
  <h2>Personal RAG Assistant (MVP)</h2>
  <div>
    <label>Upload file (txt/pdf/docx/md):</label>
    <input id="file" type="file" />
    <button onclick="upload()">Upload & Ingest</button>
    <span id="uploadStatus"></span>
  </div>
  <hr/>
  <div id="chat"></div>
  <input id="question" style="width:70%" placeholder="Ask a question..." />
  <button onclick="ask()">Ask</button>

  <script>
    const apiBase = "";
    let history = [];

    function appendMessage(who, text) {
      const d = document.createElement("div");
      d.className = "msg";
      d.innerHTML = `<strong class="${who==='user'?'user':'bot'}">${who}:</strong> ${text}`;
      document.getElementById("chat").appendChild(d);
      document.getElementById("chat").scrollTop = document.getElementById("chat").scrollHeight;
    }

    async function upload() {
      const f = document.getElementById("file").files[0];
      if (!f) return alert("Pick a file");
      const fd = new FormData();
      fd.append("file", f);
      document.getElementById("uploadStatus").innerText = "Uploading...";
      const res = await fetch("/upload", {method:"POST", body:fd});
      const j = await res.json();
      if (res.ok) {
        document.getElementById("uploadStatus").innerText = `${j.status} (${j.chunks} chunks)`;
      } else {
        document.getElementById("uploadStatus").innerText = `Error: ${j.detail || JSON.stringify(j)}`;
      }
    }

    async function ask() {
      const q = document.getElementById("question").value;
      if (!q) return;
      appendMessage('user', q);
      document.getElementById("question").value = "";
      const res = await fetch("/chat", {
        method:"POST",
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({question: q, chat_history: history})
      });
      const j = await res.json();
      if (res.ok) {
        appendMessage('assistant', j.answer || '(no answer)');
        history.push([q, j.answer]);
        if (j.source_documents) {
          appendMessage('assistant', '<small>Sources:</small><br/>' + j.source_documents.map(s => s.page_content).join('<hr/>'));
        }
      } else {
        appendMessage('assistant', 'Error: ' + (j.detail || JSON.stringify(j)));
      }
    }
  </script>
</body>
</html>
"""

app = FastAPI(title="Single-file RAG Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helpers
# Minimal fallback loader used when LangChain loaders aren't available
class _SimpleTextLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.path = path
        self.encoding = encoding

    def load(self):
        with open(self.path, "r", encoding=self.encoding, errors="ignore") as f:
            class Doc:
                def __init__(self, text):
                    self.page_content = text
            return [Doc(f.read())]

SimpleTextLoader = _SimpleTextLoader

def get_loader_for_file(path: str):
    lower = path.lower()
    if lower.endswith(".pdf"):
        if PyPDFLoader is not None:
            return PyPDFLoader(path)
        return SimpleTextLoader(path)
    if lower.endswith(".txt") or lower.endswith(".md"):
        if TextLoader is not None:
            return TextLoader(path, encoding="utf-8")
        return SimpleTextLoader(path, encoding="utf-8")
    if lower.endswith(".docx") or lower.endswith(".doc"):
        if UnstructuredWordDocumentLoader is not None:
            return UnstructuredWordDocumentLoader(path)
        return SimpleTextLoader(path, encoding="utf-8")
    # fallback to text loader
    if TextLoader is not None:
        return TextLoader(path, encoding="utf-8")
    return SimpleTextLoader(path, encoding="utf-8")

def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def ingest_file(path: str):
    """
    Load file, split into chunks, create embeddings and save FAISS index.
    Overwrites existing index (simple MVP).
    Returns number of chunks created.
    """
    if FAISS is None:
        raise RuntimeError(
            "FAISS vector store is not available. Install 'faiss-cpu' and ensure a compatible langchain version. "
            "On Windows prefer using conda: 'conda install -c conda-forge faiss-cpu'. Or use WSL/Docker."
        )
    if OpenAIEmbeddings is None:
        raise RuntimeError(
            "OpenAIEmbeddings is not available. Ensure 'langchain' and 'openai' packages are installed."
        )

    loader = get_loader_for_file(path)
    docs = loader.load()
    split_docs = split_documents(docs)
    # embeddings uses OPENAI_API_KEY from env
    embed = OpenAIEmbeddings()
    # create FAISS index (in-memory) and persist
    index = FAISS.from_documents(split_docs, embed)
    os.makedirs(FAISS_DIR, exist_ok=True)
    index.save_local(FAISS_DIR)
    return len(split_docs)

def load_index():
    """Return a FAISS index object or raise when dependencies are missing"""
    if FAISS is None:
        raise RuntimeError(
            "FAISS vector store is not available. Install 'faiss-cpu' and ensure a compatible langchain version."
        )
    if not os.path.exists(FAISS_DIR):
        return None
    if OpenAIEmbeddings is None:
        raise RuntimeError("OpenAIEmbeddings is not available. Ensure 'langchain' and 'openai' packages are installed.")
    embed = OpenAIEmbeddings()
    index = FAISS.load_local(FAISS_DIR, embed)
    return index

# API models
class Query(BaseModel):
    question: str
    chat_history: list = []

# Routes
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(content=FRONTEND_HTML, status_code=200)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    dest = os.path.join(UPLOAD_DIR, file.filename)
    # Save uploaded file
    try:
        with open(dest, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    # Ingest and build index
    try:
        chunks = ingest_file(dest)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")
    return JSONResponse({"status": "ingested", "chunks": chunks, "filename": file.filename})

@app.post("/chat")
async def chat(q: Query):
    index = load_index()
    if index is None:
        raise HTTPException(status_code=400, detail="No index found. Upload documents first.")
    retriever = index.as_retriever()
    llm = OpenAI(temperature=0)
    qa = ConversationalRetrievalChain.from_llm(llm, retriever)
    try:
        result = qa({"question": q.question, "chat_history": q.chat_history})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM chain error: {e}")
    # Truncate long source doc contents for response
    sources = []
    for d in result.get("source_documents", []):
        text = getattr(d, "page_content", str(d))[:800]
        sources.append({"page_content": text})
    return JSONResponse({"answer": result.get("answer"), "source_documents": sources})

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    # Run with: python app.py
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=True)