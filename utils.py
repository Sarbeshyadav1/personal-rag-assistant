import os
from typing import Any

# Document loader imports may be optional (not installed in minimal env).
# Try importing the standard LangChain loaders; if unavailable, provide simple fallbacks
try:
    from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader  # type: ignore
except Exception:
    TextLoader = None
    PyPDFLoader = None
    UnstructuredWordDocumentLoader = None

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    # Fallback simple splitter when langchain's splitter is unavailable
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

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

def get_loader_for_file(path: str) -> Any:
    lower = path.lower()
    if lower.endswith(".pdf"):
        if PyPDFLoader is not None:
            return PyPDFLoader(path)
        # fall back to simple text loader (may not extract text from binary PDFs)
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
