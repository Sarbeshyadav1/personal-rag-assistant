import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import Document
from .utils import get_loader_for_file, split_documents
from dotenv import load_dotenv

load_dotenv()

FAISS_DIR = os.path.join(os.path.dirname(__file__), "..", "faiss_index")

def ingest_file(path: str):
    """
    Load file, split into docs, create embeddings and save FAISS index to disk.
    Overwrites any existing index (simple MVP).
    """
    loader = get_loader_for_file(path)
    docs = loader.load()  # list[Document]
    # split
    split_docs = split_documents(docs)
    # embeddings
    embed = OpenAIEmbeddings()
    # create FAISS index
    index = FAISS.from_documents(split_docs, embed)
    # ensure save dir
    os.makedirs(FAISS_DIR, exist_ok=True)
    index.save_local(FAISS_DIR)
    return len(split_docs)

def load_index():
    """
    Returns FAISS index object (or None if not present)
    """
    if not os.path.exists(FAISS_DIR):
        return None
    embed = OpenAIEmbeddings()
    index = FAISS.load_local(FAISS_DIR, embed)
    return index