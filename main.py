import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from .ingest import ingest_file, load_index
from .utils import UPLOAD_DIR
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

load_dotenv()
PORT = int(os.getenv("PORT", 8000))

app = FastAPI(title="Personal RAG Assistant")

# allow local frontend to call
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    chat_history: list = []

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    dest = os.path.join(UPLOAD_DIR, file.filename)
    # save file
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # ingest
    count = ingest_file(dest)
    return {"status": "ingested", "chunks": count, "filename": file.filename}

@app.post("/chat")
async def chat(q: Query):
    index = load_index()
    if index is None:
        raise HTTPException(status_code=400, detail="No index found. Upload documents first.")
    retriever = index.as_retriever()
    llm = OpenAI(temperature=0)
    qa = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = qa({"question": q.question, "chat_history": q.chat_history})
    # truncate source docs for client
    sources = [{"page_content": d.page_content[:400]} for d in result.get("source_documents", [])]
    return {"answer": result.get("answer"), "source_documents": sources}