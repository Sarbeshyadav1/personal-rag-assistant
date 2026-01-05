
# Personal RAG Assistant 🤖📄

A **Personal RAG (Retrieval-Augmented Generation) Assistant**  smart AI-powered Personal RAG Assistant that enables users to upload documents and interact with them through natural language questions using Retrieval-Augmented Generation (RAG).

---

## 🔍 What is RAG?
Retrieval-Augmented Generation (RAG) is an AI approach that combines:
- Document retrieval
- Large Language Models (LLMs)

to generate accurate, context-aware answers from user-provided documents.

---

## ✨ Features
- 📂 Upload and process documents
- 🧠 AI-powered question answering
- 📊 Tracks document and question count
- ⚡ Clean and user-friendly interface
- 🔐 Personal document-based AI assistant

---

## 🧠 System Architecture (Theory)
1. Document upload by user  
2. Text extraction from documents  
3. Embedding generation  
4. Storage in vector database  
5. Query embedding creation  
6. Context retrieval  
7. AI-generated response  

---

## 🛠️ Tech Stack
- Frontend: HTML, CSS, JavaScript (or React)
- Backend: Python (Flask / FastAPI)
- AI Model: Large Language Model (LLM)
- Vector Store: FAISS / ChromaDB
- Embeddings: OpenAI / HuggingFace

---

## 🚀 How to Run the Project Locally

### 🔹 Prerequisites
- Python 3.10+
- Git

### 🔹 Steps
```bash
git clone https://github.com/Sarbeshyadav1/personal-rag-assistant.git
cd personal-rag-assistant

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
python app.py

http://localhost:8000


personal-rag-assistant/
│── app.py
│── requirements.txt
│── templates/
│── static/
│── uploads/
│── README.md
