@echo off
echo 🚀 Starting Personal RAG Assistant...

REM Check if .env exists
if not exist .env (
    echo ⚠️  .env file not found. Creating from .env.example...
    copy .env.example .env
    echo ✏️  Please edit .env and add your OPENAI_API_KEY
    pause
    exit /b 1
)

REM Check if venv exists
if not exist .venv (
    echo 📦 Creating virtual environment...
    python -m venv .venv
)

REM Activate venv
call .venv\Scripts\activate.bat

REM Install requirements
echo 📥 Installing dependencies...
pip install -q -r requirements.txt

REM Run the app
echo ✅ Starting server on http://localhost:8000
uvicorn backend.main:app --reload --port 8000
