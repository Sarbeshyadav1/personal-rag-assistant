#!/bin/bash

# Personal RAG Assistant - Quick Start Script

echo "🚀 Starting Personal RAG Assistant..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from .env.example..."
    cp .env.example .env
    echo "✏️  Please edit .env and add your OPENAI_API_KEY"
    exit 1
fi

# Check if venv exists
if [ ! -d .venv ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install requirements
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Run the app
echo "✅ Starting server on http://localhost:8000"
uvicorn backend.main:app --reload --port 8000
