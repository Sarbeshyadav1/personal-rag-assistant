FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PDF/DOCX processing
RUN apt-get update && apt-get install -y \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
ENV PORT=8000
EXPOSE 8000

# Run the application (using modular structure)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
