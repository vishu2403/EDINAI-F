FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/backend:$PYTHONPATH \
    # Optional but useful so Tesseract knows where to find traineddata
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

WORKDIR /app

# Copy only requirements first to leverage Docker layer cache
COPY backend/requirements.txt ./requirements.txt

# Install system dependencies: Tesseract, Ghostscript, etc.
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        build-essential \
        libpq-dev \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-hin \
        tesseract-ocr-guj \
        ghostscript \
    && pip install --upgrade pip \
    && pip install -r requirements.txt \
    # Make sure pytesseract (and optionally ocrmypdf) are available
    && pip install pytesseract ocrmypdf \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy the backend code
COPY backend ./backend

EXPOSE 8000

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
