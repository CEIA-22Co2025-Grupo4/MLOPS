FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY docker/fastapi/requirements.txt /requirements.txt
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt

# Copy application code
COPY src/api/app.py /app/app.py
COPY src/api/preprocessing.py /app/preprocessing.py

EXPOSE 8800
