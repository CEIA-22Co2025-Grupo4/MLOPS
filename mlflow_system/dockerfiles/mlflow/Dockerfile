FROM python:3.11-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt /requirements.txt

# Instalar dependencias Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /requirements.txt

# Exponer puerto MLFlow
EXPOSE 5000

# Comando por defecto (se sobrescribe en docker-compose)
CMD ["mlflow", "server", "--host", "0.0.0.0"]
