FROM python:3.12-slim

# Dependencias nativas para xgboost/matplotlib/s3fs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /trainer

# Instala requirements específicos del trainer
COPY dockerfiles/trainer/requirements.txt /trainer/requirements.txt
RUN pip install --no-cache-dir -r /trainer/requirements.txt

# Copia el código
COPY mlflow_scripts/ /trainer/

# Variables por defecto
ENV AWS_ACCESS_KEY_ID=minio \
    AWS_SECRET_ACCESS_KEY=minio123 \
    MLFLOW_S3_ENDPOINT_URL=http://s3:9000 \
    AWS_ENDPOINT_URL_S3=http://s3:9000 \
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    MINIO_ENDPOINT=http://s3:9000 \
    MINIO_BUCKET=data \
    MINIO_PREFIX=ml-ready-data

CMD ["python", "mlflow_xgboost_poc_docker.py"]
