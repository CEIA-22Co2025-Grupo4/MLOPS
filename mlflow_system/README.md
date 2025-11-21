# MLFlow System - Infraestructura Docker

## 🏗️ Arquitectura

Este directorio contiene la configuración de Docker para el sistema MLFlow completo:

- **PostgreSQL** (Puerto 5432): Almacena metadatos (experimentos, runs, parámetros, métricas)
- **MinIO** (Puertos 9000, 9001): Almacenamiento S3-compatible para artefactos (modelos, gráficos)
- **MLFlow Server** (Puerto 5001): API REST + UI Web

## 🚀 Inicio Rápido

### Iniciar servicios
```bash
cd mlflow_system
docker compose up -d
```

### Verificar estado
```bash
docker compose ps
```

### Ver logs
```bash
# Todos los servicios
docker compose logs -f

# Solo MLFlow
docker compose logs -f mlflow
```

### Detener servicios
```bash
docker compose down
```

### Limpiar todo (incluye volúmenes)
```bash
docker compose down --volumes --rmi all
```

## 🌐 URLs de Acceso

- **MLFlow UI**: http://localhost:5001
- **MinIO Console**: http://localhost:9001
  - Usuario: `minio`
  - Password: `minio123`
- **PostgreSQL**: localhost:5432
  - Usuario: `postgres`
  - Password: `postgres`
  - Database: `mlflow_db`

## 📝 Configuración Cliente Python

```python
import os
import mlflow

# Variables de entorno para MinIO
os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
os.environ['AWS_ENDPOINT_URL_S3'] = 'http://localhost:9000'

# Conectar al servidor MLFlow
mlflow.set_tracking_uri('http://localhost:5001')

# Verificar conexión
print(f"✅ Conectado a: {mlflow.get_tracking_uri()}")
```

## 🔍 Troubleshooting

### Verificar salud de servicios
```bash
# PostgreSQL
docker exec mlflow_postgres pg_isready -U postgres

# MinIO
docker exec mlflow_minio mc ready local

# MLFlow
curl http://localhost:5001/health
```

### Recrear servicios
```bash
docker compose down
docker compose up -d --build
```

### Ver logs de errores
```bash
docker compose logs mlflow | grep -i error
```

## 📊 Volúmenes Docker

Los datos persisten en volúmenes Docker:
- `db_data`: Base de datos PostgreSQL
- `minio_data`: Artefactos en MinIO

Para hacer backup:
```bash
docker run --rm -v mlflow_system_db_data:/data -v $(pwd):/backup alpine tar czf /backup/db_backup.tar.gz /data
```

## ⚠️ Notas Importantes

1. **Primera ejecución**: Los servicios pueden tardar ~30 segundos en estar completamente listos
2. **Puertos**: Asegúrate de que los puertos 5001, 5432, 9000, 9001 estén disponibles
3. **Credenciales**: Cambiar passwords en producción
4. **Volúmenes**: Los datos persisten entre reinicios hasta hacer `docker compose down --volumes`

## 🔄 Actualizar MLFlow

1. Editar `dockerfiles/mlflow/requirements.txt`
2. Reconstruir imagen:
   ```bash
   docker compose build mlflow
   docker compose up -d mlflow
   ```
