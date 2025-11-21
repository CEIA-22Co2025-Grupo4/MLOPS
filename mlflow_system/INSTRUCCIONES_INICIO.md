# 🚀 Instrucciones para Iniciar MLFlow System

## ⚠️ Problema Detectado

Docker Desktop no está corriendo actualmente.

## 📋 Pasos para Iniciar

### 1️⃣ Iniciar Docker Desktop

**Opción A - Inicio Manual:**
1. Buscar "Docker Desktop" en el menú de Windows
2. Hacer clic para iniciar
3. Esperar ~30 segundos a que esté completamente cargado
4. Verificar que el icono de Docker en la bandeja del sistema esté verde

**Opción B - Inicio desde PowerShell:**
```powershell
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
```

### 2️⃣ Verificar Docker Funcionando

```powershell
# Verificar que Docker está listo
docker --version
docker ps
```

Deberías ver:
```
Docker version X.X.X, build XXXXX
CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES
```

### 3️⃣ Iniciar Servicios MLFlow

```powershell
cd C:\Users\nicoc\Desktop\MLOPS\mlflow_system
docker compose up -d
```

### 4️⃣ Verificar Servicios Corriendo

```powershell
docker compose ps
```

Deberías ver 4 servicios:
- ✅ `mlflow_postgres` - running
- ✅ `mlflow_minio` - running  
- ✅ `mlflow_server` - running
- ✅ `mlflow_create_bucket` - exited (esto es normal)

### 5️⃣ Verificar Logs (Opcional)

```powershell
# Ver todos los logs
docker compose logs

# Ver solo MLFlow
docker compose logs mlflow

# Seguir logs en tiempo real
docker compose logs -f mlflow
```

### 6️⃣ Verificar Acceso Web

Abre tu navegador y visita:
- **MLFlow UI**: http://localhost:5001
- **MinIO Console**: http://localhost:9001
  - Usuario: `minio`
  - Password: `minio123`

## ✅ Verificación Final

Una vez que todo esté corriendo, ejecuta este script Python para verificar:

```python
import os
import mlflow

# Configurar conexión
os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
os.environ['AWS_ENDPOINT_URL_S3'] = 'http://localhost:9000'

mlflow.set_tracking_uri('http://localhost:5001')

# Verificar conexión
try:
    experiments = mlflow.search_experiments()
    print(f"✅ Conexión exitosa a MLFlow!")
    print(f"✅ Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"✅ Experimentos encontrados: {len(experiments)}")
except Exception as e:
    print(f"❌ Error de conexión: {e}")
```

## 🔄 Comandos Útiles

```powershell
# Ver estado de servicios
docker compose ps

# Ver logs
docker compose logs -f

# Reiniciar un servicio
docker compose restart mlflow

# Detener todos los servicios
docker compose down

# Detener y limpiar todo
docker compose down --volumes --rmi all
```

## 🆘 Troubleshooting

### Error: "port is already allocated"
Otro servicio está usando los puertos. Detenerlo o cambiar puertos en docker-compose.yml

### Error: "no space left on device"
Limpiar imágenes y volúmenes viejos:
```powershell
docker system prune -a --volumes
```

### Servicios no inician
Verificar logs:
```powershell
docker compose logs postgres
docker compose logs s3
docker compose logs mlflow
```

## 📞 Próximo Paso

Una vez que todos los servicios estén corriendo y MLFlow UI sea accesible en http://localhost:5001, estaremos listos para:

✅ **FASE 1 COMPLETADA**  
➡️ **FASE 2**: Crear notebook con XGBoost + MLFlow

---

**Estado Actual**: Esperando que inicies Docker Desktop y ejecutes `docker compose up -d`
