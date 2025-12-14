# Chicago Crime Arrest Prediction - MLOps Pipeline

Predicción de arrestos en crímenes reportados en la Ciudad de Chicago mediante un pipeline MLOps end-to-end.

## 👥 Autores

- Daniel Eduardo Peñaranda Peralta
- Jorge Adrián Alvarez
- María Belén Cattaneo
- Nicolás Valentín Ciarrapico
- Sabrina Daiana Pryszczuk

---

## ⚙️ Paso 1: Configuración Inicial

### Requisitos

- **Docker** y **Docker Compose** instalados
- **8GB RAM** mínimo disponible
- **10GB** espacio en disco

### Configurar Socrata API Token (Requerido)

El pipeline ETL descarga datos del **Chicago Data Portal** usando la API de Socrata. **Debes configurar un App Token antes de instalar**.

#### Obtener el Token

1. Ir a https://data.cityofchicago.org/
2. Crear una cuenta o iniciar sesión (click en "Sign In" arriba a la derecha)
3. Una vez logueado, ir a tu perfil (click en tu nombre) → **"Developer Settings"**
4. Click en **"Create New App Token"**
5. Completar el formulario:
   - **Application Name**: Nombre descriptivo (ej: "MLOps CEIA")
   - **Description**: Descripción breve
   - **Website** (opcional): Puede dejarse vacío
6. Click en **"Save"** y copiar el **App Token** generado

#### Configurar el Token en el Proyecto

Abrir el archivo `Makefile` y reemplazar el token en la **línea 4**:

```makefile
SOCRATA_TOKEN ?= tu_token_generado_aqui
```

> ⚠️ **Importante**: Sin el token configurado, el pipeline ETL fallará o será extremadamente lento.

### Siguiente paso

Una vez configurado el token, continúa con la instalación según tu sistema operativo:

- **macOS / Linux** → [Ir a instalación](#-paso-2a-instalación-macos--linux)
- **Windows** → [Ir a instalación](#-paso-2b-instalación-windows-wsl2)

---

## 📦 Paso 2a: Instalación (macOS / Linux)

```bash
cd MLOPS-main
make install
```

Este comando automáticamente:
- Crea directorios necesarios (`airflow/logs`, etc.)
- Configura permisos
- Crea archivo `.env` con configuración por defecto
- Construye todos los contenedores Docker
- Levanta todos los servicios

#### Verificar instalación

```bash
make status
```

Todos los servicios deben mostrar `(healthy)`. Esperar ~2 minutos si algunos servicios aún están iniciando.

#### Desinstalación

```bash
# Limpieza completa (elimina datos, logs, .env)
make uninstall

# Solo detener servicios (mantiene datos)
make down
```

**Siguiente paso** → [Ir a Ejecución](#-paso-3-ejecución)

---

## 📦 Paso 2b: Instalación (Windows WSL2)

Windows requiere **WSL2** (Windows Subsystem for Linux) ya que el proyecto usa comandos Unix.

### 1. Instalar WSL2

```powershell
# En PowerShell como Administrador
wsl --install
```

Reiniciar el equipo después de la instalación.

### 2. Instalar Docker Desktop

1. Descargar [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Durante la instalación, habilitar **WSL2 backend**
3. En Docker Desktop → Settings → Resources → WSL Integration → Habilitar para tu distro

### 3. Instalar el proyecto

```bash
# Abrir terminal WSL2 (Ubuntu)
wsl

# Navegar al proyecto
cd /mnt/c/Users/TuUsuario/MLOPS-main

# Instalar
make install
```

### 4. Verificar

```bash
make status
```

Todos los servicios deben mostrar `(healthy)`.

### Desinstalación

```bash
wsl
make uninstall
```

**Siguiente paso** → [Ir a Ejecución](#-paso-3-ejecución)

---

## 🔄 Paso 3: Ejecución

### Servicios disponibles

| Servicio | URL | Credenciales |
|----------|-----|--------------|
| Airflow | http://localhost:8080 | `airflow` / `airflow` |
| MLflow | http://localhost:5001 | — |
| MinIO | http://localhost:9001 | `minio` / `minio123` |
| API | http://localhost:8800/docs | — |

### Comandos (ejecutar en orden)

```bash
make airflow    # 1. Abrir Airflow → ejecutar DAG 'etl_with_taskflow' → esperar ~15min
make train      # 2. Entrenar modelo XGBoost
make champion   # 3. Asignar como champion
make reload     # 4. Cargar modelo en API
make api        # 5. Abrir documentación API
```

<details>
<summary><strong>Detalles de cada paso (click para expandir)</strong></summary>

#### 1. Ejecutar ETL Pipeline

En la UI de Airflow (http://localhost:8080):
- Buscar DAG: `etl_with_taskflow`
- Activar el toggle (si está pausado)
- Click ▶️ para ejecutar
- Esperar ~10-15 minutos hasta que todas las tareas estén en verde ✅

#### 2. Entrenar modelo

El comando `make train` entrena un modelo XGBoost y lo registra en MLflow.

#### 3. Asignar modelo como Champion

El comando `make champion` asigna el alias `champion` a la última versión del modelo.

#### 4. Cargar modelo en API

El comando `make reload` carga el modelo champion en la API.

#### 5. Usar la API

Ejemplo de predicción con cURL:

```bash
curl -X POST "http://localhost:8800/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "iucr_freq": 0.03,
    "primary_type_freq": 0.1,
    "location_description_freq": 0.05,
    "day_of_week_sin": 0.5,
    "x_coordinate_standardized": 0.12,
    "y_coordinate_standardized": -0.34,
    "distance_crime_to_police_station_standardized": 0.56
  }'
```

</details>

---

## 📋 Comandos Disponibles

```bash
make help         # Ver todos los comandos

# Instalación
make install      # Instalación completa
make uninstall    # Desinstalación completa
make setup        # Solo crear directorios y .env

# Servicios
make up           # Iniciar servicios
make down         # Detener servicios
make restart      # Reiniciar servicios
make status       # Ver estado
make logs         # Ver logs en tiempo real
make clean        # Eliminar contenedores y volúmenes

# ML Pipeline
make train        # Entrenar modelo XGBoost
make champion     # Asignar modelo como champion
make reload       # Recargar modelo en API

# Abrir interfaces
make airflow      # http://localhost:8080
make mlflow       # http://localhost:5001
make minio        # http://localhost:9001
make api          # http://localhost:8800/docs
```

---

## 🏗️ Arquitectura

<details>
<summary><strong>Click para expandir</strong></summary>

### Estructura del Proyecto

```
MLOPS-main/
├── src/                          # Código fuente
│   ├── api/                      # FastAPI
│   │   └── app.py
│   └── training/                 # Scripts ML
│       ├── train_xgboost.py
│       ├── champion_challenger.py
│       └── predictor.py
├── airflow/                      # Apache Airflow
│   ├── dags/
│   │   ├── etl_process_taskflow.py
│   │   └── etl_helpers/
│   └── secrets/
├── docker/                       # Dockerfiles
│   ├── airflow/
│   ├── fastapi/
│   ├── mlflow/
│   ├── postgres/
│   └── trainer/
├── tests/
├── docs/
├── docker-compose.yaml
├── Makefile
└── README.md
```

### Flujo MLOps

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  ETL        │ ──▶ │  Training   │ ──▶ │  Registry   │ ──▶ │  Serving    │
│  (Airflow)  │     │  (XGBoost)  │     │  (MLflow)   │     │  (FastAPI)  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
     │                    │                   │                   │
     ▼                    ▼                   ▼                   ▼
  MinIO               MLflow              Model              /predict
  s3://data          Tracking            Registry           Endpoint
```

</details>

---

## 🌐 API Reference

<details>
<summary><strong>Click para expandir</strong></summary>

### Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Predicción individual |
| POST | `/predict/batch` | Predicción por lote (max 1000) |
| GET | `/model/info` | Info del modelo |
| POST | `/model/reload` | Recargar modelo |

### Input Features (7 total)

| Feature | Descripción |
|---------|-------------|
| `iucr_freq` | Frequency encoding del código IUCR |
| `primary_type_freq` | Frequency encoding del tipo de crimen |
| `location_description_freq` | Frequency encoding de la ubicación |
| `day_of_week_sin` | Encoding cíclico del día de semana |
| `x_coordinate_standardized` | Coordenada X estandarizada |
| `y_coordinate_standardized` | Coordenada Y estandarizada |
| `distance_crime_to_police_station_standardized` | Distancia a comisaría (estandarizada) |

### Ejemplo Response

```json
{
  "prediction": true,
  "probability": 0.78,
  "model_version": "2",
  "timestamp": "2025-12-14T15:30:00Z"
}
```

</details>

---

## 🔧 ETL Pipeline

<details>
<summary><strong>Click para expandir</strong></summary>

### Etapas

1. **Setup S3** - Crea buckets en MinIO
2. **Download Data** - Descarga datos de Chicago Data Portal
3. **Enrich Data** - Agrega features geoespaciales y temporales
4. **Split Data** - División train/test (80/20)
5. **Process Outliers** - Manejo de outliers
6. **Encode Data** - Encoding de categóricas
7. **Scale Data** - Estandarización
8. **Balance Data** - SMOTE + Undersampling
9. **Extract Features** - Selección con Mutual Information
10. **Pipeline Summary** - Métricas consolidadas

### Datos de salida

```
s3://data/ml-ready-data/
├── train_{date}.csv    # ~173k registros × 7 features
└── test_{date}.csv     # ~46k registros × 7 features
```

</details>

---

## 🆘 Troubleshooting

<details>
<summary><strong>Click para expandir</strong></summary>

### Servicios no inician

```bash
make logs      # Ver logs
make restart   # Reiniciar
```

### Permisos en airflow/logs (Linux/Mac)

```bash
sudo chmod 777 airflow/logs
make restart
```

### Puerto 5000 ocupado (macOS)

MLflow usa puerto 5001 por defecto para evitar conflicto con AirPlay.

### ETL falla en download_data

Verificar que el Socrata Token está configurado correctamente en el `Makefile`.

### ETL falla en balance_data

Verificar que el ETL completo se ejecutó. Si hay errores previos, los datos pueden tener NaN.

### Modelo no carga en API

```bash
make champion  # Verificar que existe el alias
make reload    # Recargar
```

### Windows: make command not found

Usar WSL2:
```bash
wsl
cd /mnt/c/path/to/MLOPS-main
make install
```

</details>

---

## 📚 Documentación Adicional

- [Consigna del Proyecto](docs/CONSIGNAS.md)
- [API Documentation](http://localhost:8800/docs) (requiere servicios activos)
- [MLflow UI](http://localhost:5001) (requiere servicios activos)

---

## 📄 Licencia

MIT License - Ver archivo [LICENSE](LICENSE)
