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
make drift        # Ver instrucciones de drift monitoring

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
│   │   ├── app.py
│   │   └── preprocessing.py      # Preprocesamiento para inference
│   └── training/                 # Scripts ML
│       ├── train_xgboost.py
│       ├── champion_challenger.py
│       └── predictor.py
├── airflow/                      # Apache Airflow
│   ├── dags/
│   │   ├── etl_process_taskflow.py
│   │   ├── drift_process_taskflow.py  # Drift monitoring
│   │   └── etl_helpers/
│   │       ├── inference_preprocessing.py
│   │       └── ...
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
                                                                 │
                          ┌──────────────────────────────────────┘
                          ▼
                   ┌─────────────┐
                   │   Drift     │  ◀── Weekly monitoring
                   │  Monitoring │
                   └─────────────┘
                          │
                          ▼
                   Retrain if needed
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
| POST | `/predict` | Predicción individual (features preprocesadas) |
| POST | `/predict/batch` | Predicción por lote (features preprocesadas, max 1000) |
| POST | `/predict/raw` | Predicción individual (datos crudos) |
| POST | `/predict/raw/batch` | Predicción por lote (datos crudos, max 1000) |
| GET | `/model/info` | Info del modelo |
| POST | `/model/reload` | Recargar modelo |

### Predicción con datos crudos

Los endpoints `/predict/raw` y `/predict/raw/batch` aceptan datos sin preprocesar:

```bash
curl -X POST "http://localhost:8800/predict/raw" \
  -H "Content-Type: application/json" \
  -d '{
    "iucr": "0820",
    "primary_type": "THEFT",
    "location_description": "STREET",
    "date": "2024-01-15 14:30:00",
    "x_coordinate": 1176096.0,
    "y_coordinate": 1912547.0,
    "distance_crime_to_police_station": 1250.5
  }'
```

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

## 📊 Drift Monitoring

<details>
<summary><strong>Click para expandir</strong></summary>

### Prerequisitos

Antes de ejecutar drift monitoring, es necesario:

1. **Ejecutar el ETL** - Para tener datos de entrenamiento
2. **Entrenar el modelo** - `make train` crea el archivo de referencia automáticamente
3. **Tener el modelo en la API** - `make champion && make reload`

### Primera Ejecución

Si ejecutas el DAG de drift **sin haber entrenado el modelo**, verás este warning:

```
NO REFERENCE DATA AVAILABLE
Drift monitoring requires a reference dataset from training.
Please run 'make train' to create the reference data.
```

**Solución**: Ejecuta `make train` primero. El entrenamiento crea automáticamente el archivo `drift/reference/reference_{fecha}.csv`.

### Ejecución Normal

Una vez que existe el archivo de referencia:

1. Ir a Airflow: http://localhost:8080
2. Buscar DAG: `drift_with_taskflow`
3. Click en "Trigger DAG" (play button)
4. Configurar parámetros si es necesario (ver abajo)
5. Click "Trigger"

### Parámetros del DAG

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `test_mode` | `false` | Si es `true`, usa delay mínimo de 2 días (para testing) |
| `data_delay_days` | `null` | Override manual del delay (null = usar config) |

**Para testing** (datos más recientes):
```
test_mode: true
```

**Para override específico**:
```
data_delay_days: 3
```

### Qué hace el DAG

1. **Descarga datos recientes** de Chicago Data Portal (últimos 7 días, con delay configurable)
2. **Preprocesa** los datos (encoding, scaling) igual que en entrenamiento
3. **Obtiene predicciones** del modelo via API (`/predict/batch`)
4. **Calcula métricas de drift**:
   - **Feature Drift** (PSI, KS-test) - cambios en distribución de features
   - **Prediction Drift** - cambios en distribución de predicciones
   - **Concept Drift** - degradación de accuracy (si hay labels)
5. **Alerta** si se detecta drift significativo

### Tipos de Drift y Umbrales

| Tipo | Métrica | Umbral | Interpretación |
|------|---------|--------|----------------|
| Feature Drift | PSI | > 0.2 | Cambio significativo en distribución |
| Feature Drift | KS | > 0.1 | Test Kolmogorov-Smirnov significativo |
| Concept Drift | Accuracy Delta | > 0.05 | Degradación de accuracy vs referencia |

**PSI (Population Stability Index)**:
- < 0.1: Sin cambio significativo
- 0.1 - 0.2: Cambio moderado
- \> 0.2: Cambio significativo (requiere atención)

### Archivos generados

```
s3://data/drift/
├── reference/
│   └── reference_{fecha}.csv  # Creado por make train
├── current/
│   └── current_{fecha}.csv    # Datos actuales (cada ejecución)
└── results/
    └── drift_{fecha}.csv      # Métricas de drift
```

### Flujo Completo (Nuevo Proyecto)

```bash
# 1. Instalar y levantar servicios
make install

# 2. Ejecutar ETL (en Airflow UI)
make airflow
# -> Trigger 'etl_with_taskflow', esperar ~15 min

# 3. Entrenar modelo (crea referencia automáticamente)
make train

# 4. Configurar modelo en API
make champion
make reload

# 5. Ejecutar drift monitoring (en Airflow UI)
# -> Trigger 'drift_with_taskflow' con test_mode: true
```

### Troubleshooting

**Error: "No crime data available for period..."**
- Chicago Data Portal tiene delay de publicación (3-7 días)
- Solución: Aumentar `data_delay_days` o usar datos de fecha anterior

**Error: "No reference dataset found..."**
- No se ha entrenado el modelo
- Solución: Ejecutar `make train`

**Error: "API call failed..."**
- El modelo no está cargado en la API
- Solución: `make champion && make reload`

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
