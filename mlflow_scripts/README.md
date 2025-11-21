# Scripts de MLFlow - Documentación Completa

Este directorio contiene todos los scripts relacionados con MLFlow para el proyecto de ML de Chicago Crimes.

## 📋 Índice

1. [Resumen del Proyecto](#resumen-del-proyecto)
2. [Scripts Disponibles](#scripts-disponibles)
3. [Fases Completadas](#fases-completadas)
4. [Prerequisitos](#prerequisitos)
5. [Guía de Uso](#guía-de-uso)
6. [Resultados y Métricas](#resultados-y-métricas)

---

## 📊 Resumen del Proyecto

### Objetivo
Implementar un sistema completo de MLOps usando MLFlow para gestionar el ciclo de vida de modelos de Machine Learning para predicción de arrestos en crímenes de Chicago.

### Problema de Negocio
**Clasificación binaria**: Predecir si un crimen resultará en arresto (Arrest_tag: 0=No, 1=Sí)
- Dataset desbalanceado (55% No Arrest, 45% Arrest)
- Dataset: 194,897 muestras de entrenamiento, 50,744 de test
- 7 features después de preprocesamiento

### Arquitectura MLFlow
- **PostgreSQL** (Puerto 5432): Metadatos
- **MinIO** (Puerto 9000/9001): Artefactos S3-compatible
- **MLFlow Server** (Puerto 5001): API + UI Web

---

## 📜 Scripts Disponibles

### 1. verify_mlflow_setup.py
**Propósito**: Verificar la infraestructura y conectividad de MLFlow

**Descripción**: 
- Verifica la conexión al servidor MLFlow
- Valida la configuración de MinIO (S3)
- Prueba la creación de experimentos y runs
- Valida todas las librerías Python requeridas

**Uso**:
```bash
python mlflow_scripts/verify_mlflow_setup.py
```

**Salida esperada**: Estado de conexión, versiones de librerías, creación de experimento de prueba

---

### 2. mlflow_xgboost_poc.py
**Propósito**: Prueba de Concepto - Modelo XGBoost con tracking de MLFlow

**Descripción**:
- Carga el dataset de Chicago Crimes (train/test)
- Entrena clasificador XGBoost
- Registra todos los parámetros, métricas y artefactos en MLFlow
- Crea visualizaciones (matriz de confusión, importancia de features)
- Registra el modelo en MLFlow Model Registry

**Uso**:
```bash
python mlflow_scripts/mlflow_xgboost_poc.py
```

**Salidas**:
- Experimento MLFlow: `chicago_crimes_xgboost`
- Modelo registrado: `xgboost_chicago_crimes`
- Métricas: Accuracy, Precision, Recall, F1, AUC, MCC
- Artefactos: Matriz de confusión, gráficos de importancia de features, reporte de clasificación

**Resultados** (Conjunto de Test):
- Accuracy: 90.81%
- MCC: 0.5624
- AUC: 87.90%
- Tiempo de entrenamiento: ~2 segundos

---

### 3. check_mlflow_model.py
**Propósito**: Verificar modelos registrados y runs recientes

**Descripción**:
- Lista todos los modelos registrados en MLFlow
- Muestra versiones y stages de modelos
- Despliega runs recientes en experimentos
- Ayuda a verificar el estado de registro de modelos

**Uso**:
```bash
python mlflow_scripts/check_mlflow_model.py
```

---

### 4. mlflow_training_helper.py
**Propósito**: Funciones helper reutilizables para entrenamiento con MLFlow

**Descripción**:
- Módulo con funciones utilitarias para simplificar el tracking con MLFlow
- Configuración automática del entorno MLFlow
- Cálculo estandarizado de métricas de clasificación
- Creación de visualizaciones (matriz de confusión, importancia de features)
- Función principal `train_and_log_model()` que encapsula todo el flujo

**Funciones principales**:
- `setup_mlflow_environment()`: Configura variables de entorno
- `get_or_create_experiment()`: Obtiene o crea experimento
- `train_and_log_model()`: Entrena modelo y registra todo en MLFlow
- `calculate_classification_metrics()`: Calcula métricas completas
- `create_confusion_matrix_plot()`: Genera matriz de confusión
- `create_feature_importance_plot()`: Genera gráfico de importancia
- `compare_models()`: Compara modelos de un experimento

**Uso**:
```python
from mlflow_training_helper import train_and_log_model

model, run_id, metrics = train_and_log_model(
    model=XGBClassifier(),
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    experiment_name="my_experiment",
    run_name="xgboost_v1",
    model_name="my_model"
)
```

---

### 5. train_multiple_models.py
**Propósito**: Ejemplo de entrenamiento de múltiples modelos usando helpers

**Descripción**:
- Demuestra el uso de `mlflow_training_helper`
- Entrena 5 modelos diferentes:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - AdaBoost
  - XGBoost
- Compara resultados automáticamente
- Registra todos los modelos en MLFlow

**Uso**:
```bash
python mlflow_scripts/train_multiple_models.py
```

**Salidas**:
- Experimento MLFlow: `chicago_crimes_all_models`
- 5 modelos registrados con sus métricas
- Comparación automática por MCC score

---

### 6. train_all_8_models.py ⭐
**Propósito**: Migración completa de los 8 modelos originales a MLFlow

**Descripción**:
- Replica la implementación original de `modelos/machine_learning.ipynb`
- Entrena los 8 modelos con configuraciones idénticas:
  1. Logistic Regression
  2. K-Nearest Neighbors (k=10)
  3. SVM Linear
  4. Decision Tree
  5. Random Forest (20 estimators)
  6. Bagging (Logistic Regression, 20 estimators)
  7. AdaBoost (max_depth=5, 20 estimators)
  8. XGBoost (100 estimators)
- Compara todos los modelos por MCC
- Valida resultados contra métricas originales

**Uso**:
```bash
python mlflow_scripts/train_all_8_models.py
```

**Salidas**:
- Experimento MLFlow: `chicago_crimes_8_models`
- 8 modelos registrados en Model Registry
- Tabla comparativa completa
- Ranking por MCC score

**Resultados**:
- ✅ XGBoost: MCC 0.5624 (Mejor modelo)
- ✅ Random Forest: MCC 0.5230
- ✅ AdaBoost: MCC 0.4999
- Tiempo total: ~2 minutos

### 7. champion_challenger.py
**Propósito**: Sistema de gestión Champion/Challenger para Model Registry

**Descripción**:
- Clase `ChampionChallengerManager` para gestión completa del ciclo de vida
- Asignación y gestión de aliases (champion, challenger, previous_champion)
- Comparación automática de modelos (métricas almacenadas y evaluación en vivo)
- Promoción automática con backup del champion anterior
- Capacidad de rollback a versión anterior
- Soporte para A/B testing

**Funciones principales**:
- `set_alias()` / `delete_alias()`: Gestión de aliases
- `compare_models()`: Comparación completa entre champion y challenger
- `promote_challenger()`: Promoción segura con backup
- `rollback_champion()`: Revertir a champion anterior
- `get_model_versions()`: Ver historial de versiones

**Uso**:
```python
from champion_challenger import ChampionChallengerManager

manager = ChampionChallengerManager()
manager.set_alias("xgboost_chicago", "champion", "1")
manager.set_alias("random_forest_chicago", "challenger", "1")

comparison = manager.compare_models(
    champion_model_name="xgboost_chicago",
    challenger_model_name="random_forest_chicago"
)
```

---

### 8. demo_champion_challenger.py
**Propósito**: Demostración completa del workflow Champion/Challenger

**Descripción**:
- Workflow completo de 6 pasos
- Setup inicial de champion y challenger
- Comparación con métricas almacenadas
- Evaluación en vivo con datos de test
- Decisión automática de promoción (threshold configurable)
- Demostración de rollback
- Evaluación de múltiples challengers

**Uso**:
```bash
python mlflow_scripts/demo_champion_challenger.py
```

**Salidas**:
- Comparación detallada de métricas
- Recomendación de promoción
- Tabla de resultados en vivo
- Decisión basada en threshold (default: 1% mejora en MCC)

---

## 🎯 Fases Completadas

### ✅ Fase 1: Infraestructura MLFlow (Docker)
**Objetivo**: Configurar y verificar infraestructura Docker de MLFlow

**Logros**:
- Docker Compose con PostgreSQL, MinIO y MLFlow Server
- Configuración de volúmenes persistentes
- Healthchecks para todos los servicios
- Documentación de inicio y troubleshooting

**Archivos**: `../mlflow_system/docker-compose.yml`, `Dockerfile`, `requirements.txt`

**Verificación**: 
- MLFlow UI: http://localhost:5001 ✓
- MinIO Console: http://localhost:9001 ✓
- PostgreSQL: localhost:5432 ✓

---

### ✅ Fase 2: POC XGBoost con MLFlow
**Objetivo**: Crear prueba de concepto con modelo XGBoost

**Logros**:
- Script completo de entrenamiento con tracking
- Logging de parámetros, métricas y artefactos
- Visualizaciones (confusion matrix, feature importance)
- Registro en Model Registry
- Manejo de compatibilidad Windows/MLFlow

**Resultados**:
- Test Accuracy: 90.81%
- Test MCC: 0.5624 (mejor métrica para datos desbalanceados)
- Test AUC: 87.90%
- Tiempo de entrenamiento: ~2 segundos

**Archivo**: `mlflow_xgboost_poc.py`

---

### ✅ Fase 3: Funciones Helper Reutilizables
**Objetivo**: Crear módulo de funciones helper para simplificar el tracking

**Logros**:
- Módulo `mlflow_training_helper.py` con 7 funciones principales
- Función `train_and_log_model()` que automatiza todo el flujo
- Reducción de ~93% de código repetitivo
- Soporte para cualquier modelo compatible con scikit-learn
- Cálculo automático de métricas completas
- Generación automática de visualizaciones

**Funciones implementadas**:
1. `setup_mlflow_environment()` - Configuración automática
2. `get_or_create_experiment()` - Gestión de experimentos
3. `calculate_classification_metrics()` - Métricas completas
4. `create_confusion_matrix_plot()` - Visualización
5. `create_feature_importance_plot()` - Visualización
6. `train_and_log_model()` - Función principal ⭐
7. `compare_models()` - Comparación de modelos

**Archivo**: `mlflow_training_helper.py`

---

### ✅ Fase 4: Migración de 8 Modelos
**Objetivo**: Migrar todos los modelos originales a MLFlow

**Logros**:
- 8/8 modelos migrados exitosamente
- Configuraciones idénticas a implementación original
- Validación de resultados contra métricas originales
- Todos los modelos registrados en Model Registry
- Comparación automática por MCC

**Modelos migrados**:
1. Logistic Regression - MCC: 0.2127
2. K-Nearest Neighbors (k=10) - MCC: 0.1706
3. SVM Linear - MCC: 0.2149
4. Decision Tree - MCC: 0.4192
5. Random Forest (20 est.) - MCC: 0.5230
6. Bagging (LR, 20 est.) - MCC: 0.2128
7. AdaBoost (20 est.) - MCC: 0.4999
8. XGBoost (100 est.) - MCC: 0.5624 ⭐ **CHAMPION**

**Experimento**: `chicago_crimes_8_models`
**Archivo**: `train_all_8_models.py`

---

### ✅ Fase 5: Sistema Champion/Challenger
**Objetivo**: Implementar patrón Champion/Challenger para gestión de modelos

**Logros**:
- Clase `ChampionChallengerManager` completa
- Sistema de aliases (champion, challenger, previous_champion)
- Comparación automática de modelos
- Promoción segura con backup
- Rollback en un comando
- Evaluación en vivo con datos de test
- Soporte para múltiples challengers
- Decisión automática basada en threshold

**Características**:
- Comparación usando métricas almacenadas
- Evaluación en vivo cargando modelos desde artifacts
- Tabla comparativa automática
- Threshold configurable (default: 1% mejora)
- Backup automático del champion actual
- Capacidad de rollback completa

**Archivos**: `champion_challenger.py`, `demo_champion_challenger.py`

---

## 📋 Prerequisitos

1. **Infraestructura Docker de MLFlow en ejecución**:
   ```bash
   cd mlflow_system
   docker compose up -d
   ```

2. **Entorno virtual activado**:
   ```bash
   .\.venv\Scripts\activate
   ```

3. **Acceso a MLFlow UI**: http://localhost:5001

## Variables de Entorno

Todos los scripts configuran automáticamente:
- `AWS_ACCESS_KEY_ID=minio`
- `AWS_SECRET_ACCESS_KEY=minio123`
- `MLFLOW_S3_ENDPOINT_URL=http://localhost:9000`
- `MLFLOW_TRACKING_URI=http://localhost:5001`

## 📚 Guía de Uso

### Inicio Rápido

1. **Iniciar infraestructura MLFlow**:
```bash
cd mlflow_system
docker compose up -d
```

2. **Verificar instalación**:
```bash
.\.venv\Scripts\activate
python mlflow_scripts/verify_mlflow_setup.py
```

3. **Entrenar todos los modelos**:
```bash
python mlflow_scripts/train_all_8_models.py
```

4. **Configurar Champion/Challenger**:
```bash
python mlflow_scripts/demo_champion_challenger.py
```

### Workflows Comunes

#### Entrenar un modelo individual
```python
from mlflow_training_helper import train_and_log_model
from sklearn.ensemble import RandomForestClassifier

model, run_id, metrics = train_and_log_model(
    model=RandomForestClassifier(n_estimators=100),
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    experiment_name="my_experiment",
    run_name="rf_v1",
    model_name="my_rf_model"
)
```

#### Comparar modelos
```python
from champion_challenger import ChampionChallengerManager

manager = ChampionChallengerManager()
comparison = manager.compare_models(
    champion_model_name="xgboost_chicago",
    challenger_model_name="random_forest_chicago",
    X_test=X_test,
    y_test=y_test
)
```

#### Promover un modelo
```python
# Si el challenger supera al champion
manager.promote_challenger(
    model_name="xgboost_chicago",
    champion_alias="champion",
    challenger_alias="challenger"
)
```

---

## 📊 Resultados y Métricas

### Ranking Final de Modelos (por MCC)

| Posición | Modelo | Test MCC | Test Accuracy | Test AUC | Tiempo (s) | Estado |
|----------|--------|----------|---------------|----------|------------|--------|
| 🥇 1 | **XGBoost** | **0.5624** | **90.81%** | **87.90%** | 2.83 | **CHAMPION** |
| 🥈 2 | Random Forest | 0.5230 | 89.49% | 85.47% | 6.70 | Challenger |
| 🥉 3 | AdaBoost | 0.4999 | 89.04% | 86.26% | 31.00 | - |
| 4 | Decision Tree | 0.4192 | 84.48% | 73.77% | 5.34 | - |
| 5 | SVM Linear | 0.2149 | 65.90% | N/A | 0.32 | - |
| 6 | Bagging (LR) | 0.2128 | 66.43% | 67.21% | 23.33 | - |
| 7 | Logistic Regression | 0.2127 | 66.41% | 67.21% | 13.25 | - |
| 8 | KNN (k=10) | 0.1706 | 75.88% | 64.38% | 1.27 | - |

### Comparación Champion vs Challenger

**XGBoost (Champion)** vs **Random Forest (Challenger)**:

| Métrica | Champion | Challenger | Diferencia | Ganador |
|---------|----------|------------|------------|---------|
| MCC | 0.5624 | 0.5230 | -0.0394 | Champion |
| Accuracy | 90.81% | 89.49% | -1.32% | Champion |
| AUC | 87.90% | 85.47% | -2.43% | Champion |
| F1-Score | 89.99% | 89.04% | -0.95% | Champion |

**Decisión**: Mantener XGBoost como champion ✓

### Métricas Clave del Champion (XGBoost)

- **MCC**: 0.5624 (métrica más robusta para datos desbalanceados)
- **Accuracy**: 90.81%
- **Precision**: 90.00%
- **Recall**: 90.81%
- **F1-Score**: 89.99%
- **AUC**: 87.90%
- **Tiempo de entrenamiento**: 2.83s
- **Tiempo de predicción**: 0.90s

### Estadísticas del Proyecto

- **Total de modelos entrenados**: 8
- **Modelos registrados en MLFlow**: 8
- **Experimentos creados**: 3
  - `test_verification` - Verificación inicial
  - `chicago_crimes_xgboost` - POC XGBoost
  - `chicago_crimes_8_models` - Todos los modelos
- **Total de runs**: 10+
- **Artefactos generados**: 24+ (matrices de confusión, feature importance, reports)

---

## 🔗 Enlaces Útiles

- **MLFlow UI**: http://localhost:5001
- **MinIO Console**: http://localhost:9001 (user: minio, pass: minio123)
- **Documentación MLFlow**: https://mlflow.org/docs/latest/index.html
- **Repositorio del proyecto**: (agregar URL si aplica)

---

## 📝 Notas Importantes

1. **Código en Inglés**: Todo el código y comentarios están en inglés siguiendo las mejores prácticas
2. **Documentación en Español**: Los archivos .md están en español para facilitar la comprensión
3. **Compatibilidad Windows**: Todos los scripts manejan correctamente encoding Unicode
4. **Versionado**: Todos los modelos están versionados en MLFlow Model Registry
5. **Reproducibilidad**: Todos los experimentos incluyen seeds aleatorios fijos (random_state=42)

---

### 9. predictor.py ⭐
**Propósito**: Clase de producción para deployment de modelos

**Descripción**:
- Clase `ChicagoCrimePredictor` lista para producción
- Carga modelos por alias desde MLFlow Model Registry
- Validación automática de features de entrada
- Soporte para predicciones individuales y por lotes
- Explicaciones detalladas de predicciones
- Logging opcional de predicciones a MLFlow
- Manejo robusto de errores

**Funciones principales**:
- `__init__()`: Inicializa predictor cargando modelo por alias
- `predict()`: Predicción con validación
- `predict_proba()`: Probabilidades de clase positiva
- `predict_with_explanation()`: Predicción con explicación detallada
- `get_model_info()`: Metadata completa del modelo
- `batch_predict()`: Predicciones por lotes para datasets grandes
- `log_prediction()`: Logging de predicciones para monitoreo

**Uso**:
```python
from predictor import ChicagoCrimePredictor

# Initialize with champion model
predictor = ChicagoCrimePredictor(
    model_name="xgboost_chicago",
    alias="champion"
)

# Make prediction
prediction = predictor.predict(crime_features)
probability = predictor.predict_proba(crime_features)

# Get detailed explanation
explanation = predictor.predict_with_explanation(crime_features)
```

**Características**:
- Detección automática de features desde el modelo
- Validación de entrada (features faltantes, valores nulos)
- Soporte para dict, DataFrame y list de dicts
- Batch processing con progress tracking
- Feature importance en explicaciones
- Compatible con cualquier modelo scikit-learn

---

### 10. demo_predictor.py
**Propósito**: Demostración completa del predictor en producción

**Descripción**:
- 8 demos diferentes mostrando todas las funcionalidades
- Carga y uso del champion model
- Predicciones individuales y por lotes
- Comparación champion vs challenger
- Validación y manejo de errores
- Evaluación en datos reales

**Uso**:
```bash
python mlflow_scripts/demo_predictor.py
```

**Demos incluidos**:
1. Inicialización del predictor
2. Información del modelo
3. Predicción individual (dict input)
4. Predicción con explicación
5. Predicciones múltiples (DataFrame)
6. Batch prediction en test data
7. Comparación champion vs challenger
8. Validación y manejo de errores

**Resultados**:
- Batch Accuracy: 90.40% en 1,000 muestras
- Agreement champion/challenger: 95.30%
- Todas las validaciones funcionando correctamente

---

## 🎉 Proyecto Completado (6/6 Fases)

**Estado**: ✅ TODAS LAS FASES COMPLETADAS (100%)

### Resumen de Logros:
- ✅ Infraestructura MLFlow operativa
- ✅ 8 modelos migrados y versionados
- ✅ Sistema Champion/Challenger funcional
- ✅ Clase Predictor lista para producción
- ✅ Documentación completa y unificada
- ✅ 10 scripts Python funcionales
- ✅ Demos y ejemplos de uso

---

## 🚀 Deployment en Producción

### Opciones de Deployment:

#### 1. Uso Directo (Python)
```python
from predictor import ChicagoCrimePredictor

predictor = ChicagoCrimePredictor("xgboost_chicago", "champion")
prediction = predictor.predict(crime_data)
```

#### 2. API REST (Próximo paso opcional)
Crear endpoint Flask/FastAPI:
```python
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = predictor.predict(data)
    return jsonify({'prediction': int(prediction[0])})
```

#### 3. Batch Processing
```python
# Para grandes volúmenes de datos
predictions = predictor.batch_predict(large_dataset, batch_size=1000)
```

---

## 📞 Soporte

Para problemas o preguntas:
1. Revisar logs de Docker: `docker compose logs -f mlflow`
2. Verificar conectividad: `python mlflow_scripts/verify_mlflow_setup.py`
3. Consultar documentación en `../mlflow_system/README.md`

---

## 🛠️ Stack Tecnológico

### Infraestructura
- Docker & Docker Compose
- PostgreSQL 15
- MinIO (S3-compatible)
- MLFlow 2.9.2 (servidor) / 3.6.0 (cliente)

### Python
- Python 3.11.13
- scikit-learn 1.7.2
- XGBoost 3.1.2
- pandas 2.3.2
- numpy 2.3.3
- matplotlib 3.10.6
- seaborn 0.13.2

### MLOps
- MLFlow Tracking
- MLFlow Model Registry
- MLFlow Projects (implícito)
- Artifact Storage (MinIO)
- Metadata Storage (PostgreSQL)

---

## 🎓 Aprendizajes Clave

### Técnicos

1. **MLFlow Version Compatibility**:
   - Cliente 3.6.0 con servidor 2.9.2 requiere workarounds
   - Usar pickle manual para evitar APIs nuevas
   - Deshabilitar autologging para control total

2. **Windows Encoding**:
   - Emojis Unicode causan `UnicodeEncodeError`
   - Solución: Redireccionar stdout o usar texto ASCII
   - Configurar `MLFLOW_ENABLE_EMOJI=false`

3. **Model Registry**:
   - Aliases son más flexibles que Stages
   - Backup automático antes de promoción
   - Rollback en un comando

4. **Métricas para Datos Desbalanceados**:
   - MCC es la métrica más robusta
   - Accuracy puede ser engañosa
   - AUC complementa bien a MCC

### MLOps

1. **Automatización**:
   - Funciones helper reducen 93% de código
   - Estandarización mejora reproducibilidad
   - Comparación automática acelera decisiones

2. **Versionado**:
   - Todos los modelos versionados
   - Historial completo en MLFlow
   - Trazabilidad de cambios

3. **Seguridad**:
   - Backup antes de promoción
   - Rollback disponible
   - Threshold configurable

---

## 📝 Comandos Rápidos

```bash
# Iniciar infraestructura
cd mlflow_system && docker compose up -d

# Verificar setup
python mlflow_scripts/verify_mlflow_setup.py

# Entrenar todos los modelos
python mlflow_scripts/train_all_8_models.py

# Demo Champion/Challenger
python mlflow_scripts/demo_champion_challenger.py

# Demo Predictor
python mlflow_scripts/demo_predictor.py

# Ver logs
docker compose logs -f mlflow

# Detener todo
docker compose down
```

---
