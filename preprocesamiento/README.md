# Etapa de Preprocesamiento de Datos

## Descripción general

Esta sección del proyecto contiene los notebooks correspondientes a la **etapa de preprocesamiento y preparación del dataset** utilizados para el desarrollo del modelo predictivo de arrestos en la ciudad de Chicago.  
El objetivo principal de esta etapa es garantizar la **calidad, consistencia y adecuación estadística de los datos** antes de la fase de modelado.

El proceso se desarrolla de forma secuencial a través de seis notebooks, cada uno enfocado en una fase específica del tratamiento y análisis de los datos.

Estos notebooks constituyen la **base metodológica y analítica** sobre la cual se construyen los modelos predictivos posteriores.  
Garantizan que las decisiones de modelado se apoyen en datos **robustos, normalizados y representativos**, alineados con criterios estadísticos y de calidad propios del análisis de datos avanzado.

---

## Estructura y propósito de los notebooks

### 1. Creación del Dataset ([`1_Creacion_dataset_AndD.ipynb`](./1_Creacion_dataset_AndD.ipynb))
Integración de las fuentes de datos públicas de criminalidad y estaciones de policía.  
Incluye la carga, limpieza inicial y construcción del dataset base con variables geográficas, temporales y contextuales relevantes.

### 2. Análisis Exploratorio ([`2_EDA_dataset_procesado_AndD.ipynb`](./2_EDA_dataset_procesado_AndD.ipynb))
Exploración descriptiva y visual de los datos.  
Identificación de distribuciones, patrones, correlaciones y posibles problemas de calidad.  
Establece las primeras hipótesis sobre las relaciones entre variables.

### 3. Tratamiento de Outliers y Codificación ([`3_Outliers_Encoding.ipynb`](./3_Outliers_Encoding.ipynb))
Aplicación de transformaciones logarítmicas y métodos estadísticos (IQR, desviación estándar) para la detección y manejo de valores atípicos.  
Codificación de variables categóricas mediante técnicas de frecuencia y one-hot encoding, según su naturaleza y cardinalidad.

### 4. Escalado de Variables ([`4_Escalado.ipynb`](./4_Escalado.ipynb))
Normalización de las variables numéricas utilizando distintos escaladores (`StandardScaler`, `MinMaxScaler`, `RobustScaler`, `QuantileTransformer`).  
Selección final de **StandardScaler** por su simplicidad, interpretabilidad y preservación de las relaciones lineales entre variables.

### 5. Balanceo de Clases ([`5_Balanceo.ipynb`](./5_Balanceo.ipynb))
Corrección del desbalance existente en la variable objetivo (`Arrest_tag`) mediante una estrategia híbrida:
1. **Oversampling (SMOTE)** para aumentar la clase minoritaria.  
2. **Undersampling aleatorio** para reducir la clase mayoritaria y alcanzar una proporción 80/100.  
El resultado es un dataset balanceado que mejora la equidad y desempeño de los modelos de clasificación.

### 6. Selección y Extracción de Features ([`6_Feature_Selection_and_Extraction.ipynb`](./6_Feature_Selection_and_Extraction.ipynb))
Evaluación de la relevancia de las variables predictoras mediante:
- **Correlación de Pearson**  
- **Prueba ANOVA (F-test)**  
- **Información Mutua (Mutual Information)**  

Posteriormente, se aplica **Análisis de Componentes Principales (PCA)** para la reducción de dimensionalidad:
- Eliminación de multicolinealidad.  
- Identificación de los componentes más representativos.  
- Retención del 93% de la varianza total utilizando ocho componentes principales.

---

## Resultados de la etapa

- Dataset final limpio, escalado y balanceado, listo para la fase de modelado.  
- Reducción efectiva de la dimensionalidad y eliminación de redundancias.  
- Identificación de las variables más relevantes para la predicción de arrestos, destacando los factores **tipo de delito**, **ubicación geográfica** y **contexto temporal**.

---
