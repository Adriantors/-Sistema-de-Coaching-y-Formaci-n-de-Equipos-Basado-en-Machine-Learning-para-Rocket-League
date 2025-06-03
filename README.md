# 🚀 Rocket League Coaching Platform

Una plataforma avanzada de análisis para Rocket League que utiliza inteligencia artificial y algoritmos de clustering para mejorar el rendimiento individual y formar equipos óptimos.

## ✨ Características Principales

### 🎮 Análisis Individual
- **Predicción de Rango**: Utiliza un modelo LightGBM entrenado para predecir tu rango basado en estadísticas de juego
- **Análisis Comparativo**: Compara tus estadísticas con promedios de rangos superiores
- **Recomendaciones Personalizadas**: Identifica fortalezas y áreas de mejora específicas
- **Métricas Avanzadas**: Calcula más de 60 métricas personalizadas de rendimiento
- **Explicabilidad SHAP**: Utiliza SHAP para explicar las predicciones del modelo

### 👥 Análisis de Equipos
- **Formación de Equipos Óptimos**: Encuentra las mejores combinaciones de 3 jugadores usando 64+ métricas
- **Búsqueda de Tercer Jugador**: Interfaz mejorada para seleccionar dúo base y evaluar candidatos por separado
- **Análisis de Sinergia Avanzado**: 
  - Utiliza todas las métricas del rank_predictor (64 características completas)
  - Análisis de complementariedad en 7 dimensiones diferentes
  - Diversidad de roles y estilos de juego optimizada
  - Sinergia de boost, posicionamiento, mecánicas y química de equipo
- **Datos del Birmingham Major**: Utiliza patrones de equipos profesionales para optimización
- **Interfaz Separada**: Modo completamente independiente del análisis individual

## 🛠️ Tecnologías Utilizadas

- **Frontend**: Streamlit con CSS personalizado
- **Machine Learning**: LightGBM, scikit-learn, K-means clustering, SHAP
- **Análisis de Datos**: pandas, numpy, plotly, scipy
- **API**: Ballchasing.com para procesamiento de replays
- **Algoritmos**: Clustering, análisis de correlación, cálculo de sinergias
- **Optimización**: Optuna para hyperparameter tuning

## 🧠 Decisiones Técnicas y Justificación

### 🤔 Por qué LightGBM en lugar de Random Forest o XGBoost?

Durante la propuesta inicial del TFG se consideraron **Random Forest y XGBoost** como algoritmos principales. Sin embargo, tras experimentación rigurosa con datos reales, se decidió usar **LightGBM** por las siguientes razones basadas en evidencia empírica:

#### **Resultados Experimentales Comparativos**

| Algoritmo | QWK Score | RMSE | Tiempo Entrenamiento | Estabilidad |
|-----------|-----------|------|---------------------|-------------|
| **LightGBM** | **0.8038 ± 0.004** | **0.7504 ± 0.004** | **0.29s ± 0.29** | **Más estable** |
| Random Forest | 0.7664 ± 0.005 | 0.7941 ± 0.004 | 0.61s ± 0.05 | Menos estable |
| Neural Network | 0.7533 ± 0.006 | 0.9220 ± 0.010 | 5.57s ± 0.05 | Menos estable |

#### **Ventajas de LightGBM Verificadas**

1. **Mejor Rendimiento**: QWK Score 4.9% superior a Random Forest
2. **Mayor Eficiencia**: 52% más rápido que Random Forest, 95% más rápido que Neural Networks
3. **Mejor Estabilidad**: Menor desviación estándar (0.004 vs 0.005-0.006)
4. **Mejor RMSE**: 5.5% menor error que Random Forest
5. **Optimización Superior**: Mejor respuesta a hyperparameter tuning con Optuna

#### **¿Por qué NO Redes Neuronales?**

Aunque se propusieron como opción, las redes neuronales mostraron:
- **Rendimiento inferior**: QWK 6.2% peor que LightGBM
- **Mayor tiempo de entrenamiento**: 19x más lento
- **Menor estabilidad**: Mayor variabilidad entre folds
- **Mayor complejidad**: Requieren más preprocessing (escalado)
- **Overfitting**: Tendencia a memorizar en datasets pequeños

### 🎯 Por qué K-means para Clustering de Roles?

El análisis de equipos requería detectar **roles de jugadores**. Se evaluaron múltiples algoritmos de clustering:

#### **Resultados Experimentales de Clustering**

| Algoritmo | Silhouette Score | Calinski-Harabasz | Davies-Bouldin | Tiempo | Interpretabilidad |
|-----------|------------------|-------------------|----------------|---------|-------------------|
| **K-means** | 0.1493 | **9.21** | 1.6572 | 0.343s | **4 roles claros** |
| Gaussian Mixture | **0.1529** | 9.03 | **1.6393** | 0.013s | 4 roles variables |
| Hierarchical | 0.1456 | 8.51 | 1.6976 | 0.115s | 4 roles jerárquicos |
| DBSCAN | - | - | - | - | **0 clusters válidos** |

#### **Ventajas de K-means Verificadas**

1. **Interpretabilidad Superior**: Clusters corresponden a roles reconocibles (Delantero, Creador, Defensor, Polivalente)
2. **Estabilidad**: Resultados consistentes entre ejecuciones
3. **Rendimiento Balanceado**: Segundo mejor en métricas técnicas, pero mejor en aplicabilidad
4. **Simplicidad**: Fácil de explicar y validar con expertos en Rocket League
5. **Escalabilidad**: Funciona bien con diferentes tamaños de datasets

#### **¿Por qué NO otros algoritmos?**

- **DBSCAN**: No pudo formar clusters válidos con los datos disponibles
- **Gaussian Mixture**: Aunque técnicamente superior, clusters menos interpretables
- **Hierarchical**: Bueno técnicamente, pero computacionalmente más costoso para datos grandes

### 📊 Decisiones de Feature Engineering

#### **Métricas Derivadas vs Raw Stats**

Se crearon **64+ métricas derivadas** en lugar de usar solo estadísticas brutas porque:

1. **Mejor Rendimiento**: Las métricas normalizadas (ratios, eficiencias) son más predictivas
2. **Comparabilidad**: Permiten comparar jugadores independientemente del tiempo de juego
3. **Interpretabilidad**: Métricas como "precisión de tiro" son más intuitivas que "goles totales"
4. **Robustez**: Menos sensibles a outliers y variaciones en duración de partidos

#### **Ejemplos de Feature Engineering**

```python
# En lugar de solo "goles" y "tiros"
shot_accuracy = safe_divide(goals_per_game, shots_per_game)

# En lugar de solo estadísticas de boost
boost_efficiency = safe_divide(amount_collected_per_game, time_zero_boost_per_game)

# Métricas de posicionamiento compuestas
offensive_positioning = safe_divide(time_offensive_third, time_defensive_third)
```

### 🔧 Decisiones de Arquitectura

#### **Ensemble vs Modelo Único**

Se usa un **ensemble de 5 modelos LightGBM** con diferentes semillas porque:

1. **Mayor Robustez**: Reduce variabilidad entre entrenamientos
2. **Mejor Generalización**: Menor riesgo de overfitting a particularidades del dataset
3. **Confianza en Predicciones**: Permite calcular intervalos de confianza
4. **Evidencia Empírica**: Mejoró estabilidad de 0.0007 desviación estándar

#### **Regresión vs Clasificación**

Se eligió **regresión** en lugar de clasificación multiclase porque:

1. **Orden Natural**: Los rangos tienen orden natural (Bronze < Silver < Gold...)
2. **Granularidad**: Permite predicciones "intermedias" (ej: entre Gold y Platinum)
3. **Métricas Apropiadas**: QWK (Quadratic Weighted Kappa) penaliza errores distantes
4. **Flexibilidad**: Fácil conversión a clasificación cuando sea necesario

### 🎛️ Hyperparameter Optimization

#### **¿Por qué Optuna sobre Grid Search?**

1. **Eficiencia**: Búsqueda inteligente vs fuerza bruta
2. **Mejor Exploración**: Algoritmos de optimización bayesiana
3. **Paralelización**: Soporte nativo para múltiples workers
4. **Pruning**: Detención temprana de trials poco prometedores

#### **Configuración de Optuna**

- **50 trials** (balance entre tiempo y exploración)
- **Tree-structured Parzen Estimator** para búsqueda eficiente
- **Métricas de validación cruzada** para evitar overfitting

### 📈 Validación y Testing

#### **Cross-Validation Estratificado**

Se usa **5-fold stratified CV** porque:

1. **Distribución Balanceada**: Mantiene proporción de rangos en cada fold
2. **Robustez Estadística**: 5 folds proveen estimación confiable
3. **Eficiencia Computacional**: Balance entre precisión y tiempo

#### **Métricas de Evaluación**

- **QWK (Quadratic Weighted Kappa)**: Métrica principal, penaliza errores grandes
- **RMSE**: Error absoluto para comparación técnica
- **Análisis por Rango**: Validación específica para cada nivel

Esta metodología **data-driven** garantiza que todas las decisiones técnicas estén respaldadas por evidencia empírica en lugar de suposiciones teóricas.

## 📊 Algoritmos de Análisis de Equipos

### Clustering de Roles
- **K-means (k=4)**: Identifica 4 roles principales
  - **Delantero**: Alto ratio goles/tiros, posicionamiento ofensivo
  - **Creador**: Altas asistencias, buen control de pelota
  - **Defensor**: Altas paradas, tiempo en campo defensivo
  - **Polivalente**: Estadísticas balanceadas

### Cálculo de Sinergia
La sinergia del equipo se calcula usando un análisis multidimensional avanzado:

```python
# Análisis de complementariedad en 7 dimensiones
complementarity_scores = [
    core_complementarity,      # Balance en rendimiento base
    boost_synergy,            # Diversidad en gestión de boost
    movement_synergy,         # Complementariedad en mecánicas
    positioning_synergy,      # Varianza en posicionamiento
    chemistry_synergy,        # Rotaciones y control de pelota
    playstyle_synergy,        # Diversidad en estilos de juego
    feature_diversity         # Diversidad general usando 64 métricas
]

# Cálculo final optimizado
synergy_score = (
    0.2 * role_diversity_bonus +     # Diversidad de roles
    0.5 * weighted_complementarity + # Complementariedad ponderada
    0.3 * feature_diversity          # Diversidad de características
)
```

### Datos del Birmingham Major
- **16 equipos profesionales** analizados
- **Patrones de éxito** identificados en equipos Top 4
- **Métricas de referencia** para comparación

## 🚀 Instalación y Uso

### Requisitos
```bash
pip install -r requirements.txt
```

### Dependencias Principales
```
streamlit==1.32.0
pandas==2.2.0
numpy==1.26.4
plotly==5.18.0
python-dotenv==1.0.1
lightgbm>=4.0.0
scikit-learn>=1.3.0
scipy>=1.10.0
requests>=2.31.0
joblib>=1.3.0
optuna>=3.0.0
shap>=0.42.0
```

### Configuración
1. **API Key de Ballchasing**: Configura tu API key en el archivo `app.py` o como variable de entorno
2. **Datos de entrenamiento**: Los modelos pre-entrenados están incluidos en `outputs/`

### Ejecución
```bash
streamlit run app.py
```

## 📁 Estructura del Proyecto

```
├── app.py                          # Aplicación principal Streamlit
├── team_analyzer.py                # Módulo de análisis de equipos
├── rank_predictor.py               # Predictor de rangos individual
├── ballchasing_api.py              # Cliente API de Ballchasing
├── train_rank_model_lgb.py         # Entrenamiento del modelo de predicción
├── creategroups.py                 # Gestión de grupos de replays
├── anonymize_and_filter_players.py # Anonimización de datos
├── merge_players_with_rank.py      # Fusión de datos con rangos
├── model_comparison_simple.py      # Comparación de algoritmos ML
├── clustering_comparison.py        # Comparación de algoritmos clustering
├── appbackup.py                    # Backup de la aplicación
├── log modelo.txt                  # Log del entrenamiento del modelo
├── data/
│   ├── 1-birmingham-major-*.csv    # Datos del Birmingham Major
│   ├── players_anonymous_per_game.csv # Datos anonimizados de entrenamiento
│   ├── all_players_with_rank.csv   # Datos combinados con rangos
│   └── *-players.csv               # Datos por rango (SSL, GC, etc.)
├── outputs/                        # Modelos entrenados y artefactos
│   ├── rank_predictor_gbdt_*.txt   # Ensemble de 5 modelos LightGBM
│   ├── rank_stats.parquet          # Estadísticas por rango
│   └── shap_explainer.pkl          # Explainer SHAP para interpretabilidad
└── requirements.txt                # Dependencias del proyecto
```

## 🎯 Modos de Análisis de Equipos

### 1. Encontrar Mejor Trío
- Selecciona 3+ jugadores candidatos
- Evalúa todas las combinaciones posibles de 3
- Ordena por puntuación de sinergia + habilidad
- Muestra top 5 combinaciones con análisis detallado

### 2. Encontrar Tercer Jugador
- Define un dúo base (2 jugadores)
- Evalúa candidatos para el tercer puesto
- Calcula complementariedad de roles
- Recomienda basado en sinergia del equipo completo

## 📈 Métricas de Evaluación

### Sinergia del Equipo
- **Diversidad de Roles**: Bonus por tener roles diferentes
- **Balance Ofensivo/Defensivo**: Equilibrio en estadísticas
- **Gestión de Boost**: Patrones complementarios de recolección
- **Posicionamiento**: Varianza en tiempo como último/primer jugador

### Puntuación Individual
- **Habilidad Base**: Score, goles, asistencias, paradas
- **Eficiencia**: Ratios de precisión y efectividad
- **Contribución al Equipo**: Acciones ofensivas y defensivas

## 🏆 Datos de Referencia

### Birmingham Major 2024
- **Equipos Top 4**: Karmine Corp, The Ultimates, Furia, NRG
- **Métricas Promedio Exitosas**:
  - Goles por jugador: 0.64
  - Asistencias por jugador: 0.50
  - Paradas por jugador: 1.83
  - Score por jugador: 356

## 🔧 API de Ballchasing

### Configuración
```python
API_KEY = "tu_api_key_aqui"
api = BallchasingAPI(API_KEY)
```

### Funcionalidades
- Subida automática de replays
- Creación de grupos de análisis
- Procesamiento de estadísticas por jugador
- Cálculo de métricas personalizadas

## 📊 Métricas Calculadas

### Core Stats
- Puntuación, goles, asistencias, paradas por partido
- Precisión de tiro, ratio de paradas
- Acciones ofensivas y defensivas totales

### Boost Management
- Eficiencia de recolección y robo de boost
- Tiempo sin boost, gestión de supersónico
- Negación de boost al rival

### Posicionamiento
- Tiempo en diferentes tercios del campo
- Balance ofensivo/defensivo
- Control de pelota y rotaciones

### Movimiento
- Gestión de velocidad
- Balance suelo/aire
- Uso de powerslide

## 🤖 Modelo de Predicción de Rango

### Arquitectura
- **Algoritmo**: LightGBM (Gradient Boosting Decision Trees)
- **Ensemble**: 5 modelos con diferentes semillas para robustez
- **Métricas**: 64+ características derivadas de estadísticas de juego
- **Optimización**: Hyperparameter tuning con Optuna (50 trials)
- **Validación**: 5-fold cross-validation con Cohen's Kappa Score

### Rendimiento del Modelo
- **Métrica Principal**: Quadratic Weighted Kappa (QWK) = 0.7025
- **Estabilidad**: Desviación estándar = 0.0007 (muy estable)
- **Interpretabilidad**: SHAP values para explicar predicciones

### Características del Modelo
- **Objetivo**: Regresión para predicción continua de rango
- **Early Stopping**: 60 iteraciones sin mejora
- **Regularización**: L1 y L2 optimizadas automáticamente
- **Profundidad**: 9 niveles máximo para evitar overfitting

## 🎮 Uso de la Aplicación

### Modo Individual
1. Sube archivos .replay
2. Selecciona tu jugador
3. Configura rango actual y objetivo
4. Recibe análisis personalizado con explicaciones SHAP

### Modo Equipos
1. Sube replays del equipo
2. Selecciona modo de análisis
3. Elige jugadores candidatos
4. Obtén recomendaciones optimizadas

## 🔄 Pipeline de Datos

### 1. Recolección de Datos
- **creategroups.py**: Gestiona grupos de replays en Ballchasing API
- **Filtrado automático**: Por rango, fecha y modo de juego
- **Paginación**: Manejo eficiente de grandes volúmenes de datos

### 2. Procesamiento de Datos
- **merge_players_with_rank.py**: Combina estadísticas con información de rango
- **anonymize_and_filter_players.py**: Anonimiza datos y filtra métricas relevantes
- **Validación**: Limpieza automática de datos inconsistentes

### 3. Entrenamiento del Modelo
- **train_rank_model_lgb.py**: Pipeline completo de entrenamiento
- **Feature Engineering**: Derivación automática de 64+ métricas
- **Hyperparameter Optimization**: Búsqueda automática de mejores parámetros
- **Model Persistence**: Guardado automático de modelos y artefactos

## 🧪 Testing y Validación

### Validación del Modelo
- **Cross-validation**: 5-fold estratificado
- **Métricas de evaluación**: QWK, RMSE, precisión por rango
- **Análisis de residuos**: Detección de sesgos por rango

### Validación de Equipos
- **Datos de referencia**: Birmingham Major 2024
- **Métricas de sinergia**: Validadas contra equipos profesionales
- **A/B Testing**: Comparación de algoritmos de clustering

## 🤝 Contribución

Este proyecto utiliza un enfoque "MoneyBall" para Rocket League, aplicando análisis de datos avanzado para optimizar el rendimiento individual y la formación de equipos.

### Metodología
- **Data-Driven**: Todas las decisiones basadas en datos reales
- **Científico**: Validación estadística rigurosa
- **Escalable**: Arquitectura modular para fácil extensión

## 📝 Licencia

Proyecto académico desarrollado por Adrián Torremocha para TFG.

## 🔗 Enlaces

- [Ballchasing.com](https://ballchasing.com) - API de análisis de replays
- [Birmingham Major 2024](https://liquipedia.net/rocketleague/Birmingham_Major_2024) - Datos de referencia
- [LightGBM Documentation](https://lightgbm.readthedocs.io/) - Documentación del modelo
- [SHAP Documentation](https://shap.readthedocs.io/) - Explicabilidad del modelo

---

**Nota**: Requiere API key válida de Ballchasing.com para funcionalidad completa.

## 1.6. Breve Sumario de Productos Obtenidos

Como resultado del trabajo realizado hasta la fecha, el proyecto ha generado una serie de productos tangibles y artefactos clave que constituyen el núcleo del sistema desarrollado:

Conjunto de Datos Procesado y Anonimizado: Un dataset extenso (`players_anonymous_per_game.csv`) con estadísticas de juego anonimizadas de más de 200,000 repeticiones de Rocket League, abarcando múltiples rangos y modos de juego. A partir de este dataset base, se derivan más de 60 características que son utilizadas para el entrenamiento de los modelos. Se incluyen también estadísticas agregadas por rango (`rank_stats.parquet`).

Modelos de Machine Learning Entrenados y Validados:

*   Un ensemble de cinco modelos LightGBM para la predicción del rango de los jugadores, optimizados mediante Optuna, que alcanzan un rendimiento robusto (QWK ≈ 0.702 ± 0.001) en validación cruzada estratificada. Los modelos entrenados se almacenan en formato `.txt`.
*   `outputs/shap_explainer.pkl` (Explainer SHAP para el primer modelo del ensemble LightGBM)
*   Modelos de clustering K-means configurados para la identificación de roles de jugadores.

Prototipo Funcional de la Plataforma Web (`app.py`): Una aplicación interactiva desarrollada en Streamlit que integra los siguientes módulos:

*   Análisis Individual: Permite la carga de replays locales por parte del usuario, realiza el parsing (a través de la API de Ballchasing), ejecuta el modelo de predicción de rango y muestra la predicción junto con un análisis de las métricas más influyentes (visualización SHAP) y comparativas estadísticas.
*   Análisis de Equipos: Ofrece funcionalidades para (a) encontrar la mejor combinación de tres jugadores a partir de un grupo de candidatos y (b) buscar el tercer jugador ideal para complementar un dúo preexistente, basándose en la identificación de roles y un algoritmo de cálculo de sinergia multidimensional.

Código Fuente y Scripts: Un repositorio de código Python que incluye:

*   Scripts para la adquisición y preprocesamiento de datos (`creategroups.py`, `merge_players_with_rank.py`, `anonymize_and_filter_players.py`).
*   Script para el entrenamiento y la optimización de los modelos LightGBM (`train_rank_model_lgb.py`).
*   Módulos de la aplicación (`ballchasing_api.py`, `rank_predictor.py`, `team_analyzer.py`).
*   Scripts de comparación de algoritmos (`model_comparison_simple.py`, `clustering_comparison.py`).
*   Archivo de dependencias (`requirements.txt`).

Documentación del Proyecto: La presente memoria de TFG, junto con los informes de seguimiento previos y el `README.md` del repositorio de código, que documentan la metodología, el desarrollo, los resultados y las decisiones técnicas tomadas.

Estos productos constituyen un sistema coherente y funcional que aborda los objetivos planteados, sentando las bases para futuras mejoras y validaciones con la comunidad de Rocket League. 

- **Resultados Experimentales**: CSVs con resultados de comparación
  - `model_comparison_results.csv`
  - `clustering_comparison_results.csv`
- **Modelos Entrenados**:
  - `outputs/rank_predictor_gbdt_0.txt` a `outputs/rank_predictor_gbdt_4.txt`
  - `outputs/shap_explainer.pkl` (Explainer SHAP para el primer modelo del ensemble LightGBM)
- **Datasets**:
  - `data/players_anonymous_per_game.csv`
