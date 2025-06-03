"""
Simplified Model Comparison Experiments

This module compares available machine learning algorithms for rank prediction
to justify the final choice of LightGBM over Random Forest and Neural Networks.

Author: Adrián Torremocha
Date: 02-05-2025
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

def safe_divide(a, b, fill_value=0):
    """Dividir a/b evitando inf y NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.divide(a, b)
    if isinstance(res, (pd.Series, pd.DataFrame)):
        return res.replace([np.inf, -np.inf], fill_value).fillna(fill_value)
    return fill_value if np.isinf(res) or np.isnan(res) else res

def load_and_preprocess_data():
    """Cargar y preprocesar una muestra de los datos."""
    print("Cargando datos...")
    df = pd.read_csv("data/players_anonymous_per_game.csv", sep=";")
    
    # Tomar una muestra más pequeña para acelerar la comparación
    df = df.sample(n=min(10000, len(df)), random_state=727)
    
    # Mapeo de rangos a números
    rank_order = ["Bronze", "Silver", "Gold", "Platinum", "Diamond", 
                  "Champion", "Grand Champion", "Supersonic Legend"]
    rank_to_id = {rank: i for i, rank in enumerate(rank_order)}
    df["rank_id"] = df["Rank"].map(rank_to_id)
    df = df.dropna(subset=["rank_id"])
    
    # Ingeniería de características básica
    a = df.copy()
    
    # Métricas principales
    a["shot_accuracy"] = safe_divide(a["goals per game"], a["shots per game"])
    a["save_ratio"] = safe_divide(a["saves per game"], a["shots conceded per game"])
    
    # Limpiar nombres de columnas
    a.columns = [c.replace(" ", "_") for c in a.columns]
    num_cols = a.select_dtypes("number").columns
    a[num_cols] = a[num_cols].replace([np.inf, -np.inf], np.nan).fillna(a[num_cols].mean())
    
    X = a.select_dtypes("number").drop(columns=["rank_id"])
    y = a["rank_id"]
    
    return X, y, rank_order

def qwk_score(y_true, y_pred):
    """Calcular Quadratic Weighted Kappa."""
    return cohen_kappa_score(y_true, np.clip(np.rint(y_pred), 0, 7), weights="quadratic")

def evaluate_model(model, X, y, model_name, cv_folds=3):
    """Evaluar un modelo usando cross-validation."""
    print(f"\nEvaluando {model_name}...")
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=727)
    
    qwk_scores = []
    rmse_scores = []
    times = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Entrenar modelo
        start_time = time.time()
        
        if isinstance(model, MLPRegressor):
            # Escalar datos para Red Neuronal
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
        
        train_time = time.time() - start_time
        times.append(train_time)
        
        # Calcular métricas
        qwk = qwk_score(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        qwk_scores.append(qwk)
        rmse_scores.append(rmse)
        
        print(f"  Fold {fold+1}: QWK={qwk:.4f}, RMSE={rmse:.4f}, Time={train_time:.2f}s")
    
    # Calcular estadísticas finales
    results = {
        'model': model_name,
        'qwk_mean': np.mean(qwk_scores),
        'qwk_std': np.std(qwk_scores),
        'rmse_mean': np.mean(rmse_scores),
        'rmse_std': np.std(rmse_scores),
        'time_mean': np.mean(times),
        'time_std': np.std(times)
    }
    
    print(f"  {model_name} Resultados Finales:")
    print(f"    QWK: {results['qwk_mean']:.4f} ± {results['qwk_std']:.4f}")
    print(f"    RMSE: {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}")
    print(f"    Tiempo: {results['time_mean']:.2f} ± {results['time_std']:.2f}s")
    
    return results

def main():
    """Ejecutar la comparación de modelos."""
    print("=== COMPARACIÓN DE ALGORITMOS DE MACHINE LEARNING ===")
    print("Objetivo: Justificar la elección de LightGBM")
    
    # Cargar datos
    X, y, rank_order = load_and_preprocess_data()
    print(f"Datos cargados: {X.shape[0]} muestras, {X.shape[1]} características")
    
    # Definir modelos a comparar
    models = {
        'LightGBM': lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            num_leaves=50,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            random_state=727,
            verbose=-1
        ),
        
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=727,
            n_jobs=-1
        ),
        
        'Neural Network': MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=200,
            random_state=727
        )
    }
    
    # Evaluar cada modelo
    results = []
    for name, model in models.items():
        result = evaluate_model(model, X, y, name)
        results.append(result)
    
    # Crear tabla de comparación
    print("\n=== RESUMEN DE RESULTADOS ===")
    print(f"{'Modelo':<15} {'QWK':<12} {'RMSE':<12} {'Tiempo (s)':<12}")
    print("-" * 50)
    
    for result in results:
        print(f"{result['model']:<15} "
              f"{result['qwk_mean']:.4f}±{result['qwk_std']:.3f} "
              f"{result['rmse_mean']:.4f}±{result['rmse_std']:.3f} "
              f"{result['time_mean']:.2f}±{result['time_std']:.2f}")
    
    # Análisis de resultados
    print("\n=== ANÁLISIS DE RESULTADOS ===")
    
    # Encontrar el mejor modelo por QWK
    best_qwk = max(results, key=lambda x: x['qwk_mean'])
    print(f"Mejor QWK: {best_qwk['model']} ({best_qwk['qwk_mean']:.4f})")
    
    # Encontrar el más rápido
    fastest = min(results, key=lambda x: x['time_mean'])
    print(f"Más rápido: {fastest['model']} ({fastest['time_mean']:.2f}s)")
    
    # Encontrar el más estable
    most_stable = min(results, key=lambda x: x['qwk_std'])
    print(f"Más estable: {most_stable['model']} (std: {most_stable['qwk_std']:.4f})")
    
    # Guardar resultados
    results_df = pd.DataFrame(results)
    results_df.to_csv("model_comparison_results.csv", index=False)
    print(f"\nResultados guardados en: model_comparison_results.csv")
    
    return results

if __name__ == "__main__":
    results = main() 