"""
Clustering Algorithm Comparison

This module compares different clustering algorithms for player role detection
to justify the choice of K-means clustering in the team analysis system.

Author: Adrián Torremocha
Date: 02-05-2025
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import time
import warnings
warnings.filterwarnings('ignore')

def load_team_data():
    """Cargar datos para análisis de clustering."""
    print("Cargando datos para análisis de clustering...")
    
    # Usar datos del Birmingham Major como referencia
    df = pd.read_csv("data/1-birmingham-major-pg7kax34a0-players.csv", sep=";")
    
    # Seleccionar métricas clave para clustering de roles
    key_metrics = [
        'score per game', 'goals per game', 'assists per game', 'saves per game',
        'shots per game', 'time offensive third per game', 'time defensive third per game',
        'time most forward per game', 'time most back per game',
        'amount collected per game', 'demos inflicted per game', 'demos taken per game'
    ]
    
    # Filtrar métricas disponibles
    available_metrics = [col for col in key_metrics if col in df.columns]
    X = df[available_metrics].fillna(df[available_metrics].mean())
    
    return X, available_metrics

def evaluate_clustering(algorithm, X, algorithm_name, n_clusters=4):
    """Evaluar un algoritmo de clustering."""
    print(f"\nEvaluando {algorithm_name}...")
    
    # Escalar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    start_time = time.time()
    
    try:
        if algorithm_name == 'DBSCAN':
            # DBSCAN no requiere número de clusters
            labels = algorithm.fit_predict(X_scaled)
            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
        else:
            labels = algorithm.fit_predict(X_scaled)
            n_clusters_found = n_clusters
        
        train_time = time.time() - start_time
        
        # Verificar si hay suficientes clusters para calcular métricas
        if n_clusters_found < 2:
            print(f"  {algorithm_name}: No se formaron suficientes clusters ({n_clusters_found})")
            return None
            
        # Calcular métricas de evaluación
        silhouette = silhouette_score(X_scaled, labels)
        calinski = calinski_harabasz_score(X_scaled, labels)
        davies_bouldin = davies_bouldin_score(X_scaled, labels)
        
        # Contar muestras por cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique_labels, counts))
        
        results = {
            'algorithm': algorithm_name,
            'n_clusters': n_clusters_found,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski,
            'davies_bouldin_score': davies_bouldin,
            'training_time': train_time,
            'cluster_sizes': cluster_sizes
        }
        
        print(f"  {algorithm_name} Resultados:")
        print(f"    Clusters formados: {n_clusters_found}")
        print(f"    Silhouette Score: {silhouette:.4f}")
        print(f"    Calinski-Harabasz: {calinski:.2f}")
        print(f"    Davies-Bouldin: {davies_bouldin:.4f}")
        print(f"    Tiempo: {train_time:.3f}s")
        
        return results
        
    except Exception as e:
        print(f"  {algorithm_name}: Error durante clustering: {e}")
        return None

def main():
    """Ejecuta la comparación de algoritmos de clustering."""
    print("=== COMPARACIÓN DE ALGORITMOS DE CLUSTERING ===")
    print("Objetivo: Justificar la elección de K-means para detección de roles")
    
    # Cargar datos
    X, metrics = load_team_data()
    print(f"Datos cargados: {X.shape[0]} jugadores, {X.shape[1]} métricas")
    print(f"Métricas utilizadas: {metrics}")
    
    # Definir algoritmos a comparar
    algorithms = {
        'K-means': KMeans(n_clusters=4, random_state=727, n_init=10),
        'Hierarchical': AgglomerativeClustering(n_clusters=4),
        'Gaussian Mixture': GaussianMixture(n_components=4, random_state=727),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=3)
    }
    
    # Evaluar cada algoritmo
    results = []
    for name, algorithm in algorithms.items():
        result = evaluate_clustering(algorithm, X, name)
        if result:
            results.append(result)
    
    if not results:
        print("No se pudieron evaluar algoritmos de clustering.")
        return
    
    # Crear tabla de comparación
    print("\n=== RESUMEN DE RESULTADOS ===")
    print(f"{'Algoritmo':<20} {'Clusters':<10} {'Silhouette':<12} {'Calinski-H':<12} {'Davies-B':<12} {'Tiempo':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['algorithm']:<20} "
              f"{result['n_clusters']:<10} "
              f"{result['silhouette_score']:<12.4f} "
              f"{result['calinski_harabasz_score']:<12.2f} "
              f"{result['davies_bouldin_score']:<12.4f} "
              f"{result['training_time']:<10.3f}")
    
    # Análisis de resultados
    print("\n=== ANÁLISIS DE RESULTADOS ===")
    
    # Mejor silhouette score (más alto es mejor)
    best_silhouette = max(results, key=lambda x: x['silhouette_score'])
    print(f"Mejor Silhouette Score: {best_silhouette['algorithm']} ({best_silhouette['silhouette_score']:.4f})")
    
    # Mejor Calinski-Harabasz (más alto es mejor)
    best_calinski = max(results, key=lambda x: x['calinski_harabasz_score'])
    print(f"Mejor Calinski-Harabasz: {best_calinski['algorithm']} ({best_calinski['calinski_harabasz_score']:.2f})")
    
    # Mejor Davies-Bouldin (más bajo es mejor)
    best_davies = min(results, key=lambda x: x['davies_bouldin_score'])
    print(f"Mejor Davies-Bouldin: {best_davies['algorithm']} ({best_davies['davies_bouldin_score']:.4f})")
    
    # Más rápido
    fastest = min(results, key=lambda x: x['training_time'])
    print(f"Más rápido: {fastest['algorithm']} ({fastest['training_time']:.3f}s)")
    
    # Análisis de interpretabilidad
    print(f"\n=== ANÁLISIS DE INTERPRETABILIDAD ===")
    for result in results:
        algorithm = result['algorithm']
        n_clusters = result['n_clusters']
        sizes = result['cluster_sizes']
        
        print(f"{algorithm}:")
        print(f"  Clusters: {n_clusters}")
        if -1 not in sizes:  # No hay outliers
            balance = max(sizes.values()) / min(sizes.values()) if min(sizes.values()) > 0 else float('inf')
            print(f"  Balance de clusters: {balance:.2f} (ideal: ~1.0)")
        else:
            print(f"  Outliers detectados: {sizes.get(-1, 0)}")
        print(f"  Tamaños: {list(sizes.values())}")
    
    # Guardar resultados
    results_df = pd.DataFrame(results)
    results_df.to_csv("clustering_comparison_results.csv", index=False)
    print(f"\nResultados guardados en: clustering_comparison_results.csv")
    
    return results

if __name__ == "__main__":
    results = main() 