"""
Rank Predictor Module

This module implements a rank prediction system for Rocket League players based on their in-game statistics.
It uses a pre-trained LightGBM model to predict player ranks and provides detailed comparisons with rank statistics.

Author: Adrián Torremocha
Date: 02-05-2025
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

class RankPredictor:
    """
    A class for predicting player ranks and comparing player statistics with rank averages.
    
    This class handles:
    - Loading and managing the rank prediction model
    - Processing player statistics
    - Making rank predictions
    - Comparing player statistics with rank averages
    """
    
    def __init__(self, models_dir="outputs"):
        """
        Initialize the RankPredictor with a pre-trained model.
        
        Args:
            models_dir (str): Directory containing pre-trained model files
        """
        self.models_dir = Path(models_dir)
        self.models = []
        self.rank_stats = None
        self.shap_explainer = None
        self.rank_order = [
            "Bronce", "Plata", "Oro", "Platino", "Diamante",
            "Champion", "Grand Champion", "Supersonic Legend"
        ]
        
        # Mapeo de nombres de la API a nombres del modelo
        self.api_to_model = {
            # Estadísticas principales
            'score': 'score_per_game',
            'goals': 'goals_per_game',
            'assists': 'assists_per_game',
            'saves': 'saves_per_game',
            'shots': 'shots_per_game',
            'shots_against': 'shots_conceded_per_game',
            'goals_against': 'goals_conceded_per_game',
            'goals_against_while_last_defender': 'goals_conceded_while_last_defender_per_game',
            # Estadísticas de boost
            'bpm': 'bpm_per_game',
            'avg_amount': 'avg_boost_amount_per_game',
            'amount_collected': 'amount_collected_per_game',
            'amount_collected_big': 'amount_collected_big_pads_per_game',
            'amount_collected_small': 'amount_collected_small_pads_per_game',
            'count_collected_big': 'count_collected_big_pads_per_game',
            'count_collected_small': 'count_collected_small_pads_per_game',
            'amount_stolen': 'amount_stolen_per_game',
            'amount_stolen_big': 'amount_stolen_big_pads_per_game',
            'amount_stolen_small': 'amount_stolen_small_pads_per_game',
            'count_stolen_big': 'count_stolen_big_pads_per_game',
            'count_stolen_small': 'count_stolen_small_pads_per_game',
            'time_zero_boost': '0_boost_time_per_game',
            'time_full_boost': '100_boost_time_per_game',
            'amount_used_while_supersonic': 'amount_used_while_supersonic_per_game',
            'amount_overfill': 'amount_overfill_total_per_game',
            'amount_overfill_stolen': 'amount_overfill_stolen_per_game',
            # Estadísticas de movimiento
            'avg_speed': 'avg_speed_per_game',
            'total_distance': 'total_distance_per_game',
            'time_slow_speed': 'time_slow_speed_per_game',
            'time_boost_speed': 'time_boost_speed_per_game',
            'time_supersonic_speed': 'time_supersonic_speed_per_game',
            'time_ground': 'time_on_ground_per_game',
            'time_low_air': 'time_low_in_air_per_game',
            'time_high_air': 'time_high_in_air_per_game',
            'time_powerslide': 'time_powerslide_per_game',
            'avg_powerslide_duration': 'avg_powerslide_time_per_game',
            'count_powerslide': 'count_powerslide_per_game',
            # Estadísticas de posicionamiento
            'time_most_back': 'time_most_back_per_game',
            'time_most_forward': 'time_most_forward_per_game',
            'time_infront_ball': 'time_in_front_of_ball_per_game',
            'time_behind_ball': 'time_behind_ball_per_game',
            'time_defensive_half': 'time_defensive_half_per_game',
            'time_offensive_half': 'time_offensive_half_per_game',
            'time_defensive_third': 'time_defensive_third_per_game',
            'time_neutral_third': 'time_neutral_third_per_game',
            'time_offensive_third': 'time_offensive_third_per_game',
            'avg_distance_to_ball': 'avg_distance_to_ball_per_game',
            'avg_distance_to_ball_possession': 'avg_distance_to_ball_has_possession_per_game',
            'avg_distance_to_ball_no_possession': 'avg_distance_to_ball_no_possession_per_game',
            # Estadísticas de demoliciones
            'inflicted': 'demos_inflicted_per_game',
            'taken': 'demos_taken_per_game',
        }
        
        self.load_models()
        
    def safe_divide(self, a: float, b: float, fill_value=0) -> float:
        """
        Safely divide two numbers, returning 0 if the denominator is 0.
        
        Args:
            a (float): Numerator
            b (float): Denominator
            fill_value (float): Value to return if division is undefined
            
        Returns:
            float: Result of division or 0 if denominator is 0
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            res = np.divide(a, b)
        if isinstance(res, (pd.Series, pd.DataFrame)):
            return res.replace([np.inf, -np.inf], fill_value).fillna(fill_value)
        return fill_value if np.isinf(res) or np.isnan(res) else res
        
    def load_models(self):
        """Cargar los modelos y artefactos necesarios."""
        # Cargar modelos
        for i in range(5):
            model_path = self.models_dir / f"rank_predictor_gbdt_{i}.txt"
            if model_path.exists():
                self.models.append(lgb.Booster(model_file=str(model_path)))
        
        # Cargar estadísticas por rango
        stats_path = self.models_dir / "rank_stats.parquet"
        if stats_path.exists():
            self.rank_stats = pd.read_parquet(stats_path)
            
        # Cargar explainer SHAP
        shap_path = self.models_dir / "shap_explainer.pkl"
        if shap_path.exists():
            self.shap_explainer = joblib.load(shap_path)
    
    def prepare_features(self, stats: Dict) -> np.ndarray:
        """
        Prepare player statistics into a feature vector for prediction.
        
        Args:
            stats (Dict): Dictionary containing player statistics with _per_game suffix
            
        Returns:
            np.ndarray: Feature vector ready for prediction
        """
        try:
            # Si las estadísticas ya vienen procesadas (con _per_game)
            if 'shots_per_game' in stats:
                # Crear vector de características en el orden esperado por el modelo
                features = [
                    # Estadísticas principales
                    stats.get('score_per_game', 0),
                    stats.get('goals_per_game', 0),
                    stats.get('assists_per_game', 0),
                    stats.get('saves_per_game', 0),
                    stats.get('shots_per_game', 0),
                    stats.get('shots_against_per_game', 0),
                    stats.get('goals_against_per_game', 0),
                    stats.get('goals_against_while_last_defender_per_game', 0),
                    stats.get('bpm_per_game', 0),
                    
                    # Estadísticas de boost
                    stats.get('avg_amount_per_game', 0),
                    stats.get('amount_collected_per_game', 0),
                    stats.get('amount_collected_big_per_game', 0),
                    stats.get('amount_collected_small_per_game', 0),
                    stats.get('count_collected_big_per_game', 0),
                    stats.get('count_collected_small_per_game', 0),
                    stats.get('amount_stolen_per_game', 0),
                    stats.get('amount_stolen_big_per_game', 0),
                    stats.get('amount_stolen_small_per_game', 0),
                    stats.get('count_stolen_big_per_game', 0),
                    stats.get('count_stolen_small_per_game', 0),
                    stats.get('time_zero_boost_per_game', 0),
                    stats.get('time_full_boost_per_game', 0),
                    stats.get('amount_used_while_supersonic_per_game', 0),
                    stats.get('amount_overfill_per_game', 0),
                    stats.get('amount_overfill_stolen_per_game', 0),
                    
                    # Estadísticas de movimiento
                    stats.get('avg_speed_per_game', 0),
                    stats.get('total_distance_per_game', 0),
                    stats.get('time_slow_speed_per_game', 0),
                    stats.get('time_boost_speed_per_game', 0),
                    stats.get('time_supersonic_speed_per_game', 0),
                    stats.get('time_ground_per_game', 0),
                    stats.get('time_low_air_per_game', 0),
                    stats.get('time_high_air_per_game', 0),
                    stats.get('time_powerslide_per_game', 0),
                    stats.get('avg_powerslide_duration_per_game', 0),
                    stats.get('count_powerslide_per_game', 0),
                    
                    # Estadísticas de posicionamiento
                    stats.get('time_most_back_per_game', 0),
                    stats.get('time_most_forward_per_game', 0),
                    stats.get('time_infront_ball_per_game', 0),
                    stats.get('time_behind_ball_per_game', 0),
                    stats.get('time_defensive_half_per_game', 0),
                    stats.get('time_offensive_half_per_game', 0),
                    stats.get('time_defensive_third_per_game', 0),
                    stats.get('time_neutral_third_per_game', 0),
                    stats.get('time_offensive_third_per_game', 0),
                    stats.get('avg_distance_to_ball_per_game', 0),
                    stats.get('avg_distance_to_ball_possession_per_game', 0),
                    stats.get('avg_distance_to_ball_no_possession_per_game', 0),
                    0,  # Marcador de posición para mantener la dimensionalidad del modelo
                    
                    # Estadísticas de demoliciones
                    stats.get('inflicted_per_game', 0),
                    stats.get('taken_per_game', 0),
                    
                    # Métricas personalizadas (ya vienen calculadas)
                    stats.get('shot_accuracy', 0),
                    stats.get('save_ratio', 0),
                    stats.get('boost_collection_efficiency', 0),
                    stats.get('boost_steal_ratio', 0),
                    stats.get('ground_air_ratio', 0),
                    stats.get('offensive_positioning', 0),
                    stats.get('ball_control', 0),
                    stats.get('speed_management', 0),
                    stats.get('rotation_balance', 0),
                    stats.get('defensive_presence', 0),
                    0,  # Marcador de posición para teammate_spacing
                    stats.get('demo_efficiency', 0),
                    stats.get('boost_denial', 0),
                    stats.get('total_off_actions', 0),
                    stats.get('total_def_actions', 0)
                ]
                
                return np.array(features).reshape(1, -1)
            
            # Si recibimos la respuesta completa de la API, buscar el jugador
            elif 'players' in stats:
                if not stats['players']:
                    print("Error: No se encontraron jugadores en las estadísticas")
                    return None
                # Usar el primer jugador (asumiendo que ya se ha filtrado el correcto)
                return self.prepare_features(stats['players'][0])
            
            # Si tenemos game_average, usar esa estructura
            elif 'game_average' in stats:
                game_avg = stats['game_average']
                
            else:
                print("Error: Formato de estadísticas no reconocido")
                print("Estructura recibida:", list(stats.keys()))
                return None
                
        except Exception as e:
            print(f"Error al preparar características: {e}")
            return None
    
    def predict_rank(self, player_stats: Dict) -> Tuple[str, float]:
        """
        Predict the rank of a player based on their statistics.
        
        Args:
            player_stats (Dict): Dictionary containing player statistics
            
        Returns:
            Tuple[str, float]: Predicted rank and confidence score
        """
        if not self.models:
            return None, None
            
        # Preparar características
        features = self.prepare_features(player_stats)
        if features is None:
            return None, None
        
        # Hacer predicciones con cada modelo
        predictions = []
        for model in self.models:
            pred = model.predict(features)
            predictions.append(pred[0])
        
        # Calcular predicción promedio y redondear al rango más cercano
        avg_pred = np.mean(predictions)
        rank_id = int(np.clip(np.round(avg_pred), 0, len(self.rank_order)-1))
        rank_name = self.rank_order[rank_id]
        
        # Calcular confianza basada en la varianza de las predicciones
        confidence = 1 - (np.std(predictions) / (len(self.rank_order)-1))
        
        return rank_name, confidence
    
    def get_rank_comparison(self, player_stats: Dict) -> Dict[str, Dict[str, float]]:
        """
        Compare player statistics with the average statistics of a specific rank.
        
        Args:
            player_stats (Dict): Dictionary containing player statistics
            
        Returns:
            Dict[str, Dict[str, float]]: Dictionary containing comparison results
            
        Note:
            The comparison includes:
            - Player's value for each statistic
            - Rank's average value
            - Standard deviation
            - Z-score indicating how many standard deviations the player is from the rank average
        """
        if self.rank_stats is None:
            return None
            
        rank_name, _ = self.predict_rank(player_stats)
        if rank_name is None:
            return None
            
        # Obtener estadísticas del rango predicho
        rank_stats = self.rank_stats.loc[self.rank_order.index(rank_name)]
        
        # Preparar características del jugador
        player_features = self.prepare_features(player_stats)
        if player_features is None:
            return None
            
        # Lista de nombres de características en el mismo orden que se preparan
        feature_names = [
            # Estadísticas principales
            'score_per_game', 'goals_per_game', 'assists_per_game', 'saves_per_game',
            'shots_per_game', 'shots_conceded_per_game', 'goals_conceded_per_game',
            'goals_conceded_while_last_defender_per_game', 'bpm_per_game',
            
            # Estadísticas de boost
            'avg_boost_amount_per_game', 'amount_collected_per_game',
            'amount_collected_big_pads_per_game', 'amount_collected_small_pads_per_game',
            'count_collected_big_pads_per_game', 'count_collected_small_pads_per_game',
            'amount_stolen_per_game', 'amount_stolen_big_pads_per_game',
            'amount_stolen_small_pads_per_game', 'count_stolen_big_pads_per_game',
            'count_stolen_small_pads_per_game', '0_boost_time_per_game',
            '100_boost_time_per_game', 'amount_used_while_supersonic_per_game',
            'amount_overfill_total_per_game', 'amount_overfill_stolen_per_game',
            
            # Estadísticas de movimiento
            'avg_speed_per_game', 'total_distance_per_game', 'time_slow_speed_per_game',
            'time_boost_speed_per_game', 'time_supersonic_speed_per_game',
            'time_on_ground_per_game', 'time_low_in_air_per_game',
            'time_high_in_air_per_game', 'time_powerslide_per_game',
            'avg_powerslide_time_per_game', 'count_powerslide_per_game',
            
            # Estadísticas de posicionamiento
            'time_most_back_per_game', 'time_most_forward_per_game',
            'time_in_front_of_ball_per_game', 'time_behind_ball_per_game',
            'time_defensive_half_per_game', 'time_offensive_half_per_game',
            'time_defensive_third_per_game', 'time_neutral_third_per_game',
            'time_offensive_third_per_game', 'avg_distance_to_ball_per_game',
            'avg_distance_to_ball_has_possession_per_game',
            'avg_distance_to_ball_no_possession_per_game',
            
            # Estadísticas de demoliciones
            'demos_inflicted_per_game', 'demos_taken_per_game',
            
            # Métricas personalizadas
            'shot_accuracy', 'save_ratio', 'boost_collection_efficiency',
            'boost_steal_ratio', 'ground_air_ratio', 'offensive_positioning',
            'ball_control', 'speed_management', 'rotation_balance',
            'defensive_presence', 'teammate_spacing', 'demo_efficiency',
            'boost_denial', 'total_off_actions', 'total_def_actions'
        ]
        
        # Comparar cada característica
        comparison = {}
        for i, feature in enumerate(feature_names):
            try:
                player_value = player_features[0][i]
                rank_mean = rank_stats[feature]['mean']
                rank_std = rank_stats[feature]['std']
                
                # Calcular z-score
                z_score = (player_value - rank_mean) / rank_std if rank_std != 0 else 0
                
                comparison[feature] = {
                    'player_value': player_value,
                    'rank_mean': rank_mean,
                    'rank_std': rank_std,
                    'z_score': z_score
                }
            except KeyError as e:
                print(f"Warning: Feature {feature} not found in rank_stats: {e}")
                continue
        
        return comparison
    
    def get_feature_importance(self, player_stats: Dict) -> pd.DataFrame:
        """
        Get feature importance for the prediction model.
        
        Args:
            player_stats (Dict): Dictionary containing player statistics
            
        Returns:
            pd.DataFrame: DataFrame containing feature importance
        """
        if self.shap_explainer is None:
            return None
            
        features = self.prepare_features(player_stats)
        shap_values = self.shap_explainer.shap_values(features)
        
        # Calcular importancia promedio
        importance = pd.DataFrame({
            'feature': features[0],
            'importance': np.abs(shap_values[0]).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        return importance 