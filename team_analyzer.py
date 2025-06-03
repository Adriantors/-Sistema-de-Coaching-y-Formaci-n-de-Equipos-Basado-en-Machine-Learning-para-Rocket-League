"""
Team Analyzer Module

This module implements team formation algorithms using clustering and correlation analysis
to detect optimal synergies between players based on their playing styles and roles.

Author: Adrián Torremocha
Date: 02-05-2025
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from typing import Dict, List, Tuple, Optional
import itertools
from pathlib import Path

class TeamAnalyzer:
    """
    A class for analyzing team compositions and finding optimal player combinations.
    
    This class handles:
    - Loading and processing Birmingham Major data
    - Clustering players by playing styles
    - Calculating team synergies and correlations
    - Finding optimal team compositions
    """
    
    def __init__(self, data_path="data/1-birmingham-major-pg7kax34a0-players.csv"):
        """
        Initialize the TeamAnalyzer with Birmingham Major data.
        
        Args:
            data_path (str): Path to the Birmingham Major data file
        """
        self.data_path = data_path
        self.players_data = None
        self.team_results = None
        self.scaler = StandardScaler()
        self.player_clusters = None
        self.team_synergy_model = None
        
        # Clasificaciones de equipos del Birmingham Major
        self.team_rankings = {
            "KARMINE CORP": 1,
            "THE ULTIMATES": 2,
            "FURIA": 3.5,  # 3/4
            "NRG ESPORTS": 3.5,  # 3/4
            "TEAM VITALITY": 5.5,  # 5/6
            "TEAM FALCONS": 5.5,  # 5/6
            "TWISTED MINDS": 7.5,  # 7/8
            "GEEKAY ESPORTS": 7.5,  # 7/8
            "DIGNITAS": 10,  # 9/11
            "GENG MOBIL 1": 10,  # 9/11
            "LUMINOSITY": 10,  # 9/11
            "WILDCARD": 13,  # 12/14
            "TEAM SECRET": 13,  # 12/14
            "COMPLEXITY": 13,  # 12/14
            "HELFIE CHIEFS": 15.5,  # 15/16
            "FUT ESPORTS": 15.5  # 15/16
        }
        
        self.load_and_process_data()
        
    def load_and_process_data(self):
        """Cargar y procesar los datos del Birmingham Major."""
        # Cargar datos
        df = pd.read_csv(self.data_path, sep=';')
        
        # Filtrar solo columnas 'per game'
        per_game_cols = [col for col in df.columns if 'per game' in col.lower()]
        essential_cols = ['team name', 'player name', 'win percentage']
        
        # Seleccionar columnas relevantes
        selected_cols = essential_cols + per_game_cols
        self.players_data = df[selected_cols].copy()
        
        # Añadir clasificaciones de equipos
        self.players_data['team_ranking'] = self.players_data['team name'].map(self.team_rankings)
        
        # Calcular puntuación de éxito del equipo (menor ranking = mayor éxito)
        self.players_data['team_success'] = 17 - self.players_data['team_ranking']  # Invertir ranking
        
        # Procesar datos a nivel de equipo
        self.team_results = self.players_data.groupby('team name').agg({
            'win percentage': 'first',
            'team_ranking': 'first',
            'team_success': 'first'
        }).reset_index()
        
    def extract_player_features(self, player_stats: Dict) -> np.ndarray:
        """
        Extract comprehensive feature vector from player statistics for team analysis.
        Uses the same extensive feature set as the rank predictor for maximum accuracy.
        
        Args:
            player_stats (Dict): Player statistics dictionary
            
        Returns:
            np.ndarray: Comprehensive feature vector for team analysis
        """
        # Características completas que coinciden con rank_predictor para máxima precisión
        features = [
            # Estadísticas principales
            player_stats.get('score_per_game', 0),
            player_stats.get('goals_per_game', 0),
            player_stats.get('assists_per_game', 0),
            player_stats.get('saves_per_game', 0),
            player_stats.get('shots_per_game', 0),
            player_stats.get('shots_conceded_per_game', 0),
            player_stats.get('goals_conceded_per_game', 0),
            player_stats.get('goals_conceded_while_last_defender_per_game', 0),
            player_stats.get('bpm_per_game', 0),
            
            # Estadísticas de boost (completas)
            player_stats.get('avg_boost_amount_per_game', 0),
            player_stats.get('amount_collected_per_game', 0),
            player_stats.get('amount_collected_big_pads_per_game', 0),
            player_stats.get('amount_collected_small_pads_per_game', 0),
            player_stats.get('count_collected_big_pads_per_game', 0),
            player_stats.get('count_collected_small_pads_per_game', 0),
            player_stats.get('amount_stolen_per_game', 0),
            player_stats.get('amount_stolen_big_pads_per_game', 0),
            player_stats.get('amount_stolen_small_pads_per_game', 0),
            player_stats.get('count_stolen_big_pads_per_game', 0),
            player_stats.get('count_stolen_small_pads_per_game', 0),
            player_stats.get('0_boost_time_per_game', 0),
            player_stats.get('100_boost_time_per_game', 0),
            player_stats.get('amount_used_while_supersonic_per_game', 0),
            player_stats.get('amount_overfill_total_per_game', 0),
            player_stats.get('amount_overfill_stolen_per_game', 0),
            
            # Estadísticas de movimiento (completas)
            player_stats.get('avg_speed_per_game', 0),
            player_stats.get('total_distance_per_game', 0),
            player_stats.get('time_slow_speed_per_game', 0),
            player_stats.get('time_boost_speed_per_game', 0),
            player_stats.get('time_supersonic_speed_per_game', 0),
            player_stats.get('time_on_ground_per_game', 0),
            player_stats.get('time_low_in_air_per_game', 0),
            player_stats.get('time_high_in_air_per_game', 0),
            player_stats.get('time_powerslide_per_game', 0),
            player_stats.get('avg_powerslide_time_per_game', 0),
            player_stats.get('count_powerslide_per_game', 0),
            
            # Estadísticas de posicionamiento (completas)
            player_stats.get('time_most_back_per_game', 0),
            player_stats.get('time_most_forward_per_game', 0),
            player_stats.get('time_in_front_of_ball_per_game', 0),
            player_stats.get('time_behind_ball_per_game', 0),
            player_stats.get('time_defensive_half_per_game', 0),
            player_stats.get('time_offensive_half_per_game', 0),
            player_stats.get('time_defensive_third_per_game', 0),
            player_stats.get('time_neutral_third_per_game', 0),
            player_stats.get('time_offensive_third_per_game', 0),
            player_stats.get('avg_distance_to_ball_per_game', 0),
            player_stats.get('avg_distance_to_ball_has_possession_per_game', 0),
            player_stats.get('avg_distance_to_ball_no_possession_per_game', 0),
            
            # Estadísticas de demoliciones
            player_stats.get('demos_inflicted_per_game', 0),
            player_stats.get('demos_taken_per_game', 0),
            
            # Métricas personalizadas (ratios calculados y estadísticas avanzadas)
            player_stats.get('shot_accuracy', 0),
            player_stats.get('save_ratio', 0),
            player_stats.get('boost_collection_efficiency', 0),
            player_stats.get('boost_steal_ratio', 0),
            player_stats.get('ground_air_ratio', 0),
            player_stats.get('offensive_positioning', 0),
            player_stats.get('ball_control', 0),
            player_stats.get('speed_management', 0),
            player_stats.get('rotation_balance', 0),
            player_stats.get('defensive_presence', 0),
            player_stats.get('demo_efficiency', 0),
            player_stats.get('boost_denial', 0),
            player_stats.get('total_off_actions', 0),
            player_stats.get('total_def_actions', 0)
        ]
        
        return np.array(features)
    
    def analyze_player_roles(self):
        """Analizar y agrupar jugadores por sus roles y estilos de juego."""
        # Preparar matriz de características
        feature_cols = [col for col in self.players_data.columns if 'per game' in col.lower()]
        X = self.players_data[feature_cols].fillna(0)
        
        # Estandarizar características
        X_scaled = self.scaler.fit_transform(X)
        
        # Aplicar clustering K-means para identificar roles de jugadores
        kmeans = KMeans(n_clusters=4, random_state=42)  # 4 roles: Delantero, Creador, Defensor, Polivalente
        self.player_clusters = kmeans.fit_predict(X_scaled)
        
        # Añadir etiquetas de cluster a los datos
        self.players_data['role_cluster'] = self.player_clusters
        
        # Analizar características del cluster
        cluster_analysis = {}
        for cluster in range(4):
            cluster_players = self.players_data[self.players_data['role_cluster'] == cluster]
            cluster_analysis[cluster] = {
                'count': len(cluster_players),
                'avg_goals': cluster_players['goals per game'].mean(),
                'avg_assists': cluster_players['assists per game'].mean(),
                'avg_saves': cluster_players['saves per game'].mean(),
                'avg_score': cluster_players['score per game'].mean(),
                'avg_team_success': cluster_players['team_success'].mean()
            }
        
        # Asignar nombres de roles basados en las características
        role_names = {
            0: "Delantero",    # High goals, moderate assists
            1: "Creador",      # High assists, moderate goals
            2: "Defensor",     # High saves, low goals
            3: "Polivalente"   # Balanced stats
        }
        
        # Encontrar el cluster con más goles para Striker
        striker_cluster = max(cluster_analysis.keys(), 
                            key=lambda x: cluster_analysis[x]['avg_goals'])
        
        # Encontrar el cluster con más asistencias para Playmaker
        playmaker_cluster = max(cluster_analysis.keys(), 
                              key=lambda x: cluster_analysis[x]['avg_assists'])
        
        # Encontrar el cluster con más saves para Defender
        defender_cluster = max(cluster_analysis.keys(), 
                             key=lambda x: cluster_analysis[x]['avg_saves'])
        
        # El cluster restante es Polivalente
        all_clusters = set(range(4))
        assigned_clusters = {striker_cluster, playmaker_cluster, defender_cluster}
        allrounder_cluster = list(all_clusters - assigned_clusters)[0]
        
        self.role_mapping = {
            striker_cluster: "Delantero",
            playmaker_cluster: "Creador", 
            defender_cluster: "Defensor",
            allrounder_cluster: "Polivalente"
        }
        
        self.players_data['role'] = self.players_data['role_cluster'].map(self.role_mapping)
        
        return cluster_analysis
    
    def calculate_team_synergy(self, player_stats_list: List[Dict]) -> float:
        """
        Calculate comprehensive synergy score for a team of 3 players using all available metrics.
        
        Args:
            player_stats_list (List[Dict]): List of 3 player statistics
            
        Returns:
            float: Team synergy score (0-1)
        """
        if len(player_stats_list) != 3:
            return 0.0
        
        # Extraer vectores de características completos para cada jugador
        features = [self.extract_player_features(stats) for stats in player_stats_list]
        features_array = np.array(features)
        
        # Normalizar características para mejor comparación
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features_array)
        
        # 1. Análisis de diversidad de roles (mejorado)
        roles = []
        for stats in player_stats_list:
            # Determinación de rol más sofisticada usando múltiples métricas
            goals_ratio = stats.get('shot_accuracy', 0)
            assists_ratio = stats.get('assists_per_game', 0)
            saves_ratio = stats.get('save_ratio', 0)
            defensive_presence = stats.get('defensive_presence', 0)
            offensive_positioning = stats.get('offensive_positioning', 0)
            
            # Asignación de rol multifactorial
            if goals_ratio > 0.3 and offensive_positioning > 1.2:
                roles.append("Delantero")
            elif assists_ratio > 0.5 and stats.get('ball_control', 0) > 1.0:
                roles.append("Creador")
            elif saves_ratio > 0.3 and defensive_presence > 1.2:
                roles.append("Defensor")
            else:
                roles.append("Polivalente")
        
        unique_roles = len(set(roles))
        role_diversity_bonus = unique_roles / 3.0
        
        # 2. Análisis de complementariedad mejorado
        complementarity_scores = []
        
        # Complementariedad de rendimiento principal
        total_goals = sum(stats.get('goals_per_game', 0) for stats in player_stats_list)
        total_saves = sum(stats.get('saves_per_game', 0) for stats in player_stats_list)
        total_assists = sum(stats.get('assists_per_game', 0) for stats in player_stats_list)
        total_score = sum(stats.get('score_per_game', 0) for stats in player_stats_list)
        
        # Ratios ideales basados en equipos exitosos del Birmingham Major
        ideal_goals_per_team = 2.1
        ideal_saves_per_team = 5.2
        ideal_assists_per_team = 1.6
        ideal_score_per_team = 1200
        
        goals_score = max(0, 1.0 - abs(total_goals - ideal_goals_per_team) / ideal_goals_per_team)
        saves_score = max(0, 1.0 - abs(total_saves - ideal_saves_per_team) / ideal_saves_per_team)
        assists_score = max(0, 1.0 - abs(total_assists - ideal_assists_per_team) / ideal_assists_per_team)
        score_balance = max(0, 1.0 - abs(total_score - ideal_score_per_team) / ideal_score_per_team)
        
        core_complementarity = (goals_score + saves_score + assists_score + score_balance) / 4.0
        complementarity_scores.append(core_complementarity)
        
        # 3. Sinergia avanzada de boost
        boost_efficiency = [stats.get('boost_collection_efficiency', 0) for stats in player_stats_list]
        boost_steal_ratio = [stats.get('boost_steal_ratio', 0) for stats in player_stats_list]
        boost_denial = [stats.get('boost_denial', 0) for stats in player_stats_list]
        
        # Diversidad en estilos de gestión de boost
        boost_eff_variance = np.var(boost_efficiency) if boost_efficiency else 0
        boost_steal_variance = np.var(boost_steal_ratio) if boost_steal_ratio else 0
        boost_denial_variance = np.var(boost_denial) if boost_denial else 0
        
        boost_synergy = min((boost_eff_variance + boost_steal_variance + boost_denial_variance) / 3, 1.0)
        complementarity_scores.append(boost_synergy)
        
        # 4. Sinergia de movimiento y mecánicas
        speed_management = [stats.get('speed_management', 0) for stats in player_stats_list]
        ground_air_ratio = [stats.get('ground_air_ratio', 0) for stats in player_stats_list]
        supersonic_time = [stats.get('time_supersonic_speed_per_game', 0) for stats in player_stats_list]
        
        speed_variance = np.var(speed_management) if speed_management else 0
        air_variance = np.var(ground_air_ratio) if ground_air_ratio else 0
        supersonic_variance = np.var(supersonic_time) if supersonic_time else 0
        
        movement_synergy = min((speed_variance + air_variance + supersonic_variance / 1000) / 3, 1.0)
        complementarity_scores.append(movement_synergy)
        
        # 5. Sinergia de posicionamiento mejorada
        back_times = [stats.get('time_most_back_per_game', 0) for stats in player_stats_list]
        forward_times = [stats.get('time_most_forward_per_game', 0) for stats in player_stats_list]
        defensive_third = [stats.get('time_defensive_third_per_game', 0) for stats in player_stats_list]
        offensive_third = [stats.get('time_offensive_third_per_game', 0) for stats in player_stats_list]
        
        back_variance = np.var(back_times) if back_times else 0
        forward_variance = np.var(forward_times) if forward_times else 0
        def_third_variance = np.var(defensive_third) if defensive_third else 0
        off_third_variance = np.var(offensive_third) if offensive_third else 0
        
        positioning_synergy = min((back_variance + forward_variance + def_third_variance + off_third_variance) / 40000, 1.0)
        complementarity_scores.append(positioning_synergy)
        
        # 6. Química de equipo (rotación y control de pelota)
        rotation_balance = [stats.get('rotation_balance', 0) for stats in player_stats_list]
        ball_control = [stats.get('ball_control', 0) for stats in player_stats_list]
        demo_efficiency = [stats.get('demo_efficiency', 0) for stats in player_stats_list]
        
        rotation_variance = np.var(rotation_balance) if rotation_balance else 0
        ball_variance = np.var(ball_control) if ball_control else 0
        demo_variance = np.var(demo_efficiency) if demo_efficiency else 0
        
        chemistry_synergy = min((rotation_variance + ball_variance + demo_variance) / 30, 1.0)
        complementarity_scores.append(chemistry_synergy)
        
        # 7. Diversidad general de características usando similitud coseno
        feature_diversity = 0.0
        for i in range(len(normalized_features)):
            for j in range(i + 1, len(normalized_features)):
                similarity = cosine_similarity([normalized_features[i]], [normalized_features[j]])[0][0]
                feature_diversity += (1 - similarity)
        feature_diversity /= 3  # Diversidad promedio
        
        # Calcular puntuación de sinergia ponderada
        if len(complementarity_scores) >= 5:
            weights = [0.25, 0.15, 0.15, 0.15, 0.15, 0.15][:len(complementarity_scores)]
            weighted_complementarity = np.average(complementarity_scores, weights=weights)
        else:
            weighted_complementarity = np.mean(complementarity_scores)
        
        # Cálculo final de sinergia con pesos optimizados
        synergy_score = (
            0.2 * role_diversity_bonus +
            0.5 * weighted_complementarity +
            0.3 * feature_diversity
        )
        
        return max(0.0, min(1.0, synergy_score))
    
    def find_optimal_team_from_players(self, available_players: List[Dict], 
                                     mode: str = "best_trio") -> List[Dict]:
        """
        Find the optimal team composition from available players.
        
        Args:
            available_players (List[Dict]): List of available player statistics
            mode (str): "best_trio" or "find_third" (for premade duo)
            
        Returns:
            List[Dict]: Optimal team composition with synergy scores
        """
        if mode == "best_trio":
            return self._find_best_trio(available_players)
        elif mode == "find_third":
            return self._find_best_third_player(available_players)
        else:
            raise ValueError("Mode must be 'best_trio' or 'find_third'")
    
    def _find_best_trio(self, available_players: List[Dict]) -> List[Dict]:
        """Encontrar el mejor trío de todos los jugadores disponibles."""
        if len(available_players) < 3:
            return []
        
        best_teams = []
        
        # Generar todas las combinaciones posibles de 3 jugadores
        for combo in itertools.combinations(available_players, 3):
            synergy_score = self.calculate_team_synergy(list(combo))
            
            # Calcular nivel de habilidad individual
            avg_score = sum(player.get('score_per_game', 0) for player in combo) / 3
            avg_goals = sum(player.get('goals_per_game', 0) for player in combo) / 3
            avg_saves = sum(player.get('saves_per_game', 0) for player in combo) / 3
            
            skill_score = (avg_score / 400 + avg_goals / 1 + avg_saves / 2) / 3
            
            # Puntuación combinada (sinergia + habilidad)
            combined_score = 0.6 * synergy_score + 0.4 * skill_score
            
            team_info = {
                'players': [
                    {
                        'name': player.get('name', 'Unknown'),
                        'stats': player
                    } for player in combo
                ],
                'synergy_score': synergy_score,
                'skill_score': skill_score,
                'combined_score': combined_score,
                'roles': self._get_team_roles(list(combo))
            }
            
            best_teams.append(team_info)
        
        # Ordenar por puntuación combinada y devolver top 5
        best_teams.sort(key=lambda x: x['combined_score'], reverse=True)
        return best_teams[:5]
    
    def _find_best_third_player(self, available_players: List[Dict]) -> List[Dict]:
        """Encontrar el mejor tercer jugador para un dúo preformado."""
        if len(available_players) < 3:
            return []
        
        # Asumir que los primeros dos jugadores son el dúo preformado
        duo = available_players[:2]
        candidates = available_players[2:]
        
        best_thirds = []
        
        for candidate in candidates:
            team = duo + [candidate]
            synergy_score = self.calculate_team_synergy(team)
            
            # Calcular qué tan bien el tercer jugador complementa al dúo
            duo_roles = self._get_team_roles(duo)
            full_team_roles = self._get_team_roles(team)
            
            # Bonificación por llenar roles faltantes
            role_completion_bonus = len(set(full_team_roles)) - len(set(duo_roles))
            role_completion_bonus = max(0, role_completion_bonus) / 2.0
            
            # Habilidad individual del candidato
            candidate_skill = (
                candidate.get('score_per_game', 0) / 400 +
                candidate.get('goals_per_game', 0) / 1 +
                candidate.get('saves_per_game', 0) / 2
            ) / 3
            
            # Puntuación combinada
            combined_score = 0.5 * synergy_score + 0.3 * candidate_skill + 0.2 * role_completion_bonus
            
            third_info = {
                'player': {
                    'name': candidate.get('name', 'Unknown'),
                    'stats': candidate
                },
                'synergy_score': synergy_score,
                'skill_score': candidate_skill,
                'role_completion_bonus': role_completion_bonus,
                'combined_score': combined_score,
                'team_roles': full_team_roles
            }
            
            best_thirds.append(third_info)
        
        # Ordenar por puntuación combinada y devolver top 5
        best_thirds.sort(key=lambda x: x['combined_score'], reverse=True)
        return best_thirds[:5]
    
    def _get_team_roles(self, team: List[Dict]) -> List[str]:
        """Obtener los roles de los jugadores en un equipo."""
        roles = []
        for player in team:
            goals_ratio = player.get('goals_per_game', 0) / max(player.get('shots_per_game', 1), 1)
            assists_ratio = player.get('assists_per_game', 0)
            saves_ratio = player.get('saves_per_game', 0)
            
            if goals_ratio > 0.25 and player.get('goals_per_game', 0) > 0.6:
                roles.append("Striker")
            elif assists_ratio > 0.5:
                roles.append("Playmaker")
            elif saves_ratio > 1.5:
                roles.append("Defender")
            else:
                roles.append("All-rounder")
        
        return roles
    
    def get_team_analysis_insights(self) -> Dict:
        """Obtener insights del análisis de equipos del Birmingham Major."""
        if self.players_data is None:
            return {}
        
        # Analizar equipos exitosos vs no exitosos
        successful_teams = self.players_data[self.players_data['team_ranking'] <= 4]
        unsuccessful_teams = self.players_data[self.players_data['team_ranking'] > 8]
        
        insights = {
            'successful_team_patterns': {
                'avg_goals_per_player': successful_teams['goals per game'].mean(),
                'avg_assists_per_player': successful_teams['assists per game'].mean(),
                'avg_saves_per_player': successful_teams['saves per game'].mean(),
                'avg_score_per_player': successful_teams['score per game'].mean(),
                'avg_boost_efficiency': successful_teams['bpm per game'].mean()
            },
            'unsuccessful_team_patterns': {
                'avg_goals_per_player': unsuccessful_teams['goals per game'].mean(),
                'avg_assists_per_player': unsuccessful_teams['assists per game'].mean(),
                'avg_saves_per_player': unsuccessful_teams['saves per game'].mean(),
                'avg_score_per_player': unsuccessful_teams['score per game'].mean(),
                'avg_boost_efficiency': unsuccessful_teams['bpm per game'].mean()
            },
            'key_differences': {},
            'top_teams': self.team_results.nsmallest(4, 'team_ranking')['team name'].tolist()
        }
        
        # Calcular diferencias clave
        for metric in ['avg_goals_per_player', 'avg_assists_per_player', 'avg_saves_per_player', 
                      'avg_score_per_player', 'avg_boost_efficiency']:
            diff = insights['successful_team_patterns'][metric] - insights['unsuccessful_team_patterns'][metric]
            insights['key_differences'][metric] = diff
        
        return insights 