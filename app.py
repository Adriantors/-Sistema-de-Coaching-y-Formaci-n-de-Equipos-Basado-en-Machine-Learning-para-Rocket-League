"""
Rocket League Coaching Platform

This module implements a web application for analyzing Rocket League replays and providing
player performance insights. It integrates with the Ballchasing API and uses a custom
rank prediction model to analyze player performance.

Author: Adrián Torremocha
Date: 02-05-2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
from datetime import datetime
import time
from ballchasing_api import BallchasingAPI
from rank_predictor import RankPredictor
from team_analyzer import TeamAnalyzer
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import plotly.graph_objects as go
from collections import defaultdict
import json

def safe_divide(a, b, fill_value=0):
    """
    Divide a/b evitando inf y NaN.
    
    Args:
        a: Numerador
        b: Denominador
        fill_value: Valor a devolver si la división no es posible
        
    Returns:
        float: Resultado de la división o fill_value si no es posible
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.divide(a, b)
    if isinstance(res, (pd.Series, pd.DataFrame)):
        return res.replace([np.inf, -np.inf], fill_value).fillna(fill_value)
    return fill_value if np.isinf(res) or np.isnan(res) else res

# Configuración de la página
st.set_page_config(
    page_title="Rocket League Coaching",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Configuración de la API de Ballchasing
API_KEY = "JR1R8xOTPOw7grXWLw1VUDHthdQM9gAn5j5wqsYr"
api = BallchasingAPI(API_KEY)

# Inicializar predictor de rango y analizador de equipos
rank_predictor = RankPredictor()
team_analyzer = TeamAnalyzer()

# Lista de rangos disponibles
RANKS = ["Bronce", "Plata", "Oro", "Platino", "Diamante", "Champion", "Grand Champion", "Supersonic Legend"]

# Diccionario de alias: API -> modelo
API_TO_MODEL_STATS = {
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
    # Las métricas personalizadas ya están bien mapeadas
}

# Diccionario de traducciones para las métricas
METRIC_TRANSLATIONS = {
    # Estadísticas principales
    'score_per_game': 'Puntuación por partido',
    'goals_per_game': 'Goles por partido',
    'assists_per_game': 'Asistencias por partido',
    'saves_per_game': 'Paradas por partido',
    'shots_per_game': 'Tiros por partido',
    'shots_conceded_per_game': 'Tiros recibidos por partido',
    'goals_conceded_per_game': 'Goles recibidos por partido',
    'goals_conceded_while_last_defender_per_game': 'Goles recibidos como último defensor',
    'bpm_per_game': 'Boost Medio por Minuto',
    
    # Estadísticas de boost
    'avg_boost_amount_per_game': 'Cantidad media de boost',
    'amount_collected_per_game': 'Boost total recogido',
    'amount_collected_big_pads_per_game': 'Boost recogido de pads grandes',
    'amount_collected_small_pads_per_game': 'Boost recogido de pads pequeños',
    'count_collected_big_pads_per_game': 'Número de pads grandes recogidos',
    'count_collected_small_pads_per_game': 'Número de pads pequeños recogidos',
    'amount_stolen_per_game': 'Boost robado total',
    'amount_stolen_big_pads_per_game': 'Boost robado de pads grandes',
    'amount_stolen_small_pads_per_game': 'Boost robado de pads pequeños',
    'count_stolen_big_pads_per_game': 'Número de pads grandes robados',
    'count_stolen_small_pads_per_game': 'Número de pads pequeños robados',
    '0_boost_time_per_game': 'Tiempo sin boost',
    '100_boost_time_per_game': 'Tiempo con boost completo',
    'amount_used_while_supersonic_per_game': 'Boost usado en supersónico',
    'amount_overfill_total_per_game': 'Boost desperdiciado total',
    'amount_overfill_stolen_per_game': 'Boost desperdiciado al robar',
    
    # Estadísticas de movimiento
    'avg_speed_per_game': 'Velocidad media',
    'total_distance_per_game': 'Distancia total recorrida',
    'time_slow_speed_per_game': 'Tiempo a velocidad baja',
    'time_boost_speed_per_game': 'Tiempo a velocidad de boost',
    'time_supersonic_speed_per_game': 'Tiempo en supersónico',
    'time_on_ground_per_game': 'Tiempo en suelo',
    'time_low_in_air_per_game': 'Tiempo en vuelo bajo',
    'time_high_in_air_per_game': 'Tiempo en vuelo alto',
    'time_powerslide_per_game': 'Tiempo en powerslide',
    'avg_powerslide_time_per_game': 'Duración media de powerslide',
    'count_powerslide_per_game': 'Número de powerslides',
    
    # Estadísticas de posicionamiento
    'time_most_back_per_game': 'Tiempo como último defensor',
    'time_most_forward_per_game': 'Tiempo como delantero',
    'time_in_front_of_ball_per_game': 'Tiempo delante de la pelota',
    'time_behind_ball_per_game': 'Tiempo detrás de la pelota',
    'time_defensive_half_per_game': 'Tiempo en campo defensivo',
    'time_offensive_half_per_game': 'Tiempo en campo ofensivo',
    'time_defensive_third_per_game': 'Tiempo en tercio defensivo',
    'time_neutral_third_per_game': 'Tiempo en tercio neutral',
    'time_offensive_third_per_game': 'Tiempo en tercio ofensivo',
    'avg_distance_to_ball_per_game': 'Distancia media a la pelota',
    'avg_distance_to_ball_has_possession_per_game': 'Distancia a pelota con posesión',
    'avg_distance_to_ball_no_possession_per_game': 'Distancia a pelota sin posesión',
    
    # Estadísticas de demoliciones
    'demos_inflicted_per_game': 'Demoliciones realizadas',
    'demos_taken_per_game': 'Demoliciones recibidas',
    
    # Métricas personalizadas
    'shot_accuracy': 'Precisión de tiro',
    'save_ratio': 'Efectividad de paradas',
    'boost_collection_efficiency': 'Eficiencia en recogida de boost',
    'boost_steal_ratio': 'Eficiencia en robo de boost',
    'ground_air_ratio': 'Balance suelo/aire',
    'offensive_positioning': 'Posicionamiento ofensivo',
    'ball_control': 'Control de pelota',
    'speed_management': 'Gestión de velocidad',
    'rotation_balance': 'Balance de rotaciones',
    'defensive_presence': 'Presencia defensiva',
    'demo_efficiency': 'Eficiencia de demoliciones',
    'boost_denial': 'Negación de boost',
    'total_off_actions': 'Acciones ofensivas totales',
    'total_def_actions': 'Acciones defensivas totales'
}

# Diccionario de conjuntos de métricas y sus videos tutoriales
METRIC_SETS = {
    'Ataque': {
        'metrics': [
            'goals_per_game',
            'shots_per_game',
            'shot_accuracy',
            'assists_per_game',
            'total_off_actions'
        ],
        'video_url': '',  # Se implementará más adelante
        'video_title': 'Guía Completa de Ataque en Rocket League'
    },
    'Defensa': {
        'metrics': [
            'saves_per_game',
            'save_ratio',
            'goals_conceded_per_game',
            'goals_conceded_while_last_defender_per_game',
            'total_def_actions'
        ],
        'video_url': '',  # Se implementará más adelante
        'video_title': 'Cómo Defender como un Pro'
    },
    'Turbo': {
        'metrics': [
            'avg_boost_amount_per_game',
            'boost_collection_efficiency',
            'boost_steal_ratio',
            'amount_collected_per_game',
            'amount_stolen_per_game',
            '0_boost_time_per_game',
            'boost_denial',
            'amount_used_while_supersonic_per_game'
        ],
        'video_url': '',  # Se implementará más adelante
        'video_title': 'Gestión de Boost Avanzada'
    },
    'Posicionamiento': {
        'metrics': [
            'avg_distance_to_ball_per_game',
            'time_defensive_half_per_game',
            'time_offensive_half_per_game',
            'time_neutral_third_per_game',
            'rotation_balance',
            'time_most_forward_per_game',
            'time_most_back_per_game',
            'ball_control',
            'time_defensive_third_per_game',
            'time_offensive_third_per_game',
            'time_in_front_of_ball_per_game',
            'time_behind_ball_per_game',
            'defensive_presence',
            'offensive_positioning'
        ],
        'video_url': '',  # Se implementará más adelante
        'video_title': 'Posicionamiento y Rotaciones'
    },
    'Movimiento': {
        'metrics': [
            'avg_speed_per_game',
            'time_supersonic_speed_per_game',
            'time_boost_speed_per_game',
            'time_powerslide_per_game',
            'time_low_in_air_per_game',
            'time_high_in_air_per_game',
            'ground_air_ratio',
            'speed_management'
        ],
        'video_url': '',  # Se implementará más adelante
        'video_title': 'Mecánicas Esenciales de Movimiento'
    }
}

def initialize_session_state():
    """
    Initialize the Streamlit session state variables.
    
    This function sets up the initial state for:
    - API key
    - Replay processing status
    - Player statistics
    - Rank predictions
    """
    if 'api_verified' not in st.session_state:
        st.session_state['api_verified'] = api.check_auth()
    if 'processed_replays' not in st.session_state:
        st.session_state['processed_replays'] = []
    if 'player_names' not in st.session_state:
        st.session_state['player_names'] = set()
    if 'processing_complete' not in st.session_state:
        st.session_state['processing_complete'] = False
    if 'is_processing' not in st.session_state:
        st.session_state['is_processing'] = False
    if 'current_group_id' not in st.session_state:
        st.session_state['current_group_id'] = None
    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = None
    if 'start_processing' not in st.session_state:
        st.session_state['start_processing'] = False
    if 'show_player_selection' not in st.session_state:
        st.session_state['show_player_selection'] = False
    if 'show_file_list' not in st.session_state:
        st.session_state['show_file_list'] = False
    if 'selected_player' not in st.session_state:
        st.session_state['selected_player'] = None
    if 'current_rank' not in st.session_state:
        st.session_state['current_rank'] = None
    if 'target_rank' not in st.session_state:
        st.session_state['target_rank'] = None
    if 'start_analysis' not in st.session_state:
        st.session_state['start_analysis'] = False
    if 'team_mode' not in st.session_state:
        st.session_state['team_mode'] = False
    if 'selected_players_team' not in st.session_state:
        st.session_state['selected_players_team'] = []
    if 'team_formation_mode' not in st.session_state:
        st.session_state['team_formation_mode'] = 'best_trio'

def extract_player_names(replay_stats):
    """
    Extrae los nombres de los jugadores de las estadísticas de la replay.
    
    Args:
        replay_stats (dict): Diccionario con las estadísticas de la replay.
        
    Returns:
        set: Conjunto de nombres de jugadores únicos encontrados en la replay.
    """
    player_names = set()
    if replay_stats and "teams" in replay_stats:
        for team in replay_stats["teams"]:
            for player in team["players"]:
                player_names.add(player["name"])
    return player_names

def handle_file_upload():
    """
    Maneja la subida de archivos de replay.
    Actualiza el estado de la sesión con los archivos subidos y reinicia el conjunto de nombres de jugadores.
    """
    if st.session_state['file_uploader'] is not None:
        st.session_state['uploaded_files'] = st.session_state['file_uploader']
        st.session_state['show_file_list'] = True
        st.session_state['player_names'] = set()

def wait_for_group_processing(api, group_id, max_attempts=100, delay=1000):
    """
    Espera a que el grupo termine de procesarse.
    
    Args:
        api: Instancia de BallchasingAPI
        group_id: ID del grupo a esperar
        max_attempts: Número máximo de intentos
        delay: Tiempo de espera entre intentos en milisegundos
        
    Returns:
        dict: Información del grupo o None si no se pudo procesar
    """
    for attempt in range(max_attempts):
        group_info = api.get_group_info(group_id)
        if group_info and group_info.get("status") == "ok":
            return group_info
        time.sleep(delay/1000)  # Convertir a segundos
    return None

def wait_for_group_players(api, group_id, max_attempts=100, delay=1000):
    """
    Espera a que los datos de los jugadores estén disponibles en el grupo.
    
    Args:
        api: Instancia de BallchasingAPI
        group_id: ID del grupo a esperar
        max_attempts: Número máximo de intentos
        delay: Tiempo de espera entre intentos en milisegundos
        
    Returns:
        dict: Información del grupo con datos de jugadores o None si no se encontraron
    """
    last_response = None
    for attempt in range(max_attempts):
        group_info = api.get_group_info(group_id)
        last_response = group_info  # Guardar la última respuesta para debug
        if (group_info and 
            group_info.get("status") == "ok" and 
            'players' in group_info and 
            group_info['players'] and
            any('game_average' in player for player in group_info['players'])):
            return group_info
        time.sleep(delay/1000)  # Convertir a segundos
    
    # Si llegamos aquí, guardamos la última respuesta para debug
    if last_response:
        debug_json = json.dumps(last_response, indent=2, ensure_ascii=False)
        st.download_button(
            label='Descargar última respuesta de la API (Debug)',
            data=debug_json,
            file_name='last_api_response.json',
            mime='application/json'
        )
    return None

def process_replays():
    """
    Procesa las replays subidas a través de la API de Ballchasing.
    Maneja la subida de archivos, el procesamiento y la extracción de estadísticas.
    """
    if st.session_state['uploaded_files'] is None:
        return

    # Creación de un grupo para el análisis de las replays
    if not st.session_state['current_group_id']:
        group_name = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with st.spinner("Creando grupo de análisis..."):
            group_id = api.create_group(group_name)
            if group_id:
                st.session_state['current_group_id'] = group_id
            else:
                st.error("Error al crear el grupo de análisis")
                st.session_state['is_processing'] = False
                return

    # Configuración de la interfaz para mostrar el progreso
    st.markdown('<div class="processing-info">', unsafe_allow_html=True)
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # Procesamiento de las replays
    replay_ids = []
    processed_stats = []
    all_player_names = set()
    temp_files = []
    
    files = st.session_state['uploaded_files']
    total_steps = len(files) * 2  # Subida + procesamiento para cada replay
    current_step = 0
    
    # Fase 1: Subida de replays
    status_text.text("Estado: Subiendo repeticiones...")
    for i, file in enumerate(files):
        # Manejo temporal de archivos
        with tempfile.NamedTemporaryFile(delete=False, suffix='.replay') as tmp_file:
            tmp_file.write(file.getvalue())
            temp_files.append(tmp_file.name)
            
            try:
                # Subida de la replay a la API
                response = api.upload_replay(tmp_file.name, st.session_state['current_group_id'])
                if response and 'id' in response:
                    replay_ids.append(response['id'])
            except Exception as e:
                st.error(f"Error al subir {file.name}: {str(e)}")
        
        current_step += 1
        progress_bar.progress(current_step / total_steps)

    # Después de subir todas las replays, esperar a que el grupo se procese
    if st.session_state['current_group_id']:
        status_text.text("Esperando a que el grupo termine de procesarse...")
        group_info = wait_for_group_processing(api, st.session_state['current_group_id'])
        if not group_info:
            st.error("El grupo no se procesó correctamente. Intenta de nuevo.")
            return

        # Esperar a que los datos de los jugadores estén disponibles
        status_text.text("Esperando a que se agreguen los datos de jugadores al grupo...")
        group_info = wait_for_group_players(api, st.session_state['current_group_id'])
        if not group_info or not group_info.get('players'):
            st.error("No se encontraron datos de jugadores en el grupo. Intenta de nuevo más tarde.")
            return

        # Procesar estadísticas de jugadores
        all_player_stats = []
        for player in group_info['players']:
            player_stats = {}
            
            # Extraer datos de game_average
            if 'game_average' in player:
                game_avg = player['game_average']
                
                # Core stats
                if 'core' in game_avg:
                    for key, value in game_avg['core'].items():
                        player_stats[f"{key}_per_game"] = value
                
                # Boost stats
                if 'boost' in game_avg:
                    for key, value in game_avg['boost'].items():
                        player_stats[f"{key}_per_game"] = value
                
                # Movement stats
                if 'movement' in game_avg:
                    for key, value in game_avg['movement'].items():
                        player_stats[f"{key}_per_game"] = value
                
                # Positioning stats
                if 'positioning' in game_avg:
                    for key, value in game_avg['positioning'].items():
                        if key != 'avg_distance_to_mates':  # Excluir esta métrica
                            player_stats[f"{key}_per_game"] = value
                
                # Demo stats
                if 'demo' in game_avg:
                    for key, value in game_avg['demo'].items():
                        player_stats[f"{key}_per_game"] = value

                # Calcular métricas personalizadas
                player_stats['shot_accuracy'] = safe_divide(
                    game_avg['core'].get('goals', 0),
                    game_avg['core'].get('shots', 0)
                )
                player_stats['save_ratio'] = safe_divide(
                    game_avg['core'].get('saves', 0),
                    game_avg['core'].get('shots_against', 0)
                )
                player_stats['boost_collection_efficiency'] = safe_divide(
                    game_avg['boost'].get('amount_collected_small', 0),
                    game_avg['boost'].get('amount_collected', 0)
                )
                player_stats['boost_steal_ratio'] = safe_divide(
                    game_avg['boost'].get('amount_stolen', 0),
                    game_avg['boost'].get('amount_collected', 0)
                )
                player_stats['ground_air_ratio'] = safe_divide(
                    game_avg['movement'].get('time_ground', 0),
                    game_avg['movement'].get('time_low_air', 0) + game_avg['movement'].get('time_high_air', 0) + 1
                )
                player_stats['offensive_positioning'] = safe_divide(
                    game_avg['positioning'].get('time_offensive_third', 0),
                    game_avg['positioning'].get('time_defensive_third', 0)
                )
                player_stats['ball_control'] = safe_divide(
                    game_avg['positioning'].get('time_infront_ball', 0),
                    game_avg['positioning'].get('time_behind_ball', 0)
                )
                player_stats['speed_management'] = safe_divide(
                    game_avg['movement'].get('time_supersonic_speed', 0),
                    game_avg['movement'].get('time_boost_speed', 0)
                )
                player_stats['rotation_balance'] = safe_divide(
                    game_avg['positioning'].get('time_most_back', 0),
                    game_avg['positioning'].get('time_most_forward', 0)
                )
                player_stats['defensive_presence'] = safe_divide(
                    game_avg['positioning'].get('time_defensive_half', 0),
                    game_avg['positioning'].get('time_offensive_half', 0)
                )
                player_stats['demo_efficiency'] = safe_divide(
                    game_avg['demo'].get('inflicted', 0),
                    max(game_avg['demo'].get('taken', 0), 1)
                )
                player_stats['boost_denial'] = safe_divide(
                    game_avg['boost'].get('amount_stolen_big', 0) + game_avg['boost'].get('amount_stolen_small', 0),
                    game_avg['boost'].get('amount_collected', 0)
                )
                player_stats['total_off_actions'] = (
                    game_avg['core'].get('goals', 0) +
                    game_avg['core'].get('assists', 0) +
                    game_avg['core'].get('shots', 0)
                )
                player_stats['total_def_actions'] = (
                    game_avg['core'].get('saves', 0) +
                    game_avg['demo'].get('taken', 0)
                )

            # Agregar nombre del jugador
            player_stats['name'] = player.get('name', 'Unknown')
            all_player_names.add(player_stats['name'])
            all_player_stats.append(player_stats)

        st.session_state['all_player_stats_group'] = all_player_stats
        st.session_state['player_names'] = all_player_names

    # Limpieza de archivos temporales
    for tmp_path in temp_files:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    # Actualización del estado de la sesión
    st.session_state['processed_replays'] = processed_stats
    st.session_state['processing_complete'] = True
    st.session_state['is_processing'] = False
    st.session_state['show_player_selection'] = True
    
    status_text.text("Estado: Procesamiento completado!")
    progress_bar.progress(1.0)
    st.markdown('</div>', unsafe_allow_html=True)

def local_css():
    """
    Define los estilos CSS personalizados para la interfaz de la aplicación.
    Incluye estilos para ocultar elementos por defecto de Streamlit y personalizar la apariencia.
    """
    st.markdown("""
        <style>
        /* Ocultar elementos por defecto de Streamlit */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Estilos personalizados */
        .main {
            padding: 0rem 1rem;
            background-color: #0E1117;
        }
        .stButton button {
            background-color: #FF4B4B;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #FF6B6B;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(255, 75, 75, 0.2);
        }
        .title-container {
            text-align: center;
            padding: 2rem 0;
            margin-bottom: 2rem;
            background: linear-gradient(90deg, #1E2A3A 0%, #0E1117 100%);
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        .subtitle {
            color: #9BA5B7;
            font-size: 1.1rem;
            margin-bottom: 1rem;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
        }
        p {
            color: #B8C2CC !important;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
        }
        .section-container {
            background: rgba(30, 42, 58, 0.4);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stSelectbox {
            margin-bottom: 1rem;
        }
        .stSelectbox > div > div {
            background-color: #1E2A3A;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 5px;
        }
        .processing-info {
            background: rgba(30, 42, 58, 0.4);
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .file-list {
            background: rgba(30, 42, 58, 0.4);
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        .file-item {
            padding: 0.5rem;
            margin: 0.25rem 0;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 3px;
            color: #D1D5DB;
        }
        .rank-selectors {
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .rank-selector {
            flex: 1;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #E5E7EB !important;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        /* Mejoras de contraste y visibilidad */
        .stMarkdown div {
            color: #D1D5DB !important;
        }
        .stSpinner, .stProgress {
            color: #E5E7EB !important;
        }
        /* Ocultar etiquetas style */
        style {
            display: none !important;
        }
        </style>
    """, unsafe_allow_html=True)

def get_rank_index(rank: str) -> int:
    """
    Obtiene el índice numérico de un rango.
    """
    return RANKS.index(rank)

def get_rank_difference_description(current_rank: str, predicted_rank: str) -> str:
    """
    Genera una descripción motivadora de la diferencia entre el rango actual y el predicho.
    """
    current_idx = get_rank_index(current_rank)
    predicted_idx = get_rank_index(predicted_rank)
    diff = predicted_idx - current_idx
    
    if diff == 0:
        return "¡Excelente! Tu rango actual refleja perfectamente tu nivel de juego. 🎯"
    elif diff > 0:
        return f"¡Buenas noticias! Tus estadísticas muestran que tienes potencial para alcanzar {predicted_rank}. 🚀"
    else:
        return f"Estás en un momento perfecto para consolidar tus habilidades en {current_rank}. 💪"

def suggest_target_rank(predicted_rank: str) -> str:
    """
    Sugiere un rango objetivo basado en el rango predicho.
    """
    predicted_idx = get_rank_index(predicted_rank)
    suggested_idx = min(predicted_idx + 2, len(RANKS) - 1)
    return RANKS[suggested_idx]

def validate_current_rank(current_rank: str, predicted_rank: str) -> Tuple[bool, str]:
    """
    Valida si el rango actual es coherente con el predicho.
    """
    current_idx = get_rank_index(current_rank)
    predicted_idx = get_rank_index(predicted_rank)
    diff = abs(current_idx - predicted_idx)
    
    if diff > 2:
        return False, "⚠️ El rango que has seleccionado difiere significativamente de tus estadísticas. ¿Estás seguro de que es tu rango actual?"
    return True, ""

def main():
    """
    Main application function that sets up the Streamlit interface and handles user interactions.
    """
    # Aplicación de estilos CSS
    local_css()
    
    # Configuración de la interfaz de Streamlit
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    # Inicialización de estados
    initialize_session_state()
    
    # Configuración de la navegación principal
    st.markdown('<div style="display: flex; justify-content: center; gap: 1rem; margin-bottom: 2rem;">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🎮 Entrenamiento Individual", use_container_width=True):
            st.session_state['team_mode'] = False
            st.rerun()
    with col2:
        if st.button("👥 Análisis de Equipo", use_container_width=True):
            st.session_state['team_mode'] = True
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Configuración del encabezado principal
    st.markdown('<div class="title-container">', unsafe_allow_html=True)
    if st.session_state.get('team_mode', False):
        st.title("👥 Análisis de Equipos - Rocket League")
        st.markdown('<p class="subtitle">Forma equipos con sinergias óptimas usando algoritmos avanzados</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #7C8DA6;">Sube repeticiones de tu equipo y descubre las mejores combinaciones de jugadores basadas en datos del Birmingham Major.</p>', unsafe_allow_html=True)
    else:
        st.title("🚀 Entrenador Automático de Rocket League")
        st.markdown('<p class="subtitle">Mejora tu juego con análisis automatizado de tus repeticiones</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #7C8DA6;">Sube tus repeticiones y recibe un análisis detallado de tu rendimiento, comparado con jugadores de rangos superiores.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Contenedor principal de la aplicación
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    
    # Verificar el modo actual
    if st.session_state.get('team_mode', False):
        # Modo de análisis de equipos
        handle_team_analysis_mode()
    else:
        # Modo individual (código existente)
        # Gestión de la subida de archivos
        if not st.session_state['is_processing'] and not st.session_state['show_player_selection']:
            uploaded_files = st.file_uploader(
                "📁 Selecciona tus archivos de repetición",
                accept_multiple_files=True,
                type=['replay'],
                key='file_uploader',
                on_change=handle_file_upload
            )
        
        # Visualización de archivos subidos
        if 'uploaded_files' in st.session_state and st.session_state['uploaded_files']:
            st.markdown('<div class="file-list">', unsafe_allow_html=True)
            st.markdown("### 📋 Archivos seleccionados:")
            for file in st.session_state['uploaded_files']:
                st.markdown(f'<div class="file-item">📄 {file.name}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.button("🔄 Analizar Repeticiones", type="primary", use_container_width=True, 
                     on_click=lambda: setattr(st.session_state, 'start_processing', True) or setattr(st.session_state, 'is_processing', True))

    # Procesamiento de replays
    if st.session_state['start_processing']:
        process_replays()
        st.session_state['start_processing'] = False

    # Selección de jugador y configuración de rangos (solo en modo individual)
    if st.session_state['show_player_selection'] and not st.session_state.get('team_mode', False):
        st.markdown("### 👤 Identifica tu jugador")
        st.markdown('<p class="subtitle">Selecciona tu nombre en las repeticiones</p>', unsafe_allow_html=True)
        
        # Selector de jugador
        player_names = sorted(list(st.session_state['player_names']))
        selected_player = st.selectbox(
            "Selecciona tu nombre de jugador",
            player_names,
            key='player_selector',
            label_visibility="collapsed"
        )
        
        # Realizar predicción inicial si hay jugador seleccionado
        predicted_rank = None
        confidence = None
        if selected_player and st.session_state.get('all_player_stats_group'):
            all_player_stats = st.session_state['all_player_stats_group']
            selected_stats = next((stats for stats in all_player_stats if stats.get('name') == selected_player), None)
            if selected_stats:
                predicted_rank, confidence = rank_predictor.predict_rank(selected_stats)

        # Selectores de rango con validación y sugerencias
        st.markdown('<div class="rank-selectors">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="rank-selector">', unsafe_allow_html=True)
            st.markdown("#### Rango actual")
            current_rank = st.selectbox(
                "Selecciona tu rango actual",
                RANKS,
                key="current_rank",
                label_visibility="collapsed"
            )
            if current_rank and predicted_rank:
                is_valid, message = validate_current_rank(current_rank, predicted_rank)
                if not is_valid:
                    st.warning(message)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="rank-selector">', unsafe_allow_html=True)
            st.markdown("#### Rango objetivo")
            if predicted_rank:
                suggested_rank = suggest_target_rank(predicted_rank)
                st.info(f"💡 Sugerencia: Basado en tus estadísticas, te sugerimos apuntar a {suggested_rank}")
            target_rank = st.selectbox(
                "Selecciona tu rango objetivo",
                RANKS,
                key="target_rank",
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Realizar análisis automáticamente cuando se tienen todos los datos
        if selected_player and current_rank and target_rank:
            if st.session_state.get('all_player_stats_group'):
                all_player_stats = st.session_state['all_player_stats_group']
                selected_stats = next((stats for stats in all_player_stats if stats.get('name') == selected_player), None)
                
                if selected_stats:
                    predicted_rank, confidence = rank_predictor.predict_rank(selected_stats)
                    if predicted_rank:
                        st.markdown("### 📊 Análisis de Rendimiento")
                        
                        # Mensaje inicial positivo
                        st.markdown(get_rank_difference_description(current_rank, predicted_rank))
                        
                        # Información del rango en un contenedor más discreto
                        with st.expander("ℹ️ Detalles de la predicción", expanded=True):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Rango predicho:** {predicted_rank}")
                            with col2:
                                st.markdown(f"**Nivel de confianza:** {confidence:.1%}")
                        
                        # Análisis comparativo
                        st.markdown("---")
                        
                        # Manejo especial para Supersonic Legend
                        if predicted_rank == "Supersonic Legend":
                            st.success("""
                                ### 🏆 ¡Increíble! Tienes nivel de Supersonic Legend
                                
                                Tus estadísticas indican que ya estás en el nivel más alto del juego. En este punto, el enfoque debe ser:
                                
                                - **Mantener la consistencia**: Seguir jugando para mantener tu nivel
                                - **Perfeccionar detalles**: Trabajar en aspectos muy específicos
                                - **Competir al más alto nivel**: Participar en torneos y ligas
                                - **Ayudar a otros**: Compartir tu conocimiento con jugadores de menor nivel
                                
                                Aún así, siempre hay margen para la mejora. Aquí tienes un análisis de tus fortalezas y áreas de perfeccionamiento:
                            """)
                            
                            # Mostrar análisis incluso para SSL
                            comparison = rank_predictor.get_rank_comparison(selected_stats)
                            if comparison:
                                show_key_differences_zscore(comparison, predicted_rank)
                            else:
                                st.warning('No se pudo calcular la comparación detallada.')
                                
                        elif target_rank != predicted_rank:
                            # Contexto motivador sobre el objetivo
                            target_idx = get_rank_index(target_rank)
                            predicted_idx = get_rank_index(predicted_rank)
                            diff = target_idx - predicted_idx
                            
                            if diff > 2:
                                st.markdown("""
                                    #### 🎯 Sobre tu objetivo
                                    Has establecido un objetivo ambicioso, ¡eso es genial! Ten en cuenta que alcanzar este objetivo requerirá:
                                    - Dedicación constante
                                    - Mejoras en varios aspectos del juego
                                    - Paciencia y persistencia
                                """)
                            elif diff > 0:
                                st.markdown("""
                                    #### 🎯 Sobre tu objetivo
                                    ¡Has elegido un objetivo realista y alcanzable! Con práctica focalizada en las áreas clave, podrás alcanzarlo.
                                """)
                            elif diff < 0:
                                st.info("""
                                    #### 💡 Sobre tu objetivo
                                    Tu rendimiento actual sugiere que podrías apuntar más alto. ¡Tienes potencial para más!
                                """)
                            
                            comparison = rank_predictor.get_rank_comparison(selected_stats)
                            if comparison:
                                show_key_differences_zscore(comparison, target_rank)
                            else:
                                st.warning('No se pudo calcular la comparación con el rango objetivo.')
                                
                        elif target_rank == predicted_rank and predicted_rank != "Supersonic Legend":
                            st.success("""
                                ### 🌟 ¡Felicidades!
                                Tu rendimiento ya está al nivel de tu objetivo. ¿Por qué no intentas apuntar un poco más alto?
                                Siempre hay espacio para crecer y mejorar.
                            """)
                            
                            # Mostrar análisis también cuando coinciden los rangos
                            comparison = rank_predictor.get_rank_comparison(selected_stats)
                            if comparison:
                                show_key_differences_zscore(comparison, target_rank)
                            else:
                                st.warning('No se pudo calcular la comparación detallada.')
                    else:
                        st.warning(f"No se encontraron estadísticas para el jugador {selected_player}")

    # --- Botón para descargar los datos crudos de la API para debug ---
    if st.session_state.get('processed_replays'):
        raw_json = json.dumps(st.session_state['processed_replays'], indent=2, ensure_ascii=False)
        st.download_button(
            label='Descargar datos crudos de la API (JSON)',
            data=raw_json,
            file_name='replays_api_raw.json',
            mime='application/json'
        )

    st.markdown('</div>', unsafe_allow_html=True)

def display_player_stats(player_stats: Dict):
    """
    Display player statistics in a formatted manner.
    
    Args:
        player_stats (Dict): Dictionary containing player statistics
        
    The display includes:
    - Basic statistics (goals, assists, saves)
    - Advanced metrics (shot accuracy, save ratio)
    - Boost statistics
    - Movement and positioning metrics
    """
    # Implementation of display_player_stats function
    pass

def display_rank_comparison(comparison: Dict[str, Dict[str, float]]):
    """
    Display the comparison between player statistics and rank averages.
    
    Args:
        comparison (Dict[str, Dict[str, float]]): Dictionary containing comparison data
        
    The display includes:
    - Player's value for each statistic
    - Rank's average value
    - Standard deviation
    - Z-score indicating performance relative to rank average
    """
    # Implementation of display_rank_comparison function
    pass

def show_key_differences_zscore(comparison, target_rank):
    """
    Muestra las 3 mejores y 3 peores métricas basadas en z-score en un sistema de tarjetas animadas.
    """
    # Ordenar las características por z-score
    sorted_features = sorted(comparison.items(), key=lambda x: x[1]['z_score'])
    
    # Obtener las 3 peores y 3 mejores métricas
    worst_features = sorted_features[:3]  # Z-scores más negativos
    best_features = sorted_features[-3:]  # Z-scores más positivos

    # Analizar conjuntos de métricas para encontrar el que necesita más mejora
    weakest_set, avg_z_score, weak_metrics = analyze_metric_sets(comparison)

    # Actualizar los estilos CSS
    st.markdown("""
    <style>
    /* Animaciones para el carrusel */
    @keyframes slideInFromRight {
        from { transform: translateX(100%); opacity: 0; }
        to   { transform: translateX(0);    opacity: 1; }
    }
    
    @keyframes slideInFromLeft {
        from { transform: translateX(-100%); opacity: 0; }
        to   { transform: translateX(0);     opacity: 1; }
    }
    
    /* Clases que aplican la animación */
    .slide-in-right {
        animation: slideInFromRight 0.5s ease-out;
    }
    
    .slide-in-left {
        animation: slideInFromLeft 0.5s ease-out;
    }

    /* Estilos para las tarjetas */
    .metric-card {
        background: #1E2A3A;
        border-radius: 15px;
        padding: 2rem;
        height: 500px;
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }

    .non-active-card {
        opacity: 0.4;
    }

    .metric-card:hover {
        transform: scale(1.02);
    }

    .metric-title {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        color: #ffffff;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }

    .metric-values {
        display: flex;
        justify-content: space-between;
        background: rgba(13, 17, 23, 0.6);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    .metric-value-container {
        text-align: left;
    }

    .metric-label {
        font-size: 1rem;
        color: #a0aec0;
        margin-bottom: 0.5rem;
    }

    .metric-number {
        font-size: 2rem;
        font-weight: bold;
    }

    .metric-number.positive {
        color: #2ecc71;
    }

    .metric-number.negative {
        color: #e74c3c;
    }

    .metric-number.reference {
        color: #3498db;
    }

    .metric-explanation {
        flex-grow: 1;
        background: rgba(13, 17, 23, 0.6);
        border-radius: 10px;
        font-size: 1.1rem;
        color: #e0e0e0;
        line-height: 1.6;
        padding: 1.5rem;
    }

    .strengths-card {
        border-left: 3px solid #2ecc71;
    }

    .weaknesses-card {
        border-left: 3px solid #e74c3c;
    }

    .carousel-title {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1.5rem;
        color: #ffffff;
    }

    .carousel-container {
        margin-bottom: 3rem;
    }

    .card-container {
        padding: 0.5rem;
        transition: all 0.3s ease;
    }

    .nav-dots {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-top: 20px;
    }

    .nav-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: rgba(255,255,255,0.3);
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .nav-dot.active {
        background-color: #ffffff;
    }

    .nav-buttons {
        margin-bottom: 1rem;
    }

    /* Estilos para la tarjeta de video tutorial */
    .video-card {
        background: #1E2A3A;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        border-left: 3px solid #3498db;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    .video-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 1rem;
    }
    
    .video-description {
        color: #a0aec0;
        margin-bottom: 1.5rem;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    .video-metrics {
        background: rgba(13, 17, 23, 0.6);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .video-button {
        display: inline-block;
        background: #3498db;
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        text-decoration: none;
        transition: all 0.3s ease;
        margin-top: 1rem;
    }
    
    .video-button:hover {
        background: #2980b9;
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

    # Función para generar las tarjetas HTML
    def generate_metric_cards(features, is_strengths=True):
        cards = []
        card_type = "strengths-card" if is_strengths else "weaknesses-card"
        
        for i, (metric_name, metric_data) in enumerate(features):
            metric_translation = METRIC_TRANSLATIONS.get(metric_name, metric_name.replace('_', ' ').title())
            value_class = "positive" if is_strengths else "negative"
            
            # Obtener la explicación de la métrica
            explanation = get_metric_explanation(metric_name, is_strengths, metric_data['z_score'])
            
            card_html = f"""
            <div class="metric-card {card_type}">
                <div class="metric-title">{metric_translation}</div>
                <div class="metric-values">
                    <div class="metric-value-container">
                        <div class="metric-label">Tu valor</div>
                        <div class="metric-number {value_class}">
                            {metric_data['player_value']:.2f}
                        </div>
                    </div>
                    <div class="metric-value-container">
                        <div class="metric-label">Media {target_rank}</div>
                        <div class="metric-number reference">
                            {metric_data['rank_mean']:.2f}
                        </div>
                    </div>
                </div>
                <div class="metric-explanation">
                    {explanation}
                </div>
            </div>
            """
            cards.append(card_html)
        return cards

    # Función para mostrar conjunto de tarjetas (fortalezas o debilidades)
    def show_metric_cards(features, is_strengths=True):
        """
        Muestra un conjunto de tarjetas métricas con navegación y animaciones.
        """
        carousel_id = "strengths" if is_strengths else "weaknesses"
        title = "🌟 Tus Puntos Fuertes" if is_strengths else "📈 Oportunidades de Mejora"
        
        st.markdown(f"<h3 class='carousel-title'>{title}</h3>", unsafe_allow_html=True)
        
        # Generar las tarjetas
        cards = generate_metric_cards(features, is_strengths)
        
        # Inicializar variables de estado para este carrusel
        if f'{carousel_id}_index' not in st.session_state:
            st.session_state[f'{carousel_id}_index'] = 1  # Índice central (para 3 tarjetas)
        if f'{carousel_id}_direction' not in st.session_state:
            st.session_state[f'{carousel_id}_direction'] = 'right'  # Dirección inicial
        
        # Botones de navegación
        col1, col2, col3 = st.columns([1, 8, 1])
        with col1:
            if st.button('←', key=f"{carousel_id}_left", use_container_width=True):
                st.session_state[f'{carousel_id}_direction'] = 'left'
                st.session_state[f'{carousel_id}_index'] = (st.session_state[f'{carousel_id}_index'] - 1) % len(cards)
                st.rerun()
                
        with col3:
            if st.button('→', key=f"{carousel_id}_right", use_container_width=True):
                st.session_state[f'{carousel_id}_direction'] = 'right'
                st.session_state[f'{carousel_id}_index'] = (st.session_state[f'{carousel_id}_index'] + 1) % len(cards)
                st.rerun()
        
        # Mostrar las tarjetas
        cols = st.columns(3)
        for i in range(3):
            # Calcular qué tarjeta mostrar en cada posición
            card_index = (st.session_state[f'{carousel_id}_index'] - 1 + i) % len(cards)
            
            # Determinar si esta tarjeta es la activa (central)
            is_active = i == 1
            opacity_class = "" if is_active else "non-active-card"
            
            # Determinar la clase de animación según la dirección
            direction = st.session_state[f'{carousel_id}_direction']
            slide_class = 'slide-in-right' if direction == 'right' else 'slide-in-left'
            
            # Mostrar la tarjeta
            with cols[i]:
                st.markdown(f'<div class="card-container {opacity_class} {slide_class}">{cards[card_index]}</div>', unsafe_allow_html=True)
        
        # Indicadores de navegación
        st.markdown('<div class="nav-dots">', unsafe_allow_html=True)
        for i in range(len(cards)):
            is_active = i == st.session_state[f'{carousel_id}_index']
            st.markdown(f'<div class="nav-dot {" active" if is_active else ""}"></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Función para obtener la explicación de la métrica
    def get_metric_explanation(metric_name: str, is_strength: bool, z_score: float) -> str:
        explanations = {
            # Core stats
            'score_per_game': {
                'high': "Excelente contribución general al equipo. Tu alta puntuación refleja una participación activa en todas las facetas del juego.",
                'low': "Podrías aumentar tu impacto en el juego. Intenta involucrarte más en jugadas tanto ofensivas como defensivas."
            },
            'goals_per_game': {
                'high': "Gran capacidad goleadora. Tu habilidad para convertir oportunidades en goles es un gran activo para tu equipo.",
                'low': "Podrías mejorar tu efectividad goleadora. Practica diferentes tipos de tiros y trabaja en tu posicionamiento ofensivo."
            },
            'assists_per_game': {
                'high': "Excelente capacidad de creación de juego. Tus pases y centros generan muchas oportunidades para tu equipo.",
                'low': "Podrías mejorar tu juego de pases. Trabaja en la precisión de los centros y en identificar mejor a tus compañeros."
            },
            'shot_accuracy': {
                'high': "Tu precisión de tiro es excepcional. Mantienes un excelente balance entre tiros y goles, maximizando las oportunidades de anotar.",
                'low': "Tu precisión de tiro podría mejorar. Intenta practicar tiros desde diferentes ángulos y distancias, y considera si estás tomando tiros demasiado arriesgados."
            },
            'save_ratio': {
                'high': "Eres muy efectivo defendiendo la portería. Tu capacidad para realizar paradas es un gran activo para tu equipo.",
                'low': "Tu efectividad en las paradas podría mejorar. Practica el posicionamiento defensivo y anticipa mejor los tiros del oponente."
            },
            
            # Boost management
            'boost_collection_efficiency': {
                'high': "Excelente gestión de boost. Aprovechas muy bien los pads pequeños, lo que te mantiene con recursos constantes.",
                'low': "Podrías mejorar tu recolección de boost. Aprende las rutas de los pads pequeños y optimiza tus movimientos."
            },
            'boost_steal_ratio': {
                'high': "Gran control sobre el boost del campo. Privas efectivamente al equipo rival de recursos mientras mantienes los tuyos.",
                'low': "Podrías mejorar tu control del boost rival. Intenta robar más boost al oponente durante tus rotaciones."
            },
            'boost_denial': {
                'high': "Excelente trabajo negando recursos al rival. Tu control sobre los pads de boost dificulta el juego del oponente.",
                'low': "Podrías mejorar tu control sobre el boost del campo. Considera más el aspecto estratégico de la negación de recursos."
            },
            
            # Movement and mechanics
            'ground_air_ratio': {
                'high': "Excelente balance entre juego aéreo y terrestre. Tu versatilidad te permite adaptarte a diferentes situaciones.",
                'low': "Podrías mejorar tu balance entre juego aéreo y terrestre. Practica las mecánicas que menos domines."
            },
            'speed_management': {
                'high': "Gran gestión de la velocidad. Sabes cuándo acelerar y cuándo ir más lento.",
                'low': "Podrías mejorar tu gestión de velocidad. Trabaja en mantener velocidad supersónica sin desperdiciar boost."
            },
            
            # Positioning
            'offensive_positioning': {
                'high': "Excelente posicionamiento ofensivo. Te mantienes bien colocado para aprovechar oportunidades de ataque.",
                'low': "Podrías mejorar tu posicionamiento ofensivo. Trabaja en mantener más presión y mejor spacing en ataque."
            },
            'defensive_presence': {
                'high': "Gran presencia defensiva. Tu posicionamiento defensivo da seguridad al equipo y dificulta el ataque rival.",
                'low': "Podrías mejorar tu presencia defensiva. Trabaja en el posicionamiento y las rotaciones defensivas."
            },
            'rotation_balance': {
                'high': "Excelentes rotaciones. Mantienes un buen balance entre ataque y defensa, adaptándote a las necesidades del equipo.",
                'low': "Podrías mejorar tus rotaciones. Trabaja en alternar mejor entre posiciones ofensivas y defensivas."
            },
            'ball_control': {
                'high': "Gran control de pelota. Mantienes la posesión efectivamente y creas oportunidades para tu equipo.",
                'low': "Podrías mejorar tu control de pelota. Practica el dribbling y el manejo general del balón."
            },
            
            # Team play
            'demo_efficiency': {
                'high': "Uso efectivo de las demoliciones. Sabes cuándo buscar demos sin comprometer tu posición.",
                'low': "Podrías mejorar tu uso de demoliciones. Busca oportunidades más estratégicas para demos."
            },
            'total_off_actions': {
                'high': "Gran participación ofensiva. Contribuyes consistentemente al ataque del equipo.",
                'low': "Podrías aumentar tu participación ofensiva. Busca más oportunidades para involucrarte en el ataque."
            },
            'total_def_actions': {
                'high': "Excelente contribución defensiva. Tu equipo puede confiar en ti para defender cuando es necesario.",
                'low': "Podrías mejorar tu contribución defensiva. Trabaja en estar más presente en defensa cuando el equipo lo necesite."
            }
        }
        
        # Explicación por defecto si no está en el diccionario
        default_explanation = {
            'high': f"Destacas significativamente en esta métrica. Sigue aprovechando esta fortaleza.",
            'low': f"Hay margen de mejora en esta métrica. Considera dedicar tiempo a desarrollar este aspecto."
        }
        
        metric_dict = explanations.get(metric_name, default_explanation)
        return metric_dict['high'] if is_strength else metric_dict['low']

    # Mostrar los carruseles
    show_metric_cards(best_features, is_strengths=True)
    show_metric_cards(worst_features, is_strengths=False)
    
    # Mostrar recomendación de video tutorial si se encontró un conjunto débil
    if weakest_set and avg_z_score < -0.5:  # Solo mostrar si el z-score promedio es menor a -0.5
        set_info = METRIC_SETS[weakest_set]
        weak_metrics_translated = [METRIC_TRANSLATIONS[m] for m in weak_metrics if m in METRIC_TRANSLATIONS]
        
        st.markdown("""
        <div class="video-card">
            <div class="video-title">🎓 Video Tutorial Recomendado</div>
            <div class="video-description">
                Hemos detectado que el área que más podrías mejorar es <strong>{}</strong>. 
                Específicamente, podrías trabajar en:
            </div>
            <div class="video-metrics">
                <ul>
                    {}
                </ul>
            </div>
            <div class="video-description">
                Te recomendamos ver este video tutorial que te ayudará a mejorar en estos aspectos:
            </div>
            <a href="{}" target="_blank" class="video-button">
                👉 Ver "{}"
            </a>
        </div>
        """.format(
            weakest_set,
            ''.join(f'<li>{metric}</li>' for metric in weak_metrics_translated),
            set_info['video_url'],
            set_info['video_title']
        ), unsafe_allow_html=True)

def analyze_metric_sets(comparison: Dict[str, Dict[str, float]]) -> Tuple[str, float, List[str]]:
    """
    Analiza los conjuntos de métricas para encontrar el que necesita más mejora.
    
    Args:
        comparison (Dict[str, Dict[str, float]]): Diccionario con las comparaciones de métricas
        
    Returns:
        Tuple[str, float, List[str]]: (Nombre del conjunto más débil, Z-score promedio, Lista de métricas específicas a mejorar)
    """
    set_scores = {}
    set_weak_metrics = defaultdict(list)
    
    for set_name, set_info in METRIC_SETS.items():
        # Calcular z-score promedio para las métricas disponibles en este conjunto
        valid_metrics = [m for m in set_info['metrics'] if m in comparison]
        if not valid_metrics:
            continue
            
        z_scores = [comparison[m]['z_score'] for m in valid_metrics]
        avg_z_score = sum(z_scores) / len(z_scores)
        set_scores[set_name] = avg_z_score
        
        # Identificar métricas específicas que necesitan mejora
        weak_metrics = [m for m in valid_metrics if comparison[m]['z_score'] < -1.0]
        set_weak_metrics[set_name].extend(weak_metrics)
    
    # Encontrar el conjunto con el z-score más bajo
    if not set_scores:
        return None, 0, []
        
    weakest_set = min(set_scores.items(), key=lambda x: x[1])
    return weakest_set[0], weakest_set[1], set_weak_metrics[weakest_set[0]]

def average_stats_nested(all_player_stats):
    avg_stats = defaultdict(lambda: defaultdict(float))
    count_stats = defaultdict(lambda: defaultdict(int))
    for stats in all_player_stats:
        for category, values in stats.items():
            for stat, value in values.items():
                if isinstance(value, (int, float)):
                    stat_key = API_TO_MODEL_STATS.get(stat, stat)
                    avg_stats[category][stat_key] += value
                    count_stats[category][stat_key] += 1
    # Calcular promedios solo donde hay datos
    for category in avg_stats:
        for stat in avg_stats[category]:
            if count_stats[category][stat] > 0:
                avg_stats[category][stat] /= count_stats[category][stat]
            else:
                avg_stats[category][stat] = 0
    # Convertir defaultdict a dict normal
    return {cat: dict(stats) for cat, stats in avg_stats.items()}

def handle_team_analysis_mode():
    """
    Maneja el modo de análisis de equipos.
    """
    # Gestión de la subida de archivos para equipos
    if not st.session_state['is_processing'] and not st.session_state['show_player_selection']:
        uploaded_files = st.file_uploader(
            "📁 Selecciona archivos de repetición del equipo",
            accept_multiple_files=True,
            type=['replay'],
            key='team_file_uploader',
            on_change=handle_team_file_upload
        )
        
        # Visualización de archivos subidos
        if 'uploaded_files' in st.session_state and st.session_state['uploaded_files']:
            st.markdown('<div class="file-list">', unsafe_allow_html=True)
            st.markdown("### 📋 Archivos seleccionados:")
            for file in st.session_state['uploaded_files']:
                st.markdown(f'<div class="file-item">📄 {file.name}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.button("🔄 Analizar Repeticiones del Equipo", type="primary", use_container_width=True, 
                     on_click=lambda: setattr(st.session_state, 'start_processing', True) or setattr(st.session_state, 'is_processing', True))

    # Procesamiento de replays
    if st.session_state['start_processing']:
        process_replays()
        st.session_state['start_processing'] = False

    # Selección de jugadores para análisis de equipo (solo en modo equipos)
    if st.session_state['show_player_selection'] and st.session_state.get('team_mode', False):
        st.markdown("### 👥 Selección de Jugadores para Análisis de Equipo")
        st.markdown('<p class="subtitle">Selecciona los jugadores que quieres analizar</p>', unsafe_allow_html=True)
        
        # Selector de modo de formación de equipo
        st.markdown("#### 🎯 Modo de Análisis")
        team_mode = st.radio(
            "Selecciona el tipo de análisis:",
            ["🔍 Encontrar mejor trío", "👥 Encontrar tercer jugador para dúo"],
            key='team_formation_mode_radio'
        )
        
        player_names = sorted(list(st.session_state['player_names']))
        
        if "mejor trío" in team_mode:
            st.session_state['team_formation_mode'] = 'best_trio'
            
            st.markdown("#### 👥 Jugadores Candidatos")
            selected_players = st.multiselect(
                "Selecciona al menos 3 jugadores para encontrar la mejor combinación:",
                player_names,
                key='trio_player_selector'
            )
            
            if len(selected_players) >= 3:
                if st.button("🚀 Analizar Mejores Tríos", type="primary", use_container_width=True):
                    # Obtener estadísticas de los jugadores seleccionados
                    all_player_stats = st.session_state.get('all_player_stats_group', [])
                    selected_stats = []
                    
                    for player_name in selected_players:
                        player_stats = next((stats for stats in all_player_stats if stats.get('name') == player_name), None)
                        if player_stats:
                            selected_stats.append(player_stats)
                    
                    if len(selected_stats) >= 3:
                        show_team_analysis_results(selected_stats, 'best_trio')
                    else:
                        st.error("No se pudieron obtener las estadísticas de todos los jugadores seleccionados.")
            else:
                st.info("Selecciona al menos 3 jugadores para continuar.")
                
        else:  # Modo encontrar tercer jugador
            st.session_state['team_formation_mode'] = 'find_third'
            
            # Selección del dúo
            st.markdown("#### 🤝 Selecciona tu Dúo Base")
            duo_players = st.multiselect(
                "Selecciona exactamente 2 jugadores que forman el dúo:",
                player_names,
                max_selections=2,
                key='duo_player_selector'
            )
            
            if len(duo_players) == 2:
                # Mostrar información del dúo seleccionado
                with st.expander("👥 Información del Dúo Seleccionado", expanded=True):
                    col1, col2 = st.columns(2)
                    all_player_stats = st.session_state.get('all_player_stats_group', [])
                    
                    for i, player_name in enumerate(duo_players):
                        player_stats = next((stats for stats in all_player_stats if stats.get('name') == player_name), None)
                        if player_stats:
                            with [col1, col2][i]:
                                st.markdown(f"**{player_name}**")
                                st.metric("Goles/partido", f"{player_stats.get('goals_per_game', 0):.2f}")
                                st.metric("Asistencias/partido", f"{player_stats.get('assists_per_game', 0):.2f}")
                                st.metric("Paradas/partido", f"{player_stats.get('saves_per_game', 0):.2f}")
                
                # Selección de candidatos para tercer jugador (excluyendo el dúo)
                available_candidates = [name for name in player_names if name not in duo_players]
                
                st.markdown("#### 🎯 Candidatos para Tercer Jugador")
                third_candidates = st.multiselect(
                    "Selecciona los candidatos a evaluar para el tercer puesto:",
                    available_candidates,
                    key='third_player_selector'
                )
                
                if len(third_candidates) >= 1:
                    if st.button("🚀 Analizar Candidatos para Tercer Jugador", type="primary", use_container_width=True):
                        # Combinar dúo + candidatos
                        all_selected = duo_players + third_candidates
                        
                        # Obtener estadísticas
                        selected_stats = []
                        for player_name in all_selected:
                            player_stats = next((stats for stats in all_player_stats if stats.get('name') == player_name), None)
                            if player_stats:
                                selected_stats.append(player_stats)
                        
                        if len(selected_stats) >= 3:
                            show_team_analysis_results(selected_stats, 'find_third')
                        else:
                            st.error("No se pudieron obtener las estadísticas de todos los jugadores seleccionados.")
                else:
                    st.info("Selecciona al menos 1 candidato para el tercer puesto.")
            elif len(duo_players) > 0:
                st.warning("Selecciona exactamente 2 jugadores para formar el dúo base.")
            else:
                st.info("Selecciona 2 jugadores para formar el dúo base.")

def handle_team_file_upload():
    """
    Maneja la subida de archivos para el modo de equipos.
    """
    if st.session_state['team_file_uploader'] is not None:
        st.session_state['uploaded_files'] = st.session_state['team_file_uploader']
        st.session_state['show_file_list'] = True
        st.session_state['player_names'] = set()

def show_team_analysis_results(selected_stats: List[Dict], mode: str):
    """
    Muestra los resultados del análisis de equipos.
    
    Args:
        selected_stats (List[Dict]): Estadísticas de los jugadores seleccionados
        mode (str): Modo de análisis ('best_trio' o 'find_third')
    """
    st.markdown("---")
    st.markdown("## 📊 Resultados del Análisis de Equipos")
    
    # Obtener insights del Birmingham Major
    insights = team_analyzer.get_team_analysis_insights()
    
    # Mostrar patrones de equipos exitosos
    with st.expander("🏆 Patrones de Equipos Exitosos (Birmingham Major)", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ✅ Equipos Top 4")
            successful = insights['successful_team_patterns']
            st.metric("Goles promedio por jugador", f"{successful['avg_goals_per_player']:.2f}")
            st.metric("Asistencias promedio por jugador", f"{successful['avg_assists_per_player']:.2f}")
            st.metric("Paradas promedio por jugador", f"{successful['avg_saves_per_player']:.2f}")
            st.metric("Puntuación promedio por jugador", f"{successful['avg_score_per_player']:.0f}")
        
        with col2:
            st.markdown("### ❌ Equipos Bottom 8")
            unsuccessful = insights['unsuccessful_team_patterns']
            st.metric("Goles promedio por jugador", f"{unsuccessful['avg_goals_per_player']:.2f}")
            st.metric("Asistencias promedio por jugador", f"{unsuccessful['avg_assists_per_player']:.2f}")
            st.metric("Paradas promedio por jugador", f"{unsuccessful['avg_saves_per_player']:.2f}")
            st.metric("Puntuación promedio por jugador", f"{unsuccessful['avg_score_per_player']:.0f}")
    
    # Realizar análisis según el modo
    if mode == 'best_trio':
        st.markdown("### 🔍 Mejores Combinaciones de Trío")
        results = team_analyzer.find_optimal_team_from_players(selected_stats, mode="best_trio")
        
        if results:
            for i, team in enumerate(results):
                with st.expander(f"🥇 Opción {i+1} - Puntuación: {team['combined_score']:.3f}", expanded=(i==0)):
                    col1, col2, col3 = st.columns(3)
                    
                    # Mostrar información de cada jugador
                    for j, player_info in enumerate(team['players']):
                        with [col1, col2, col3][j]:
                            st.markdown(f"**{player_info['name']}**")
                            st.markdown(f"*Rol: {team['roles'][j]}*")
                            stats = player_info['stats']
                            st.metric("Goles/partido", f"{stats.get('goals_per_game', 0):.2f}")
                            st.metric("Asistencias/partido", f"{stats.get('assists_per_game', 0):.2f}")
                            st.metric("Paradas/partido", f"{stats.get('saves_per_game', 0):.2f}")
                    
                    # Métricas del equipo
                    st.markdown("#### 📈 Métricas del Equipo")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sinergia", f"{team['synergy_score']:.3f}")
                    with col2:
                        st.metric("Habilidad", f"{team['skill_score']:.3f}")
                    with col3:
                        st.metric("Puntuación Total", f"{team['combined_score']:.3f}")
        else:
            st.warning("No se pudieron generar combinaciones de equipo con los jugadores seleccionados.")
    
    elif mode == 'find_third':
        if len(selected_stats) < 3:
            st.error("Necesitas al menos 3 jugadores: 2 para el dúo y candidatos para el tercer puesto.")
            return
            
        st.markdown("### 👥 Mejores Terceros Jugadores para tu Dúo")
        
        # Mostrar información del dúo
        duo = selected_stats[:2]
        st.markdown("#### 🤝 Tu Dúo Actual")
        col1, col2 = st.columns(2)
        
        for i, player in enumerate(duo):
            with [col1, col2][i]:
                st.markdown(f"**{player.get('name', 'Unknown')}**")
                st.metric("Goles/partido", f"{player.get('goals_per_game', 0):.2f}")
                st.metric("Asistencias/partido", f"{player.get('assists_per_game', 0):.2f}")
                st.metric("Paradas/partido", f"{player.get('saves_per_game', 0):.2f}")
        
        # Encontrar mejores terceros jugadores
        results = team_analyzer.find_optimal_team_from_players(selected_stats, mode="find_third")
        
        if results:
            st.markdown("#### 🎯 Candidatos para Tercer Jugador")
            for i, candidate in enumerate(results):
                with st.expander(f"🥇 Opción {i+1} - {candidate['player']['name']} - Puntuación: {candidate['combined_score']:.3f}", expanded=(i==0)):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Estadísticas del Candidato**")
                        stats = candidate['player']['stats']
                        st.metric("Goles/partido", f"{stats.get('goals_per_game', 0):.2f}")
                        st.metric("Asistencias/partido", f"{stats.get('assists_per_game', 0):.2f}")
                        st.metric("Paradas/partido", f"{stats.get('saves_per_game', 0):.2f}")
                        st.metric("Puntuación/partido", f"{stats.get('score_per_game', 0):.0f}")
                    
                    with col2:
                        st.markdown("**Análisis de Compatibilidad**")
                        st.metric("Sinergia del Equipo", f"{candidate['synergy_score']:.3f}")
                        st.metric("Habilidad Individual", f"{candidate['skill_score']:.3f}")
                        st.metric("Complemento de Roles", f"{candidate['role_completion_bonus']:.3f}")
                        
                        st.markdown("**Roles del Equipo Completo:**")
                        for role in candidate['team_roles']:
                            st.markdown(f"• {role}")
        else:
            st.warning("No se pudieron encontrar candidatos adecuados para el tercer puesto.")

if __name__ == "__main__":
    main() 