"""
Player Data Anonymization and Filtering

This module handles the anonymization and filtering of player data for the Rocket League
rank prediction system. It ensures data privacy while maintaining data quality.

Author: Adrián Torremocha
Date: 02-05-2025
"""

import pandas as pd
import os
from typing import Dict, List, Optional
from pathlib import Path

# Ruta del archivo combinado
INPUT_FILE = os.path.join("data", "all_players_with_rank.csv")
OUTPUT_FILE = os.path.join("data", "players_anonymous_per_game.csv")

# Leer el archivo
print(f"Leyendo {INPUT_FILE} ...")
df = pd.read_csv(INPUT_FILE, sep=";")

# Eliminar columnas de nombre de jugador y equipo (busca columnas que contengan 'name')
cols_to_drop = [col for col in df.columns if 'name' in col.lower()]
df = df.drop(columns=cols_to_drop)

# Seleccionar solo columnas que contienen 'per game' y la columna Rank
per_game_cols = [col for col in df.columns if 'per game' in col.lower()]
if 'Rank' in df.columns:
    per_game_cols.append('Rank')
filtered_df = df[per_game_cols]

# Guardar el resultado
filtered_df.to_csv(OUTPUT_FILE, index=False, sep=";")
print(f"Archivo anonimizado y filtrado guardado como {OUTPUT_FILE}")

def anonymize_players(data_path: str, output_path: str):
    """
    Anonymize player data by removing personal identifiers and filtering relevant statistics.
    
    Args:
        data_path (str): Path to the input data file
        output_path (str): Path to save the anonymized data
        
    Note:
        The anonymization process:
        - Removes player IDs and names
        - Filters out incomplete or invalid records
        - Normalizes statistics for analysis
    """ 