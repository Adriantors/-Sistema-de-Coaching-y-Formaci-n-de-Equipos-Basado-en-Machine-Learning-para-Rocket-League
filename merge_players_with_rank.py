"""
Player Data and Rank Merging

This module merges player statistics with their corresponding rank information
to create a comprehensive dataset for the rank prediction system.

Author: Adrián Torremocha
Date: 02-05-2025
"""

import os
import pandas as pd
import re
from typing import Dict, List, Optional
from pathlib import Path

# Carpeta donde están los archivos
DATA_FOLDER = "data"

# Expresión regular para extraer el rango del nombre del archivo
RANK_REGEX = re.compile(r"^(.*?)\-2s\-replays.*\-players\.csv$")

# Lista para guardar los DataFrames
dfs = []

for filename in os.listdir(DATA_FOLDER):
    if filename.endswith("-players.csv"):
        match = RANK_REGEX.match(filename)
        if match:
            rank = match.group(1).replace("-", " ").title()
        else:
            rank = "Unknown"
        file_path = os.path.join(DATA_FOLDER, filename)
        df = pd.read_csv(file_path, sep=";")
        df["Rank"] = rank
        dfs.append(df)

# Unir todos los DataFrames
if dfs:
    all_data = pd.concat(dfs, ignore_index=True)
    # Guardar el resultado
    all_data.to_csv(os.path.join(DATA_FOLDER, "all_players_with_rank.csv"), index=False, sep=";")
    print("Archivo combinado guardado como all_players_with_rank.csv")
else:
    print("No se encontraron archivos -players.csv en la carpeta data.")

def merge_player_rank_data(players_path: str, ranks_path: str, output_path: str):
    """
    Merge player statistics with rank information.
    
    Args:
        players_path (str): Path to the player statistics file
        ranks_path (str): Path to the rank information file
        output_path (str): Path to save the merged data
        
    Note:
        The merging process:
        - Matches players with their ranks
        - Handles missing or inconsistent data
        - Ensures data integrity
    """ 