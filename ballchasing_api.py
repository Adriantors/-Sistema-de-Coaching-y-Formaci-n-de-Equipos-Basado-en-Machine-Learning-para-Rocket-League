"""
Ballchasing API Module

This module provides functionality to interact with the Ballchasing API for Rocket League replay analysis.
It handles API requests, data processing, and custom metric calculations.

Author: Adrián Torremocha
Date: 02-05-2025
"""

import requests
import os
from datetime import datetime
import time
from typing import List, Dict, Optional, Union
from pathlib import Path

class BallchasingAPI:
    """
    A class for interacting with the Ballchasing API.
    
    This class handles:
    - API authentication and requests
    - Replay upload and processing
    - Data retrieval and processing
    - Custom metric calculations
    """
    
    BASE_URL = "https://ballchasing.com/api"
    
    def __init__(self, api_key: str):
        """
        Initialize the BallchasingAPI with an API key.
        
        Args:
            api_key (str): Ballchasing API key for authentication
        """
        self.api_key = api_key
        self.headers = {"Authorization": api_key}
        
    def check_auth(self) -> bool:
        """Verificar si la API key es válida."""
        try:
            response = requests.get(f"{self.BASE_URL}/", headers=self.headers)
            return response.status_code == 200
        except:
            return False
    
    def create_group(self, name: str, player_identification: str = "by-id", 
                    team_identification: str = "by-distinct-players") -> Optional[str]:
        """
        Crear un nuevo grupo para las replays.
        
        Args:
            name: Nombre del grupo
            player_identification: Método de identificación de jugadores ('by-id' o 'by-name')
            team_identification: Método de identificación de equipos
            
        Returns:
            Optional[str]: ID del grupo creado o None si falla
        """
        url = f"{self.BASE_URL}/groups"
        data = {
            "name": name,
            "player_identification": player_identification,
            "team_identification": team_identification
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=data)
            if response.status_code == 201:
                return response.json().get("id")
            print(f"Error creating group: {response.status_code} - {response.text}")
            return None
        except Exception as e:
            print(f"Exception creating group: {str(e)}")
            return None
    
    def upload_replay(self, replay_path: str, group_id: Optional[str] = None) -> Dict:
        """
        Upload a replay file to Ballchasing for analysis.
        
        Args:
            replay_path (str): Path to the replay file
            group_id (Optional[str]): ID of the group to add the replay to
            
        Returns:
            Dict: Response from the API containing replay ID and status
            
        Raises:
            Exception: If the upload fails
        """
        url = f"{self.BASE_URL}/v2/upload"
        params = {"visibility": "public"}
        if group_id:
            params["group"] = group_id
        
        try:
            with open(replay_path, 'rb') as file:
                files = {'file': file}
                response = requests.post(url, headers=self.headers, 
                                      params=params, files=files)
                
                if response.status_code in [201, 409]:  # Created o Duplicate
                    return response.json()
                print(f"Error uploading replay: {response.status_code} - {response.text}")
                raise Exception(f"Upload failed: {response.text}")
        except Exception as e:
            print(f"Exception uploading replay: {str(e)}")
            raise
    
    def get_group_info(self, group_id: str) -> Optional[Dict]:
        """
        Obtener la información de un grupo específico.
        
        Args:
            group_id: ID del grupo a consultar
            
        Returns:
            Optional[Dict]: Información del grupo o None si falla
            
        Note:
            La respuesta incluye:
            - status: Estado del procesamiento del grupo
            - players: Lista de jugadores con sus estadísticas
            - name: Nombre del grupo
            - created: Fecha de creación
        """
        url = f"{self.BASE_URL}/groups/{group_id}"
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                return response.json()
            print(f"Error getting group info: {response.status_code} - {response.text}")
            return None
        except Exception as e:
            print(f"Exception getting group info: {str(e)}")
            return None

    # Eliminar métodos relacionados con replays individuales:
    # - get_replay_status
    # - get_replay_data
    # - wait_for_replay_processing
    # - process_replay_stats (y la función fuera de la clase)
    #
    # Mantener solo la lógica de grupos y subida de replays. 