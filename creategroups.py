"""
Replay Group Management

This module handles the creation and management of replay groups in the Ballchasing API.
It provides functionality to organize and categorize replays for analysis.

Author: Adrián Torremocha
Date: 02-05-2025
"""

import requests
import time
import math
from datetime import datetime, timezone
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from typing import Dict, List, Optional
from pathlib import Path

# --- CONFIGURACIÓN ---
BALLCHASING_API_TOKEN = "JR1R8xOTPOw7grXWLw1VUDHthdQM9gAn5j5wqsYr" 
BASE_URL = "https://ballchasing.com/api"
REPLAYS_PER_GROUP = 750
REPLAYS_PER_REQUEST = 200  # Máximo permitido por la API para GET /replays
START_DATE_STR = "2024-12-01T00:00:00Z" # Formato RFC3339 UTC

# Definición de rangos (ajusta según los valores exactos de la API si es necesario)
# Los valores de min/max rank deben coincidir con los 'Choices' en la documentación
RANK_RANGES = {
    "Bronze": {"min": "bronze-1", "max": "bronze-3"},
    "Silver": {"min": "silver-1", "max": "silver-3"},
    "Gold": {"min": "gold-1", "max": "gold-3"},
    "Platinum": {"min": "platinum-1", "max": "platinum-3"},
    "Diamond": {"min": "diamond-1", "max": "diamond-3"},
    "Champion": {"min": "champion-1", "max": "champion-3"},
    "Grand Champion": {"min": "grand-champion-1", "max": "grand-champion-3"},
    "Supersonic Legend": {"min": "supersonic-legend", "max": "supersonic-legend"}

}

HEADERS = {
    "Authorization": BALLCHASING_API_TOKEN
}

# --- FUNCIONES AUXILIARES ---

def make_api_request(method, endpoint_or_url, params=None, json_data=None, retries=3, delay=5):
    """Función genérica para hacer llamadas a la API con manejo básico de errores y reintentos."""
    # --- INICIO DE CORRECCIÓN ---
    # Determinar la URL final correctamente
    if isinstance(endpoint_or_url, str) and endpoint_or_url.startswith('http'):
        url = endpoint_or_url # Ya es una URL completa
    elif isinstance(endpoint_or_url, str):
        # Asegurarse de que el endpoint relativo empiece con / si no lo hace
        safe_endpoint = endpoint_or_url
        if not endpoint_or_url.startswith('/'):
            safe_endpoint = '/' + endpoint_or_url
        url = f"{BASE_URL}{safe_endpoint}" # Es un endpoint relativo, añadir BASE_URL
    else:
         # Manejar caso si no es un string (defensivo)
         print(f"ERROR: make_api_request recibió un endpoint inválido: {endpoint_or_url}")
         return None
    # --- FIN DE CORRECCIÓN ---

    # print(f"DEBUG: Requesting URL: {url} with params: {params}") # Descomenta para depurar URLs

    for attempt in range(retries):
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=HEADERS, params=params)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=HEADERS, json=json_data)
            elif method.upper() == 'PATCH':
                response = requests.patch(url, headers=HEADERS, json=json_data)
            else:
                raise ValueError(f"Método HTTP no soportado: {method}")

            if response.status_code == 429:
                print(f"WARN: Límite de tasa alcanzado (429). Esperando {delay * (attempt + 1)} segundos...")
                time.sleep(delay * (attempt + 1))
                continue # Reintenta

            # Incluir la URL en el mensaje de error para mejor diagnóstico
            response.raise_for_status()

            if response.status_code in [200, 201]:
                if response.content:
                    try:
                        return response.json()
                    except requests.exceptions.JSONDecodeError:
                        print(f"WARN: Respuesta exitosa ({response.status_code}) pero no es JSON válido para {method} {url}")
                        return None
                else:
                     return None
            elif response.status_code == 204:
                return None
            else:
                 print(f"WARN: Código de estado inesperado {response.status_code} para {method} {url}")
                 return None

        except requests.exceptions.RequestException as e:
            # Incluir la URL final en el mensaje de error
            print(f"ERROR: Fallo en la solicitud a {method} {url}: {e}")
            if attempt < retries - 1:
                print(f"Reintentando en {delay} segundos...")
                time.sleep(delay)
            else:
                print("ERROR: Máximo de reintentos alcanzado.")
                return None
    return None


def fetch_replays(rank_min, rank_max, date_after):
    """Obtiene todos los IDs de replays para un rango y fecha dados, corrigiendo la URL 'next'."""
    all_replay_ids = []
    next_page_indicator = "/replays" # Endpoint relativo inicial
    params = {
        "playlist": "ranked-doubles",
        "min-rank": rank_min,
        "max-rank": rank_max, # Asegúrate que 'rank_max' tiene valor aquí
        "replay-date-after": date_after,
        "count": REPLAYS_PER_REQUEST,
        "sort-by": "replay-date",
        "sort-dir": "asc",
    }

    page_count = 0
    while next_page_indicator:
        page_count += 1
        
        # --- INICIO DE CORRECCIÓN PAGINACIÓN ---
        # Para la primera página, usa el endpoint relativo y los params originales
        if page_count == 1:
            current_target = next_page_indicator
            current_params = params
            print(f"  Fetching page {page_count} from {current_target} with params {current_params}...")
        # Para páginas siguientes, usa la URL completa que ya debería tener params
        else:
            current_target = next_page_indicator
            current_params = None # Los params están en la URL
            print(f"  Fetching page {page_count} from {current_target}...")
        # --- FIN DE CORRECCIÓN PAGINACIÓN ---

        response_data = make_api_request('GET', current_target, params=current_params) # Pasa params solo la primera vez

        if not response_data or 'list' not in response_data:
            print(f"WARN: No se pudo obtener la lista de replays o la respuesta está vacía/malformada en la página {page_count}.")
            break

        replays_in_page = response_data.get('list', [])
        if not replays_in_page and page_count >= 1:
             if page_count == 1:
                 print(f"  No se encontraron replays iniciales para {rank_min}-{rank_max}.")
             else:
                 print("  No more replays found in subsequent pages.")
             break

        for replay in replays_in_page:
            all_replay_ids.append(replay['id'])

        print(f"  Found {len(replays_in_page)} replays this page. Total so far: {len(all_replay_ids)}")

        # --- INICIO WORKAROUND PARA max-rank FALTANTE ---
        next_page_url = response_data.get('next')

        if next_page_url:
            # Parsear la URL 'next' proporcionada por la API
            parsed_url = urlparse(next_page_url)
            query_params_dict = parse_qs(parsed_url.query) # Devuelve dict donde los valores son listas

            # Verificar si 'max-rank' falta en los parámetros de la query
            if 'max-rank' not in query_params_dict:
                # ¡Añadirlo manualmente usando el valor original!
                print(f"  WARN: 'max-rank' ({rank_max}) missing in API's 'next' URL. Re-inserting.")
                query_params_dict['max-rank'] = [rank_max] # parse_qs espera listas como valores

                # Reconstruir la query string
                new_query_string = urlencode(query_params_dict, doseq=True)

                # Reconstruir la URL completa con la query corregida
                # Usar _replace ya que parsed_url es una tupla nombrada inmutable
                corrected_parsed_url = parsed_url._replace(query=new_query_string)
                next_page_indicator = urlunparse(corrected_parsed_url)
                # print(f"  DEBUG: Corrected next URL: {next_page_indicator}") # Descomenta para depurar
            else:
                # Si max-rank sí estaba, usar la URL 'next' tal cual
                next_page_indicator = next_page_url
        else:
            # Si no hay 'next' URL, hemos terminado
            next_page_indicator = None
        # --- FIN WORKAROUND ---

        time.sleep(0.5) # Pausa entre peticiones GET

    print(f"Total replays fetched for {rank_min}-{rank_max}: {len(all_replay_ids)}")
    return all_replay_ids


def create_group(name):
    """Crea un nuevo grupo en Ballchasing."""
    print(f"Creating group: {name}")
    endpoint = "/groups"
    payload = {
        "name": name,
        "player_identification": "by-id",        # O 'by-name'
        "team_identification": "by-player-clusters" # O 'by-distinct-players'
    }
    response_data = make_api_request('POST', endpoint, json_data=payload)
    if response_data and 'id' in response_data:
        print(f"  Group created successfully with ID: {response_data['id']}")
        return response_data['id']
    else:
        print(f"ERROR: Failed to create group '{name}'. Response: {response_data}")
        return None


def assign_replay_to_group(replay_id, group_id):
    """Asigna un replay específico a un grupo."""
    endpoint = f"/replays/{replay_id}"
    payload = {"group": group_id}
    # No esperamos contenido en la respuesta (204 No Content)
    make_api_request('PATCH', endpoint, json_data=payload)
    # Añadir una pequeña pausa aquí es CRUCIAL para evitar el rate limit del PATCH
    time.sleep(0.6) # ¡AJUSTA ESTE VALOR! Empieza conservador (más alto)


# --- LÓGICA PRINCIPAL ---

if BALLCHASING_API_TOKEN == "TU_API_TOKEN_AQUI":
    print("ERROR: Por favor, reemplaza 'TU_API_TOKEN_AQUI' con tu token real de la API de Ballchasing.")
else:
    start_time = time.time()
    print(f"--- Iniciando proceso para replays desde {START_DATE_STR} ---")

    for rank_name, ranks in RANK_RANGES.items():
        print(f"\n--- Procesando Rango: {rank_name} ({ranks['min']} a {ranks['max']}) ---")

        # 1. Obtener todos los replays para el rango y fecha
        replay_ids = fetch_replays(ranks['min'], ranks['max'], START_DATE_STR)

        if not replay_ids:
            print(f"No se encontraron replays para {rank_name}. Pasando al siguiente rango.")
            continue

        # 2. Calcular cuántos grupos se necesitan
        num_groups_needed = math.ceil(len(replay_ids) / REPLAYS_PER_GROUP)
        print(f"Se necesitan {num_groups_needed} grupo(s) para {len(replay_ids)} replays de {rank_name}.")

        # 3. Crear grupos y asignar replays
        for i in range(num_groups_needed):
            group_part = i + 1
            group_name = f"{rank_name} 2s Replays ({START_DATE_STR[:10]}) Part {group_part}"

            # Crear el grupo
            new_group_id = create_group(group_name)
            if not new_group_id:
                print(f"ERROR: No se pudo crear el grupo {group_name}. Abortando para este rango.")
                break # O podrías intentar continuar con el siguiente grupo

            # Determinar qué replays van en este grupo
            start_index = i * REPLAYS_PER_GROUP
            end_index = start_index + REPLAYS_PER_GROUP
            replays_for_this_group = replay_ids[start_index:end_index]

            print(f"  Asignando {len(replays_for_this_group)} replays al grupo '{group_name}' (ID: {new_group_id})...")

            # Asignar cada replay al grupo (¡puede tardar!)
            assigned_count = 0
            for replay_id in replays_for_this_group:
                assign_replay_to_group(replay_id, new_group_id)
                assigned_count += 1
                if assigned_count % 50 == 0: # Imprimir progreso cada 50 asignaciones
                     print(f"    {assigned_count}/{len(replays_for_this_group)} replays asignados...")
            
            print(f"  ¡Asignación completada para el grupo {group_part} de {rank_name}!")
            # Pausa adicional entre la creación de grupos/asignaciones masivas
            time.sleep(2) 

    end_time = time.time()
    print(f"\n--- Proceso completado en {end_time - start_time:.2f} segundos ---")

def create_replay_group(api_key: str, group_name: str) -> Optional[str]:
    """
    Create a new replay group in the Ballchasing API.
    
    Args:
        api_key (str): Ballchasing API key
        group_name (str): Name for the new group
        
    Returns:
        Optional[str]: ID of the created group or None if creation failed
        
    Note:
        The group creation process:
        - Validates API credentials
        - Creates a new group with specified name
        - Returns the group ID for future reference
    """
    pass

def add_replay_to_group(api_key: str, group_id: str, replay_id: str) -> bool:
    """
    Add a replay to an existing group.
    
    Args:
        api_key (str): Ballchasing API key
        group_id (str): ID of the target group
        replay_id (str): ID of the replay to add
        
    Returns:
        bool: True if the replay was added successfully, False otherwise
    """
    pass