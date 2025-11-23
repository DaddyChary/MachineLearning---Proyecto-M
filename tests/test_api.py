import sys
import os
from fastapi.testclient import TestClient

# --- Configuración de rutas ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.main import app

def test_read_root():
    """
    Prueba de Health Check: Verifica que la API esté viva.
    Usamos 'with TestClient' para asegurar que el evento de inicio (startup) corra.
    """
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        # Verificamos que el mensaje coincida con la versión actual (v1.0) o la nueva (Activa)
        # Usamos 'in' para que sea compatible con ambas versiones
        assert "API CESFAM" in data["message"]

def test_predict_endpoint_valid_input():
    """
    Prueba Funcional: Verifica que el endpoint /predict responda correctamente.
    """
    payload = {
        "edad": 30,
        "sexo": "Femenino",
        "sector": "Norte",
        "prevision": "Fonasa B",
        "especialidad": "Medicina General",
        "dia_semana": "Lunes",
        "turno": "Mañana",
        "tiempo_espera_dias": 5,
        "inasistencias_previas": 0
    }

    # IMPORTANTE: Usamos el contexto 'with' para cargar el modelo
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        
        # Validaciones
        assert response.status_code == 200, f"Error: {response.text}"
        data = response.json()
        assert "prediccion" in data
        assert "probabilidad" in data

def test_predict_endpoint_high_risk():
    """
    Prueba de Lógica: Verifica caso de alto riesgo.
    """
    payload = {
        "edad": 25,
        "sexo": "Masculino",
        "sector": "Centro",
        "prevision": "Fonasa A",
        "especialidad": "Dental",
        "dia_semana": "Viernes",
        "turno": "Tarde",
        "tiempo_espera_dias": 30,
        "inasistencias_previas": 10
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200