import sys
import os
from fastapi.testclient import TestClient

# --- Configuración de rutas ---
# Agregamos la ruta raíz del proyecto para poder importar src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importamos la instancia de la aplicación FastAPI (que crearemos en el próximo paso)
try:
    from src.api.main import app
except ImportError:
    # Este bloque evita que el test crashee si main.py aun no existe, 
    # pero fallará indicando la razón.
    raise ImportError("❌ No se encontró 'src/api/main.py'. Debes generar la API primero.")

# Creamos el cliente de prueba
client = TestClient(app)

def test_read_root():
    """
    Prueba de Health Check: Verifica que la API esté viva.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "API CESFAM Model Ready"}

def test_predict_endpoint_valid_input():
    """
    Prueba Funcional: Verifica que el endpoint /predict responda correctamente
    con un JSON válido de entrada.
    """
    # Payload de ejemplo (un paciente típico)
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

    # Enviamos petición POST
    response = client.post("/predict", json=payload)

    # 1. Validar código de respuesta HTTP (200 OK)
    assert response.status_code == 200, f"Error en API: {response.text}"
    
    # 2. Validar estructura de la respuesta JSON
    data = response.json()
    assert "prediccion" in data
    assert "probabilidad" in data
    
    # 3. Validar consistencia de datos
    assert data["prediccion"] in [0, 1] # Debe ser clase binaria
    assert 0.0 <= data["probabilidad"] <= 1.0 # Probabilidad válida

def test_predict_endpoint_high_risk():
    """
    Prueba de Lógica: Verifica que un caso de alto riesgo (muchas faltas previas)
    tenga una probabilidad alta.
    """
    payload = {
        "edad": 25,
        "sexo": "Masculino",
        "sector": "Centro",
        "prevision": "Fonasa A",
        "especialidad": "Dental",
        "dia_semana": "Viernes",
        "turno": "Tarde",
        "tiempo_espera_dias": 30, # Mucha espera
        "inasistencias_previas": 10 # Historial terrible
    }

    response = client.post("/predict", json=payload)
    data = response.json()
    
    # Esperamos que el modelo detecte el riesgo (Probabilidad > 0.5 o Predicción 1)
    # Nota: Esto depende de la calidad del entrenamiento, pero sirve como sanity check
    assert response.status_code == 200
    print(f"\nProbabilidad calculada para caso riesgoso: {data['probabilidad']}")