from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import uvicorn
import os
import sys

# --- Configuración de Rutas ---
# Agregamos la ruta raíz del proyecto para poder importar módulos propios (como model_loader)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.api.model_loader import load_model

# --- 1. Definición de la Aplicación FastAPI ---
app = FastAPI(
    title="API de Predicción No-Show CESFAM",
    description="Microservicio para predecir inasistencias a horas médicas usando Machine Learning.",
    version="1.0.0"
)

# Variable global para almacenar el modelo en memoria
model = None

# --- 2. Evento de Inicio (Carga del Modelo) ---
# Se ejecuta una sola vez al arrancar el servidor.
@app.on_event("startup")
def startup_event():
    global model
    try:
        # Carga el archivo .pkl generado por train.py
        model = load_model("model_pipeline.pkl")
        print("🚀 API Iniciada y Modelo Cargado Correctamente.")
    except Exception as e:
        print(f"❌ Error fatal al cargar el modelo: {e}")
        # No detenemos la app para permitir diagnósticos, pero /predict fallará si model es None

# --- 3. Esquema de Validación de Datos (Pydantic) ---
# Esto asegura que los datos que lleguen tengan el tipo y formato correcto.
class PacienteInput(BaseModel):
    edad: int = Field(..., ge=0, le=120, description="Edad del paciente")
    sexo: str = Field(..., description="Femenino o Masculino")
    sector: str = Field(..., description="Sector del CESFAM (Norte, Sur, Centro, Rural)")
    prevision: str = Field(..., description="Tipo de previsión (Fonasa A, B, C, D)")
    especialidad: str = Field(..., description="Especialidad médica solicitada")
    dia_semana: str = Field(..., description="Día de la cita (Lunes, Martes...)")
    turno: str = Field(..., description="Mañana o Tarde")
    tiempo_espera_dias: int = Field(..., ge=0, description="Días entre solicitud y cita")
    inasistencias_previas: int = Field(..., ge=0, description="Historial de faltas")

    # Ejemplo para la documentación automática de la API (/docs)
    class Config:
        schema_extra = {
            "example": {
                "edad": 45,
                "sexo": "Femenino",
                "sector": "Norte",
                "prevision": "Fonasa B",
                "especialidad": "Medicina General",
                "dia_semana": "Lunes",
                "turno": "Mañana",
                "tiempo_espera_dias": 5,
                "inasistencias_previas": 0
            }
        }

# --- 4. Endpoint de Predicción ---
@app.post("/predict")
def predict_no_show(data: PacienteInput):
    global model
    
    # Verificación de seguridad: ¿El modelo está cargado?
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo no está disponible. Revise los logs del servidor.")

    try:
        # A. Convertir input JSON a DataFrame (formato que espera el pipeline)
        input_df = pd.DataFrame([data.dict()])
        
        # B. Realizar predicción
        # El pipeline (model) se encarga de: Imputar nulos -> Encoding -> Escalar -> Predecir
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] # Probabilidad de la clase 1 (No Asiste)

        # C. Retornar respuesta JSON
        return {
            "prediccion": int(prediction), # 0 = Asiste, 1 = No Asiste
            "probabilidad": float(round(probability, 4)),
            "mensaje": "Alto riesgo de inasistencia" if prediction == 1 else "Bajo riesgo - Asistencia probable"
        }

    except Exception as e:
        # Captura cualquier error técnico durante la predicción
        print(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor al procesar la solicitud: {str(e)}")

# --- 5. Endpoint de Salud (Health Check) ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "API CESFAM Model Ready v1.0"}

# --- Bloque para ejecución directa ---
if __name__ == "__main__":
    # Permite ejecutar el script directamente con: python src/api/main.py
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)