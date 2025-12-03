from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import uvicorn
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.api.model_loader import load_model

app = FastAPI(
    title="API de Predicción No-Show CESFAM",
    description="Microservicio para predecir inasistencias a horas médicas usando Machine Learning.",
    version="1.0.0"
)

model = None

@app.on_event("startup")
def startup_event():
    global model
    try:

        model = load_model("model_pipeline.pkl")
        print("🚀 API Iniciada y Modelo Cargado Correctamente.")
    except Exception as e:
        print(f"❌ Error fatal al cargar el modelo: {e}")


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

@app.post("/predict")
def predict_no_show(data: PacienteInput):
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo no está disponible. Revise los logs del servidor.")

    try:
        input_df = pd.DataFrame([data.dict()])
        
      
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] 
        return {
            "prediccion": int(prediction),
            "probabilidad": float(round(probability, 4)),
            "mensaje": "Alto riesgo de inasistencia" if prediction == 1 else "Bajo riesgo - Asistencia probable"
        }

    except Exception as e:
        print(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor al procesar la solicitud: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "API CESFAM Model Ready v1.0"}

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)