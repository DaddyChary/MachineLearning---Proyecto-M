# Sistema Predictivo de Agendamiento CESFAM
## Proyecto de Minería de Datos - Noviembre 2025

Este proyecto implementa una solución de Machine Learning para optimizar la gestión de horas médicas en Centros de Salud Familiar (CESFAM), prediciendo la probabilidad de inasistencia (no-show) de los pacientes.

👥 Equipo de Trabajo

Gamaliel Moya

Erika Aristizábal

Leonardo Miranda

Matías Baeza

Luis Tobar

---

# 1. Descripción del Problema
1.1 Contexto Real
Actualmente, los Centros de Salud Familiar (CESFAM) en Chile enfrentan una saturación crónica en su sistema de agendamiento. La asignación de horas se realiza a menudo mediante métodos manuales o presenciales, resultando en una distribución ineficiente y alta frustración en la población.

- El problema crítico es la alta tasa de inasistencia (no-show rate). Los cupos asignados se pierden cuando los pacientes no asisten, generando tiempos ociosos para los médicos y listas de espera más largas.

1.2 Solución Propuesta
Desarrollamos un sistema inteligente que utiliza datos históricos para predecir la probabilidad de falta. Esto permite pasar de una gestión reactiva a una proactiva, habilitando estrategias como el sobre-agendamiento inteligente o recordatorios focalizados.

- El núcleo de la solución es un modelo de clasificación (Gradient Boosting) expuesto a través de una API REST y visualizado en un Dashboard interactivo.

---

# 2. Arquitectura del Sistema
El proyecto sigue una arquitectura modular de microservicios:

- Capa de Datos: Generación de datos sintéticos con patrones demográficos reales (data_generator.py).
- Pipeline ETL/ML: Preprocesamiento (Imputación, Encoding) y entrenamiento automatizado (pipeline.py, train.py).
- API REST: Microservicio en FastAPI que sirve el modelo (main.py).
- Dashboard: Interfaz de usuario en Streamlit para análisis y predicción (dashboard.py).

---

# 3. Estructura del Proyecto
El código está organizado de manera modular para facilitar el mantenimiento y escalabilidad :

ProyectoCesfam/

├── README.md               # Documentación general

├── requirements.txt        # Dependencias del proyecto

├── .streamlit/

│   ├── # config.toml # Modifica Fondo, letras y Tipografía

├── data/

│   ├── # Dataset generado (dataset_cesfam_v1.csv)

├── docs/ # Evidencias de pruebas funcionales y documentacion en general   

├── models/

│   └── model_pipeline.pkl  # Modelo entrenado serializado

├── src/

│   ├── api/

│   │   ├── main.py         # API FastAPI (Endpoint /predict)

│   │   └── model_loader.py # Cargador del modelo

│   ├── dashboard/

│   │   └── dashboard.py    # Interfaz Streamlit

│   ├── data_prep/

│   │   └── stream_generator.py # Simulación de flujo en tiempo real para el Dashboard

│   │   └── data_generator.py # Script de generación de datos

│   └── modeling/

│       ├── pipeline.py     # Lógica de preprocesamiento

│       └── train.py        # Script de entrenamiento

└── tests/                  # Tests unitarios (pytest)

---

# 4. Guía de Instalación y Ejecución
Sigue estos pasos en orden para ejecutar el sistema completo.

---

## Paso 1: Instalación de Dependencias
Asegúrate de tener Python 3.9+ instalado.

Bash

pip install -r requirements.txt

(Si no tienes el archivo, instala: pip install streamlit pandas seaborn matplotlib requests scikit-learn fastapi uvicorn pydantic joblib)

---

## Paso 2: Generación de Datos
Crea el dataset sintético que simula los patrones del CESFAM.

Bash

python src/data_prep/data_generator.py

---

## Paso 3: Entrenamiento del Modelo
Entrena el algoritmo y genera el archivo model_pipeline.pkl en la carpeta models/.

Bash

python src/modeling/train.py

Métricas clave: Se prioriza el Recall de la clase 1 para minimizar falsos negativos.

---

## Paso 4: Iniciar la API (Backend)
En una terminal, levanta el servidor de predicción.

Bash

python src/api/main.py
La API quedará corriendo en http://127.0.0.1:8000.

---

## Paso 5: Iniciar el Dashboard (Frontend)
En una segunda terminal, inicia la interfaz gráfica.

Bash

streamlit run src/dashboard/dashboard.py

---

# 5. Uso de la API
El sistema expone un endpoint principal para realizar predicciones.

Endpoint: POST /predict

Formato de Entrada (JSON):

JSON


{

  "edad": 45,
  
  "sexo": "Femenino",
  
  "sector": "Norte",
  
  "prevision": "Fonasa B",
  
  "especialidad": "Dental",
  
  "dia_semana": "Lunes",
  
  "turno": "Mañana",
  
  "tiempo_espera_dias": 10,
  
  "inasistencias_previas": 1
  
}

Respuesta: Predicción binaria (0/1) y probabilidad de riesgo.

---

# 6. Testing
Para ejecutar las pruebas unitarias que validan el preprocesamiento y la API:

Bash

python -m unittest tests/test_preprocess.py

o usando pytest

pytest tests/
---

### Guía Rápida para Usar la Plataforma CESFAM
Sistema de Predicción de Inasistencia (No-Show)

Esta plataforma permite analizar datos de los pacientes y predecir si una persona podría faltar a su cita. No requiere conocimientos técnicos.

1. Inicio — Información General
   Cuando entras a la página verás primero una explicación del problema y de la solución.
   Aquí puedes:
   - Leer por qué existe este sistema.
   - Saber qué hace: analiza datos y predice riesgos de inasistencia.
   - Ver instrucciones básicas que indican hacia dónde avanzar:
     - "Análisis de Datos" para conocer patrones.
     - "Predicción" para probar el modelo con un paciente.
   Esta sección es solo informativa, no necesitas realizar ninguna acción.

2. Análisis de Datos (EDA) — Ver patrones del CESFAM
   En esta sección puedes observar de forma sencilla cómo se comportan los pacientes del CESFAM.
   Encontrarás:

   Indicadores principales:
   - Total de citas históricas.
   - Tasa global de inasistencia.

   Gráficas:
   - Inasistencia por especialidad: muestra qué áreas tienen más faltas.
   - Inasistencia por edad: cuántas personas faltan según su edad.
   - Matriz de correlación: relación entre variables (solo observación).

   Aquí no se ingresan datos. Sirve únicamente para mirar y comprender la información rápidamente.

3. Predicción en Tiempo Real — Calcular riesgo de No-Show
   Esta es la parte más importante para un usuario común.
   Aquí puedes simular una cita ingresando datos y el sistema entregará la probabilidad de que la persona falte.

  ### Instrucciones de ingreso de datos

Debes ingresar lo siguiente:

1.  **Edad del paciente:** Mover la barra hasta la edad correspondiente.
2.  **Sexo:** Seleccionar Femenino o Masculino.
3.  **Previsión:** Elegir Fonasa A/B/C/D.
4.  **Día de la semana:** Seleccionar el día de la cita.
5.  **Especialidad:** Escoger la especialidad donde será atendido.
6.  **Turno:** Seleccionar mañana o tarde.
7.  **Sector:** Elegir Norte, Sur, Poniente u otro.
8.  **Inasistencias previas:** Usar los botones para indicar cuántas veces ha faltado antes.
9.  **Días de espera:** Mover la barra según cuántos días faltan para la cita.
10. **Finalización:** Presiona "Calcular Riesgo". El sistema mostrará la probabilidad de inasistencia, indicando si es baja, media o alta. Esto ayuda a decidir si conviene enviar recordatorios, reagendar o tomar medidas preventivas.

**Resumen:**
- Inicio: leer información general.
- Análisis de Datos: visualizar gráficos simples con los patrones de inasistencia.
- Predicción: ingresar datos de un paciente para obtener el riesgo de que no asista.

La plataforma está diseñada para que cualquier persona pueda utilizarla fácilmente sin conocimientos técnicos.


---

# *Desarrollado para la asignatura de Minería de Datos - 2025. MachineLearning---Proyecto-M*

---
