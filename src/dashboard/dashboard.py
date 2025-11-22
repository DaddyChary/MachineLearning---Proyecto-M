import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import os

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Dashboard CESFAM - Predicción No-Show",
    page_icon="🏥",
    layout="wide"
)

# --- URL DE LA API (Microservicio) ---
# Asumimos que la API correrá en el puerto 8000 localmente
API_URL = "http://127.0.0.1:8000/predict"

# --- FUNCIÓN DE CARGA DE DATOS ---
@st.cache_data
def load_data():
    """
    Carga el dataset generado sintéticamente para el EDA.
    Busca el archivo en la ruta relativa correcta.
    """
    # Ajustar ruta según desde dónde se ejecute el script
    # Asumimos ejecución desde la raíz del proyecto
    path = "data/raw/dataset_cesfam_v1.csv"
    
    if not os.path.exists(path):
        return None
    
    df = pd.read_csv(path)
    return df

# --- INTERFAZ LATERAL (SIDEBAR) ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
st.sidebar.title("Navegación")
page = st.sidebar.radio("Ir a:", ["Inicio", "Análisis de Datos (EDA)", "Predicción en Tiempo Real"])

st.sidebar.info(
    """
    **Proyecto M:** Optimización de Agendamiento CESFAM.
    Este sistema predice la probabilidad de inasistencia (No-Show)
    utilizando Machine Learning.
    """
)

# --- PÁGINA 1: INICIO ---
if page == "Inicio":
    st.title("🏥 Sistema de Gestión de Horas CESFAM")
    st.markdown("""
    ### Contexto del Problema
    Los Centros de Salud Familiar enfrentan una alta tasa de inasistencia (**No-Show**), 
    lo que genera ineficiencia en el uso de recursos médicos y largas listas de espera.
    
    ### Solución Propuesta
    Este Dashboard integra un modelo de **Machine Learning (XGBoost/LightGBM)** capaz de:
    1. Analizar patrones históricos de comportamiento.
    2. Predecir la probabilidad de que un paciente falte a su cita.
    3. Permitir al personal administrativo tomar decisiones proactivas (sobrecupos, recordatorios).
    
    ---
    **Instrucciones:**
    * Ve a **Análisis de Datos** para entender los patrones.
    * Ve a **Predicción** para probar el modelo con un paciente nuevo.
    """)

# --- PÁGINA 2: EDA (Exploratory Data Analysis) ---
elif page == "Análisis de Datos (EDA)":
    st.title("📊 Análisis Exploratorio de Datos")
    
    df = load_data()
    
    if df is None:
        st.error("⚠️ No se encontró el dataset. Por favor ejecuta primero: `python src/data_prep/data_generator.py`")
    else:
        # Métricas Generales
        col1, col2, col3 = st.columns(3)
        total_citas = len(df)
        tasa_noshow = df['target_no_asiste'].mean() * 100
        col1.metric("Total Citas Históricas", f"{total_citas}")
        col2.metric("Tasa Global de No-Show", f"{tasa_noshow:.2f}%")
        
        st.markdown("---")
        
        # Gráficos
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.subheader("Inasistencia por Especialidad")
            fig, ax = plt.subplots()
            sns.barplot(data=df, x='especialidad', y='target_no_asiste', errorbar=None, palette="viridis", ax=ax)
            plt.xticks(rotation=45)
            plt.ylabel("Probabilidad de No-Show")
            st.pyplot(fig)
            st.caption("Observamos qué especialidades tienen mayor riesgo de deserción.")

        with col_g2:
            st.subheader("Inasistencia por Edad")
            fig, ax = plt.subplots()
            sns.histplot(data=df, x='edad', hue='target_no_asiste', multiple="stack", bins=20, palette="coolwarm", ax=ax)
            plt.xlabel("Edad")
            st.pyplot(fig)
            st.caption("Distribución de edad diferenciada por asistencia.")

        st.subheader("Matriz de Correlación (Variables Numéricas)")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 4))
        # Seleccionamos solo numéricas para correlación
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)

# --- PÁGINA 3: PREDICCIÓN (Consumo de API) ---
elif page == "Predicción en Tiempo Real":
    st.title("🤖 Predicción de Riesgo de No-Show")
    st.markdown("Ingrese los datos de la cita para evaluar el riesgo de inasistencia.")
    
    # Formulario de entrada
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            edad = st.slider("Edad del Paciente", 0, 100, 30)
            sexo = st.selectbox("Sexo", ["Femenino", "Masculino"])
            sector = st.selectbox("Sector", ["Norte", "Sur", "Centro", "Rural"])
            
        with col2:
            prevision = st.selectbox("Previsión", ["Fonasa A", "Fonasa B", "Fonasa C", "Fonasa D"])
            especialidad = st.selectbox("Especialidad", 
                                      ['Medicina General', 'Dental', 'Matrona', 'Salud Mental', 'Kinesiologia', 'Nutricionista'])
            inasistencias = st.number_input("Inasistencias Previas", 0, 20, 0)

        with col3:
            dia = st.selectbox("Día de la Semana", ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes"])
            turno = st.radio("Turno", ["Mañana", "Tarde"])
            espera = st.slider("Días de Espera (Anticipación)", 0, 60, 5)
        
        submit_button = st.form_submit_button("Calcular Riesgo")
        
    if submit_button:
        # Preparar el payload para la API
        datos_entrada = {
            "edad": edad,
            "sexo": sexo,
            "sector": sector,
            "prevision": prevision,
            "especialidad": especialidad,
            "dia_semana": dia,
            "turno": turno,
            "tiempo_espera_dias": espera,
            "inasistencias_previas": inasistencias
        }
        
        # Llamada a la API
        try:
            with st.spinner("Consultando al oráculo del Machine Learning..."):
                # Nota: Esto fallará si la API no está corriendo.
                # Simularemos la respuesta si la API no está activa para que veas el funcionamiento visual
                try:
                    response = requests.post(API_URL, json=datos_entrada)
                    if response.status_code == 200:
                        result = response.json()
                        prediccion = result["prediccion"] # 0 o 1
                        probabilidad = result["probabilidad"] # 0.0 a 1.0
                    else:
                        st.error(f"Error en la API: {response.status_code}")
                        st.stop()
                except requests.exceptions.ConnectionError:
                    st.warning("⚠️ No se pudo conectar con la API (src/api/main.py). Asegúrate de que esté corriendo.")
                    st.info("ℹ️ Mostrando simulación visual para propósitos de demostración:")
                    # --- SIMULACIÓN (SOLO PARA DEMO VISUAL SI LA API ESTÁ OFF) ---
                    probabilidad = 0.85 if inasistencias > 2 else 0.15
                    prediccion = 1 if probabilidad > 0.5 else 0
                    # -----------------------------------------------------------

            # Visualización del Resultado
            st.markdown("---")
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                if prediccion == 1:
                    st.error("🔴 ALTO RIESGO DE NO-SHOW")
                    st.metric("Probabilidad de Falta", f"{probabilidad:.1%}")
                else:
                    st.success("🟢 ASISTENCIA PROBABLE")
                    st.metric("Probabilidad de Falta", f"{probabilidad:.1%}")
            
            with col_res2:
                # Barra de progreso visual
                st.write("Nivel de Riesgo:")
                st.progress(float(probabilidad))
                if prediccion == 1:
                    st.warning("💡 **Recomendación:** Enviar recordatorio por WhatsApp o realizar sobre-agendamiento.")
                else:
                    st.info("💡 **Recomendación:** Mantener flujo normal.")
                    
        except Exception as e:
            st.error(f"Ocurrió un error inesperado: {e}")

# --- PIE DE PÁGINA ---
st.sidebar.markdown("---")
st.sidebar.caption("Desarrollado para Minería de Datos 2025")