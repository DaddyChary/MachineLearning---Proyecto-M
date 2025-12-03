import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import os
import time
import subprocess 
import sys 
import signal 

st.set_page_config(
    page_title="Dashboard CESFAM - Predicción No-Show",
    page_icon="🏥",
    layout="wide"
)

plt.style.use('seaborn-v0_8-whitegrid') 
CELSTE_PRINCIPAL = "#B80B9B"
AZUL_CLARO = "#16E643" 

API_URL = "http://127.0.0.1:8000/predict"
GENERATOR_SCRIPT = "src/data_prep/stream_generator.py"

if 'stream_pid' not in st.session_state:
    st.session_state.stream_pid = None
if 'stream_active' not in st.session_state:
    st.session_state.stream_active = False

def load_data(path="data/raw/dataset_cesfam_stream.csv"):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df
    except (pd.errors.EmptyDataError, Exception):
        return None

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
st.sidebar.markdown("<h3 style='color: #006dfc;'>Navegación</h3>", unsafe_allow_html=True)
page = st.sidebar.radio("Ir a:", ["Inicio", "Análisis de Datos (EDA)", "Predicción en Tiempo Real"])

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='color: #FF5733;'>⚙️ Control de Datos</h3>", unsafe_allow_html=True)

col_btn1, col_btn2 = st.sidebar.columns(2)

with col_btn1:
    if st.button("▶️ Iniciar", disabled=st.session_state.stream_active):
        try:
            process = subprocess.Popen([sys.executable, GENERATOR_SCRIPT])
            st.session_state.stream_pid = process.pid
            st.session_state.stream_active = True
            st.success("Streaming ON")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

with col_btn2:
    if st.button("⏹️ Detener", disabled=not st.session_state.stream_active):
        if st.session_state.stream_pid:
            try:
                os.kill(st.session_state.stream_pid, signal.SIGTERM)
                st.session_state.stream_pid = None
                st.session_state.stream_active = False
                st.warning("Streaming OFF")
                st.rerun()
            except Exception as e:
                st.error(f"Error al detener: {e}")
                st.session_state.stream_pid = None
                st.session_state.stream_active = False

if st.session_state.stream_active:
    st.sidebar.markdown("🟢 **Generando datos en vivo...**")
else:
    st.sidebar.markdown("🔴 **Generación detenida.**")

st.sidebar.info("**Proyecto M:** Optimización de Agendamiento CESFAM.")

if page == "Inicio":
    st.title("🏥 Sistema de Gestión de Horas CESFAM")
    st.markdown("""
    ### Contexto del Problema
    Los Centros de Salud Familiar enfrentan una alta tasa de inasistencia (**No-Show**), 
    lo que genera ineficiencia en el uso de recursos médicos y largas listas de espera.
    
    ### Solución Propuesta
    Este Dashboard integra un modelo de **Machine Learning** capaz de:
    1. Analizar patrones históricos de comportamiento.
    2. Predecir la probabilidad de que un paciente falte a su cita.
    3. Permitir al personal administrativo tomar decisiones proactivas.

    ---
    **Instrucciones:**
    - Ve a **Análisis de Datos** para entender los patrones.
    - Ve a **Predicción** para probar el modelo.
    - Presiona **▶️ Iniciar** para simular datos en tiempo real.
    """)

# --- PÁGINA 2: EDA ---
elif page == "Análisis de Datos (EDA)":
    st.title("📊 Análisis Exploratorio de Datos (Proactivo)")
    
    update_container = st.empty()
    streaming_mode = st.session_state.stream_active
    
    while True:
        df = load_data()
        
        with update_container.container():
            st.info(f"Estado del Sistema: {'🟢 ONLINE' if st.session_state.stream_active else '🔴 OFFLINE'} | Última Lectura: **{pd.Timestamp.now().strftime('%H:%M:%S')}**")
            
            if df is None:
                st.warning("⚠️ Esperando datos... Presiona '▶️ Iniciar'.")
            else:
                col1, col2, col3 = st.columns(3)
                total_citas = len(df)
                tasa_noshow = df['target_no_asiste'].mean() * 100
                
                col1.metric("Total Citas Acumuladas", f"{total_citas}")
                col2.metric("Tasa Global de No-Show", f"{tasa_noshow:.2f}%")
                
                file_path = load_data.__defaults__[0]
                if os.path.exists(file_path):
                    col3.metric("Tamaño del Archivo", f"{os.path.getsize(file_path) / (1024*1024):.2f} MB")
                
                st.markdown("---")
                
                col_g1, col_g2 = st.columns(2)

                with col_g1:
                    st.subheader("Inasistencia por Especialidad")
                    fig, ax = plt.subplots()
                    sns.barplot(data=df, x='especialidad', y='target_no_asiste',
                                errorbar=None, palette="Blues_r", ax=ax)
                    plt.xticks(rotation=45)
                    plt.ylabel("Probabilidad de No-Show")
                    st.pyplot(fig, clear_figure=True)

                with col_g2:
                    st.subheader("Inasistencia por Edad")
                    fig, ax = plt.subplots()
                    sns.histplot(data=df, x='edad', hue='target_no_asiste',
                                 multiple="stack", bins=20,
                                 palette=[AZUL_CLARO, CELSTE_PRINCIPAL], ax=ax)
                    st.pyplot(fig, clear_figure=True)

                st.subheader("Matriz de Correlación")
                fig_corr, ax_corr = plt.subplots(figsize=(10, 4))
                numeric_df = df.select_dtypes(include=['float64', 'int64'])
                sns.heatmap(numeric_df.corr(), annot=True, cmap='mako', ax=ax_corr)
                plt.xticks(rotation=45, ha="right")
                plt.yticks(rotation=0)
                plt.tight_layout()
                st.pyplot(fig_corr, clear_figure=True)

        if not st.session_state.stream_active:
            break

        time.sleep(3)

elif page == "Predicción en Tiempo Real":
    st.title("🤖 Predicción de Riesgo de No-Show")
    st.markdown("Ingrese los datos de la cita para evaluar el riesgo de inasistencia.")
    
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
        
        try:
            with st.spinner("Consultando al oráculo del Machine Learning..."):
                try:
                    response = requests.post(API_URL, json=datos_entrada)
                    if response.status_code == 200:
                        result = response.json()
                        prediccion = result["prediccion"] 
                        probabilidad = result["probabilidad"] 
                    else:
                        st.error(f"Error en la API: {response.status_code}")
                        st.stop()
                except requests.exceptions.ConnectionError:
                    st.warning("⚠️ No se pudo conectar con la API.")
                    st.info("ℹ️ Mostrando simulación:")
                    probabilidad = 0.85 if inasistencias > 2 or espera > 30 else 0.15
                    prediccion = 1 if probabilidad > 0.5 else 0

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
                st.write("Nivel de Riesgo:")
                st.progress(float(probabilidad))
                if prediccion == 1:
                    st.warning("💡 Recomendación: Enviar recordatorio o sobre-agendar.")
                else:
                    st.info("💡 Flujo normal.")
                        
        except Exception as e:
            st.error(f"Ocurrió un error inesperado: {e}")

st.sidebar.markdown("---")
st.sidebar.caption(
    "© 2025 Sistema Predictivo de Agendamiento CESFAM\n"
    "Creadores: leonidasSpartano, LuisTobar765, DaddyChary, TheBlackSoldier1, peulsa"
)
