import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import os
import time # [AÑADIDO] Importado para la pausa de 3 segundos (simulación de tiempo real)

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Dashboard CESFAM - Predicción No-Show",
    page_icon="🏥",
    layout="wide"
)

# --- CONFIGURACIÓN DE ESTILO PARA GRÁFICOS ---
plt.style.use('seaborn-v0_8-whitegrid') 
CELSTE_PRINCIPAL = "#B80B9B"
AZUL_CLARO = "#16E643" 


# --- URL DE LA API (Microservicio) ---
API_URL = "http://127.0.0.1:8000/predict"

# --- FUNCIÓN DE CARGA DE DATOS ---
def load_data(path="data/raw/dataset_cesfam_stream.csv"):
    """
    Carga el dataset leyendo siempre la versión más reciente del archivo
    para reflejar los 50 registros añadidos cada 3 segundos.
    [MODIFICADO] Se cambió la ruta de 'dataset_cesfam_v1.csv' a 'dataset_cesfam_stream.csv'.
    """
    if not os.path.exists(path):
        return None
    
    try:
        df = pd.read_csv(path)
        return df
    except pd.errors.EmptyDataError:
        return None
    except Exception:
        return None

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
st.sidebar.markdown("<h3 style='color: #006dfc;'>Navegación</h3>", 
    unsafe_allow_html=True
)
page = st.sidebar.radio("Ir a:", ["Inicio", "Análisis de Datos (EDA)", "Predicción en Tiempo Real"])

st.sidebar.info(
    """
    **Proyecto M:** Optimización de Agendamiento CESFAM.
    Este sistema predice la probabilidad de inasistencia (No-Show).
    """
)

# --- PÁGINA 1: INICIO (Sin cambios) ---
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
    * Ve a **Análisis de Datos** para entender los patrones en tiempo real.
    * Ve a **Predicción** para probar el modelo con un paciente nuevo.
    """)

# --- PÁGINA 2: EDA (Exploratory Data Analysis) - PROACTIVO ---
elif page == "Análisis de Datos (EDA)":
    st.title("📊 Análisis Exploratorio de Datos (Proactivo)")
    st.markdown("Los datos y métricas se actualizan cada 3 segundos al crecer el archivo CSV.")
    
    update_container = st.empty()
    # releer el archivo y redibujar los gráficos periódicamente.
    while True:
        
        # 1. Leer los datos más recientes (sin caché)
        df = load_data()
        
        # 2. Re-dibujar todo el contenido dentro del contenedor (sobrescribiendo el anterior)
        with update_container.container():
            
            # Muestra la hora de actualización para confirmar el ciclo
            st.info(f"Última Lectura de Datos: **{pd.Timestamp.now().strftime('%H:%M:%S')}**")
            
            if df is None:
                st.error("⚠️ No se encontró el dataset de streaming o está vacío. Asegúrate de ejecutar el script generador de datos.")
            else:
                # Métricas Generales
                col1, col2, col3 = st.columns(3)
                total_citas = len(df) # Este valor debe crecer con el tiempo
                tasa_noshow = df['target_no_asiste'].mean() * 100
                
                col1.metric("Total Citas Acumuladas", f"{total_citas}")
                col2.metric("Tasa Global de No-Show", f"{tasa_noshow:.2f}%")
                
                # Confirma que el archivo está creciendo
                file_path = load_data.__defaults__[0]
                if os.path.exists(file_path):
                     col3.metric("Tamaño del Archivo", f"{os.path.getsize(file_path) / (1024*1024):.2f} MB")
                else:
                    col3.metric("Tamaño del Archivo", "N/A")

                st.markdown("---")
                
                # Gráficos (Generados DENTRO del bucle para actualizarse)
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    st.subheader("Inasistencia por Especialidad")
                    fig, ax = plt.subplots()
                    # MODIFICADO: Paleta secuencial celeste
                    sns.barplot(data=df, x='especialidad', y='target_no_asiste', errorbar=None, palette="Blues_r", ax=ax)
                    plt.xticks(rotation=45)
                    plt.ylabel("Probabilidad de No-Show")
                    # [BUENA PRÁCTICA] Se usa clear_figure=True para liberar la memoria de Matplotlib
                    st.pyplot(fig, clear_figure=True)
                    st.caption("Observamos qué especialidades tienen mayor riesgo de deserción.")

                with col_g2:
                    st.subheader("Inasistencia por Edad")
                    fig, ax = plt.subplots()
                    # MODIFICADO: Paleta con los colores celestes definidos
                    sns.histplot(data=df, x='edad', hue='target_no_asiste', multiple="stack", bins=20, palette=[AZUL_CLARO, CELSTE_PRINCIPAL], ax=ax)
                    plt.xlabel("Edad")
                    st.pyplot(fig, clear_figure=True)
                    st.caption("Distribución de edad diferenciada por asistencia.")

                st.subheader("Matriz de Correlación (Variables Numéricas)")
                fig_corr, ax_corr = plt.subplots(figsize=(10, 4))
                numeric_df = df.select_dtypes(include=['float64', 'int64'])
                # MODIFICADO: Mapa de calor azulado/celeste
                sns.heatmap(numeric_df.corr(), annot=True, cmap='mako', ax=ax_corr)
                st.pyplot(fig_corr, clear_figure=True)

        # 3. Pausa de 3 segundos para sincronizar con el script de generación.
        time.sleep(3)

# --- PÁGINA 3: PREDICCIÓN (Consumo de API - Sin cambios estructurales) ---
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
                    st.warning("⚠️ No se pudo conectar con la API (http://127.0.0.1:8000).")
                    st.info("ℹ️ Mostrando simulación visual para propósitos de demostración:")
                    # --- SIMULACIÓN (Fallback) ---
                    probabilidad = 0.85 if inasistencias > 2 or espera > 30 else 0.15
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