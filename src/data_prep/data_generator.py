import pandas as pd
import numpy as np
import os
import random

# Configuración de semilla para reproducibilidad
np.random.seed(42)
random.seed(42)

def generar_dataset_cesfam(n_registros=10000, guardar_path="data/dataset_cesfam_v1.csv"):
    """
    Genera un dataset sintético para el problema de agendamiento del CESFAM.
    Simula patrones lógicos para que el modelo de ML pueda aprender.
    """
    
    print(f"Generando {n_registros} registros sintéticos...")

    # --- 1. Generación de Variables Independientes (Features) ---

    # ID del paciente
    ids = range(1, n_registros + 1)

    # Edad: Distribución más realista (más niños y adultos mayores, menos jóvenes)
    # Simulamos una mezcla de distribuciones
    edades = np.concatenate([
        np.random.randint(0, 15, int(n_registros * 0.15)),  # Niños
        np.random.randint(15, 65, int(n_registros * 0.60)), # Adultos
        np.random.randint(65, 95, int(n_registros * 0.25))  # Adultos Mayores
    ])
    np.random.shuffle(edades) # Mezclar para perder el orden

    # Sexo
    sexos = np.random.choice(['Femenino', 'Masculino'], n_registros, p=[0.55, 0.45])

    # Sector del CESFAM (Variable geográfica)
    sectores = np.random.choice(['Norte', 'Sur', 'Centro', 'Rural'], n_registros, p=[0.3, 0.3, 0.3, 0.1])

    # Previsión (Fonasa Tramos)
    prevision = np.random.choice(['Fonasa A', 'Fonasa B', 'Fonasa C', 'Fonasa D'], n_registros)

    # Especialidad de la cita
    especialidades = np.random.choice(
        ['Medicina General', 'Dental', 'Matrona', 'Salud Mental', 'Kinesiologia', 'Nutricionista'], 
        n_registros, 
        p=[0.40, 0.20, 0.15, 0.10, 0.10, 0.05]
    )

    # Día de la semana (0=Lunes, 4=Viernes) - Asumimos que CESFAM no atiende fines de semana para este ejemplo
    dias_semana = np.random.choice(['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes'], n_registros)

    # Turno (Mañana o Tarde)
    turnos = np.random.choice(['Mañana', 'Tarde'], n_registros)

    # Tiempo de espera (Días entre solicitud y la cita)
    # Usamos una distribución exponencial: muchas citas se piden cerca de la fecha, pocas con mucha antelación
    tiempo_espera_dias = np.random.exponential(scale=10, size=n_registros).astype(int)

    # Historial de inasistencias previas (0 a 5)
    inasistencias_previas = np.random.poisson(lam=0.5, size=n_registros) # La mayoría tiene 0

    # --- 2. Simulación de la Variable Objetivo (Lógica de Negocio) ---
    
    # Creamos un "Score de Riesgo" base
    # Si el score supera cierto umbral, asignamos "No Asiste" (1)
    
    probabilidad_base = 0.15 # 15% de inasistencia base
    
    scores = np.random.uniform(0, 1, n_registros)
    
    # Ajustamos la probabilidad según patrones lógicos (Feature Engineering inverso)
    
    # Patrón A: Mayor tiempo de espera = Mayor probabilidad de olvido
    mask_espera_larga = tiempo_espera_dias > 20
    scores[mask_espera_larga] -= 0.15 # Hace más probable caer en el rango de inasistencia

    # Patrón B: Viernes en la tarde = Mayor inasistencia
    mask_viernes_tarde = (dias_semana == 'Viernes') & (turnos == 'Tarde')
    scores[mask_viernes_tarde] -= 0.10

    # Patrón C: Adultos jóvenes (20-35) faltan más por trabajo/estudios
    mask_jovenes = (edades >= 20) & (edades <= 35)
    scores[mask_jovenes] -= 0.05

    # Patrón D: Historial de inasistencias es fuerte predictor
    # Si ha faltado mucho antes, es muy probable que falte de nuevo
    scores -= (inasistencias_previas * 0.05)
    
    # Patrón E: Adultos mayores suelen ser más responsables (asisten más)
    mask_mayores = edades > 65
    scores[mask_mayores] += 0.10

    # Patrón F: Salud Mental y Dental suelen tener tasas más altas de abandono
    mask_complejos = np.isin(especialidades, ['Salud Mental', 'Dental'])
    scores[mask_complejos] -= 0.05

    # Definir la etiqueta final (1 = No Asiste, 0 = Asiste)
    # Si el score modificado es bajo (menor que un umbral aleatorio), es inasistencia
    # Esto asegura que no sea determinístico, pero sí probabilístico
    target = np.where(scores < 0.20, 1, 0)

    # --- 3. Consolidación del DataFrame ---
    
    df = pd.DataFrame({
        'paciente_id': ids,
        'edad': edades,
        'sexo': sexos,
        'sector': sectores,
        'prevision': prevision,
        'especialidad': especialidades,
        'dia_semana': dias_semana,
        'turno': turnos,
        'tiempo_espera_dias': tiempo_espera_dias,
        'inasistencias_previas': inasistencias_previas,
        'target_no_asiste': target # Variable Objetivo
    })

    # --- 4. Guardado ---
    
    # Asegurar que el directorio existe
    os.makedirs(os.path.dirname(guardar_path), exist_ok=True)
    
    df.to_csv(guardar_path, index=False)
    print(f"Dataset guardado exitosamente en: {guardar_path}")
    print(f"Tasa de inasistencia simulada: {df['target_no_asiste'].mean()*100:.2f}%")
    
    return df

if __name__ == "__main__":
    generar_dataset_cesfam()