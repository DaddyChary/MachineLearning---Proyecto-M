import pandas as pd
import numpy as np
import os
import random
import time

# --- Configuración Inicial ---
GUARDAR_PATH = "data/raw/dataset_cesfam_stream.csv"
SEMBRAR_INICIAL = 10000 
SEMBRAR_INCREMENTO = 50  # 50 REGISTROS A AÑADIR CADA 3 SEGUNDOS
INTERVALO_SEGUNDOS = 3

# Configuración de semilla para reproducibilidad (solo para la siembra inicial)
np.random.seed(42)
random.seed(42)

def generar_registros_cesfam(n_registros, start_id):
    """
    Genera un lote de registros sintéticos.
    """
    if n_registros <= 0:
        return pd.DataFrame()

    # --- Lógica de Generación de Datos (Se mantiene igual) ---
    ids = range(start_id, start_id + n_registros)

    edades = np.concatenate([
        np.random.randint(0, 15, int(n_registros * 0.15)),
        np.random.randint(15, 65, int(n_registros * 0.60)),
        np.random.randint(65, 95, int(n_registros * 0.25))
    ])
    edades = np.resize(edades, n_registros) 
    np.random.shuffle(edades) 

    sexos = np.random.choice(['Femenino', 'Masculino'], n_registros, p=[0.55, 0.45])
    sectores = np.random.choice(['Norte', 'Sur', 'Centro', 'Rural'], n_registros, p=[0.3, 0.3, 0.3, 0.1])
    prevision = np.random.choice(['Fonasa A', 'Fonasa B', 'Fonasa C', 'Fonasa D'], n_registros)
    especialidades = np.random.choice(
        ['Medicina General', 'Dental', 'Matrona', 'Salud Mental', 'Kinesiologia', 'Nutricionista'], 
        n_registros, 
        p=[0.40, 0.20, 0.15, 0.10, 0.10, 0.05]
    )

    dias_semana = np.random.choice(['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes'], n_registros)
    turnos = np.random.choice(['Mañana', 'Tarde'], n_registros)
    tiempo_espera_dias = np.random.exponential(scale=10, size=n_registros).astype(int)
    inasistencias_previas = np.random.poisson(lam=0.5, size=n_registros)

    # Lógica de la Variable Objetivo
    scores = np.random.uniform(0, 1, n_registros)
    
    mask_espera_larga = tiempo_espera_dias > 20
    scores[mask_espera_larga] -= 0.15
    mask_viernes_tarde = (dias_semana == 'Viernes') & (turnos == 'Tarde')
    scores[mask_viernes_tarde] -= 0.10
    mask_jovenes = (edades >= 20) & (edades <= 35)
    scores[mask_jovenes] -= 0.05
    scores -= (inasistencias_previas * 0.05)
    mask_mayores = edades > 65
    scores[mask_mayores] += 0.10
    mask_complejos = np.isin(especialidades, ['Salud Mental', 'Dental'])
    scores[mask_complejos] -= 0.05

    target = np.where(scores < 0.20, 1, 0)

    # Consolidación del DataFrame
    df_lote = pd.DataFrame({
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
        'target_no_asiste': target
    })

    return df_lote

def simular_streaming_cesfam(guardar_path=GUARDAR_PATH, inicial=SEMBRAR_INICIAL, incremento=SEMBRAR_INCREMENTO, intervalo_segundos=INTERVALO_SEGUNDOS):
    """
    Genera 10,000 registros iniciales (instantáneo) y luego añade 50 cada 3 segundos.
    """
    os.makedirs(os.path.dirname(guardar_path), exist_ok=True)
    
    # --- 1. Generación de 10,000 registros INICIALES (instantáneo) ---
    print(f"⌛ Generando siembra inicial de {inicial} registros...")
    df_inicial = generar_registros_cesfam(inicial, start_id=1)
    df_inicial.to_csv(guardar_path, index=False) # Guarda todo el lote de 10k
    
    print(f"✅ Siembra inicial de 10,000 registros guardada instantáneamente.")
    
    siguiente_id = inicial + 1
    lote_count = 0

    # --- 2. Simulación de Streaming (50 registros cada 3 segundos) ---
    print("\n--- 🚀 INICIANDO SIMULACIÓN DE STREAMING ---")
    print(f"Añadiendo **{incremento} nuevos registros** cada {intervalo_segundos} segundos. Presiona Ctrl+C para detener.")

    try:
        while True:
            # Generar el nuevo lote de 50
            df_nuevo_lote = generar_registros_cesfam(incremento, start_id=siguiente_id)
            
            if not df_nuevo_lote.empty:
                # Añadir al archivo CSV existente sin reescribir la cabecera
                df_nuevo_lote.to_csv(guardar_path, mode='a', header=False, index=False)
                
                siguiente_id += incremento
                lote_count += 1
                
                tasa_lote = df_nuevo_lote['target_no_asiste'].mean() * 100
                
                # Imprimir confirmación
                print(f"\n[{time.strftime('%H:%M:%S')}] Lote #{lote_count} | **{incremento} REGISTROS AÑADIDOS**.")
                print(f"  Registros totales acumulados: {siguiente_id-1} | Tasa de inasistencia del lote: {tasa_lote:.2f}%")
                print("  Primeros 5 registros del lote:")
                print(df_nuevo_lote.head().to_markdown(index=False, numalign="left", stralign="left")) 
                print("--------------------------------------------------------------------------------")

            # Pausa de 3 segundos
            time.sleep(intervalo_segundos)

    except KeyboardInterrupt:
        print("\n--- 🛑 SIMULACIÓN DE STREAMING DETENIDA POR EL USUARIO. ---")
        print(f"Total de lotes generados: {lote_count}")


if __name__ == "__main__":
    # La ejecución comienza aquí y genera primero los 10k y luego los lotes de 50.
    simular_streaming_cesfam()