import unittest
import pandas as pd
import numpy as np
import sys
import os

# --- Configuración de rutas ---
# Agregamos la ruta raíz para importar src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.modeling.pipeline import get_preprocessing_pipeline

class TestPreprocessing(unittest.TestCase):
    
    def setUp(self):
        """
        Se ejecuta antes de cada test. Crea un DataFrame de prueba pequeño
        con la misma estructura que el dataset real del CESFAM.
        """
        self.pipeline = get_preprocessing_pipeline()
        
        # Datos de ejemplo (1 fila completa)
        self.sample_data = pd.DataFrame({
            'edad': [45],
            'tiempo_espera_dias': [10],
            'inasistencias_previas': [2],
            'sexo': ['Femenino'],
            'sector': ['Norte'],
            'prevision': ['Fonasa B'],
            'especialidad': ['Medicina General'],
            'dia_semana': ['Lunes'],
            'turno': ['Mañana']
        })

    def test_pipeline_creation(self):
        """Prueba que la función devuelve un objeto pipeline válido."""
        self.assertIsNotNone(self.pipeline)

    def test_basic_transformation(self):
        """
        Prueba que el pipeline pueda transformar datos limpios sin errores.
        Debe devolver un array numérico (listo para el modelo).
        """
        # Ajustamos el pipeline con los datos (fit) y transformamos
        processed_data = self.pipeline.fit_transform(self.sample_data)
        
        # Verificaciones
        self.assertIsInstance(processed_data, np.ndarray) # Debe ser un array numpy
        self.assertTrue(processed_data.shape[1] > 0) # Debe tener columnas (features)

    def test_missing_values_handling(self):
        """
        Prueba CRÍTICA: Verifica que el pipeline no falle si llegan datos Nulos (NaN).
        El pipeline debe imputarlos (rellenarlos) automáticamente.
        """
        # Creamos datos con nulos (simulando un error en el sistema de origen)
        dirty_data = pd.DataFrame({
            'edad': [np.nan], # Falta la edad
            'tiempo_espera_dias': [5],
            'inasistencias_previas': [0],
            'sexo': [np.nan], # Falta el sexo
            'sector': ['Sur'],
            'prevision': ['Fonasa A'],
            'especialidad': ['Dental'],
            'dia_semana': ['Martes'],
            'turno': ['Tarde']
        })

        try:
            # Intentamos transformar. Si el pipeline está mal configurado, esto lanzará error.
            processed_dirty = self.pipeline.fit_transform(dirty_data)
            self.assertIsNotNone(processed_dirty)
            print("\n✅ El pipeline manejó correctamente los valores Nulos (NaN).")
        except Exception as e:
            self.fail(f"El pipeline falló al recibir valores nulos: {e}")

    def test_unknown_category(self):
        """
        Prueba que pasa si llega una categoría nueva que no conocíamos
        (ej: un sector nuevo 'Sector X'). El OneHotEncoder debe ignorarlo y no romper.
        """
        # Entrenamos primero con datos conocidos
        self.pipeline.fit(self.sample_data)
        
        # Datos nuevos con una categoría desconocida en 'sector'
        new_data = self.sample_data.copy()
        new_data['sector'] = ['Sector_Desconocido_Nuevo']
        
        try:
            transformed = self.pipeline.transform(new_data)
            self.assertIsNotNone(transformed)
            print("\n✅ El pipeline manejó correctamente una categoría desconocida.")
        except Exception as e:
            self.fail(f"El pipeline falló con una categoría desconocida: {e}")

if __name__ == '__main__':
    unittest.main()