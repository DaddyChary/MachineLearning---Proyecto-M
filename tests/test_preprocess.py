import unittest
import pandas as pd
import numpy as np
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.modeling.pipeline import get_preprocessing_pipeline

class TestPreprocessing(unittest.TestCase):
    
    def setUp(self):
        
        self.pipeline = get_preprocessing_pipeline()
        
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
        self.assertIsNotNone(self.pipeline)

    def test_basic_transformation(self):
        
        processed_data = self.pipeline.fit_transform(self.sample_data)
        
        self.assertIsInstance(processed_data, np.ndarray)
        self.assertTrue(processed_data.shape[1] > 0) 

    def test_missing_values_handling(self):
        
        dirty_data = pd.DataFrame({
            'edad': [np.nan],
            'tiempo_espera_dias': [5],
            'inasistencias_previas': [0],
            'sexo': [np.nan],
            'sector': ['Sur'],
            'prevision': ['Fonasa A'],
            'especialidad': ['Dental'],
            'dia_semana': ['Martes'],
            'turno': ['Tarde']
        })

        try:
            processed_dirty = self.pipeline.fit_transform(dirty_data)
            self.assertIsNotNone(processed_dirty)
            print("\n✅ El pipeline manejó correctamente los valores Nulos (NaN).")
        except Exception as e:
            self.fail(f"El pipeline falló al recibir valores nulos: {e}")

    def test_unknown_category(self):
        
        self.pipeline.fit(self.sample_data)
        
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