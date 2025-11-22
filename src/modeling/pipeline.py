import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_preprocessing_pipeline():
    """
    Crea y devuelve un objeto Pipeline de scikit-learn para el preprocesamiento de datos.
    
    Este pipeline maneja:
    1. Variables Numéricas: Imputación de nulos (mediana) + Escalado Estándar.
    2. Variables Categóricas: Imputación de nulos (moda) + One-Hot Encoding.
    
    Returns:
        sklearn.compose.ColumnTransformer: El transformador de columnas configurado.
    """
    
    # --- 1. Definición de Variables ---
    # Es crucial que estos nombres coincidan con las columnas del dataset generado
    numeric_features = ['edad', 'tiempo_espera_dias', 'inasistencias_previas']
    categorical_features = ['sexo', 'sector', 'prevision', 'especialidad', 'dia_semana', 'turno']

    # --- 2. Transformador para Variables Numéricas ---
    # Estrategia:
    # - SimpleImputer(strategy='median'): Si viene un nulo, lo rellena con la mediana (robusto a outliers).
    # - StandardScaler: Escala los datos para que tengan media 0 y varianza 1 (ayuda a modelos lineales y redes neuronales, 
    #   y aunque XGBoost no lo exige estrictamente, mejora la estabilidad).
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # --- 3. Transformador para Variables Categóricas ---
    # Estrategia:
    # - SimpleImputer(strategy='most_frequent'): Rellena nulos con el valor más común.
    # - OneHotEncoder: Convierte categorías en columnas binarias (ej: Sexo -> Sexo_F, Sexo_M).
    #   handle_unknown='ignore' es vital para producción: si llega una categoría nueva desconocida, no rompe la API.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # --- 4. Ensamble del Preprocesador (ColumnTransformer) ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        verbose_feature_names_out=False # Mantiene nombres de columnas limpios
    )

    return preprocessor

if __name__ == "__main__":
    # Bloque de prueba simple para verificar que el pipeline se construye sin errores
    try:
        pipeline = get_preprocessing_pipeline()
        print("✅ Pipeline de preprocesamiento construido exitosamente.")
        print(pipeline)
    except Exception as e:
        print(f"❌ Error al construir el pipeline: {e}")