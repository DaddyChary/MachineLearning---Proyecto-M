import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_preprocessing_pipeline():
    
    numeric_features = ['edad', 'tiempo_espera_dias', 'inasistencias_previas']
    categorical_features = ['sexo', 'sector', 'prevision', 'especialidad', 'dia_semana', 'turno']

   
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        verbose_feature_names_out=False 
    )

    return preprocessor

if __name__ == "__main__":
    try:
        pipeline = get_preprocessing_pipeline()
        print("✅ Pipeline de preprocesamiento construido exitosamente.")
        print(pipeline)
    except Exception as e:
        print(f"❌ Error al construir el pipeline: {e}")