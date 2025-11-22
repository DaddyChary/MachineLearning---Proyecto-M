import pandas as pd
import sys
import os
import joblib

# Agregamos el directorio raíz al path para poder importar módulos propios
sys.path.append(os.getcwd())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Importamos nuestro pipeline de preprocesamiento definido anteriormente
try:
    from src.modeling.pipeline import get_preprocessing_pipeline
except ImportError:
    # Fallback por si se ejecuta desde dentro de la carpeta modeling
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from src.modeling.pipeline import get_preprocessing_pipeline

def train_model():
    print("🚀 Iniciando proceso de entrenamiento del modelo CESFAM...")

    # --- 1. Carga de Datos ---
    data_path = "data/raw/dataset_cesfam_v1.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontró el dataset en {data_path}. Ejecuta primero data_generator.py")
    
    df = pd.read_csv(data_path)
    print(f"✅ Datos cargados: {df.shape[0]} registros.")

    # --- 2. Separación de Variables (X) y Objetivo (y) ---
    target = 'target_no_asiste'
    X = df.drop(columns=[target, 'paciente_id']) # Eliminamos ID porque no predice nada
    y = df[target]

    # --- 3. División Train/Test (Requisito Rúbrica) ---
    # Usamos 80% para entrenar y 20% para validar.
    # stratify=y asegura que la proporción de 'no-shows' sea igual en ambos grupos.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"🔹 Datos de entrenamiento: {X_train.shape[0]}")
    print(f"🔹 Datos de prueba: {X_test.shape[0]}")

    # --- 4. Definición del Modelo y Pipeline Completo ---
    # Usamos GradientBoostingClassifier (similar a XGBoost)
    # Justificación de Hiperparámetros:
    # - n_estimators=100: Cantidad de árboles de decisión (suficiente para este volumen).
    # - learning_rate=0.1: Paso de aprendizaje estándar para evitar overfitting.
    # - max_depth=3: Árboles poco profundos para mantener el modelo generalizable.
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    # Unimos el preprocesador (pipeline.py) con el modelo
    full_pipeline = Pipeline([
        ('preprocessor', get_preprocessing_pipeline()),
        ('classifier', model)
    ])

    # --- 5. Entrenamiento ---
    print("⏳ Entrenando el modelo (esto puede tardar unos segundos)...")
    full_pipeline.fit(X_train, y_train)
    print("✅ Entrenamiento completado.")

    # --- 6. Evaluación y Métricas (Requisito Rúbrica) ---
    print("\n--- 📊 Evaluación del Modelo (Set de Prueba) ---")
    y_pred = full_pipeline.predict(X_test)
    y_proba = full_pipeline.predict_proba(X_test)[:, 1] # Probabilidad de clase 1

    # Reporte de clasificación (Precision, Recall, F1)
    print(classification_report(y_test, y_pred))

    # Métrica ROC-AUC (Indica qué tan bueno es separando clases)
    auc = roc_auc_score(y_test, y_proba)
    print(f"🏆 ROC-AUC Score: {auc:.4f}")

    # Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nMatriz de Confusión:")
    print(f"Verdaderos Negativos (Asiste predicho OK): {tn}")
    print(f"Falsos Positivos (Error tipo 1): {fp}")
    print(f"Falsos Negativos (Error grave - No asiste y no avisamos): {fn}")
    print(f"Verdaderos Positivos (No asiste detectado): {tp}")

    # --- 7. Serialización (Guardado del Modelo) ---
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model_pipeline.pkl")
    
    joblib.dump(full_pipeline, model_path)
    print(f"\n💾 Modelo guardado exitosamente en: {model_path}")
    print("Listo para ser usado por la API.")

if __name__ == "__main__":
    train_model()