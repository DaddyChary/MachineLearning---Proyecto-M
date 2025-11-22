import joblib
import os
import sys

def load_model(model_filename="model_pipeline.pkl"):
    """
    Carga el modelo serializado desde la carpeta de modelos.
    
    Args:
        model_filename (str): Nombre del archivo del modelo.
        
    Returns:
        sklearn.pipeline.Pipeline: El modelo entrenado cargado en memoria.
        
    Raises:
        FileNotFoundError: Si el archivo del modelo no existe.
    """
    
    # 1. Construir la ruta absoluta al modelo
    # Esto busca la carpeta 'models' subiendo dos niveles desde 'src/api/'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    model_path = os.path.join(project_root, 'models', model_filename)

    print(f"🔄 Intentando cargar el modelo desde: {model_path}")

    # 2. Verificar existencia del archivo
    if not os.path.exists(model_path):
        error_msg = (
            f"❌ Error Crítico: No se encontró el modelo en '{model_path}'. "
            "Asegúrate de haber ejecutado 'src/modeling/train.py' primero."
        )
        print(error_msg)
        raise FileNotFoundError(error_msg)

    # 3. Cargar el modelo usando joblib
    try:
        model = joblib.load(model_path)
        print("✅ Modelo cargado exitosamente en memoria.")
        return model
    except Exception as e:
        print(f"❌ Error al deserializar el modelo: {e}")
        raise e

if __name__ == "__main__":
    # Bloque de prueba simple para verificar la carga aislada
    try:
        model = load_model()
        print(f"Tipo de objeto cargado: {type(model)}")
    except Exception as e:
        print("La prueba de carga falló.")