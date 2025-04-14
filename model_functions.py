import joblib
import pandas as pd
from pathlib import Path

def guardar_modelo(modelo, vectorizadores, scaler, df, ruta='modelos_guardados'):
    """
    Guarda todos los componentes del sistema de recomendación
    
    Args:
        modelo: Modelo NearestNeighbors entrenado
        vectorizadores: Diccionario con los vectorizadores (mlb_genres, tfidf)
        scaler: Objeto MinMaxScaler
        df: DataFrame con los datos originales
        ruta: Carpeta donde se guardarán los archivos
    """
    # Crear directorio si no existe
    Path(ruta).mkdir(parents=True, exist_ok=True)
    
    # Guardar cada componente
    joblib.dump(modelo, f'{ruta}/nearest_neighbors_model.joblib')
    joblib.dump(vectorizadores['mlb_genres'], f'{ruta}/mlb_genres.joblib')
    joblib.dump(vectorizadores['tfidf'], f'{ruta}/tfidf_author.joblib')
    joblib.dump(scaler, f'{ruta}/rating_scaler.joblib')
    
    # Guardar el DataFrame (usamos pickle para pandas)
    df.to_pickle(f'{ruta}/books_dataframe.pkl')
    
    print(f"Modelo y componentes guardados en la carpeta '{ruta}'")

def cargar_modelo(ruta='modelos_guardados'):
    """
    Carga todos los componentes del sistema de recomendación guardados
    
    Args:
        ruta: Carpeta donde están los archivos guardados
    
    Returns:
        Tupla con (modelo, vectorizadores, scaler, df)
    """
    try:
        modelo = joblib.load(f'{ruta}/nearest_neighbors_model.joblib')
        mlb_genres = joblib.load(f'{ruta}/mlb_genres.joblib')
        tfidf = joblib.load(f'{ruta}/tfidf_author.joblib')
        scaler = joblib.load(f'{ruta}/rating_scaler.joblib')
        df = pd.read_pickle(f'{ruta}/books_dataframe.pkl')
        
        print("Modelo y componentes cargados exitosamente")
        return modelo, {'mlb_genres': mlb_genres, 'tfidf': tfidf}, scaler, df
    
    except FileNotFoundError:
        print("No se encontraron archivos del modelo. Debes entrenar primero.")
        return None, None, None, None