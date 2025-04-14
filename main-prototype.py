import pandas as pd
import joblib
import pickle
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

def get_valid_range(df_length):
    """Obtiene un rango válido del usuario"""
    while True:
        try:
            print(f"\n--- Selección de rango de libros (actualmente tenemos {df_length} libros) ---")
            initial = int(input("Ingresa el número inicial del rango: ")) - 1
            final = int(input("Ingresa el número final del rango: "))
            
            if initial < 0 or final <= 0:
                print("Por favor ingresa números positivos.")
                continue
            if initial >= final:
                print("El número inicial debe ser menor que el final.")
                continue
            if final - initial > 20:
                print("Por favor selecciona un rango no mayor a 20 libros.")
                continue
            if final > df_length:
                print(f"El dataset solo tiene {df_length} libros. Por favor ingresa un rango menor.")
                continue
            
            return initial, final
            
        except ValueError:
            print("Por favor ingresa números válidos.")

def select_book(df, start, end):
    """Muestra libros en el rango y permite seleccionar uno"""
    print(f"\n--- Libros disponibles ({start+1}-{end}) ---")
    for i in range(start, end):
        print(f"{i+1}. {df.iloc[i]['title']} (Rating: {df.iloc[i]['rating']})")
    
    while True:
        try:
            selection = int(input(f"\nSelecciona un libro ({start+1}-{end}): "))
            if (start+1) <= selection <= end:
                return selection - 1
            print(f"Por favor ingresa un número entre {start+1} y {end}")
        except ValueError:
            print("Entrada inválida. Por favor ingresa un número.")

# Cargar y preparar los datos
df = pd.read_csv("data/books.csv")
df = df.dropna()
df = df.drop(columns=["price"])
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df = df.dropna(subset=["rating"])


# Preprocesamiento mejorado
# Separar los generos,Siempre que existan datos categoricos, agrupados en una sola linea se deben separar
# Por ejemplo, si hay un libro con generos "Fantasy, Drama, Horror, Mystery, Autor 1, Autor 2, Autor 3, Rating"
# Se debe separar en 8 columnas, 1 por genero, 1 por autor, 1 por rating
df["genres"] = df["genres"].apply(lambda x: x.split(","))

# Crear características de contenido
# HERRAMIENTA: MultiLabelBinarizer es un transformador que convierte una lista de etiquetas en una matriz binaria
#              Usar MultiLabelBinarizer para los generos, 1 por genero y 0 por no genero
# PORQUE USARLO: Se usa porque en este caso cada libro puede tener varios generos, Es ideal para 
#                características categóricas donde un item puede pertenecer a varias categorías
# CUANDO USARLO: Se usa cuando , tienes multiples categorias no excluyentes
#                cuando necesitas representar presencia/ausencia de multiples atributos
#                  
mlb_genres = MultiLabelBinarizer()
genre_matrix = mlb_genres.fit_transform(df["genres"])


# Usar TF-IDF para autores
# HERRAMIENTA: TfidfVectorizer es un transformador que convierte texto en una matriz de frecuencias
#              Usar TfidfVectorizer para los autores, 1 por autor y 0 por no autor
# PORQUE USARLO: Se usa porque en este caso cada libro tiene un autor, Es ideal para 
#                características categóricas donde un item puede pertenecer a una categoría
# CUANDO USARLO: Se usa cuando , tienes una categoria excluyente
#                cuando necesitas representar presencia/ausencia de un atributo
#  Ejemplo:
#       Un solo autor por libro: A diferencia de los géneros, cada libro tiene un único autor
#       Captura importancia relativa: Algunos autores son más únicos/raros que otros
#       Ponderación automática: Los autores muy comunes (que aparecen en muchos libros) tendrán menos peso                
#           Si "J.K. Rowling" aparece en muchos libros, tendrá menos peso que un autor único
#           Esto ayuda a encontrar similitudes más interesantes
tfidf = TfidfVectorizer()
author_matrix = tfidf.fit_transform(df["author"])

# Normalizar rating
# HERRAMIENTA: MinMaxScaler es un transformador que normaliza valores entre 0 y 1
#              Usar MinMaxScaler para el rating, 1 por libro y 0 por no libro
# PORQUE USARLO: Para que todas las características estén en la misma escala
#                Evita que una característica domine a las otras solo por su magnitud
#                El modelo de vecinos más cercanos es sensible a escalas diferente
# CUANDO USARLO: Cuando conoces los límites de tus datos
#                Cuando quieres preservar la distribución original pero en otra escala
#  Ejemplo:
#       Un solo autor por libro: A diferencia de los géneros, cada libro tiene un único autor
scaler = MinMaxScaler()
normalized_rating = scaler.fit_transform(df[["rating"]])

# Convertir rating a matriz dispersa
# HERRAMIENTA: Matriz dispersa, se usa para representar datos sparsos
#              Usar CSR (Compressed Sparse Row) para la matriz dispersa
# PORQUE USARLO: Para representar datos sparsos de manera eficiente
#                Evita usar memoria innecesaria para valores 0
#                Permite operaciones rápidas sobre filas
# CUANDO USARLO: Cuando tienes datos sparsos (muchos 0s)
#  Ejemplo:
#       Las matrices de géneros y autores ya son dispersas (muchos ceros)
#       Para mantener consistencia en el formato
rating_matrix = csr_matrix(normalized_rating)

# Combinar características (convertir todo a CSR para compatibilidad)
# hstack: Combina matrices horizontalmente (añade columnas)
# Tocsr(): Convertir a CSR para mejor soporte de indexación
# Ponderacion: Asigna relevancia a cada característica
#  Ejemplo:
#       Géneros con peso 0.5
#       Autores con peso 0.3
#       Rating con peso 0.2
# 
# PORQUE PONDERAR: Los generos suelen ser mas importantes que las recomendaciones
#                  los autores son importantes pero menos que los generos
#                  el rating es menos importante que los generos y los autores  
csr_matrix(normalized_rating)

# Combinar características (convertir todo a CSR para compatibilidad)
content_features = hstack([
    genre_matrix * 0.5,      # Géneros con peso 0.5
    author_matrix * 0.3,     # Autores con peso 0.3
    rating_matrix * 0.2      # Rating con peso 0.2
]).tocsr()  # Convertir a CSR para mejor soporte de indexación


# Modelo de recomendación basado en contenido
# HERRAMIENTA: NearestNeighbors es un algoritmo de vecinos más cercanos basado en distancia
# Metrica cosine: Mide el angula entre vectores (ideal para datos dispersos)
# Algorithm brute: Busqueda exhaustiva (no es eficiente para grandes datasets) 10.000 a 50.000 elementos si tienes pocos atributos
# Porque usarlo

content_model = NearestNeighbors(metric="cosine", algorithm="brute")
content_model.fit(content_features)

# Interacción con el usuario
while True:
    # Obtener rango del usuario
    initial, final = get_valid_range(len(df))
    
    # Seleccionar libro en el rango
    selected_index = select_book(df, initial, final)
    
    # Mostrar información del libro seleccionado
    selected_book = df.iloc[selected_index]
    print(f"\n📖 Libro seleccionado: {selected_book['title']}")
    print(f"✍️ Autor: {selected_book['author']}")
    print(f"🏷️ Géneros: {', '.join(selected_book['genres'])}")
    print(f"⭐ Rating: {selected_book['rating']}\n")
    
    # Generar recomendaciones dinámicamente
    # Convertir el índice a una matriz CSR de una sola fila
    query = content_features[selected_index:selected_index+1]
    
    distances, indices = content_model.kneighbors(
        query, 
        n_neighbors=6  # 5 recomendaciones + el libro mismo
    )
    
    # Filtrar el libro mismo (primera recomendación)
    recommendations = indices[0][1:]
    
    print("🔍 --- Recomendaciones basadas en contenido ---")
    for i, idx in enumerate(recommendations, 1):
        book = df.iloc[idx]
        print(f"{i}. {book['title']} (Autor: {book['author']}, Rating: {book['rating']})")
    
    # Preguntar si quiere continuar
    continuar = input("\n¿Quieres buscar otro libro? (s/n): ").lower()
    if continuar != 's':
        break

print("\n¡Gracias por usar el sistema de recomendación! 📚")