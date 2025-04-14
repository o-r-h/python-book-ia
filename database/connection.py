import psycopg2
from psycopg2 import OperationalError
from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD

def create_connection():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        print("¡Conexión exitosa!")
        return conn
    except OperationalError as e:
        print(f"Error al conectar a la base de datos: {e}")
        return None