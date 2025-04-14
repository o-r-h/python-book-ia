from database.connection import create_connection
from database.models import Usuario

def get_all_users():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email FROM users")
    users = cursor.fetchall()
    cursor.close()
    conn.close()
    return [Usuario(id=row[0], name=row[1], email=row[2]) for row in users]
    #return users

def insert_user(name, age):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (name, email) VALUES (%s, %s)",
        (name, age)
    )
    conn.commit()
    cursor.close()
    conn.close()

def find_user_by_email(email):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return Usuario(id=user[0], name=user[1], email=user[2]) if user else None   
    #return user

