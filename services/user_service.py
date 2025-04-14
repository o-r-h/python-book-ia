from database.queries_user import get_all_users, insert_user

def get_users():
    return get_all_users()

def create_user(name, email):
    insert_user(name, email)
    print(f"Usuario {name} creado exitosamente.")