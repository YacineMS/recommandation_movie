"""
Configuration pour la connexion PostgreSQL
Version avec force IPv4 pour Docker
"""

import os
import socket

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_batch
from sqlalchemy import create_engine

load_dotenv()


DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL non définie")


def get_db_engine():
    return create_engine(DATABASE_URL)


def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    return conn


def get_connection():
    """
    Retourne une connexion PostgreSQL
    Force IPv4 pour éviter les problèmes de réseau dans Docker
    """
    db_host = os.getenv("DB_HOST")

    # Résoudre le hostname en IPv4 seulement
    try:
        ipv4_address = socket.getaddrinfo(db_host, None, socket.AF_INET)[  # Force IPv4
            0
        ][4][0]
        print(f"[INFO] Résolution DNS: {db_host} -> {ipv4_address}")
    except Exception as e:
        print(f"[WARNING] Erreur résolution DNS: {e}, utilisation du hostname")
        ipv4_address = db_host

    return psycopg2.connect(
        host=ipv4_address,  # Utilise l'adresse IPv4 résolue
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def test_connection():
    """
    Test la connexion PostgreSQL
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"Connexion PostgreSQL reussie!")
        print(f"Version: {version[0]}")
        conn.close()
        return True
    except Exception as e:
        print(f"Erreur de connexion: {e}")
        return False


if __name__ == "__main__":
    test_connection()
