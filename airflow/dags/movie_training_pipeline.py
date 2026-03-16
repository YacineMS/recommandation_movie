import logging
import os
from datetime import datetime, timedelta

import requests
from airflow.providers.standard.operators.python import PythonOperator
from dotenv import load_dotenv

from airflow import DAG

load_dotenv()

# Configuration
TRAINER_API_URL = "http://movie_trainer_api:8000"
FASTAPI_KNN_URL = "http://knn_api:8000"

# Récupérer le token depuis les variables d'environnement
API_KNN_TOKEN = os.getenv("API_KNN_TOKEN")
if not API_KNN_TOKEN:
    raise ValueError(
        "API_KNN_TOKEN n'est pas défini dans les variables d'environnement"
    )

# Headers pour l'authentification Bearer
AUTH_HEADERS = {
    "Authorization": f"Bearer {API_KNN_TOKEN}",
    "Content-Type": "application/json",
}

# Credentials pour l'API KNN
KNN_USERNAME = "admin"
KNN_PASSWORD = "RecoFilm!2025"

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "movie_training_pipeline",
    default_args=default_args,
    description="Pipeline de collecte et training",
    schedule="@daily",
    catchup=False,
    tags=["ml", "movies"],
)


def insert_data():
    """Insère un chunk de données (ratings, tags, genome_scores)"""
    logging.info("🔍 Insertion de données...")

    response = requests.post(
        f"{TRAINER_API_URL}/insert-data",
        json={"force_insert": False},  # force_insert à False par défaut
        headers=AUTH_HEADERS,
        timeout=60,
    )
    response.raise_for_status()

    data = response.json()

    if data.get("status") == "no_insertion_needed":
        logging.info(f"⏭️  Aucune insertion nécessaire : {data.get('message')}")
    else:
        logging.info(f"✅ Données insérées : {data}")
        logging.info(
            f"   - Ratings: {data['results']['ratings']['inserted_rows']} lignes"
        )
        logging.info(f"   - Tags: {data['results']['tags']['inserted_rows']} lignes")
        logging.info(
            f"   - Genome-scores: {data['results']['genome-scores']['inserted_rows']} lignes"
        )

    return data


def trigger_training():
    """Déclenche le training du modèle SVD"""
    logging.info("🚀 Déclenchement du training SVD...")

    response = requests.post(
        f"{TRAINER_API_URL}/training",
        json={"model_type": "svd", "params": {"n_factors": 100, "n_epochs": 20}},
        headers=AUTH_HEADERS,
        timeout=600,
    )
    response.raise_for_status()

    result = response.json()
    logging.info(f"✅ Training SVD terminé : {result}")
    return result


def trigger_training_knn():
    """Déclenche le training du modèle KNN"""
    logging.info("🔐 Obtention du token pour l'API KNN...")

    # Obtenir le token
    token_response = requests.post(
        f"{FASTAPI_KNN_URL}/token",
        data={"username": KNN_USERNAME, "password": KNN_PASSWORD},
        timeout=30,
    )
    token_response.raise_for_status()

    token_data = token_response.json()
    access_token = token_data.get("access_token")

    if not access_token:
        raise ValueError("Token d'accès non reçu de l'API KNN")

    logging.info("✅ Token obtenu avec succès")

    # Headers avec le token
    knn_headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    logging.info("🚀 Déclenchement du training KNN...")

    # Appeler l'endpoint de training
    response = requests.post(
        f"{FASTAPI_KNN_URL}/training",
        headers=knn_headers,
        timeout=600,
    )
    response.raise_for_status()

    result = response.json()
    logging.info(f"✅ Training KNN terminé : {result}")
    return result


# Définition des tâches
task_insert = PythonOperator(
    task_id="insert_data",
    python_callable=insert_data,
    dag=dag,
)

task_train = PythonOperator(
    task_id="trigger_training",
    python_callable=trigger_training,
    dag=dag,
)

task_train_knn = PythonOperator(
    task_id="trigger_training_knn",
    python_callable=trigger_training_knn,
    dag=dag,
)

# Ordre d'exécution
# Après insert_data, les deux trainings s'exécutent en parallèle
task_insert >> [task_train, task_train_knn]
