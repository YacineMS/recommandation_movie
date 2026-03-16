"""
app.py
API FastAPI pour entraîner et servir un modèle de recommandation SVD via MLflow
"""

import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import mlflow.pyfunc
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, HTTPException, Security, status
from fastapi.openapi.utils import get_openapi
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
    OAuth2PasswordBearer,
    OAuth2PasswordRequestForm,
)
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import create_engine
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

import mlflow
from mlflow import MlflowClient
from shared.svd_wrapper import SurpriseSVDWrapper

# -----------------------------
# CONFIG
# -----------------------------
RATING_SCALE = (0.5, 5.0)
CHUNK_SIZE = 30
MODEL_NAME = "svd_model"
MODEL_ARTIFACT_PATH = "svd_model_artifact"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# Charger les variables d'environnement
load_dotenv()

# Configuration JWT
SECRET_KEY = "cle_secrete"  # Remplace par une clé sécurisée et stocke-la dans .env
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


# Modèles Pydantic
class Token(BaseModel):
    access_token: str
    token_type: str
    userid: int


class TokenData(BaseModel):
    username: Optional[str] = None


class User(BaseModel):
    username: str
    disabled: Optional[bool] = None


class UserInDB(User):
    hashed_password: str


DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL non définie")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------
# FASTAPI INITIALIZATION
# -----------------------------
app = FastAPI(title="Training & Serving SVD Model API")

security = HTTPBearer()
app.openapi_schema = None


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # 1️⃣ Declare le Bearer
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}
    }

    # 2️⃣ Lier le Bearer aux routes protégées
    for path in openapi_schema["paths"].values():
        for operation in path.values():
            if "security" not in operation:
                continue
            # Si FastAPI a mis un security vide, on le remplace
            operation["security"] = [{"BearerAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# -----------------------------
# MODELS
# -----------------------------
class DataInsertRequest(BaseModel):
    force_insert: bool = False


# -----------------------------
# DATABASE HELPERS
# -----------------------------
def get_db_engine():
    return create_engine(DATABASE_URL)


def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    return conn


def load_ratings_from_db():
    engine = get_db_engine()
    try:
        return pd.read_sql(
            "SELECT userid, movieid, rating FROM ratings_preprocessed", engine
        )
    finally:
        engine.dispose()


# -----------------------------
# SURPRISE DATASET
# -----------------------------
def prepare_surprise_dataset(ratings: pd.DataFrame):
    reader = Reader(rating_scale=RATING_SCALE)
    return Dataset.load_from_df(ratings[["userid", "movieid", "rating"]], reader)


# -----------------------------
# TRAINING PIPELINE
# -----------------------------
def train_and_evaluate(trainset, testset, params):
    algo = SVD(**params)
    algo.fit(trainset)
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    return algo, rmse, mae


def log_model_and_metrics(algo, params, rmse, mae):
    mlflow.log_params(params)
    mlflow.log_metrics({"rmse": rmse, "mae": mae})

    input_example = pd.DataFrame({"userid": [1], "movieid": [1]})

    mlflow.pyfunc.log_model(
        artifact_path=MODEL_ARTIFACT_PATH,
        python_model=SurpriseSVDWrapper(algo),
        input_example=input_example,
        signature=mlflow.models.infer_signature(input_example, pd.Series([3.5])),
    )


def promote_model(client: MlflowClient, run_id: str, rmse: float, mae: float):
    model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}"

    try:
        client.create_registered_model(MODEL_NAME)
    except Exception:
        pass

    mv = client.create_model_version(name=MODEL_NAME, source=model_uri, run_id=run_id)
    new_version = int(mv.version)

    try:
        prod = client.get_model_version_by_alias(MODEL_NAME, "production")
        prod_run = client.get_run(prod.run_id)
        prod_rmse = prod_run.data.metrics.get("rmse", 9999)
        prod_mae = prod_run.data.metrics.get("mae", 9999)

        if rmse < prod_rmse and mae < prod_mae:
            client.set_registered_model_alias(MODEL_NAME, "production", new_version)
            client.set_registered_model_alias(MODEL_NAME, "staging", int(prod.version))
            stage = "production"
        else:
            client.set_registered_model_alias(MODEL_NAME, "staging", new_version)
            stage = "staging"
    except Exception:
        client.set_registered_model_alias(MODEL_NAME, "production", new_version)
        stage = "production"

    return stage


def train_svd_model():
    logger.info("=== Entraînement SVD ===")

    ratings = load_ratings_from_db()
    data = prepare_surprise_dataset(ratings)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    params = {"n_factors": 50, "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02}

    mlflow.set_experiment("recofilm-svd-recommender")

    with mlflow.start_run() as run:
        algo, rmse, mae = train_and_evaluate(trainset, testset, params)
        log_model_and_metrics(algo, params, rmse, mae)

        client = MlflowClient()
        stage = promote_model(client, run.info.run_id, rmse, mae)

    return {"rmse": rmse, "mae": mae, "run_id": run.info.run_id, "alias": stage}


# -----------------------------
# MODEL LOADING (PRODUCTION)
# -----------------------------
def load_production_model():
    try:
        return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@production")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur chargement modèle MLflow : {e}"
        )


# -----------------------------
# DAILY UPDATE DATA
# -----------------------------
def check_and_update_daily_counts(conn, force_insert=False):
    """Vérifie et met à jour la table daily_counts. Retourne True si une insertion est nécessaire."""
    today = date.today()
    with conn.cursor() as cur:
        # Vérifier si une ligne avec id = 1 existe
        cur.execute("SELECT id, date, count FROM daily_counts WHERE id = 1;")
        result = cur.fetchone()

        if result is None:
            # Insérer une nouvelle ligne avec id = 1
            insert_sql = "INSERT INTO daily_counts (id, date, count) VALUES (1, %s, 0) RETURNING count;"
            cur.execute(insert_sql, (today,))
            count = 0
            needs_insertion = True
        else:
            id, existing_date, count = result
            if force_insert or existing_date < today:
                # Mettre à jour la date et incrémenter le compteur
                update_sql = "UPDATE daily_counts SET date = %s, count = count + 1 WHERE id = 1 RETURNING count;"
                cur.execute(update_sql, (today,))
                count = cur.fetchone()[0]
                needs_insertion = True
            else:
                needs_insertion = False  # Même date, ne pas insérer de données
        conn.commit()
        return needs_insertion, count


def get_csv_file_size(table_name):
    """Retourne le nombre total de lignes dans le fichier CSV."""
    data_dir = Path("data") / "raw_data"
    csv_path = data_dir / f"{table_name}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {csv_path}")

    # Compter le nombre de lignes dans le fichier CSV
    with open(csv_path, "r") as f:
        num_lines = sum(1 for line in f) - 1  # Soustraire 1 pour l'en-tête
    return num_lines


def insert_data_chunk(conn, table_name, count):
    """Insère un chunk de données dans la table spécifiée."""
    start_idx = count * CHUNK_SIZE
    end_idx = start_idx + CHUNK_SIZE

    # Chemin vers les fichiers CSV
    data_dir = Path("data") / "raw_data"
    csv_path = data_dir / f"{table_name}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {csv_path}")

    # Vérifier la taille totale du fichier CSV
    total_lines = get_csv_file_size(table_name)
    if start_idx >= total_lines:
        return 0  # Aucune ligne à insérer

    # Lire le chunk de données
    chunk = pd.read_csv(csv_path, skiprows=range(1, start_idx + 1), nrows=CHUNK_SIZE)

    if chunk.empty:
        return 0

    # Définir les colonnes attendues pour chaque table
    table_columns = {
        "ratings": ["userId", "movieId", "rating", "timestamp"],
        "tags": ["userId", "movieId", "tag", "timestamp"],
        "genome-scores": ["movieId", "tagId", "relevance"],
    }

    if table_name not in table_columns:
        raise ValueError(f"Table inconnue: {table_name}")

    expected_columns = table_columns[table_name]

    # Vérifier que les colonnes attendues existent dans le chunk
    missing_columns = [col for col in expected_columns if col not in chunk.columns]
    if missing_columns:
        raise ValueError(f"Colonnes manquantes dans le fichier CSV: {missing_columns}")

    # Insérer les données dans la table
    with conn.cursor() as cur:
        if table_name == "ratings":
            for _, row in chunk.iterrows():
                cur.execute(
                    "INSERT INTO ratings (userid, movieid, rating, timestamp) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING;",
                    (row["userId"], row["movieId"], row["rating"], row["timestamp"]),
                )
        elif table_name == "tags":
            for _, row in chunk.iterrows():
                cur.execute(
                    "INSERT INTO tags (userid, movieid, tag, timestamp) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING;",
                    (row["userId"], row["movieId"], row["tag"], row["timestamp"]),
                )
        elif table_name == "genome-scores":
            for _, row in chunk.iterrows():
                cur.execute(
                    "INSERT INTO genome_scores (movieid, tagid, relevance) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;",
                    (row["movieId"], row["tagId"], row["relevance"]),
                )

    conn.commit()
    return len(chunk)


# -----------------------------
# API ENDPOINTS
# -----------------------------
@app.post("/training")
def training(_: HTTPAuthorizationCredentials = Security(security)):
    try:
        return train_svd_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur entraînement : {e}")


@app.get("/health")
def health():
    """
    Vérifie que l'API et le modèle MLflow sont opérationnels.
    """
    try:
        conn = get_db_connection()
        conn.cursor().execute("SELECT 1;")
        conn.close()
        return {"status": "healthy", "bdd": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.post("/insert-data")
def insert_data(
    request: DataInsertRequest = Body(...),
    _: HTTPAuthorizationCredentials = Security(security),
):
    """
    Insère un chunk de données dans les tables ratings, tags et genome_scores.
    Vérifie et met à jour daily_counts avant l'insertion.
    """
    conn = None
    try:
        conn = get_db_connection()
        needs_insertion, count = check_and_update_daily_counts(
            conn, request.force_insert
        )

        if not needs_insertion and not request.force_insert:
            return {
                "status": "no_insertion_needed",
                "message": "La date est la même que celle du jour, aucune insertion effectuée.",
                "count": count,
            }

        tables = ["ratings", "tags", "genome-scores"]
        results = {}

        for table in tables:
            try:
                inserted_rows = insert_data_chunk(conn, table, count)
                results[table] = {
                    "inserted_rows": inserted_rows,
                    "start_idx": count * CHUNK_SIZE,
                    "end_idx": count * CHUNK_SIZE + CHUNK_SIZE,
                }
            except Exception as e:
                results[table] = {"error": str(e)}

        return {"status": "success", "count": count, "results": results}
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(
            status_code=500, detail=f"Erreur lors de l'insertion: {str(e)}"
        )
    finally:
        if conn:
            conn.close()


@app.get("/daily-counts")
def get_daily_counts(_: HTTPAuthorizationCredentials = Security(security)):
    """
    Récupère les informations de daily_counts.
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT id, date, count FROM daily_counts WHERE id = 1;")
            result = cur.fetchone()
            if result is None:
                return {"id": 1, "date": None, "count": 0}
            id, date_val, count = result
            return {"id": id, "date": date_val, "count": count}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des daily_counts: {str(e)}",
        )
    finally:
        if conn:
            conn.close()
