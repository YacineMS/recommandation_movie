"""
Script de prediction pour generer des recommandations de films
"""

import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import mlflow
from mlflow import MlflowClient


def load_model(model_dir):
    """
    Charge le modele KNN et les movieids

    Args:
        model_dir: Dossier contenant les modeles

    Returns:
        model: Modele KNN
        movie_ids: Array des movieids
    """
    print("\n1. Chargement du modele...")

    client = MlflowClient()
    model_name = "recofilm-knn-recommender"

    champion = client.get_model_version_by_alias(model_name, "champion")
    run_id = champion.run_id

    print("Champion version:", champion.version)
    print("Run id:", run_id)

    model_path = model_dir / "model.pkl"
    ids_path = model_dir / "movie_ids.pkl"

    local_dir = model_dir
    local_dir.mkdir(exist_ok=True)

    mlflow.artifacts.download_artifacts(run_id=run_id, path="", dst_path=str(local_dir))

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(ids_path, "rb") as f:
        movie_ids = pickle.load(f)

    print(f"   Modele charge: {model_path}")
    print(f"   movieids charges: {len(movie_ids):,} films")

    return model, movie_ids


def get_user_profile(user_id, user_matrix_path):
    """
    Recupere le profil d'un utilisateur depuis user_matrix

    Args:
        user_id: ID de l'utilisateur
        user_matrix_path: Chemin vers user_matrix.csv

    Returns:
        user_profile: Profil de l'utilisateur (features)
    """
    print(f"\n2. Recuperation du profil utilisateur {user_id}...")

    user_matrix = pd.read_csv(user_matrix_path)

    user_data = user_matrix[user_matrix["userid"] == user_id]

    if user_data.empty:
        raise ValueError(f"Utilisateur {user_id} non trouve dans user_matrix")

    # Extraire les features (sans userid)
    user_profile = user_data.drop("userid", axis=1).values[0]

    print(f"   Profil trouve:")
    print(
        f"   - Nombre de films aimes: {int(user_data['num_ratings_given'].values[0])}"
    )
    print(f"   - Note moyenne donnee: {user_data['avg_rating_given'].values[0]:.2f}")

    return user_profile


def get_watched_movies(user_id, db_path):
    """
    Recupere la liste des films deja vus par l'utilisateur

    Args:
        user_id: ID de l'utilisateur
        db_path: Chemin vers la base de donnees

    Returns:
        watched_movies: Set des movieids deja vus
    """
    print(f"\n3. Recuperation des films deja vus...")

    conn = sqlite3.connect(db_path)

    query = f"""
        SELECT DISTINCT movieid
        FROM ratings
        WHERE userid = {user_id}
    """

    watched = pd.read_sql_query(query, conn)
    conn.close()

    watched_movies = set(watched["movieid"].values)

    print(f"   Films deja vus: {len(watched_movies)}")

    return watched_movies


def make_predictions(
    model,
    user_profile,
    movie_ids,
    watched_movies,
    movie_matrix_path,
    num_recommendations=10,
):
    """
    Genere des recommandations pour un utilisateur

    Args:
        model: Modele KNN
        user_profile: Profil de l'utilisateur
        movie_ids: Array des movieids
        watched_movies: Set des films deja vus
        movie_matrix_path: Chemin vers movie_matrix.csv
        num_recommendations: Nombre de recommandations

    Returns:
        recommendations: DataFrame des recommandations
    """
    print(f"\n4. Generation des recommandations...")

    # Trouver les films les plus proches du profil utilisateur
    distances, indices = model.kneighbors([user_profile])

    # Recuperer les movieids recommandes
    recommended_movie_ids = movie_ids[indices[0]]

    # Filtrer les films deja vus
    filtered_recommendations = [
        movie_id for movie_id in recommended_movie_ids if movie_id not in watched_movies
    ]

    # Limiter au nombre demande
    top_recommendations = filtered_recommendations[:num_recommendations]

    print(f"   Films proches trouves: {len(recommended_movie_ids)}")
    print(f"   Apres filtrage (films non vus): {len(filtered_recommendations)}")
    print(f"   Top {num_recommendations} recommandations selectionnees")

    # Charger les infos des films pour affichage
    movie_matrix = pd.read_csv(movie_matrix_path)

    # Charger les titres depuis la BDD
    db_path = Path(__file__).parent.parent.parent / "database" / "recofilm.db"
    conn = sqlite3.connect(db_path)

    recommendations = []
    for movie_id in top_recommendations:
        query = f"SELECT title, genres FROM movies WHERE movieid = {movie_id}"
        movie_info = pd.read_sql_query(query, conn)

        if not movie_info.empty:
            movie_row = movie_matrix[movie_matrix["movieid"] == movie_id]

            recommendations.append(
                {
                    "movieid": movie_id,
                    "title": movie_info["title"].values[0],
                    "genres": movie_info["genres"].values[0],
                    "avg_rating": (
                        movie_row["avg_rating"].values[0] if not movie_row.empty else 0
                    ),
                    "num_ratings": (
                        int(movie_row["num_ratings"].values[0])
                        if not movie_row.empty
                        else 0
                    ),
                }
            )

    conn.close()

    return pd.DataFrame(recommendations)


def display_recommendations(recommendations):
    """
    Affiche les recommandations de maniere lisible

    Args:
        recommendations: DataFrame des recommandations
    """
    print(f"\n5. RECOMMANDATIONS:")
    print("=" * 80)

    for idx, row in recommendations.iterrows():
        print(f"\n{idx + 1}. {row['title']}")
        print(f"   Genres: {row['genres']}")
        print(
            f"   Note moyenne: {row['avg_rating']:.2f} ({row['num_ratings']} ratings)"
        )


def main():
    """
    Fonction principale de prediction
    """
    print("=" * 60)
    print("PREDICTIONS ET RECOMMANDATIONS DE FILMS")
    print("=" * 60)

    # Chemins
    BASE_DIR = Path(__file__).resolve().parent

    db_path = BASE_DIR / "database" / "recofilm.db"
    MOVIE_MATRIX_PATH = BASE_DIR / "movie_matrix.csv"
    USER_MATRIX_PATH = BASE_DIR / "user_matrix.csv"

    user_matrix_path = USER_MATRIX_PATH
    movie_matrix_path = MOVIE_MATRIX_PATH
    model_dir = BASE_DIR / "models"

    # Utilisateur de test (vous pouvez changer)
    test_user_id = 1
    num_recommendations = 10

    print(f"\nUtilisateur de test: {test_user_id}")
    print(f"Nombre de recommandations: {num_recommendations}")

    try:
        # Charger le modele
        model, movie_ids = load_model(model_dir)

        # Recuperer le profil utilisateur
        user_profile = get_user_profile(test_user_id, user_matrix_path)

        # Recuperer les films deja vus
        watched_movies = get_watched_movies(test_user_id, db_path)

        # Generer les recommandations
        recommendations = make_predictions(
            model,
            user_profile,
            movie_ids,
            watched_movies,
            movie_matrix_path,
            num_recommendations,
        )

        # Afficher
        display_recommendations(recommendations)

        print("\n" + "=" * 60)
        print("PREDICTIONS TERMINEES AVEC SUCCES")
        print("=" * 60)

    except Exception as e:
        print(f"\nERREUR lors de la prediction: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
