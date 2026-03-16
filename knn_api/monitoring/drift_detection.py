"""
Script de détection de data drift avec Evidently
Compare les données de référence (CSV) avec les données courantes (Supabase)
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from config import get_connection
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report


def load_reference_data():
    """
    Charge les données de référence depuis les CSV

    Returns:
        movie_matrix_ref: DataFrame des films de référence
        user_matrix_ref: DataFrame des utilisateurs de référence
    """
    print("\n1. Chargement des données de référence (CSV)...")

    project_root = Path(__file__).resolve().parent
    movie_matrix_path = project_root / ".." / "api" / "movie_matrix.csv"
    user_matrix_path = project_root / ".." / "api" / "user_matrix.csv"

    if not movie_matrix_path.exists():
        raise FileNotFoundError(f"Movie matrix non trouvée: {movie_matrix_path}")

    if not user_matrix_path.exists():
        raise FileNotFoundError(f"User matrix non trouvée: {user_matrix_path}")

    movie_matrix_ref = pd.read_csv(movie_matrix_path)
    user_matrix_ref = pd.read_csv(user_matrix_path)

    print(f"   Movies: {len(movie_matrix_ref):,} films")
    print(f"   Users: {len(user_matrix_ref):,} utilisateurs")

    return movie_matrix_ref, user_matrix_ref


def load_current_data_from_supabase():
    """
    Charge les données courantes depuis Supabase et reconstruit les matrices

    Returns:
        movie_matrix_current: DataFrame des films courants
        user_matrix_current: DataFrame des utilisateurs courants
    """
    print("\n2. Chargement des données courantes (Supabase)...")

    conn = get_connection()

    # Récupérer les ratings actuels
    print("   Récupération des ratings...")
    ratings_query = "SELECT userid, movieid, rating FROM ratings"
    ratings = pd.read_sql_query(ratings_query, conn)

    # Renommer les colonnes en camelCase
    ratings.columns = ["userid", "movieid", "rating"]
    print(f"   Ratings chargés: {len(ratings):,}")

    # Récupérer les infos des films
    print("   Récupération des films...")
    movies_query = "SELECT movieid, title, genres FROM movies"
    movies = pd.read_sql_query(movies_query, conn)

    # Renommer les colonnes en camelCase
    movies.columns = ["movieid", "title", "genres"]
    print(f"   Films chargés: {len(movies):,}")

    conn.close()

    # Reconstruire movie_matrix
    print("\n3. Reconstruction de movie_matrix...")
    movie_matrix_current = reconstruct_movie_matrix(ratings, movies)

    # Reconstruire user_matrix
    print("\n4. Reconstruction de user_matrix...")
    user_matrix_current = reconstruct_user_matrix(ratings, movies)

    return movie_matrix_current, user_matrix_current


def reconstruct_movie_matrix(ratings, movies):
    """
    Reconstruit la movie_matrix à partir des ratings actuels

    Args:
        ratings: DataFrame des ratings
        movies: DataFrame des films

    Returns:
        movie_matrix: DataFrame reconstruit
    """
    # Calculer les stats par film
    movie_stats = (
        ratings.groupby("movieid").agg({"rating": ["mean", "count"]}).reset_index()
    )

    movie_stats.columns = ["movieid", "avg_rating", "num_ratings"]

    # Parser les genres
    movies_copy = movies.copy()
    genres_list = [
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "IMAX",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]

    for genre in genres_list:
        movies_copy[genre] = (
            movies_copy["genres"].str.contains(genre, na=False).astype(int)
        )

    # Merger avec les stats
    movie_matrix = movies_copy[["movieid"] + genres_list].merge(
        movie_stats, on="movieid", how="left"
    )

    # Remplir les NaN
    movie_matrix["avg_rating"] = movie_matrix["avg_rating"].fillna(0)
    movie_matrix["num_ratings"] = movie_matrix["num_ratings"].fillna(0)

    print(f"   Movie matrix reconstruite: {len(movie_matrix)} films")

    return movie_matrix


def reconstruct_user_matrix(ratings, movies):
    """
    Reconstruit la user_matrix à partir des ratings actuels
    Inclut les proportions de genres comme la matrice de référence
    """
    genres_list = [
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "IMAX",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]

    # Ajouter les colonnes de genres aux ratings
    movies_copy = movies.copy()
    for genre in genres_list:
        movies_copy[genre] = (
            movies_copy["genres"].str.contains(genre, na=False).astype(int)
        )

    # Joindre ratings avec les genres des films
    ratings_with_genres = ratings.merge(
        movies_copy[["movieid"] + genres_list], on="movieid", how="left"
    )

    # Calculer les stats de base par utilisateur
    user_stats = (
        ratings.groupby("userid").agg({"rating": ["mean", "count"]}).reset_index()
    )
    user_stats.columns = ["userid", "avg_rating_given", "num_ratings_given"]

    # Calculer les proportions de genres par utilisateur
    genre_props = (
        ratings_with_genres.groupby("userid")[genres_list].mean().reset_index()
    )

    # Merger tout ensemble
    user_matrix = user_stats.merge(genre_props, on="userid", how="left")

    # Réordonner les colonnes pour correspondre à la référence
    cols = ["userid"] + genres_list + ["avg_rating_given", "num_ratings_given"]
    user_matrix = user_matrix[cols]

    print(f"   User matrix reconstruite: {len(user_matrix)} utilisateurs")

    return user_matrix


def generate_drift_report(reference_data, current_data, output_path, report_name):
    """
    Génère un rapport de drift avec Evidently

    Args:
        reference_data: DataFrame de référence
        current_data: DataFrame courant
        output_path: Chemin de sortie du rapport
        report_name: Nom du rapport
    """
    print(f"\n5. Génération du rapport de drift: {report_name}...")

    # Créer le rapport
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])

    # Exécuter le rapport
    report.run(reference_data=reference_data, current_data=current_data)

    # Sauvegarder en HTML
    report_file = (
        output_path / f"{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    )
    report.save_html(str(report_file))

    print(f"   Rapport sauvegardé: {report_file}")

    return report_file


def main():
    """
    Fonction principale de détection de drift
    """
    print("=" * 60)
    print("DÉTECTION DE DATA DRIFT AVEC EVIDENTLY")
    print("=" * 60)

    try:
        # Créer le dossier de sortie pour les rapports
        project_root = Path(__file__).parent.parent.parent
        output_path = project_root / "reports" / "drift"
        output_path.mkdir(parents=True, exist_ok=True)

        # Charger les données de référence
        movie_matrix_ref, user_matrix_ref = load_reference_data()

        # Charger les données courantes
        movie_matrix_current, user_matrix_current = load_current_data_from_supabase()

        # Générer le rapport pour movie_matrix
        print("\n" + "=" * 60)
        print("ANALYSE DU DRIFT - MOVIE MATRIX")
        print("=" * 60)

        # Sélectionner uniquement les colonnes communes
        common_movie_cols = list(
            set(movie_matrix_ref.columns) & set(movie_matrix_current.columns)
        )
        common_movie_cols = [col for col in common_movie_cols if col != "movieid"]

        movie_report = generate_drift_report(
            movie_matrix_ref[common_movie_cols],
            movie_matrix_current[common_movie_cols],
            output_path,
            "movie_drift_report",
        )

        # Générer le rapport pour user_matrix
        print("\n" + "=" * 60)
        print("ANALYSE DU DRIFT - USER MATRIX")
        print("=" * 60)

        # Sélectionner uniquement les colonnes communes
        common_user_cols = list(
            set(user_matrix_ref.columns) & set(user_matrix_current.columns)
        )
        common_user_cols = [col for col in common_user_cols if col != "userid"]

        user_report = generate_drift_report(
            user_matrix_ref[common_user_cols],
            user_matrix_current[common_user_cols],
            output_path,
            "user_drift_report",
        )

        print("\n" + "=" * 60)
        print("DÉTECTION DE DRIFT TERMINÉE AVEC SUCCÈS")
        print("=" * 60)
        print(f"\nRapports générés:")
        print(f"  - Movie drift: {movie_report}")
        print(f"  - User drift: {user_report}")
        print(
            f"\nOuvrez ces fichiers HTML dans votre navigateur pour voir les résultats!"
        )

    except Exception as e:
        print(f"\nERREUR lors de la détection de drift: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
