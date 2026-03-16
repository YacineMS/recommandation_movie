"""
Script d'entrainement du modele de recommandation (KNN) avec MLflow + Model Registry (Aliases)
"""

import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.neighbors import NearestNeighbors

import mlflow

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def train_model(
    movie_matrix_path, n_neighbors=20, algorithm="ball_tree", metric="euclidean"
):
    """
    Entraine un modele KNN sur la movie_matrix

    Args:
        movie_matrix_path: Chemin vers movie_matrix.csv
        n_neighbors: Nombre de voisins (K)
        algorithm: Algorithme KNN ('ball_tree', 'kd_tree', 'brute', 'auto')
        metric: Metrique de distance ('euclidean', 'manhattan', etc.)

    Returns:
        model: Modele KNN entraine
        movie_ids: Liste des movieids correspondant aux indices du modele
        metrics: Dictionnaire des metriques
    """
    print("\n1. Chargement de movie_matrix...")
    movie_matrix = pd.read_csv(movie_matrix_path)
    print(f"   Nombre de films: {len(movie_matrix):,}")
    print(f"   Nombre de features: {movie_matrix.shape[1] - 1}")

    # Separer les movieids des features
    movie_ids = movie_matrix["movieid"].values
    features = movie_matrix.drop("movieid", axis=1)

    print(f"\n2. Entrainement du modele KNN...")
    print(f"   Algorithme: {algorithm}")
    print(f"   Nombre de voisins: {n_neighbors}")
    print(f"   Metrique: {metric}")

    # Mesurer le temps d'entrainement
    start_time = time.time()

    # Entrainer le modele KNN
    model = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm=algorithm, metric=metric
    )
    model.fit(features)

    training_time = time.time() - start_time

    print(f"   Modele entraine en {training_time:.2f} secondes!")

    # Metriques
    metrics = {
        "training_time": training_time,
        "n_samples": len(movie_matrix),
        "n_features": features.shape[1],
    }

    return model, movie_ids, metrics


def save_model(model, movie_ids, output_dir):
    """
    Sauvegarde le modele et les movieids

    Args:
        model: Modele KNN entraine
        movie_ids: Liste des movieids
        output_dir: Dossier de sortie
    """
    print(f"\n3. Sauvegarde du modele...")

    # Creer le dossier s'il n'existe pas
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegarder le modele
    model_path = output_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    model_size = model_path.stat().st_size / 1024
    print(f"   Modele sauvegarde: {model_path}")
    print(f"   Taille: {model_size:.2f} KB")

    # Sauvegarder les movieids (pour retrouver les films plus tard)
    ids_path = output_dir / "movie_ids.pkl"
    with open(ids_path, "wb") as f:
        pickle.dump(movie_ids, f)

    print(f"   movieids sauvegardes: {ids_path}")

    return model_size


def test_model(model, movie_ids, movie_matrix_path):
    """
    Test rapide du modele avec un film exemple

    Args:
        model: Modele KNN entraine
        movie_ids: Liste des movieids
        movie_matrix_path: Chemin vers movie_matrix.csv

    Returns:
        test_distance: Distance moyenne des 5 plus proches voisins
    """
    print(f"\n4. Test du modele...")

    # Charger movie_matrix pour avoir les titres
    movie_matrix = pd.read_csv(movie_matrix_path)

    # Tester avec le premier film
    test_film_idx = 0
    test_film_id = movie_ids[test_film_idx]

    # Trouver les films similaires
    features = movie_matrix.drop("movieid", axis=1)
    distances, indices = model.kneighbors([features.iloc[test_film_idx]])

    # Calculer la distance moyenne (metrique de qualite)
    avg_distance = distances[0][1:6].mean()

    # Afficher les resultats
    print(f"\n   Film de test: movieid={test_film_id}")
    print(f"   Films similaires trouves:")

    for i, (dist, idx) in enumerate(zip(distances[0][1:6], indices[0][1:6])):
        similar_movie_id = movie_ids[idx]
        print(f"     {i+1}. movieid={similar_movie_id} (distance={dist:.4f})")

    print(f"\n   Distance moyenne: {avg_distance:.4f}")
    print(f"   Le modele fonctionne correctement!")

    return avg_distance


def register_model(model_name, run_id, avg_distance):
    """
    Enregistre le modele dans MLflow Model Registry

    Args:
        model_name: Nom du modele dans le registry
        run_id: ID du run MLflow
        avg_distance: Metrique de qualite du modele

    Returns:
        model_version: Version du modele enregistre
    """
    print(f"\n5. Enregistrement dans MLflow Model Registry...")

    client = MlflowClient()

    # Verifier si le modele existe deja dans le registry
    try:
        client.get_registered_model(model_name)
        print(f"   Modele '{model_name}' existe deja dans le registry")
    except:
        # Creer le modele dans le registry
        client.create_registered_model(
            model_name,
            description="Modele KNN pour recommandation de films (MovieLens 10M)",
        )
        print(f"   Nouveau modele '{model_name}' cree dans le registry")

    # Enregistrer cette version du modele
    model_uri = f"runs:/{run_id}/model"
    model_version = mlflow.register_model(model_uri, model_name)

    print(f"   Version enregistree: {model_version.version}")

    # Ajouter une description a cette version
    client.update_model_version(
        name=model_name,
        version=model_version.version,
        description=f"KNN model - avg_distance={avg_distance:.4f}",
    )

    return model_version


def compare_and_promote(model_name, current_version, avg_distance):
    """
    Compare avec le modele champion et promeut si meilleur (utilise les Aliases)

    Args:
        model_name: Nom du modele dans le registry
        current_version: Version actuelle du modele
        avg_distance: Metrique du modele actuel
    """
    print(f"\n6. Comparaison avec le modele champion...")

    client = MlflowClient()

    # Chercher le modele actuellement "champion" (via alias)
    try:
        champion_version = client.get_model_version_by_alias(model_name, "champion")

        print(f"   Modele champion actuel: version {champion_version.version}")

        # Recuperer la metrique du champion
        champion_run = client.get_run(champion_version.run_id)
        champion_distance = champion_run.data.metrics.get(
            "avg_test_distance", float("inf")
        )

        print(f"   Distance champion: {champion_distance:.4f}")
        print(f"   Distance nouveau: {avg_distance:.4f}")

        # Plus la distance est faible, meilleur est le modele
        if avg_distance < champion_distance:
            print(f"   [OK] Nouveau modele MEILLEUR! Promotion en champion")

            # Retirer l'alias du champion actuel
            client.delete_model_version_alias(model_name, "champion")

            # Donner l'alias champion au nouveau modele
            client.set_registered_model_alias(model_name, "champion", current_version)

            print(f"   [OK] Version {current_version} promue champion!")

        else:
            print(f"   [INFO] Nouveau modele moins bon que le champion")
            print(f"   Version {current_version} reste sans alias")

    except Exception as e:
        # Aucun champion actuel
        print("   Aucun modele champion actuellement")
        print(f"   Promotion de la version {current_version} en champion")

        # Donner l'alias champion
        client.set_registered_model_alias(model_name, "champion", current_version)

        print("   [OK] Version promue champion!")


def main():
    """
    Fonction principale d'entrainement avec MLflow + Model Registry (Aliases)
    """
    print("=" * 60)
    print("ENTRAINEMENT + MLFLOW MODEL REGISTRY (ALIASES)")
    print("=" * 60)

    # Chemins
    BASE_DIR = Path(__file__).resolve().parent
    MOVIE_MATRIX_PATH = BASE_DIR / "movie_matrix.csv"
    movie_matrix_path = MOVIE_MATRIX_PATH
    output_dir = BASE_DIR / "models"

    # Configuration MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("recofilm-knn-recommender")

    # Nom du modele dans le registry
    MODEL_NAME = "recofilm-knn-recommender"

    # Verifier que movie_matrix existe
    if not movie_matrix_path.exists():
        print(f"\nERREUR: {movie_matrix_path} n'existe pas!")
        print("Executez d'abord: python src/data/preprocess.py")
        return

    # Parametres du modele (tu peux les modifier pour tester)
    params = {"n_neighbors": 20, "algorithm": "ball_tree", "metric": "euclidean"}

    try:
        # Demarrer un run MLflow
        with mlflow.start_run(
            run_name=f"knn-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        ) as run:
            run_id = run.info.run_id

            # Logger les parametres
            mlflow.log_params(params)

            print("\n[MLflow] Run demarre")
            print(f"   Experiment: recofilm-knn-recommender")
            print(f"   Run ID: {run_id}")

            # Entrainer le modele
            model, movie_ids, metrics = train_model(
                movie_matrix_path,
                n_neighbors=params["n_neighbors"],
                algorithm=params["algorithm"],
                metric=params["metric"],
            )

            # Sauvegarder
            model_size = save_model(model, movie_ids, output_dir)

            # Tester
            avg_distance = test_model(model, movie_ids, movie_matrix_path)

            # Logger les metriques
            mlflow.log_metric("training_time_seconds", metrics["training_time"])
            mlflow.log_metric("n_samples", metrics["n_samples"])
            mlflow.log_metric("n_features", metrics["n_features"])
            mlflow.log_metric("model_size_kb", model_size)
            mlflow.log_metric("avg_test_distance", avg_distance)

            # Logger le modele dans MLflow
            mlflow.sklearn.log_model(model, "model")

            # Logger les artifacts (fichiers)
            mlflow.log_artifact(str(output_dir / "model.pkl"))
            mlflow.log_artifact(str(output_dir / "movie_ids.pkl"))

            print("\n[MLflow] Metriques loggees:")
            print(f"   - Training time: {metrics['training_time']:.2f}s")
            print(f"   - Samples: {metrics['n_samples']:,}")
            print(f"   - Features: {metrics['n_features']}")
            print(f"   - Model size: {model_size:.2f} KB")
            print(f"   - Avg distance: {avg_distance:.4f}")

        # APRES la fin du run, enregistrer dans le Model Registry
        model_version = register_model(MODEL_NAME, run_id, avg_distance)

        # Comparer et promouvoir si meilleur
        compare_and_promote(MODEL_NAME, model_version.version, avg_distance)

        print("\n" + "=" * 60)
        print("ENTRAINEMENT TERMINE AVEC SUCCES")
        print("=" * 60)
        print(f"\n[OK] Modele enregistre: {MODEL_NAME} v{model_version.version}")
        print(f"[INFO] Pour voir le Model Registry:")
        print(f"   mlflow ui")
        print(f"   Puis aller dans 'Models' -> Colonne 'Aliases'")
        print(f"   Le modele champion aura l'alias 'champion'")

    except Exception as e:
        print(f"\nERREUR lors de l'entrainement: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
