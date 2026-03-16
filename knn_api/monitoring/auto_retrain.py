"""
Script de réentraînement automatique basé sur la détection de drift
Lance un réentraînement si le drift dépasse un seuil défini
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from drift_detection import (
    load_current_data_from_supabase,
    load_reference_data,
    reconstruct_movie_matrix,
    reconstruct_user_matrix,
)
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report


def calculate_drift_score(reference_data, current_data):
    """
    Calcule le score de drift entre deux datasets

    Args:
        reference_data: DataFrame de référence
        current_data: DataFrame courant

    Returns:
        drift_score: Score de drift (0-1)
        drift_detected: Boolean indiquant si drift détecté
    """
    print("\n Calcul du score de drift...")

    # Créer le rapport
    report = Report(metrics=[DataDriftPreset()])

    # Exécuter le rapport
    report.run(reference_data=reference_data, current_data=current_data)

    # Extraire les résultats
    results = report.as_dict()

    # Récupérer le score de drift
    dataset_drift = results["metrics"][0]["result"]["dataset_drift"]
    drift_share = results["metrics"][0]["result"]["share_of_drifted_columns"]

    print(f"   Dataset drift détecté: {dataset_drift}")
    print(f"   Part des colonnes avec drift: {drift_share:.2%}")

    return drift_share, dataset_drift


def should_retrain(movie_drift_score, user_drift_score, threshold=0.3):
    """
    Décide si un réentraînement est nécessaire

    Args:
        movie_drift_score: Score de drift pour les films
        user_drift_score: Score de drift pour les utilisateurs
        threshold: Seuil de déclenchement (défaut: 0.3 = 30%)

    Returns:
        should_retrain: Boolean
        reason: Raison de la décision
    """
    print(f"\n Évaluation du besoin de réentraînement (seuil: {threshold:.0%})...")

    reasons = []

    if movie_drift_score > threshold:
        reasons.append(f"Drift films: {movie_drift_score:.2%} > {threshold:.0%}")

    if user_drift_score > threshold:
        reasons.append(f"Drift utilisateurs: {user_drift_score:.2%} > {threshold:.0%}")

    if reasons:
        print(f"    Réentraînement NÉCESSAIRE:")
        for reason in reasons:
            print(f"      - {reason}")
        return True, " | ".join(reasons)
    else:
        print(f"    Réentraînement NON nécessaire:")
        print(f"      - Drift films: {movie_drift_score:.2%}")
        print(f"      - Drift utilisateurs: {user_drift_score:.2%}")
        return False, "Drift en dessous du seuil"


def trigger_retraining():
    """
    Lance le script d'entraînement du modèle

    Returns:
        success: Boolean
        message: Message de résultat
    """
    print("\n Lancement du réentraînement du modèle...")
    project_root = Path(__file__).resolve().parent.parent
    train_script = project_root / "api" / "train_model.py"

    if not train_script.exists():
        return False, f"Script d'entraînement non trouvé: {train_script}"

    try:
        # Lancer le script d'entraînement
        result = subprocess.run(
            [sys.executable, str(train_script)],
            capture_output=True,
            text=True,
            cwd=str(project_root),
        )

        if result.returncode == 0:
            print("    Entraînement terminé avec succès!")
            return True, "Modèle réentraîné avec succès"
        else:
            print(f"    Erreur lors de l'entraînement:")
            print(result.stderr)
            return False, f"Erreur: {result.stderr}"

    except Exception as e:
        print(f"    Exception lors de l'entraînement: {e}")
        return False, f"Exception: {str(e)}"


def log_retrain_decision(drift_scores, decision, reason, retrain_success=None):
    """
    Log la décision de réentraînement dans un fichier JSON

    Args:
        drift_scores: Dict des scores de drift
        decision: Boolean - réentraînement décidé ou non
        reason: Raison de la décision
        retrain_success: Boolean - succès du réentraînement (si lancé)
    """
    project_root = Path(__file__).parent.parent.parent
    log_dir = project_root / "logs" / "retrain"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "drift_scores": drift_scores,
        "retrain_decision": decision,
        "reason": reason,
        "retrain_success": retrain_success,
    }

    log_file = log_dir / f"retrain_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(log_file, "w") as f:
        json.dump(log_entry, f, indent=2)

    print(f"\n Décision loggée: {log_file}")


def main():
    """
    Fonction principale du réentraînement automatique
    """
    print("=" * 60)
    print("RÉENTRAÎNEMENT AUTOMATIQUE BASÉ SUR LA DÉTECTION DE DRIFT")
    print("=" * 60)

    try:
        # 1. Charger les données de référence
        print("\n Étape 1/5: Chargement des données de référence...")
        movie_matrix_ref, user_matrix_ref = load_reference_data()

        # 2. Charger les données courantes
        print("\n Étape 2/5: Chargement des données courantes...")
        movie_matrix_current, user_matrix_current = load_current_data_from_supabase()

        # 3. Calculer les scores de drift
        print("\n Étape 3/5: Calcul des scores de drift...")

        # Drift sur les films
        print("\n   Analyse drift FILMS:")
        common_movie_cols = list(
            set(movie_matrix_ref.columns) & set(movie_matrix_current.columns)
        )
        common_movie_cols = [col for col in common_movie_cols if col != "movieid"]

        movie_drift_score, movie_drift_detected = calculate_drift_score(
            movie_matrix_ref[common_movie_cols], movie_matrix_current[common_movie_cols]
        )

        # Drift sur les utilisateurs
        print("\n   Analyse drift UTILISATEURS:")
        common_user_cols = list(
            set(user_matrix_ref.columns) & set(user_matrix_current.columns)
        )
        common_user_cols = [col for col in common_user_cols if col != "userid"]

        user_drift_score, user_drift_detected = calculate_drift_score(
            user_matrix_ref[common_user_cols], user_matrix_current[common_user_cols]
        )

        drift_scores = {
            "movie_drift_score": movie_drift_score,
            "user_drift_score": user_drift_score,
            "movie_drift_detected": movie_drift_detected,
            "user_drift_detected": user_drift_detected,
        }

        # 4. Décider si réentraînement nécessaire
        print("\n Étape 4/5: Évaluation du besoin de réentraînement...")
        decision, reason = should_retrain(
            movie_drift_score, user_drift_score, threshold=0.3
        )

        # 5. Lancer le réentraînement si nécessaire
        print("\n Étape 5/5: Action...")
        retrain_success = None

        if decision:
            retrain_success, retrain_message = trigger_retraining()
            if retrain_success:
                print(f"\n {retrain_message}")
            else:
                print(f"\n {retrain_message}")
        else:
            print("\n⏭  Pas de réentraînement nécessaire. Système stable.")

        # Logger la décision
        log_retrain_decision(drift_scores, decision, reason, retrain_success)

        print("\n" + "=" * 60)
        print("PROCESSUS TERMINÉ")
        print("=" * 60)

        # Résumé
        print("\n RÉSUMÉ:")
        print(f"   Drift films: {movie_drift_score:.2%}")
        print(f"   Drift utilisateurs: {user_drift_score:.2%}")
        print(
            f"   Réentraînement: {' EFFECTUÉ' if retrain_success else '❌ NON NÉCESSAIRE' if not decision else '⚠️ ÉCHOUÉ'}"
        )

    except Exception as e:
        print(f"\n ERREUR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
