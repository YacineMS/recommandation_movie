"""
Script pour planifier l'exécution automatique du réentraînement
Utilise APScheduler pour simuler un cron job
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger


def run_auto_retrain():
    """
    Exécute le script de réentraînement automatique
    """
    print("\n" + "=" * 60)
    print(f" DÉCLENCHEMENT PLANIFIÉ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    project_root = Path(__file__).resolve().parent
    auto_retrain_script = project_root / "auto_retrain.py"

    try:
        result = subprocess.run(
            [sys.executable, str(auto_retrain_script)], cwd=str(project_root)
        )

        if result.returncode == 0:
            print("\n Exécution terminée avec succès")
        else:
            print(f"\n Erreur lors de l'exécution (code: {result.returncode})")

    except Exception as e:
        print(f"\n Exception: {e}")


def main():
    """
    Configure et lance le scheduler
    """
    print("=" * 60)
    print("PLANIFICATEUR DE RÉENTRAÎNEMENT AUTOMATIQUE")
    print("=" * 60)

    scheduler = BlockingScheduler()

    # Option 1: Tous les jours à 2h du matin
    # scheduler.add_job(
    #    run_auto_retrain,
    #    CronTrigger(hour=2, minute=0),
    #    id="daily_retrain_check",
    #    name="Vérification quotidienne du drift et réentraînement",
    #    replace_existing=True,
    # )

    # Option 2: Toutes les heures (démo)
    # scheduler.add_job(
    #     run_auto_retrain,
    #     CronTrigger(minute=0),
    #     id='hourly_retrain_check',
    #     name='Vérification horaire du drift',
    #     replace_existing=True
    # )

    # Option 3: Toutes les 5 minutes (test)
    scheduler.add_job(
        run_auto_retrain,
        "interval",
        minutes=5,
        id="test_retrain_check",
        name="Test - Vérification toutes les 5 minutes",
        replace_existing=True,
    )

    print("\n PLANIFICATION CONFIGURÉE:")
    print("    Vérification quotidienne du drift à 2h00 du matin")
    print("    Si drift > 30% → Réentraînement automatique")
    print("\n Pour tester immédiatement, exécutez:")
    print("   python src/monitoring/auto_retrain.py")
    print("\n  Appuyez sur Ctrl+C pour arrêter le scheduler")
    print("=" * 60)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n\n Arrêt du scheduler...")
        scheduler.shutdown()
        print(" Scheduler arrêté proprement")


if __name__ == "__main__":
    main()
