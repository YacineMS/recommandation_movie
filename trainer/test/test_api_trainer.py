# tests/test_api_trainer.py

import os
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from trainer import api_trainer as api  # Import corrigé pour ton arbo

TOKEN = os.getenv("API_BEARER_TOKEN")
headers = {"Authorization": f"Bearer {TOKEN}"}

# -----------------------------
# FASTAPI TEST CLIENT
# -----------------------------
client = TestClient(api.app)


# -----------------------------
# MOCK MLflow ET SURPRISE
# -----------------------------
@pytest.fixture(autouse=True)
def mock_mlflow_and_algo(monkeypatch):
    """
    Mock toutes les fonctions liées à MLflow et SVD pour que
    l'entraînement ne fasse rien de réel.
    """
    # Mock train_svd_model pour retourner des métriques fictives
    monkeypatch.setattr(
        api,
        "train_svd_model",
        lambda: {
            "rmse": 0.5,
            "mae": 0.3,
            "run_id": "mock_run_id",
            "alias": "production",
        },
    )


# -----------------------------
# MOCK DATABASE
# -----------------------------
@pytest.fixture(autouse=True)
def mock_db(monkeypatch):
    """
    Mock get_db_connection, check_and_update_daily_counts, insert_data_chunk
    et le curseur pour /daily-counts et /insert-data
    """
    fake_conn = MagicMock()
    fake_cursor = MagicMock()

    # Permet de faire: with conn.cursor() as cur
    fake_conn.cursor.return_value.__enter__.return_value = fake_cursor

    # /daily-counts : retourne une ligne factice
    fake_cursor.fetchone.return_value = (1, "2026-01-11", 0)

    # /insert-data : check_and_update_daily_counts -> True + count = 0
    monkeypatch.setattr(
        api, "check_and_update_daily_counts", lambda conn, force_insert=False: (True, 0)
    )

    # /insert-data : insert_data_chunk -> 5 lignes insérées par table
    monkeypatch.setattr(api, "insert_data_chunk", lambda conn, table, count: 5)

    # Mock de la connexion
    monkeypatch.setattr(api, "get_db_connection", lambda: fake_conn)

    yield fake_conn


# -----------------------------
# TEST /training
# -----------------------------
def test_training_endpoint():
    response = client.post("/training", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["rmse"] == 0.5
    assert data["mae"] == 0.3
    assert data["run_id"] == "mock_run_id"
    assert data["alias"] == "production"


# -----------------------------
# TEST /insert-data
# -----------------------------
def test_insert_data_endpoint():
    response = client.post("/insert-data", json={"force_insert": True}, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    for table in ["ratings", "tags", "genome-scores"]:
        assert "inserted_rows" in data["results"][table]
        assert data["results"][table]["inserted_rows"] == 5


# -----------------------------
# TEST /daily-counts
# -----------------------------
def test_daily_counts_endpoint():
    response = client.get("/daily-counts", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert "date" in data
    assert "count" in data
    assert data["id"] == 1
    assert data["count"] == 0
    assert data["date"] == "2026-01-11"
