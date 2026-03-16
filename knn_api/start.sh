#!/bin/bash

# Lancer le scheduler en arrière-plan
python ./monitoring/schedule_retrain.py &

# Lancer l'API
uvicorn api.main:app --host 0.0.0.0 --port 8000