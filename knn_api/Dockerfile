# Image de base Python 3.11
FROM python:3.11-bullseye

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier requirements.txt
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code source
COPY . .

# Créer les répertoires nécessaires s'ils n'existent pas
RUN mkdir -p /app/mlruns /app/models

# Exposer le port 8000
EXPOSE 8000

# Variable d'environnement pour Python (évite les problèmes de buffering)
ENV PYTHONUNBUFFERED=1

# Donner les droits d'exécution au script
RUN sed -i 's/\r//' /app/start.sh && chmod +x /app/start.sh

# Commande de démarrage
CMD ["./start.sh"]