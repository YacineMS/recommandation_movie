import os
import shutil
import zipfile
from pathlib import Path

import kagglehub
import requests
import streamlit as st

# -----------------------------
# CONFIGURATION
# -----------------------------
FASTAPI_TRAINING_URL = "http://movie_trainer_api:8000"
FASTAPI_PREDICTION_URL = "http://movie_predicter_api:8000"
FASTAPI_KNN_URL = "http://knn_api:8000"
DEFAULT_RECO = 100  # Nombre par défaut de recommandations
MOVIES_PER_PAGE = 5  # Nombre de films par page

DATASET_NAME = "ghrzarea/movielens-20m-posters-for-machine-learning"
POSTER_DIR = "poster_movie"
NO_POSTER = "poster_movie/no_poster.png"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

import os
import shutil
import time
from pathlib import Path

# -----------------------------
# FONCTIONS
# -----------------------------
import kagglehub
import streamlit as st

DATASET_NAME = "ghrzarea/movielens-20m-posters-for-machine-learning"
POSTER_DIR = "poster_movie"

import os
import shutil

import kagglehub
import streamlit as st

DATASET_NAME = "ghrzarea/movielens-20m-posters-for-machine-learning"
POSTER_DIR = "poster_movie"


def find_poster_folder(root):
    for dirpath, dirnames, filenames in os.walk(root):
        jpgs = [f for f in filenames if f.lower().endswith(".jpg")]
        if len(jpgs) > 1000:  # dossier qui contient les images
            return dirpath
    return None


def download_kaggle_posters():
    print(os.path.exists(POSTER_DIR))
    print(len(os.listdir(POSTER_DIR)))
    if os.path.exists(POSTER_DIR) and len(os.listdir(POSTER_DIR)) > 1000:
        return

    st.info("📥 Téléchargement Kaggle des posters… (Première connexion à l'appli)")

    dataset_path = kagglehub.dataset_download(DATASET_NAME)
    poster_source = find_poster_folder(dataset_path)

    if not poster_source:
        raise RuntimeError("Impossible de localiser les posters dans le dataset")

    os.makedirs(POSTER_DIR, exist_ok=True)

    copied = 0
    for file in os.listdir(poster_source):
        if file.endswith(".jpg"):
            src = os.path.join(poster_source, file)
            dst = os.path.join(POSTER_DIR, file)
            if not os.path.exists(dst):
                shutil.copy(src, dst)
                copied += 1

    st.success(f"🎬 Posters chargés !")


def get_local_poster(movieid: int):
    """Récupère le poster localement ou fallback no_poster"""
    poster_path = os.path.join(POSTER_DIR, f"{movieid}.jpg")
    if os.path.exists(poster_path):
        return poster_path
    return NO_POSTER


def get_recommendations(token: str, userid: int, num_recommendations: int = 10):
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"userid": userid, "numRecommendations": num_recommendations}
    response = requests.post(
        f"{FASTAPI_KNN_URL}/predict", headers=headers, json=payload
    )
    if response.status_code == 200:
        return response.json()["recommendations"]
    else:
        st.error(f"Erreur API: {response.status_code} {response.text}")
        return []


def demo():
    # -----------------------------
    # SESSION STATE INITIALIZATION
    # -----------------------------
    for key in [
        "token",
        "username",
        "userid",
        "recommendations",
        "index",
        "recommandations_ready",
    ]:
        if key not in st.session_state:
            st.session_state[key] = (
                None
                if key in ["token", "username"]
                else ([] if key == "recommendations" else 0)
            )

    # -----------------------------
    # TITLE
    # -----------------------------
    st.title("🔐 Recommandation de films")

    # -----------------------------
    # USER CONNECTED
    # -----------------------------
    if st.session_state.recommandations_ready and st.session_state.token:
        st.success(
            f"Connecté en tant que {st.session_state.username} (UserId: {st.session_state.userid})"
        )

        recs = st.session_state.recommendations
        if recs:
            start_idx = st.session_state.index
            end_idx = start_idx + MOVIES_PER_PAGE
            page = recs[start_idx:end_idx]

            # Boutons Prev / Next
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                if st.button("⬅️ Prev"):
                    st.session_state.index = (start_idx - MOVIES_PER_PAGE) % len(recs)
                    st.rerun()
            with col3:
                if st.button("Next ➡️"):
                    st.session_state.index = (start_idx + MOVIES_PER_PAGE) % len(recs)
                    st.rerun()

            # Affichage horizontal
            movie_cols = st.columns(MOVIES_PER_PAGE)
            for idx, movie in enumerate(page):
                poster_path = get_local_poster(movie["movieid"])
                with movie_cols[idx]:
                    st.caption(f"Movie ID: {movie['movieid']}")
                    st.image(poster_path, width=150)
                    st.markdown(f"**{movie['title']}**")
                    st.markdown(f"Genres: {movie['genres']}")
                    st.markdown(f"Note moyenne: `{movie['avg_rating']:.2f}`")
                    st.markdown(f"Note prédite: `{movie['svg_pred_rate']:.2f}`")

            st.caption(
                f"Page {start_idx // MOVIES_PER_PAGE + 1} / {(len(recs)-1)//MOVIES_PER_PAGE + 1}"
            )
        else:
            st.info(
                "Aucune recommandation. Cliquez sur 'Get recommendations' pour charger les films."
            )

        if st.button("Se déconnecter"):
            for key in [
                "token",
                "username",
                "userid",
                "recommendations",
                "index",
                "recommandations_ready",
            ]:
                st.session_state[key] = None
            st.rerun()

    # -----------------------------
    # USER NOT CONNECTED
    # -----------------------------
    else:
        st.warning("Veuillez vous connecter pour accéder aux recommandations.")
        with st.form("login_form"):
            username = st.text_input("Nom d'utilisateur", value="admin")
            password = st.text_input(
                "Mot de passe", type="password", value="RecoFilm!2025"
            )
            submitted = st.form_submit_button("Se connecter")

            if submitted:
                with st.spinner(
                    "Connexion en cours et récupération des recommandations..."
                ):
                    download_kaggle_posters()
                    # Login
                    response = requests.post(
                        f"{FASTAPI_KNN_URL}/token",
                        data={"username": username, "password": password},
                    )
                    if response.status_code == 200:
                        token = response.json()["access_token"]
                        userid = response.json()["userid"]
                        recommendations = get_recommendations(
                            token, userid, DEFAULT_RECO
                        )

                        st.session_state.token = token
                        st.session_state.username = username
                        st.session_state.userid = userid
                        st.session_state.recommandations_ready = True
                        st.session_state.recommendations = recommendations
                        st.session_state.index = 0
                        st.success("Connexion réussie ! Recommandations chargées ✅")
                        st.rerun()
                    else:
                        st.error("Nom d'utilisateur ou mot de passe incorrect.")
        st.markdown("---")
        st.text(
            "A la première connexion, nous allons télécharger les posters en local depuis Kaggle (20k images, ~500Mo), cela peut prendre quelques secondes. "
            "Nous générons un id aléatoire pour un utilisateur qui sera authentifié via l’API KNN (http://localhost:8002/docs) qui est sécurisée par user/mot de passe (Basic Auth). "
            "Une fois connecté, l\’API KNN est appelée pour récupérer une liste de recommandations personnalisées du user.\n\n"
            "Avec cette liste de recommandation, on appelle l'API de prédiction SVD de façon sécurisées par token d’authentification (Bearer token), afin d'avoir les notes prédites pour chaque film recommandé. "
            "L'ensemble des films ayants une note prédite supérieur à 4 sont affichés par ordre de note prédite décroissante, avec leur poster, genres, note moyenne générale et note prédite.\n\n"
            "Pour les user avec peu de recommandations (grand consommateurs), on affiche tous les films recommandés (même ceux avec note prédite < 4) pour éviter d'avoir une page vide."
        )
        st.image(
            ASSETS_DIR / "demo_recofilm.png",
            caption="Interface de recommandation",
            use_container_width=False,
            width=1000,
        )
