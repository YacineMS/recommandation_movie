import os
import textwrap
from pathlib import Path

import streamlit as st
from demo import demo
from sprint34 import afficher_slide3_4

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
TRAINER_API_URL = "http://localhost:8003/docs"
FASTAPI_KNN_URL = "http://localhost:8002/docs"
PREDICTER_URL = "http://localhost:8001/docs"
AIRFLOW_URL = "http://localhost:8085/"
MLFLOW_URL = "http://localhost:5000/"


def asset(filename: str) -> str:
    # Retourner le chemin absolu d'un fichier dans le dossier assets.
    return str(ASSETS_DIR / filename)


# ============================================================
# 1) Page config
# ============================================================
st.set_page_config(
    page_title="RecoFilm",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 2) Global CSS (same style for all pages)
# ============================================================
st.markdown(
    """
<style>
/* Global spacing */
.block-container { padding-top: 1.2rem; padding-bottom: 2.2rem; max-width: 1400px; }
[data-testid="stSidebar"] { padding-top: 1.2rem; }

/* Typography */
h1, h2, h3 { letter-spacing: -0.2px; }
.subtitle { color:#6b7280; font-size:0.95rem; margin-top:-6px; }

/* Cards */
.card {
  background: #ffffff;
  border: 1px solid #e8e8e8;
  border-radius: 18px;
  padding: 18px 18px 14px 18px;
  box-shadow: 0 1px 0 rgba(0,0,0,0.02);
}
.card h3 { margin: 0 0 .35rem 0; font-size: 1.05rem; }
.muted { color: #6b7280; font-size: 0.92rem; }

/* Metric cards row */
.metric-row{
  display:flex;
  flex-wrap:wrap;
  gap:12px;
  align-items:stretch;
  margin-top:6px;
  margin-bottom:10px;
}
.mcard{
  flex: 1 1 160px;
  min-width: 160px;
  background:#ffffff;
  border:1px solid #ececec;
  border-radius:16px;
  padding:12px 14px;
  box-shadow: 0 1px 0 rgba(0,0,0,0.02);
}
.mhead{
  display:flex;
  gap:8px;
  align-items:center;
  font-weight:800;
  color:#111827;
  font-size:0.9rem;
}
.micon{ font-size:1.05rem; }
.mval{
  font-size:1.4rem;
  font-weight:900;
  color:#111827;
  margin-top:6px;
  line-height:1.15;
}

/* Status bar */
.status-ok{
  background:#eafff1;
  border:1px solid #c8f3d9;
  color:#0f5132;
  border-radius:14px;
  padding:10px 12px;
  font-weight:800;
}
.status-wip{
  background:#fff7ed;
  border:1px solid #fed7aa;
  color:#7c2d12;
  border-radius:14px;
  padding:10px 12px;
  font-weight:800;
}

/* Tabs header hint */
.tab-hint{ color:#6b7280; font-size:0.9rem; margin-top:-6px; }

/* Capture placeholders */
.capture-box{
  border: 2px dashed #bdbdbd;
  border-radius: 14px;
  height: 220px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #ffffff;
  color: #111827;
  font-size: 0.95rem;
  text-align:center;
  padding: 10px;
}

/* Roadmap boxes */
.phase-box {
  border-radius: 16px;
  border: 0;
  padding: 12px 14px;
  height: 72px;
  display:flex;
  align-items:center;
  justify-content:center;
  text-align:center;
  font-weight: 900;
  color: #ffffff;
  box-shadow: 0 1px 0 rgba(0,0,0,0.06);
}
.arrow {
  height: 72px;
  display:flex;
  align-items:center;
  justify-content:center;
  font-size: 1.8rem;
  color:#A3A3A3;
  font-weight: 900;
}

/* Simple list pills */
.pill{
  display:inline-block;
  padding:6px 10px;
  border-radius:999px;
  border:1px solid #e5e7eb;
  background:#f9fafb;
  font-weight:700;
  font-size:0.85rem;
  margin-right:8px;
  margin-bottom:8px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# 3) Sidebar navigation
# ============================================================
st.sidebar.title("RecoFilm")
st.sidebar.markdown("### Présentation")

PAGES = [
    ("intro", "Introduction: Objectif & roadmap"),
    ("p1", "1 — Fondations: Data, DB, API"),
    ("p2", "2 — Microservice, Versionning et orchestration"),
    ("p4", "3 — Monitoring & Maintenance"),
    ("p5", "4 — Démonstration"),
    ("p6", "5 — Conclusion"),
]
PAGE_LABEL = {k: v for k, v in PAGES}

page_key = st.sidebar.radio(
    label="",
    options=[k for k, _ in PAGES],
    format_func=lambda k: PAGE_LABEL[k],
    index=0,
)


# ============================================================
# 4) UI Components
# ============================================================
def metric_cards(items):
    """
    items: list of dicts
    {"icon":"🎬", "label":"Films", "value":"27,278"}
    IMPORTANT: no leading indentation to avoid Markdown code block rendering.
    """
    parts = ['<div class="metric-row">']
    for it in items:
        parts.append(
            f'<div class="mcard">'
            f'  <div class="mhead"><span class="micon">{it["icon"]}</span><span>{it["label"]}</span></div>'
            f'  <div class="mval">{it["value"]}</div>'
            f"</div>"
        )
    parts.append("</div>")
    st.markdown("\n".join(parts), unsafe_allow_html=True)


def show_image_or_placeholder(filename: str, caption: str, height: int = 360):
    # Afficher l'image si elle existe, sinon placeholder.
    img_path = ASSETS_DIR / filename
    if img_path.exists():
        st.image(asset(filename), width="stretch", caption=caption)
    else:
        capture_placeholder(
            f"📌 Image à ajouter : <b>{caption}</b><br><span class='muted'>Fichier : {filename}</span>",
            height=height,
        )


def status_ok(text):
    st.markdown(f'<div class="status-ok">✅ {text}</div>', unsafe_allow_html=True)


def status_wip(text):
    st.markdown(f'<div class="status-wip">🟠 {text}</div>', unsafe_allow_html=True)


def capture_placeholder(label, height=220):
    st.markdown(
        f'<div class="capture-box" style="height:{height}px;">{label}</div>',
        unsafe_allow_html=True,
    )


def roadmap_boxes():
    st.markdown("#### Roadmap")
    cols = st.columns([2.2, 0.6, 2.2, 0.6, 2.2, 0.6, 2.2, 0.6, 2.2], gap="small")

    phases = [
        (
            "Phase 1<br>Fondations",
            "Données + DB + baseline + API (/training, /predict)",
            "#A78BFA",
        ),
        (
            "Phase 2<br>Microservices & Suivi",
            "MLflow: runs + comparaison + Registry (versions & stages)",
            "#6366F1",
        ),
        (
            "Phase 3<br>Orchestration & Déploiement",
            "Docker / CI-CD + tests + build & publication",
            "#A78BFA",
        ),
        (
            "Phase 4<br>Monitoring",
            "Grafana + Evidently (drift) + politique de retrain",
            "#6366F1",
        ),
        ("Phase 5<br>Frontend", "Streamlit: démo user_id → recommandations", "#A78BFA"),
    ]

    box_positions = [0, 2, 4, 6, 8]
    arrow_positions = [1, 3, 5, 7]

    for i, pos in enumerate(box_positions):
        title_html, _, color = phases[i]
        with cols[pos]:
            st.markdown(
                f"<div class='phase-box' style='background:{color};'>{title_html}</div>",
                unsafe_allow_html=True,
            )

    for pos in arrow_positions:
        with cols[pos]:
            st.markdown("<div class='arrow'>→</div>", unsafe_allow_html=True)

    desc_cols = st.columns([2.2, 0.6, 2.2, 0.6, 2.2, 0.6, 2.2, 0.6, 2.2], gap="small")
    for i, pos in enumerate(box_positions):
        _, desc, _ = phases[i]
        with desc_cols[pos]:
            st.markdown(desc)


def placeholder_page(title: str, subtitle: str):
    st.markdown(f"## {title}")
    st.markdown(f"<div class='subtitle'>{subtitle}</div>", unsafe_allow_html=True)
    st.write("")
    st.info(
        "Page prête (template). Remplir avec : preuves (captures), liens, résultats, et 3 messages clés."
    )


# ============================================================
# 5) Pages - intro
# ============================================================
def render_intro():
    st.markdown("## RECOFILM")
    st.markdown(
        "<div class='subtitle'>Système de Recommandation de Films — Architecture MLOps (5 sprints)</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    # --- (1) Un seul grand bloc pleine largeur : Équipe / Contexte / Repo / Stackholders ---
    st.markdown(
        """
<div class="card">
  <h3>Équipe</h3>
  <div style="margin-top:10px; line-height:1.55;">
    Jimmy Seyer / Yacine Madi Said / Jingzi Zhao / Monitoring : Nicolas
  </div>

  <div style="height:14px;"></div>

  <h3>Contexte</h3>
  <div style="margin-top:10px; line-height:1.55;">
    Dans le cadre de la formation DataScientest, notre projet de module MLOps est de développer un système de recommandation de films type « Netflix ». Lorsqu’un utilisateur saisit son identifiant, le système s’appuie sur les notes existantes et les informations de contenu pour générer une liste de recommandations « que vous pourriez aussi aimer ».
  </div>

  <div style="height:14px;"></div>

  <h3>Repo</h3>
  <div style="margin-top:10px; line-height:1.55;">
    DataScientest-Studio/sep25_cmlops_reco_films2
  </div>

  <div style="height:14px;"></div>

  <h3>Stackholders</h3>
  <div style="margin-top:10px; line-height:1.55;">
    <li><b>Sponsor (simulation)</b> : une plateforme/entreprise du secteur cinéma.</li>
    <li><b>Utilisateur</b> : saisit son identifiant (ex. <code>user_id</code>) et le système renvoie une liste de films recommandés (Top-N).</li>
    <li><b>Admin/Ops</b> : déclenche la mise à jour du modèle lors des rafraîchissements de données ou en cas de baisse de performance ; surveille la dérive (drift) et la qualité des recommandations.</li>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    # --- (2) Objectifs + Architecture + Roadmap dans des tabs ---
    tabs = st.tabs(["🎯 Objectifs", "🏗️ Architecture", "🗺️ Roadmap"])

    # helper local pour afficher image si dispo sinon placeholder
    def show_image_or_placeholder(path: str, caption: str, height: int = 360):
        if path and os.path.exists(path):
            st.image(path, width="stretch", caption=caption)
        else:
            capture_placeholder(
                f"📌 Image à ajouter : <b>{caption}</b><br><span class='muted'>Chemin : {path}</span>",
                height=height,
            )

    with tabs[0]:
        objectifs_html = """
<div class="card">
<h3>Objectifs</h3>

<div style="margin-top:10px; line-height:1.65;">
<ol style="margin:0; padding-left: 18px;">
<li style="margin-bottom:10px;">
<b>Démo Streamlit — Recommandation de films</b>
<ul style="margin-top:6px;">
<li>L’utilisateur saisit son <code>user_id</code> et reçoit une liste <b>Top-N</b> de films recommandés.</li>
<li>La recommandation combine deux logiques :
<ul style="margin-top:6px;">
<li><b>Filtrage collaboratif</b></li>
<li><b>Content-based</b></li>
</ul>
</li>
</ul>
</li>

<li style="margin-bottom:10px;">
<b>3 questions métier à couvrir</b>
<ul style="margin-top:6px;">
<li><b>Cold Start</b> : proposer des recommandations pour un nouvel utilisateur ou un nouveau film.</li>
<li><b>Monitoring</b> : suivre des indicateurs pour savoir si la recommandation s’améliore ou se dégrade.</li>
<li><b>Data Drift</b> : détecter une dérive des données et déclencher une mise à jour du modèle.</li>
</ul>
</li>

<li>
<b>Mise en place des composants MLOps</b>
<ul style="margin-top:6px;">
<li>Stockage des données (PostgreSQL).</li>
<li>Séparation claire des scripts <code>train</code> et <code>predict</code>.</li>
<li>API pour l’entraînement et l’inférence.</li>
<li>MLflow : suivi d’expériences et gestion des versions.</li>
<li>Déploiement : Docker, drift detection, stratégie de maintenance.</li>
</ul>
</li>
</ol>
</div>
</div>
"""
        st.markdown(objectifs_html, unsafe_allow_html=True)

    with tabs[1]:
        show_image_or_placeholder(
            path="assets/intro_architecture.png",
            caption="Architecture (services & flux)",
            height=440,
        )

    with tabs[2]:
        show_image_or_placeholder(
            path="assets/intro_roadmap.png",
            caption="Roadmap (5 phases)",
            height=440,
        )


# ============================================================
# 6) Pages - Phase 1  Fondations
# ============================================================
def render_phase1():
    st.markdown("## Phase 1 — Fondations")
    st.markdown(
        "<div class='subtitle'>Deadline : 3 Novembre 2025</div>", unsafe_allow_html=True
    )
    st.write("")

    # Top metrics
    metric_cards(
        [
            {"icon": "🎬", "label": "Films", "value": "14026"},
            {"icon": "👥", "label": "Utilisateurs", "value": "7120"},
            {"icon": "⭐", "label": "Ratings", "value": "10M"},
            {"icon": "🗄️", "label": "DB", "value": "PostgreSQL"},
            {"icon": "🤖", "label": "Modèle", "value": "KNN/SVD"},
            {"icon": "📜", "label": "Scripts", "value": "training + predict"},
        ]
    )

    status_ok("Phase 1 livrée : data + DB + ML + API")
    st.write("")

    tabs = st.tabs(
        [
            "🎯 Objectifs & Livrables",
            "🧹 Données",
            "🗄️ DB",
            "🤖 Modèle",
            "🧩 API",
            "📜 Scripts",
            "🧯 Défis",
        ]
    )

    with tabs[0]:
        left, right = st.columns([0.55, 0.45], gap="large")
        with left:
            st.markdown(
                """
<div class="card">
  <h3>Objectifs (Sprint 1)</h3>
  <ul style="margin-top:10px; line-height:1.7; padding-left:18px;">
    <li>Définir une roadmap et poser l’architecture</li>
    <li>Environnement docker + FastAPI</li>
    <li>Collecte + prétraitement des données</li>
    <li>Créer une DB</li>
    <li>Créer les modèles : KNN / SVD</li>
  </ul>
</div>
""",
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                """
<div class="card">
  <h3>Livrables (preuves)</h3>
  <ul style="margin-top:10px; line-height:1.7; padding-left:18px;">
    <li>Repo structuré (src/, scripts/, models/, pages/)</li>
    <li>DB prête (tables movies/ratings/tags/links)</li>
    <li>Modèle baseline KNN entraîné + artefact</li>
    <li>API /docs accessible + tests OK</li>
    <li>README : instructions d’exécution</li>
  </ul>
</div>
""",
                unsafe_allow_html=True,
            )

    with tabs[1]:
        st.markdown(
            """
<div class="card">
  <h3>Données MovieLens</h3>
  <div style="margin-top:10px; line-height:1.7;">
    <b>Sources :</b><br>
    • <b>MovieLens 20M</b> (fourni par le projet) : ratings, movies, tags, links<br>
    • <b>Kaggle</b> : MovieLens 20M Posters — métadonnées complémentaires<br><br>
    <b>Pré-traitement :</b><br>
    1. <b>Ingestion :</b> Import vers PostgreSQL.<br>
    2. <b>Nettoyage :</b> Suppression des données nulles ou inutiles.<br>
    3. <b>Feature Eng. :</b> Jointure SQL entre ratings et movies.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.write("")
        st.image(
            asset("1_DataFrame.png"), caption="Aperçu des données", width="stretch"
        )

    with tabs[2]:
        st.markdown("##### Base de Données PostgreSQL (Supabase)")
        left, right = st.columns([0.35, 0.65], gap="large")

        with left:
            st.markdown(
                """
<div class="card">
  <h3>Tables créées</h3>
  <ul style="margin-top:10px; line-height:1.7; padding-left:18px;">
    <li><code>movies</code> : films (titres, genres)</li>
    <li><code>ratings</code> : notes utilisateurs</li>
    <li><code>tags</code> : tags descriptifs</li>
    <li><code>links</code> : liens IMDB / TMDb</li>
  </ul>
</div>
""",
                unsafe_allow_html=True,
            )

        with right:
            st.image(
                asset("1_Database.png"),
                caption="Aperçu des données (Exemple)",
                width="stretch",
            )

    with tabs[3]:
        left, right = st.columns([0.6, 0.4], gap="large")
        with left:
            st.markdown(
                """
<div class="card">
  <h3>Baseline : KNN (Collaborative Filtering)</h3>
  <div style="margin-top:10px; line-height:1.7;">
    <b>Pourquoi KNN ?</b><br>
    • Simple et interprétable<br>
    • Bon baseline pour recommandation<br>
    • Gère bien la sparsité (avec filtrage)
  </div>
  <div style="height:10px;"></div>
  <div style="margin-top:10px; line-height:1.7;">
    <b>Configuration (ex.)</b><br>
    • K = 20 voisins<br>
    • Algo : ball_tree<br>
    • Métrique : euclidienne
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
            st.write("")
            st.markdown(
                """
<div class="card">
  <div style="margin-top:10px; line-height:1.7;">
    <b>Baseline :</b> KNN → Simple, interprétable, bon point de départ<br>
    <b>Amélioration :</b> SVD → Matrice de factorisation, meilleure précision sur données sparses<br>
    <b>Comparaison via MLflow</b> → RMSE comme métrique de sélection
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

    with tabs[4]:
        left, right = st.columns([0.6, 0.4], gap="large")
        with left:
            st.markdown(
                """
<div class="card">
  <h3>API FastAPI</h3>
  <div style="margin-top:10px; line-height:1.7;">
    <b>Endpoints</b><br>
    • GET <code>/</code> : page d’accueil<br>
    • GET <code>/health</code> : health check<br>
    • POST <code>/training</code> : entraînement du modèle<br>
    • POST <code>/predict</code> : recommandations
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
            st.write("")
            st.image(
                asset("1_FastAPI.png"),
                caption="FastAPI /docs (Swagger) + test predict",
                width="stretch",
            )

    with tabs[5]:
        left, right = st.columns([0.55, 0.45], gap="large")
        with left:
            st.markdown(
                """
<div class="card">
  <h3>Scripts Python</h3>
  <ul style="margin-top:10px; line-height:1.8; padding-left:18px;">
    <li><b>scripts/load_data.py</b> : charge les données (one-shot)</li>
    <li><b>training.py</b> : entraîne et sauvegarde le modèle</li>
    <li><b>predict.py</b> : inférence / recommandations</li>
  </ul>
</div>
""",
                unsafe_allow_html=True,
            )
        with right:
            pass

    with tabs[6]:
        left, right = st.columns([0.55, 0.45], gap="large")
        with left:
            st.markdown(
                """
<div class="card">
  <h3>Défis & Solutions</h3>
  <div style="margin-top:10px; line-height:1.8;">
    <b>Volumétrie</b> : dataset très large → <b>échantillonnage</b><br>
    <b>Migration SQL</b> : SQLite → PostgreSQL → <b>adaptation requêtes</b><br>
    <b>Performance</b> : preprocessing lent → <b>traitement par batch</b>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
        with right:
            pass


# ============================================================
# 7) Pages - Phase 2 Suivi
# ============================================================
def render_phase2():
    st.markdown("## Phase 2/3 — Microservices, Suivi & Versionning")
    st.markdown(
        "<div class='subtitle'>MLflow : runs, comparaison, registry (versions & stages)</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    metric_cards(
        [
            {"icon": "🐳", "label": "Dockerisation", "value": "8 services"},
            {"icon": "🧭", "label": "Airflow", "value": "Planification"},
            {"icon": "📊", "label": "MLFlow", "value": "Suivi des expériences"},
            {"icon": "🔄", "label": "CI/CD", "value": "GitHub Actions & DockerHub"},
        ]
    )

    tabs = st.tabs(
        [
            "🎯 Objectifs & Livrables",
            "🐳 Dockerisation",
            "🐳 FastAPI",
            "🧭 Airflow",
            "📊 MLFlow",
            "🔄 CI/CD",
        ]
    )

    with tabs[0]:
        c1, c2 = st.columns([0.55, 0.45], gap="large")
        with c1:
            st.markdown(
                """
<div class="card">
  <h3>Objectifs (Phase 2 - 3)</h3>
  <ul style="margin-top:10px; line-height:1.7; padding-left:18px;">
    <li>Dockerisation avec communication par api sécurisée</li>
    <li>Versionning, et désignation du modèle de production avec MLFlow</li>
    <li>Intégration d'une CI/CD avec GitHub Actions et DockerHub</li>
    <li>Orchestration avec Airflow</li>
  </ul>
</div>
""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
<div class="card">
  <h3>Livrables</h3>
  <ul style="margin-top:10px; line-height:1.7; padding-left:18px;">
    <li>Docker compose</li>
    <li>MLflow Experiments</li>
    <li>Pipelines GitHub Actions + Images dans DockerHub</li>
    <li>Dags Airflow</li>
  </ul>
</div>
""",
                unsafe_allow_html=True,
            )

    with tabs[1]:
        left, right = st.columns([0.6, 0.4], gap="large")
        with left:
            st.image(
                ASSETS_DIR / "mermaid_schema.png",
                caption="Architecture Docker (services & flux)",
                use_container_width=False,
                width=1000,
            )
        with right:
            st.markdown(
                """

""",
                unsafe_allow_html=True,
            )

    with tabs[2]:
        st.link_button("API KNN", "http://localhost:8002/docs")
        st.text("API sécurisée par user/mot de passe (Basic Auth)")
        st.image(
            ASSETS_DIR / "8002.png",
            caption="API KNN",
            use_container_width=False,
            width=1000,
        )
        st.link_button("API Prédiction SVD", "http://localhost:8003/docs")
        st.text("API sécurisée avec token d’authentification (Bearer <token>)")
        st.image(
            ASSETS_DIR / "8003.png",
            caption="API Prédiction SVD",
            use_container_width=False,
            width=1000,
        )
        st.link_button("API Entrainement SVD", "http://localhost:8001/docs")
        st.text("API sécurisée avec token d’authentification (Bearer <token>)")
        st.image(
            ASSETS_DIR / "8001.png",
            caption="API Entrainement SVD",
            use_container_width=False,
            width=1000,
        )
    with tabs[3]:
        st.link_button("Airflow", "http://localhost:8085")
        st.image(
            ASSETS_DIR / "airflow_interface.png",
            caption="Airflow Interface",
            use_container_width=False,
            width=1000,
        )
        st.image(
            ASSETS_DIR / "airflow_dag_exec.png",
            caption="Airflow Dag execution",
            use_container_width=False,
            width=1000,
        )

    with tabs[4]:
        st.link_button("MLFlow", "http://localhost:5001")
        st.image(
            ASSETS_DIR / "mlflow_1.png",
            caption="MLFlow Interface",
            use_container_width=False,
            width=1000,
        )
        st.image(
            ASSETS_DIR / "mlflow_2.png",
            caption="MLFlow Models Registry",
            use_container_width=False,
            width=1000,
        )
        st.image(
            ASSETS_DIR / "mlflow_5.png",
            caption="MLFlow SVD Runs (metrics + params)",
            use_container_width=False,
            width=1000,
        )
        st.image(
            ASSETS_DIR / "mlflow_6.png",
            caption="MLFlow KNN Runs (metrics + params)",
            use_container_width=False,
            width=1000,
        )
    with tabs[5]:

        st.image(
            ASSETS_DIR / "githubactions.png",
            caption="GitHub Actions",
            use_container_width=False,
            width=1000,
        )
        st.image(
            ASSETS_DIR / "dockerhub.png",
            caption="DockerHub",
            use_container_width=False,
            width=1000,
        )


# ============================================================
# 8) Pages - Phase 3 Orchestration
# ============================================================
def render_phase3():
    st.markdown("## Phase 3 — Orchestration & Déploiement")
    st.markdown(
        "<div class='subtitle'>Docker + CI/CD + build & livraison (preuve “green”)</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    metric_cards(
        [
            {"icon": "🐳", "label": "Docker", "value": "compose"},
            {"icon": "🧪", "label": "Tests", "value": "automatiques"},
            {"icon": "✅", "label": "CI", "value": "Actions green"},
            {"icon": "📦", "label": "Build", "value": "image tag"},
            {"icon": "🚀", "label": "CD", "value": "push"},
            {"icon": "📄", "label": "README", "value": "1 commande"},
        ]
    )

    status_wip(
        "Phase 3 : page prête. Ajoute les captures docker-compose + GitHub Actions."
    )
    st.write("")

    tabs = st.tabs(
        [
            "🎯 Objectifs & Livrables",
            "🐳 Docker",
            "✅ CI (Actions)",
            "🚀 CD (Release)",
            "🧯 Défis",
        ]
    )

    with tabs[0]:
        c1, c2 = st.columns([0.55, 0.45], gap="large")
        with c1:
            st.markdown(
                """
<div class="card">
  <h3>Objectifs (Phase 3)</h3>
  <ul style="margin-top:10px; line-height:1.7; padding-left:18px;">
    <li>Orchestrer les services (API, DB, MLflow si besoin)</li>
    <li>Standardiser l’exécution via Docker Compose</li>
    <li>Mettre en place CI : lint + tests + build</li>
    <li>Déployer via pipeline (CD) + tags de version</li>
  </ul>
</div>
""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
<div class="card">
  <h3>Livrables (preuves)</h3>
  <ul style="margin-top:10px; line-height:1.7; padding-left:18px;">
    <li>docker-compose.yml</li>
    <li>Workflow CI (green)</li>
    <li>Image buildée / tags</li>
    <li>README : “docker compose up”</li>
  </ul>
</div>
""",
                unsafe_allow_html=True,
            )

    with tabs[1]:
        left, right = st.columns([0.6, 0.4], gap="large")
        with left:
            capture_placeholder(
                "📌 Capture à ajouter : docker-compose.yml + schéma services",
                height=320,
            )
        with right:
            st.markdown(
                """
""",
                unsafe_allow_html=True,
            )

    with tabs[2]:
        left, right = st.columns([0.6, 0.4], gap="large")
        with left:
            capture_placeholder(
                "📌 Capture à ajouter : GitHub Actions (workflow green)", height=320
            )
        with right:
            st.markdown(
                """
""",
                unsafe_allow_html=True,
            )

    with tabs[3]:
        left, right = st.columns([0.6, 0.4], gap="large")
        with left:
            capture_placeholder(
                "📌 Capture à ajouter : Release / tags / push image", height=320
            )
        with right:
            st.markdown(
                """
""",
                unsafe_allow_html=True,
            )

    with tabs[4]:
        capture_placeholder(
            "📌 Capture à ajouter : Release / tags / push image", height=320
        )


# ============================================================
# 9) Pages - Phase 4 Monitoring & Maintenance
# ============================================================
def render_phase4():
    st.markdown("## Phase 4 — Monitoring & Maintenance")
    st.markdown(
        "<div class='subtitle'>Grafana/Prometheus + Evidently (drift) + stratégie retrain</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    metric_cards(
        [
            {"icon": "📊", "label": "Grafana", "value": "dashboard"},
            {"icon": "⏱️", "label": "Latence", "value": "p95"},
            {"icon": "🚨", "label": "Erreurs", "value": "4xx/5xx"},
            {"icon": "🌪️", "label": "Drift", "value": "Evidently"},
            {"icon": "🔁", "label": "Retrain", "value": "rule/cron"},
            {"icon": "🧾", "label": "Reports", "value": "HTML"},
        ]
    )

    status_wip("Phase 4 : page prête. Ajoute les captures Grafana + rapport Evidently.")
    st.write("")

    tabs = st.tabs(
        [
            "🎯 Objectifs & Livrables",
            "📊 Grafana",
            "🌪️ Drift (Evidently)",
            "🔁 Maintenance",
            "🧯 Défis",
        ]
    )

    with tabs[0]:
        c1, c2 = st.columns([0.55, 0.45], gap="large")
        with c1:
            st.markdown(
                """
<div class="card">
  <h3>Objectifs (Phase 4)</h3>
  <ul style="margin-top:10px; line-height:1.7; padding-left:18px;">
    <li>Surveiller l’API : latence, erreurs, throughput</li>
    <li>Détecter la dérive des données (data drift)</li>
    <li>Définir une politique de maintenance (retrain)</li>
  </ul>
</div>
""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
<div class="card">
  <h3>Livrables (preuves)</h3>
  <ul style="margin-top:10px; line-height:1.7; padding-left:18px;">
    <li>Dashboard Grafana (3–5 graphs)</li>
    <li>Rapport Evidently (Target/Data drift)</li>
    <li>Règle retrain : planifiée ou déclenchée</li>
    <li>README : section Monitoring</li>
  </ul>
</div>
""",
                unsafe_allow_html=True,
            )

    with tabs[1]:
        left, right = st.columns([0.6, 0.4], gap="large")
        with left:
            capture_placeholder(
                "📌 Capture à ajouter : Grafana dashboard (latence, erreurs, req/s)",
                height=320,
            )
        with right:
            st.markdown(
                """
<div class="card">
  <h3>Graphiques attendus</h3>
  <div style="margin-top:10px; line-height:1.8;">
    • Requests / sec<br>
    • Latence (p50/p95)<br>
    • Erreurs (4xx/5xx)<br>
    • CPU/RAM (option)
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

    with tabs[2]:
        left, right = st.columns([0.6, 0.4], gap="large")
        with left:
            capture_placeholder(
                "📌 Capture à ajouter : Evidently report (DataDriftPreset)", height=320
            )
        with right:
            st.markdown(
                """
<div class="card">
  <h3>Drift</h3>
  <div style="margin-top:10px; line-height:1.8;">
    • Détection automatique<br>
    • Seuils & alertes<br>
    • Justifier un retrain
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

    with tabs[3]:
        st.markdown(
            """
<div class="card">
  <h3>Maintenance</h3>
  <div style="margin-top:10px; line-height:1.8;">
    <b>Scénario</b> : si drift &gt; seuil → retrain (cron ou trigger) → nouveau modèle (MLflow Registry) → prod.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.write("")
        capture_placeholder(
            "📌 (Option) Capture : log retrain / pipeline simple", height=220
        )

    with tabs[4]:
        st.markdown(
            """
<div class="card">
  <h3>Défis & Solutions</h3>
  <div style="margin-top:10px; line-height:1.8;">
    <b>Choisir KPI</b> → latence, erreurs, drift<br>
    <b>Faux positifs</b> → seuils réalistes + fenêtre temporelle<br>
    <b>Maintenance</b> → retrain contrôlé (staging → prod)
  </div>
</div>
""",
            unsafe_allow_html=True,
        )


# ============================================================
# 10)Pages- Phase5 Monitoring & Maintenance
# ============================================================
def render_phase5():
    st.markdown("## Phase 5 — Frontend (Streamlit)")
    st.markdown(
        "<div class='subtitle'>UI démo pour le jury : user_id → appel API → recommandations</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    metric_cards(
        [
            {"icon": "🧑", "label": "Entrée", "value": "user_id"},
            {"icon": "🔌", "label": "Call API", "value": "/predict"},
            {"icon": "📋", "label": "Sortie", "value": "Top-N"},
            {"icon": "🖼️", "label": "Option", "value": "posters"},
            {"icon": "🏷️", "label": "Option", "value": "version modèle"},
            {"icon": "✅", "label": "Démo", "value": "30 sec"},
        ]
    )

    tabs = st.tabs(["🎯 Objectifs", "🧪 Démo"])

    with tabs[0]:
        st.markdown(
            """
<div class="card">
  <h3>Objectifs (Phase 5)</h3>
  <ul style="margin-top:10px; line-height:1.7; padding-left:18px;">
    <li>Rendre l’API utilisable par un non-tech</li>
    <li>Démo claire : navigation + preuves (captures)</li>
    <li>Uniformiser la charte visuelle</li>
  </ul>
</div>
""",
            unsafe_allow_html=True,
        )

    with tabs[1]:
        demo()


# ============================================================
# 6) Pages - Phase 6 Conclusion & Next Steps  -- à ajouter après le demo
# ============================================================
def render_phase6():
    st.markdown("## Conclusion & Next Steps")
    st.markdown(
        "<div class='subtitle'>Bilan du projet et perspectives d'évolution</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    # ---- Conclusion ----
    st.markdown(
        """
<div class="card">
  <h3>✅ Conclusion</h3>
  <div style="margin-top:12px; line-height:1.75;">
    Au cours de ce projet, nous avons conçu et implémenté un système de recommandation
    de films <b>bout-en-bout</b> suivant les principes MLOps :<br><br>
    • Données ingérées dans <b>PostgreSQL (Supabase)</b><br>
    • Deux modèles entraînés et comparés : <b>KNN + SVD</b><br>
    • <b>3 API FastAPI</b> conteneurisées dans Docker<br>
    • Versioning et élection du champion via <b>MLflow</b><br>
    • Orchestration quotidienne avec <b>Airflow</b><br>
    • Monitoring temps réel (<b>Prometheus + Grafana</b>) et détection de drift (<b>Evidently</b>)<br>
    • Pipeline <b>CI/CD</b> avec GitHub Actions → Docker Hub<br>
    • Interface de démonstration <b>Streamlit</b><br><br>
    Merci à <b>Nicolas</b> pour son accompagnement et ses retours tout au long du projet. 🙏
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")
    st.write("")

    # ---- Next Steps ----
    st.markdown("### 🚀 Next Steps")
    st.write("")

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(
            """
<div class="card" style="height:100%;">
  <h3>☁️ Déploiement Cloud</h3>
  <div class="muted" style="margin-top:8px; line-height:1.65;">
    Migrer les services Docker vers une plateforme cloud
    pour la scalabilité et la haute disponibilité.
  </div>
  <div style="margin-top:12px;">
    <span class="pill">AWS ECS</span>
    <span class="pill">GCP Cloud Run</span>
    <span class="pill">Kubernetes</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
<div class="card" style="height:100%;">
  <h3>🔗 Orchestration unifiée</h3>
  <div class="muted" style="margin-top:8px; line-height:1.65;">
    Intégrer la détection de drift dans le DAG Airflow
    pour un pipeline unifié : insertion → train → drift → retrain.
  </div>
  <div style="margin-top:12px;">
    <span class="pill">Airflow DAG</span>
    <span class="pill">Alerting</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
<div class="card" style="height:100%;">
  <h3>🔔 Alertes & Notifications</h3>
  <div class="muted" style="margin-top:8px; line-height:1.65;">
    Mettre en place des alertes automatiques
    lors de drift détecté ou d'anomalies API.
  </div>
  <div style="margin-top:12px;">
    <span class="pill">Slack</span>
    <span class="pill">E-mail</span>
    <span class="pill">Grafana Alerts</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.write("")
    st.write("")

    # ---- Merci ----
    st.markdown(
        """
<div style="text-align:center; padding:30px 0 10px 0;">
  <span style="font-size:2.2rem;">🙏</span>
  <h2 style="margin-top:8px;">Merci pour votre attention</h2>
  <p class="muted" style="font-size:1.05rem; margin-top:4px;">Des questions ?</p>
  <div style="margin-top:14px;">
    <span class="pill">Jimmy</span>
    <span class="pill">Yacine</span>
    <span class="pill">Jingzi</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# 11)Router
# ============================================================
if page_key == "intro":
    render_intro()
elif page_key == "p1":
    render_phase1()
elif page_key == "p2":
    render_phase2()
elif page_key == "p4":
    afficher_slide3_4()
elif page_key == "p5":
    render_phase5()
elif page_key == "p6":
    render_phase6()
