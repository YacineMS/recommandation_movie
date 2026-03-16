"""
Sprint 4 - Monitoring & Maintenance
"""

from pathlib import Path

import streamlit as st


def afficher_slide3_4():
    st.set_page_config(
        page_title="Sprint 4 - Monitoring", page_icon="📊", layout="wide"
    )

    st.markdown("## Phase 4 — Monitoring & Maintenance")
    st.markdown("**Grafana/Prometheus + Evidently (drift) + stratégie retrain**")

    st.markdown("---")

    # Indicateurs en haut de page
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.markdown("### 📊 Grafana")
        st.metric("dashboard", "4 panels")

    with col2:
        st.markdown("### ⏱️ Latence")
        st.metric("p95", "< 100ms")

    with col3:
        st.markdown("### ⚠️ Erreurs")
        st.metric("4xx/5xx", "0%")

    with col4:
        st.markdown("### 🔀 Drift")
        st.metric("Evidently", "< 30%")

    with col5:
        st.markdown("### 🔄 Retrain")
        st.metric("rule/cron", "Auto")

    with col6:
        st.markdown("### 📄 Reports")
        st.metric("HTML", "Drift")

    st.markdown("---")

    # Onglets principaux
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🎯 Objectifs & Livrables",
            "📊 Grafana",
            "🔍 Drift (Evidently)",
            "🔄 Maintenance",
            "⚡ Défis",
        ]
    )

    # =============================================
    # TAB 1: OBJECTIFS & LIVRABLES
    # =============================================
    with tab1:
        st.markdown("## Objectifs (Sprint 4)")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### Objectifs Principaux
            - ✅ Surveiller l'API : latence, erreurs, throughput
            - ✅ Détecter la dérive des données (data drift)
            - ✅ Définir une politique de maintenance (retrain)
            """)

        with col2:
            st.markdown("""
            ### Livrables (preuves)
            - ✅ Dashboard Grafana (3-5 graphs)
            - ✅ Rapport Evidently (Target/Data drift)
            - ✅ Règle retrain : planifiée ou déclenchée
            - ✅ README : section Monitoring
            """)

    # =============================================
    # TAB 2: GRAFANA
    # =============================================
    with tab2:
        st.markdown("## 📊 Grafana Dashboard")

        grafana_path = Path(__file__).parent / "assets" / "grafana_dashboard.png"
        st.image(
            str(grafana_path),
            caption="Dashboard Grafana - RecoFilm Monitoring",
            width=1200,
        )

        st.markdown("""
        ### Graphiques implémentés
        
        Notre dashboard Grafana contient **4 panels** :
        
        1. **Temps de réponse (secondes)** - Durée des requêtes HTTP par endpoint
        2. **Erreurs HTTP** - Nombre d'erreurs 4xx par endpoint
        3. **Requêtes HTTP par seconde** - Taux de requêtes total par endpoint et status
        4. **Nombre de requêtes actives** - Requêtes en cours de traitement
        """)

        st.markdown("---")
        st.markdown("### Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.code(
                """
    # prometheus.yml
    scrape_configs:
    - job_name: 'recofilm-api'
        static_configs:
        - targets: ['api:8000']
            """,
                language="yaml",
            )

        with col2:
            st.code(
                """
    # Métriques exposées
    - http_requests_total
    - http_request_duration_seconds
    - http_requests_in_progress
            """,
                language="python",
            )

    # =============================================
    # TAB 3: DRIFT (EVIDENTLY)
    # =============================================
    with tab3:
        st.markdown("## 🔍 Drift (Evidently)")

        evidently_path = Path(__file__).parent / "assets" / "evidently_drift.png"
        st.image(
            str(evidently_path),
            caption="Rapport Evidently - Data Drift Detection",
            width=1500,
        )

        st.markdown("""
        ### Résultat de l'analyse
        
        **Dataset Drift : NOT Detected** ✅
        - **21 colonnes** analysées
        - **2 colonnes** avec drift détecté (9.52%)
        - **Seuil de détection** : 0.5 (50%)
        - **Share of Drifted Columns** : 0.0952 (< 50% → OK)
        
        Les colonnes avec drift détecté sont `avg_rating` et `num_ratings`, ce qui est normal car ces valeurs évoluent naturellement avec le temps (nouveaux votes).
        """)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### Seuils & alertes
            - 🟢 Drift < 10% → OK
            - 🟡 Drift 10-30% → Surveillance
            - 🔴 Drift > 30% → Retrain déclenché
            """)

        with col2:
            st.code(
                """
    # Exécution manuelle
    python src/monitoring/drift_detection.py

    # Sortie
    reports/drift/
    ├─ movie_drift_report.html
    └─ user_drift_report.html
            """,
                language="bash",
            )

    # =============================================
    # TAB 4: MAINTENANCE
    # =============================================
    with tab4:
        st.markdown("## 🔄 Maintenance")

        st.markdown("""
        **Scénario :** si drift > seuil → retrain (cron ou trigger) → nouveau modèle (MLflow Registry) → prod.
        """)

        st.info("💡 **(Option) Capture à ajouter :** log retrain / pipeline simple")

        st.markdown("---")

        st.markdown("### Stratégie de réentraînement")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### 🕐 Planifié (Cron)
            - **Tous les jours à 2h00**
            - APScheduler + CronTrigger
            - Script : `schedule_retrain.py`
            """)

            st.code(
                """
    # Planification
    python src/monitoring/schedule_retrain.py

    # Cron configuré
    CronTrigger(hour=2, minute=0)
            """,
                language="python",
            )

        with col2:
            st.markdown("""
            #### ⚡ Déclenché (Trigger)
            - **Si drift > 30%**
            - Vérification automatique
            - Script : `auto_retrain.py`
            """)

            st.code(
                """
    # Vérification + retrain si nécessaire
    python src/monitoring/auto_retrain.py

    # Résultat
    logs/retrain/retrain_log_*.json
            """,
                language="python",
            )

    # =============================================
    # TAB 5: DÉFIS
    # =============================================
    with tab5:
        st.markdown("## ⚡ Défis & Solutions")

        challenges = [
            {
                "title": "🎯 Choisir KPI",
                "problem": "Trop de métriques possibles (latence, erreurs, drift)",
                "solution": "Focus sur 4 KPIs essentiels : latence p95, erreurs 4xx/5xx, requests/sec, drift score",
            },
            {
                "title": "🔧 Faux positifs",
                "problem": "Alertes drift pour variations normales",
                "solution": "Seuils réalistes (30%) + fenêtre temporelle avant alerte",
            },
            {
                "title": "⚙️ Maintenance",
                "problem": "Retrain contrôlé (staging → prod)",
                "solution": "Pipeline simple : drift → retrain planifié ou déclenché → prod",
            },
        ]

        for challenge in challenges:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### {challenge['title']}")
                st.info(f"**Problème :** {challenge['problem']}")
            with col2:
                st.markdown("### ✅ Solution")
                st.success(challenge["solution"])
            st.markdown("---")

    # Footer
    st.markdown("---")
