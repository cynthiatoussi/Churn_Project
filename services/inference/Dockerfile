# ─────────────────────────────────────────────────────────────
# services/inference/Dockerfile
# ─────────────────────────────────────────────────────────────
# IMPORTANT : ce Dockerfile est buildé depuis la RACINE du projet
# (context = .) pour pouvoir accéder à models/ à la racine.
# Les chemins COPY sont donc relatifs à la racine du projet.
#
# Ce service charge 2 modèles ML (churn_pipeline + offer_pipeline)
# qui totalisent ~24Mo. L'image finale sera ~500-600Mo avec
# toutes les dépendances sklearn/xgboost — en dessous du seuil
# de 1Go donc pas besoin de multi-stage build.
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Dossier de travail dans le conteneur
WORKDIR /app

# Installation des dépendances en premier
# Chemin depuis la racine du projet (context = .)
COPY services/inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code applicatif
# Chemin depuis la racine du projet (context = .)
COPY services/inference/app.py .

# Copie des artefacts modèles depuis models/ à la racine du repo
# Ces fichiers sont accessibles car le context est la racine (.)
COPY models/churn_pipeline.pkl  ./models/
COPY models/offer_pipeline.pkl  ./models/
COPY models/config.json         ./models/

# Port exposé par le service d'inférence
EXPOSE 8000

# Health check — vérifie que le service répond ET que les modèles
# sont bien chargés (le /health retourne 503 si pas chargés)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Démarrage avec 2 workers Uvicorn
# Dimensionné pour 700m CPU (règle : 2 × 0.7 + 1 ≈ 2)
# Permet de tenir 150 req/min sans saturer
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]