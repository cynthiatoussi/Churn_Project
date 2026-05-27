# ─────────────────────────────────────────────────────────────
# services/inference/Dockerfile
# ─────────────────────────────────────────────────────────────
# Ce service charge 2 modèles ML (churn_pipeline + offer_pipeline)
# qui totalisent ~24Mo. L'image finale sera ~500-600Mo avec
# toutes les dépendances sklearn/xgboost — en dessous du seuil
# de 1Go donc pas besoin de multi-stage build.
#
# Les modèles sont copiés depuis models/ à la racine du repo
# lors du build Docker. Ils font partie de l'image finale.
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Dossier de travail dans le conteneur
WORKDIR /app

# Installation des dépendances en premier
# (optimise le cache Docker — cette couche n'est reconstruite
#  que si requirements.txt change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code applicatif
COPY app.py .

# Copie des artefacts modèles depuis models/ à la racine du repo
# Les 2 pipelines + config.json sont embarqués dans l'image
COPY models/churn_pipeline.pkl  ./models/
COPY models/offer_pipeline.pkl  ./models/
COPY models/config.json         ./models/

# Port exposé par le service d'inférence
EXPOSE 8000

# Health check — vérifie que le service répond ET que les modèles
# sont bien chargés (le /health retourne 503 si pas chargés)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Démarrage avec Gunicorn + workers Uvicorn
# --workers 2 : dimensionné pour 700m CPU (règle : 2 × 0.7 + 1 ≈ 2)
# Permet de tenir 150 req/min sans saturer (Flask mono-threadé
# aurait saturé à ce niveau — cf. défi spécifique du Cas 3)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]