# ─────────────────────────────────────────────────────────────
# services/preprocessing/Dockerfile
# ─────────────────────────────────────────────────────────────
# Image de base : python:3.11-slim
# "slim" = image allégée sans outils inutiles (~50Mo vs ~900Mo)
# Pas besoin de multi-stage build car l'image finale
# sera bien en dessous de 1Go (pas de modèles lourds ici)
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Dossier de travail dans le conteneur
WORKDIR /app

# Copie et installation des dépendances en premier
# (couche Docker cachée — si requirements.txt ne change pas,
#  cette étape n'est pas reconstruite lors d'un rebuild)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code applicatif
COPY app.py .

# Port exposé par le service preprocessing
EXPOSE 8001

# Health check — Kubernetes vérifie toutes les 30s que le service répond
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"

# Commande de démarrage
# --host 0.0.0.0 : écoute sur toutes les interfaces réseau du conteneur
# --port 8001    : port du service preprocessing
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]