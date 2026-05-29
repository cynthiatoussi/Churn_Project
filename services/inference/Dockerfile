# ─────────────────────────────────────────────────────────────
# services/inference/Dockerfile — multi-stage build
# ─────────────────────────────────────────────────────────────
# Multi-stage build OBLIGATOIRE car l'image dépasse 1Go (1.54Go).
# Justification dans l'ADR.
#
# Problème : scikit-learn, xgboost et numpy nécessitent gcc/g++
# pour compiler leurs extensions C pendant pip install.
# Ces outils de compilation ne servent plus à RIEN une fois
# les packages installés — mais ils pèsent ~250Mo.
#
# Solution multi-stage :
#   Stage 1 (builder) : installe TOUT dont les build tools
#   Stage 2 (runtime) : copie uniquement les packages installés
#                       sans les compilateurs → image finale allégée
#
# Réduction attendue : ~1.54Go → ~1.1-1.2Go
# ─────────────────────────────────────────────────────────────

# ── Stage 1 : builder ────────────────────────────────────────
# Installe toutes les dépendances Python avec les build tools.
# Cette image intermédiaire est JETÉE après le build.
FROM python:3.11-slim AS builder

WORKDIR /app

# Installe les build tools nécessaires pour compiler
# les extensions C de numpy, scikit-learn et xgboost
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copie et installe les dépendances dans /root/.local
# --user : installe dans l'espace utilisateur (plus facile à copier)
# --no-cache-dir : pas de cache pip pour réduire la taille
COPY services/inference/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2 : runtime ─────────────────────────────────────────
# Image finale légère — NE CONTIENT PAS gcc/g++ ni les outils de build.
# Copie uniquement les packages Python installés depuis le stage builder.
FROM python:3.11-slim

WORKDIR /app

# Copie les packages Python installés depuis le builder
# /root/.local contient tous les packages installés avec --user
COPY --from=builder /root/.local /root/.local

# Copie le code applicatif
# Chemin depuis la racine du projet (context = .)
COPY services/inference/app.py .

# Copie des artefacts modèles depuis models/ à la racine du repo
COPY models/churn_pipeline.pkl  ./models/
COPY models/offer_pipeline.pkl  ./models/
COPY models/config.json         ./models/

# Ajoute /root/.local/bin au PATH pour que les commandes
# installées avec --user soient accessibles (uvicorn, etc.)
ENV PATH=/root/.local/bin:$PATH

# Port exposé par le service d'inférence
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Démarrage avec 2 workers Uvicorn
# Dimensionné pour 700m CPU (règle : 2 × 0.7 + 1 ≈ 2)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]