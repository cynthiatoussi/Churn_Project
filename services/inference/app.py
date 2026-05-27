"""
services/inference/app.py
──────────────────────────────────────────────────────────────────
Service d'inférence — cœur du pipeline CPR.

Responsabilités :
    1. Charge les 2 modèles au démarrage (une seule fois en mémoire)
    2. Reçoit le profil client brut via POST /predict
    3. Envoie les données au service preprocessing pour enrichissement
    4. Prédit le score de churn avec churn_pipeline
    5. Si score > seuil → prédit l'offre recommandée avec offer_pipeline
    6. Envoie les métriques au service monitoring (fire-and-forget)
    7. Retourne la réponse JSON au client

Format de réponse attendu par le script de charge (load_test.py) :
    {"churn_probability": float, "recommended_offer": str}

Ports :
    - Ce service    : 8000
    - Preprocessing : 8001 (via http://preprocessing-svc:8001 en K8s)
    - Monitoring    : 8002 (via http://monitoring-svc:8002 en K8s)
──────────────────────────────────────────────────────────────────
"""

import json
import os
import time
from contextlib import asynccontextmanager

import httpx
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────
# URLs des services internes — lues depuis les variables
# d'environnement pour fonctionner à la fois en local
# (docker-compose) et dans Kubernetes (ClusterIP DNS)
# ─────────────────────────────────────────────────────────────
PREPROCESSING_URL = os.getenv("PREPROCESSING_URL", "http://preprocessing:8001")
MONITORING_URL    = os.getenv("MONITORING_URL",    "http://monitoring:8002")

# ─────────────────────────────────────────────────────────────
# Chemins des artefacts — configurables via variable d'env
# En Docker  : /app/models/    (copié par le Dockerfile)
# En local   : services/inference/models/ (chemin par défaut)
# En test    : models/ à la racine du projet (via MODELS_DIR=...)
# ─────────────────────────────────────────────────────────────
MODELS_DIR = os.getenv("MODELS_DIR",os.path.join(os.path.dirname(__file__), "models"))
CHURN_PIPELINE_PATH = os.path.join(MODELS_DIR, "churn_pipeline.pkl")
OFFER_PIPELINE_PATH = os.path.join(MODELS_DIR, "offer_pipeline.pkl")
CONFIG_PATH         = os.path.join(MODELS_DIR, "config.json")

# ─────────────────────────────────────────────────────────────
# État global — modèles chargés une fois au démarrage
# et conservés en mémoire pour toutes les requêtes suivantes
# ─────────────────────────────────────────────────────────────
state = {
    "churn_pipeline": None,
    "offer_pipeline": None,
    "threshold":      None,
    "offer_classes":  None,
}


# ─────────────────────────────────────────────────────────────
# Lifespan — chargement des modèles au démarrage du service
# Utilise le pattern lifespan de FastAPI (remplace @app.on_event)
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge les modèles en mémoire au démarrage du conteneur."""
    print("[STARTUP] Chargement des modèles...")

    # Chargement du pipeline churn (ColumnTransformer + RandomForest)
    state["churn_pipeline"] = joblib.load(CHURN_PIPELINE_PATH)
    print("[STARTUP] churn_pipeline chargé ✓")

    # Chargement du pipeline offres (ColumnTransformer + XGBoost multi-classe)
    state["offer_pipeline"] = joblib.load(OFFER_PIPELINE_PATH)
    print("[STARTUP] offer_pipeline chargé ✓")

    # Chargement de la config (seuil + noms des classes d'offres)
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    state["threshold"]     = config["churn_threshold"]
    state["offer_classes"] = config["offer_classes"]
    print(f"[STARTUP] config chargée — seuil={state['threshold']} ✓")
    print("[STARTUP] Service prêt à recevoir des requêtes")

    yield  # le service tourne ici

    # Nettoyage à l'arrêt
    print("[SHUTDOWN] Arrêt du service d'inférence")


app = FastAPI(title="Inference Service", lifespan=lifespan)


# ─────────────────────────────────────────────────────────────
# Schéma d'entrée — identique au payload du script de charge
# Toutes les colonnes du dataset Telco sauf Churn et customerID
# ─────────────────────────────────────────────────────────────
class CustomerProfile(BaseModel):
    # Variables démographiques
    gender:           str
    SeniorCitizen:    int
    Partner:          str
    Dependents:       str
    # Variables de contrat
    tenure:           float
    Contract:         str
    PaperlessBilling: str
    PaymentMethod:    str
    # Services téléphoniques
    PhoneService:     str
    MultipleLines:    str
    # Services internet
    InternetService:  str
    OnlineSecurity:   str
    OnlineBackup:     str
    DeviceProtection: str
    TechSupport:      str
    StreamingTV:      str
    StreamingMovies:  str
    # Variables financières
    MonthlyCharges:   float
    TotalCharges:     float


# ─────────────────────────────────────────────────────────────
# Schéma de réponse — format attendu par load_test.py
# ─────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    churn_probability: float
    recommended_offer: str


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
async def predict(profile: CustomerProfile):
    """
    Pipeline complet de prédiction :
        1. Appel preprocessing → features enrichies
        2. Prédiction churn
        3. Si churn >= seuil → prédiction offre
        4. Log des métriques (fire-and-forget)
        5. Retour de la réponse
    """
    t_start = time.monotonic()
    recommended_offer = "maintien_standard"  # valeur par défaut si pas de churn

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:

            # ── Étape 1 : appel au service preprocessing ──────
            prep_response = await client.post(
                f"{PREPROCESSING_URL}/process",
                json=profile.model_dump(),
            )
            prep_response.raise_for_status()
            enriched_features = prep_response.json()

            # ── Étape 2 : prédiction du score de churn ────────
            # Convertit le dict en DataFrame (format attendu par sklearn Pipeline)
            df = pd.DataFrame([enriched_features])

            # predict_proba retourne [[prob_0, prob_1]]
            # on prend la probabilité de churn (classe 1)
            churn_proba = float(
                state["churn_pipeline"].predict_proba(df)[0][1]
            )

            # ── Étape 3 : prédiction de l'offre si churn élevé ─
            # Le seuil configurable est chargé depuis config.json
            if churn_proba >= state["threshold"]:
                offer_pred        = state["offer_pipeline"].predict(df)[0]
                recommended_offer = state["offer_classes"][int(offer_pred)]

            # ── Étape 4 : log des métriques (fire-and-forget) ──
            # On n'attend pas la réponse du monitoring pour ne pas
            # ajouter de latence. Si monitoring est down, on continue.
            latency = time.monotonic() - t_start
            try:
                await client.post(
                    f"{MONITORING_URL}/log",
                    json={
                        "status_code":       200,
                        "latency_s":         round(latency, 4),
                        "churn_probability": churn_proba,
                        "recommended_offer": recommended_offer,
                    },
                )
            except Exception:
                # Le monitoring ne doit jamais bloquer l'inférence
                pass

    except httpx.HTTPStatusError as e:
        # Erreur HTTP du service preprocessing
        raise HTTPException(
            status_code=502,
            detail=f"Erreur preprocessing : {e.response.status_code}"
        )
    except Exception as e:
        # Log l'erreur dans monitoring avant de remonter
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                await client.post(
                    f"{MONITORING_URL}/log",
                    json={
                        "status_code": 500,
                        "latency_s":   round(time.monotonic() - t_start, 4),
                        "error":       str(e),
                    },
                )
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Erreur inférence : {str(e)}")

    return PredictionResponse(
        churn_probability=round(churn_proba, 4),
        recommended_offer=recommended_offer,
    )


@app.get("/health")
def health():
    """
    Health check pour Kubernetes.
    Vérifie que les modèles sont bien chargés en mémoire.
    """
    models_loaded = (
        state["churn_pipeline"] is not None and
        state["offer_pipeline"] is not None
    )
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Modèles non chargés")
    return {
        "status":    "ok",
        "service":   "inference",
        "threshold": state["threshold"],
        "models":    ["churn_pipeline", "offer_pipeline"],
    }