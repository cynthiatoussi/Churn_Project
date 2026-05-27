"""
services/preprocessing/app.py
──────────────────────────────────────────────────────────────────
Service de preprocessing — reçoit les données brutes d'un client
Telco, applique le feature engineering, et retourne les features
enrichies au service d'inférence.

Ce service ne charge aucun modèle ML. Il fait uniquement des
transformations déterministes sur les colonnes du dataset Telco.
Le OneHotEncoding et le StandardScaler sont dans le Pipeline
du service d'inférence (churn_pipeline.pkl).

Endpoints :
    POST /process  ← reçoit le profil client brut, retourne les features enrichies
    GET  /health   ← health check Kubernetes
──────────────────────────────────────────────────────────────────
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Preprocessing Service")

# ─────────────────────────────────────────────────────────────
# Schéma d'entrée — toutes les colonnes du dataset Telco
# sans la colonne cible 'Churn' et sans 'customerID'
# Ces champs correspondent exactement au payload envoyé
# par le script de charge load_test.py
# ─────────────────────────────────────────────────────────────
class CustomerProfile(BaseModel):
    # Variables démographiques
    gender:           str
    SeniorCitizen:    int            # 0 ou 1
    Partner:          str            # Yes / No
    Dependents:       str            # Yes / No

    # Variables de contrat
    tenure:           float          # ancienneté en mois
    Contract:         str            # Month-to-month / One year / Two year
    PaperlessBilling: str            # Yes / No
    PaymentMethod:    str

    # Services téléphoniques
    PhoneService:     str            # Yes / No
    MultipleLines:    str            # Yes / No / No phone service

    # Services internet
    InternetService:  str            # DSL / Fiber optic / No
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
# Liste des colonnes "services" utilisées pour compter
# le nombre de services souscrits (feature num_services)
# ─────────────────────────────────────────────────────────────
SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies",
]


def engineer_features(profile: CustomerProfile) -> dict:
    """
    Applique le feature engineering sur un profil client brut.

    Features créées :
        num_services      : nombre de services souscrits (0 à 8)
        tenure_group      : new (0-12 mois) / mid (12-36) / loyal (36+)
        charge_per_tenure : ratio MonthlyCharges / (tenure + 1)
                            évite la division par zéro pour tenure=0
        no_internet       : 1 si pas d'internet, 0 sinon
                            signal fort de faible engagement

    Retourne un dict avec toutes les colonnes originales
    + les 4 nouvelles features, prêt à être consommé
    par les Pipeline sklearn du service d'inférence.
    """

    # Récupère toutes les valeurs du profil sous forme de dict
    data = profile.model_dump()

    # ── Feature 1 : nombre de services souscrits ──────────────
    # On compte les colonnes "services" qui valent "Yes"
    num_services = sum(
        1 for col in SERVICE_COLS
        if data.get(col) in ("Yes", "1", 1)
    )
    data["num_services"] = num_services

    # ── Feature 2 : groupe de tenure ──────────────────────────
    # Catégorise l'ancienneté en 3 groupes métier
    t = data["tenure"]
    if t <= 12:
        tenure_group = "new"
    elif t <= 36:
        tenure_group = "mid"
    else:
        tenure_group = "loyal"
    data["tenure_group"] = tenure_group

    # ── Feature 3 : ratio charge / tenure ─────────────────────
    # Signal de valeur client — élevé = client récent cher
    data["charge_per_tenure"] = data["MonthlyCharges"] / (t + 1)

    # ── Feature 4 : pas d'internet ────────────────────────────
    # Indicateur binaire — signal fort de faible engagement
    data["no_internet"] = 1 if data["InternetService"] == "No" else 0

    return data


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────
@app.post("/process")
def process(profile: CustomerProfile):
    """
    Reçoit un profil client brut et retourne les features enrichies.

    Appelé par le service d'inférence avant chaque prédiction.
    Le dict retourné est directement consommé par les Pipeline
    sklearn (churn_pipeline et offer_pipeline) via pd.DataFrame.
    """
    try:
        enriched = engineer_features(profile)
        return enriched
    except Exception as e:
        # Remonte l'erreur avec un message clair pour le debugging
        raise HTTPException(status_code=422, detail=f"Erreur preprocessing : {str(e)}")


@app.get("/health")
def health():
    """Health check pour Kubernetes — retourne 200 si le service est vivant."""
    return {"status": "ok", "service": "preprocessing"}