# ─────────────────────────────────────────────────────────────
# tests/test_preprocessing.py
# ─────────────────────────────────────────────────────────────
# Tests unitaires et d'intégration du service preprocessing.
#
# Couvre :
#   - engineer_features : logique de feature engineering
#   - POST /process     : endpoint FastAPI
#   - GET  /health      : endpoint health check
#
# Usage :
#   pytest tests/test_preprocessing.py -v --cov=services/preprocessing
# ─────────────────────────────────────────────────────────────

import pytest
from fastapi.testclient import TestClient

# Import du module preprocessing (ajouté au path via conftest.py)
from app import app, engineer_features, CustomerProfile

# ─────────────────────────────────────────────────────────────
# Client de test FastAPI — simule des vraies requêtes HTTP
# sans avoir besoin de lancer le serveur
# ─────────────────────────────────────────────────────────────
client = TestClient(app)


# ─────────────────────────────────────────────────────────────
# Fixture — profil client de base réutilisé dans tous les tests
# ─────────────────────────────────────────────────────────────
@pytest.fixture
def base_profile():
    """Profil client standard utilisé comme base pour les tests."""
    return {
        "gender":           "Male",
        "SeniorCitizen":    0,
        "Partner":          "Yes",
        "Dependents":       "No",
        "tenure":           24,
        "Contract":         "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod":    "Electronic check",
        "PhoneService":     "Yes",
        "MultipleLines":    "No",
        "InternetService":  "Fiber optic",
        "OnlineSecurity":   "No",
        "OnlineBackup":     "Yes",
        "DeviceProtection": "No",
        "TechSupport":      "No",
        "StreamingTV":      "Yes",
        "StreamingMovies":  "No",
        "MonthlyCharges":   75.0,
        "TotalCharges":     1800.0,
    }


# ─────────────────────────────────────────────────────────────
# Tests de engineer_features — logique métier
# ─────────────────────────────────────────────────────────────

class TestEngineerFeatures:
    """Tests unitaires de la fonction engineer_features."""

    def test_num_services_compte_correctement(self, base_profile):
        """
        Vérifie que num_services compte bien les colonnes 'Yes'.
        Avec PhoneService=Yes, OnlineBackup=Yes, StreamingTV=Yes → 3 services.
        """
        profile = CustomerProfile(**base_profile)
        result  = engineer_features(profile)
        # PhoneService=Yes, OnlineBackup=Yes, StreamingTV=Yes → 3
        assert result["num_services"] == 3

    def test_num_services_zero_si_aucun(self, base_profile):
        """Vérifie que num_services vaut 0 quand aucun service n'est souscrit."""
        base_profile.update({
            "PhoneService": "No", "MultipleLines": "No",
            "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "No", "StreamingMovies": "No",
        })
        profile = CustomerProfile(**base_profile)
        result  = engineer_features(profile)
        assert result["num_services"] == 0

    def test_tenure_group_new(self, base_profile):
        """Client avec tenure ≤ 12 mois → groupe 'new'."""
        base_profile["tenure"] = 6
        profile = CustomerProfile(**base_profile)
        result  = engineer_features(profile)
        assert result["tenure_group"] == "new"

    def test_tenure_group_mid(self, base_profile):
        """Client avec tenure entre 12 et 36 mois → groupe 'mid'."""
        base_profile["tenure"] = 24
        profile = CustomerProfile(**base_profile)
        result  = engineer_features(profile)
        assert result["tenure_group"] == "mid"

    def test_tenure_group_loyal(self, base_profile):
        """Client avec tenure > 36 mois → groupe 'loyal'."""
        base_profile["tenure"] = 48
        profile = CustomerProfile(**base_profile)
        result  = engineer_features(profile)
        assert result["tenure_group"] == "loyal"

    def test_tenure_group_limite_12(self, base_profile):
        """Client avec tenure exactement à 12 mois → groupe 'new'."""
        base_profile["tenure"] = 12
        profile = CustomerProfile(**base_profile)
        result  = engineer_features(profile)
        assert result["tenure_group"] == "new"

    def test_charge_per_tenure_calcul(self, base_profile):
        """
        Vérifie le calcul de charge_per_tenure.
        MonthlyCharges=75, tenure=24 → 75 / (24+1) = 3.0
        """
        base_profile["MonthlyCharges"] = 75.0
        base_profile["tenure"]         = 24
        profile = CustomerProfile(**base_profile)
        result  = engineer_features(profile)
        expected = 75.0 / (24 + 1)
        assert abs(result["charge_per_tenure"] - expected) < 1e-6

    def test_charge_per_tenure_tenure_zero(self, base_profile):
        """
        Vérifie que tenure=0 ne provoque pas de division par zéro.
        On utilise (tenure + 1) dans le calcul.
        """
        base_profile["tenure"]         = 0
        base_profile["MonthlyCharges"] = 50.0
        profile = CustomerProfile(**base_profile)
        result  = engineer_features(profile)
        # 50 / (0 + 1) = 50.0
        assert abs(result["charge_per_tenure"] - 50.0) < 1e-6

    def test_no_internet_true(self, base_profile):
        """Client sans internet → no_internet = 1."""
        base_profile["InternetService"] = "No"
        profile = CustomerProfile(**base_profile)
        result  = engineer_features(profile)
        assert result["no_internet"] == 1

    def test_no_internet_false(self, base_profile):
        """Client avec internet → no_internet = 0."""
        base_profile["InternetService"] = "Fiber optic"
        profile = CustomerProfile(**base_profile)
        result  = engineer_features(profile)
        assert result["no_internet"] == 0

    def test_toutes_features_presentes(self, base_profile):
        """Vérifie que les 4 nouvelles features sont bien dans le résultat."""
        profile = CustomerProfile(**base_profile)
        result  = engineer_features(profile)
        for feature in ["num_services", "tenure_group", "charge_per_tenure", "no_internet"]:
            assert feature in result, f"Feature manquante : {feature}"

    def test_features_originales_conservees(self, base_profile):
        """Vérifie que les features originales sont toujours présentes."""
        profile = CustomerProfile(**base_profile)
        result  = engineer_features(profile)
        for col in ["gender", "tenure", "MonthlyCharges", "Contract"]:
            assert col in result, f"Colonne originale manquante : {col}"


# ─────────────────────────────────────────────────────────────
# Tests de l'endpoint POST /process
# ─────────────────────────────────────────────────────────────

class TestProcessEndpoint:
    """Tests d'intégration de l'endpoint /process."""

    def test_process_retourne_200(self, base_profile):
        """Un profil valide doit retourner HTTP 200."""
        response = client.post("/process", json=base_profile)
        assert response.status_code == 200

    def test_process_retourne_json(self, base_profile):
        """La réponse doit être un JSON avec les features enrichies."""
        response = client.post("/process", json=base_profile)
        data = response.json()
        assert isinstance(data, dict)
        assert "num_services" in data
        assert "tenure_group" in data

    def test_process_payload_incomplet_retourne_422(self, base_profile):
        """Un payload incomplet (champ manquant) doit retourner HTTP 422."""
        del base_profile["tenure"]
        response = client.post("/process", json=base_profile)
        assert response.status_code == 422

    def test_process_payload_vide_retourne_422(self):
        """Un payload vide doit retourner HTTP 422."""
        response = client.post("/process", json={})
        assert response.status_code == 422


# ─────────────────────────────────────────────────────────────
# Tests de l'endpoint GET /health
# ─────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    """Tests de l'endpoint /health."""

    def test_health_retourne_200(self):
        """Le health check doit toujours retourner HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_retourne_status_ok(self):
        """La réponse du health check doit contenir status=ok."""
        response = client.get("/health")
        assert response.json()["status"] == "ok"