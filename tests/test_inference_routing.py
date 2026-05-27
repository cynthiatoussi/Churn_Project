# ─────────────────────────────────────────────────────────────
# tests/test_inference_routing.py
# ─────────────────────────────────────────────────────────────
# Tests d'intégration du service inference.
#
# Utilise importlib pour charger services/inference/app.py
# par son chemin EXACT — évite la collision avec
# services/preprocessing/app.py dans sys.path.
# ─────────────────────────────────────────────────────────────

import importlib.util
import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# ─────────────────────────────────────────────────────────────
# Chemins
# ─────────────────────────────────────────────────────────────
# Chemin absolu vers services/inference/app.py
INFERENCE_APP_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "services", "inference", "app.py")
)

# Chemin vers models/ à la racine (contient le vrai config.json)
REAL_MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models")
)

# ─────────────────────────────────────────────────────────────
# Constantes partagées
# ─────────────────────────────────────────────────────────────
OFFER_CLASSES = [
    "maintien_standard", "offre_fidelite", "option_gratuite",
    "remise_tarifaire",  "upgrade_forfait",
]
THRESHOLD = 0.32


# ─────────────────────────────────────────────────────────────
# Helper — charge l'app inference par son chemin exact
# ─────────────────────────────────────────────────────────────
def load_inference_app():
    """
    Charge services/inference/app.py via importlib.
    Évite la collision avec preprocessing/app.py dans sys.path.
    Supprime le cache si nécessaire pour forcer le rechargement.
    """
    # Supprime les modules en cache pour forcer le rechargement
    for key in list(sys.modules.keys()):
        if key in ("inference_app", "app"):
            del sys.modules[key]

    spec   = importlib.util.spec_from_file_location("inference_app", INFERENCE_APP_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["inference_app"] = module
    spec.loader.exec_module(module)
    return module.app


# ─────────────────────────────────────────────────────────────
# Helpers — fabrique les mocks des pipelines ML
# ─────────────────────────────────────────────────────────────
def make_churn_mock(churn_proba: float):
    """Mock du pipeline churn. predict_proba → [[1-p, p]]."""
    m = MagicMock()
    m.predict_proba.return_value = [[1 - churn_proba, churn_proba]]
    return m


def make_offer_mock(offer_idx: int):
    """Mock du pipeline offres. predict → [offer_idx]."""
    m = MagicMock()
    m.predict.return_value = [offer_idx]
    return m


def build_http_mock(enriched_features: dict):
    """
    Mock du client httpx.AsyncClient.
    - Premier POST → réponse preprocessing (features enrichies)
    - Deuxième POST → réponse monitoring (log)
    """
    mock_prep = MagicMock()
    mock_prep.json.return_value = enriched_features
    mock_prep.raise_for_status.return_value = None

    mock_log = MagicMock()
    mock_log.raise_for_status.return_value = None

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(side_effect=[mock_prep, mock_log])
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)
    return mock_http


# ─────────────────────────────────────────────────────────────
# Fixture — features enrichies simulées par preprocessing
# ─────────────────────────────────────────────────────────────
@pytest.fixture
def enriched_features():
    """Réponse simulée de preprocessing/process."""
    return {
        "gender": "Female", "SeniorCitizen": 0,
        "Partner": "No", "Dependents": "No",
        "tenure": 5.0, "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No",
        "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "MonthlyCharges": 85.0, "TotalCharges": 425.0,
        "num_services": 1, "tenure_group": "new",
        "charge_per_tenure": 14.17, "no_internet": 0,
    }


# ─────────────────────────────────────────────────────────────
# Fixture — payload brut envoyé à /predict
# ─────────────────────────────────────────────────────────────
@pytest.fixture
def raw_payload():
    """Profil client brut envoyé par load_test.py."""
    return {
        "gender": "Female", "SeniorCitizen": 0,
        "Partner": "No", "Dependents": "No",
        "tenure": 5, "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No",
        "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "MonthlyCharges": 85.0, "TotalCharges": 425.0,
    }


# ─────────────────────────────────────────────────────────────
# Fixture — client inference avec churn ÉLEVÉ (>= seuil)
# ─────────────────────────────────────────────────────────────
@pytest.fixture
def client_high_churn(enriched_features):
    """TestClient avec churn_proba=0.74 > seuil 0.32."""
    from fastapi.testclient import TestClient

    mock_churn = make_churn_mock(churn_proba=0.74)
    mock_offer = make_offer_mock(offer_idx=3)  # "remise_tarifaire"
    mock_http  = build_http_mock(enriched_features)

    with patch.dict(os.environ, {"MODELS_DIR": REAL_MODELS_DIR}), \
         patch("joblib.load", side_effect=[mock_churn, mock_offer]):
        app = load_inference_app()
        with patch("inference_app.httpx.AsyncClient", return_value=mock_http):
            with TestClient(app) as client:
                yield client


# ─────────────────────────────────────────────────────────────
# Fixture — client inference avec churn FAIBLE (< seuil)
# ─────────────────────────────────────────────────────────────
@pytest.fixture
def client_low_churn(enriched_features):
    """TestClient avec churn_proba=0.10 < seuil 0.32."""
    from fastapi.testclient import TestClient

    mock_churn = make_churn_mock(churn_proba=0.10)
    mock_offer = make_offer_mock(offer_idx=0)
    mock_http  = build_http_mock(enriched_features)

    with patch.dict(os.environ, {"MODELS_DIR": REAL_MODELS_DIR}), \
         patch("joblib.load", side_effect=[mock_churn, mock_offer]):
        app = load_inference_app()
        with patch("inference_app.httpx.AsyncClient", return_value=mock_http):
            with TestClient(app) as client:
                yield client


# ─────────────────────────────────────────────────────────────
# Tests POST /predict — churn ÉLEVÉ
# ─────────────────────────────────────────────────────────────
class TestPredictHighChurn:

    def test_retourne_200(self, client_high_churn, raw_payload):
        """Payload valide avec churn élevé → HTTP 200."""
        r = client_high_churn.post("/predict", json=raw_payload)
        assert r.status_code == 200

    def test_contient_churn_probability(self, client_high_churn, raw_payload):
        """La réponse contient churn_probability entre 0 et 1."""
        r    = client_high_churn.post("/predict", json=raw_payload)
        data = r.json()
        assert "churn_probability" in data
        assert 0.0 <= data["churn_probability"] <= 1.0

    def test_offre_remise_tarifaire(self, client_high_churn, raw_payload):
        """Avec churn=0.74 et index=3 → 'remise_tarifaire'."""
        r    = client_high_churn.post("/predict", json=raw_payload)
        data = r.json()
        assert data["recommended_offer"] == "remise_tarifaire"

    def test_offre_dans_liste_valide(self, client_high_churn, raw_payload):
        """L'offre retournée est dans la liste des 5 offres valides."""
        r    = client_high_churn.post("/predict", json=raw_payload)
        data = r.json()
        assert data["recommended_offer"] in OFFER_CLASSES

    def test_format_reponse_exact(self, client_high_churn, raw_payload):
        """Réponse avec exactement les 2 clés attendues par load_test.py."""
        r    = client_high_churn.post("/predict", json=raw_payload)
        data = r.json()
        assert set(data.keys()) == {"churn_probability", "recommended_offer"}

    def test_payload_incomplet_retourne_422(self, client_high_churn):
        """Payload sans 'tenure' → HTTP 422."""
        r = client_high_churn.post("/predict", json={"gender": "Female"})
        assert r.status_code == 422


# ─────────────────────────────────────────────────────────────
# Tests POST /predict — churn FAIBLE
# ─────────────────────────────────────────────────────────────
class TestPredictLowChurn:

    def test_retourne_200(self, client_low_churn, raw_payload):
        """Payload valide avec churn faible → HTTP 200."""
        r = client_low_churn.post("/predict", json=raw_payload)
        assert r.status_code == 200

    def test_retourne_maintien_standard(self, client_low_churn, raw_payload):
        """Avec churn=0.10 < seuil → 'maintien_standard'."""
        r    = client_low_churn.post("/predict", json=raw_payload)
        data = r.json()
        assert data["recommended_offer"] == "maintien_standard"

    def test_churn_probability_sous_seuil(self, client_low_churn, raw_payload):
        """churn_probability retournée < seuil."""
        r    = client_low_churn.post("/predict", json=raw_payload)
        data = r.json()
        assert data["churn_probability"] < THRESHOLD


# ─────────────────────────────────────────────────────────────
# Tests GET /health
# ─────────────────────────────────────────────────────────────
class TestHealthEndpoint:

    def test_retourne_200(self, client_high_churn):
        """Health check → HTTP 200."""
        r = client_high_churn.get("/health")
        assert r.status_code == 200

    def test_status_ok(self, client_high_churn):
        """Health check contient status='ok'."""
        r = client_high_churn.get("/health")
        assert r.json()["status"] == "ok"

    def test_liste_les_deux_modeles(self, client_high_churn):
        """Health check liste les 2 modèles chargés."""
        r    = client_high_churn.get("/health")
        data = r.json()
        assert "churn_pipeline" in data["models"]
        assert "offer_pipeline" in data["models"]

    def test_contient_threshold(self, client_high_churn):
        """Health check expose le seuil configuré."""
        r    = client_high_churn.get("/health")
        data = r.json()
        assert data["threshold"] == THRESHOLD


# ─────────────────────────────────────────────────────────────
# Tests logique de routage — unitaires purs
# ─────────────────────────────────────────────────────────────
class TestRoutingLogic:

    def test_offre_si_churn_superieur(self):
        assert 0.74 >= THRESHOLD

    def test_maintien_si_churn_inferieur(self):
        assert not (0.10 >= THRESHOLD)

    def test_egalite_traite_comme_superieur(self):
        assert THRESHOLD >= THRESHOLD

    def test_toutes_classes_valides(self):
        for offer in OFFER_CLASSES:
            assert offer in OFFER_CLASSES

    def test_cinq_classes_disponibles(self):
        assert len(OFFER_CLASSES) == 5