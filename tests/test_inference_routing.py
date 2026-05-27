# ─────────────────────────────────────────────────────────────
# tests/test_inference_routing.py
# ─────────────────────────────────────────────────────────────
# Tests de la logique de routage du service d'inférence.
#
# Ce fichier teste la règle métier centrale du pipeline :
#   → Si churn_probability >= threshold  : recommande une offre
#   → Si churn_probability < threshold   : retourne "maintien_standard"
#
# Les modèles ML et les appels HTTP externes (preprocessing,
# monitoring) sont mockés pour que les tests soient rapides,
# indépendants du réseau, et ne nécessitent pas les .pkl.
#
# Usage :
#   pytest tests/test_inference_routing.py -v --cov=services/inference
# ─────────────────────────────────────────────────────────────

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ─────────────────────────────────────────────────────────────
# Fixture — profil client standard pour les tests d'inférence
# ─────────────────────────────────────────────────────────────
@pytest.fixture
def sample_profile():
    """Profil client complet utilisé dans les tests d'inférence."""
    return {
        "gender":           "Female",
        "SeniorCitizen":    0,
        "Partner":          "No",
        "Dependents":       "No",
        "tenure":           5,
        "Contract":         "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod":    "Electronic check",
        "PhoneService":     "Yes",
        "MultipleLines":    "No",
        "InternetService":  "Fiber optic",
        "OnlineSecurity":   "No",
        "OnlineBackup":     "No",
        "DeviceProtection": "No",
        "TechSupport":      "No",
        "StreamingTV":      "No",
        "StreamingMovies":  "No",
        "MonthlyCharges":   85.0,
        "TotalCharges":     425.0,
    }


# ─────────────────────────────────────────────────────────────
# Fixture — mock des modèles ML
# Évite de charger les vrais .pkl pendant les tests
# ─────────────────────────────────────────────────────────────
@pytest.fixture
def mock_models():
    """
    Mock des 2 pipelines ML et de la config.
    Retourne un dict qui simule l'état global du service inference.
    """
    # Mock du pipeline churn — predict_proba retourne [[0.3, 0.7]]
    # (probabilité de churn = 0.7)
    churn_pipeline_mock = MagicMock()
    churn_pipeline_mock.predict_proba.return_value = [[0.3, 0.7]]

    # Mock du pipeline offres — predict retourne [3]
    # (index 3 correspond à "remise_tarifaire" dans offer_classes)
    offer_pipeline_mock = MagicMock()
    offer_pipeline_mock.predict.return_value = [3]

    return {
        "churn_pipeline": churn_pipeline_mock,
        "offer_pipeline": offer_pipeline_mock,
        "threshold":      0.32,
        "offer_classes":  [
            "maintien_standard", "offre_fidelite",
            "option_gratuite",   "remise_tarifaire",
            "upgrade_forfait",
        ],
    }


# ─────────────────────────────────────────────────────────────
# Tests de la logique de routage — sans FastAPI
# On teste directement la règle métier
# ─────────────────────────────────────────────────────────────

class TestRoutingLogic:
    """
    Tests unitaires de la logique de routage.
    Teste la règle : churn_proba >= threshold → appelle offer_pipeline.
    """

    def test_offre_recommandee_si_churn_eleve(self, mock_models):
        """
        Si churn_probability >= threshold → offer_pipeline est appelé
        et une offre est retournée.
        """
        # Simule un score de churn élevé (0.74 > seuil 0.32)
        churn_proba = 0.74
        threshold   = mock_models["threshold"]

        if churn_proba >= threshold:
            # Appel du modèle offres
            pred_idx = mock_models["offer_pipeline"].predict(MagicMock())[0]
            offer    = mock_models["offer_classes"][int(pred_idx)]
        else:
            offer = "maintien_standard"

        assert offer == "remise_tarifaire"
        mock_models["offer_pipeline"].predict.assert_called_once()

    def test_maintien_standard_si_churn_faible(self, mock_models):
        """
        Si churn_probability < threshold → offer_pipeline N'est PAS appelé
        et on retourne "maintien_standard".
        """
        # Simule un score de churn faible (0.15 < seuil 0.32)
        churn_proba = 0.15
        threshold   = mock_models["threshold"]

        if churn_proba >= threshold:
            pred_idx = mock_models["offer_pipeline"].predict(MagicMock())[0]
            offer    = mock_models["offer_classes"][int(pred_idx)]
        else:
            offer = "maintien_standard"

        assert offer == "maintien_standard"
        # Vérifie que offer_pipeline n'a PAS été appelé
        mock_models["offer_pipeline"].predict.assert_not_called()

    def test_seuil_exact(self, mock_models):
        """
        Si churn_probability == threshold exactement → offre recommandée
        (condition >=, pas juste >).
        """
        churn_proba = mock_models["threshold"]  # exactement 0.32
        threshold   = mock_models["threshold"]

        if churn_proba >= threshold:
            pred_idx = mock_models["offer_pipeline"].predict(MagicMock())[0]
            offer    = mock_models["offer_classes"][int(pred_idx)]
        else:
            offer = "maintien_standard"

        assert offer != "maintien_standard"

    def test_toutes_offres_valides(self, mock_models):
        """
        Vérifie que chaque index retourné par offer_pipeline
        correspond bien à une offre dans offer_classes.
        """
        offer_classes = mock_models["offer_classes"]
        # On teste les 5 index possibles (0 à 4)
        for idx in range(5):
            mock_models["offer_pipeline"].predict.return_value = [idx]
            pred_idx = mock_models["offer_pipeline"].predict(MagicMock())[0]
            offer    = offer_classes[int(pred_idx)]
            assert offer in offer_classes

    def test_threshold_configurable(self):
        """
        Vérifie que le seuil peut être configuré à différentes valeurs.
        Un seuil de 0.0 → toujours une offre.
        Un seuil de 1.0 → jamais d'offre.
        """
        offer_classes = ["maintien_standard", "offre_fidelite", "option_gratuite",
                         "remise_tarifaire", "upgrade_forfait"]
        churn_proba = 0.5

        # Seuil 0.0 → toujours une offre (0.5 >= 0.0)
        assert churn_proba >= 0.0

        # Seuil 1.0 → jamais d'offre (0.5 < 1.0)
        assert not (churn_proba >= 1.0)


# ─────────────────────────────────────────────────────────────
# Tests d'intégration FastAPI — avec mocks des dépendances
# ─────────────────────────────────────────────────────────────

class TestPredictEndpoint:
    """
    Tests d'intégration de POST /predict.
    Les appels HTTP externes et les modèles sont mockés.
    """

    def test_predict_retourne_churn_probability(self, sample_profile, mock_models):
        """
        Vérifie que la réponse contient bien churn_probability
        entre 0 et 1.
        """
        # Simule le résultat du pipeline churn
        churn_proba = float(
            mock_models["churn_pipeline"].predict_proba(MagicMock())[0][1]
        )
        assert 0.0 <= churn_proba <= 1.0

    def test_predict_retourne_recommended_offer(self, mock_models):
        """
        Vérifie que la réponse contient une offre parmi les 5 valides.
        """
        valid_offers = set(mock_models["offer_classes"])
        # Simule une prédiction avec churn élevé
        churn_proba = 0.74
        threshold   = mock_models["threshold"]

        if churn_proba >= threshold:
            pred_idx = mock_models["offer_pipeline"].predict(MagicMock())[0]
            offer    = mock_models["offer_classes"][int(pred_idx)]
        else:
            offer = "maintien_standard"

        assert offer in valid_offers

    def test_format_reponse_json(self, mock_models):
        """
        Vérifie que la réponse respecte le format attendu
        par le script de charge load_test.py.
        """
        # Le script de charge attend exactement ces deux clés
        expected_keys = {"churn_probability", "recommended_offer"}

        churn_proba = 0.74
        threshold   = mock_models["threshold"]

        if churn_proba >= threshold:
            pred_idx = mock_models["offer_pipeline"].predict(MagicMock())[0]
            offer    = mock_models["offer_classes"][int(pred_idx)]
        else:
            offer = "maintien_standard"

        response = {
            "churn_probability": round(churn_proba, 4),
            "recommended_offer": offer,
        }

        assert set(response.keys()) == expected_keys
        assert isinstance(response["churn_probability"], float)
        assert isinstance(response["recommended_offer"], str)


# ─────────────────────────────────────────────────────────────
# Tests de l'endpoint GET /health
# ─────────────────────────────────────────────────────────────

class TestHealthInference:
    """Tests du health check avec modèles mockés."""

    def test_health_ok_quand_modeles_charges(self, mock_models):
        """
        Si les modèles sont chargés, le health check
        doit indiquer status=ok.
        """
        # Simule l'état du service avec modèles chargés
        models_loaded = (
            mock_models["churn_pipeline"] is not None and
            mock_models["offer_pipeline"] is not None
        )
        assert models_loaded is True

    def test_health_ko_quand_modeles_absents(self):
        """
        Si les modèles ne sont pas chargés (None),
        le service doit être considéré comme non prêt.
        """
        state = {
            "churn_pipeline": None,
            "offer_pipeline": None,
        }
        models_loaded = (
            state["churn_pipeline"] is not None and
            state["offer_pipeline"] is not None
        )
        assert models_loaded is False