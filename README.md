# projet-CPR — Prédiction de churn et recommandation d'offre

Pipeline ML multi-services déployé sur Kubernetes (Minikube).

## Déploiement

```bash
# 1. Démarrer Minikube
minikube start --cpus=4 --memory=6144 --driver=docker

# 2. Créer le namespace et appliquer les quotas
kubectl create namespace projet-CPR
kubectl apply -f k8s/ -n projet-CPR

# 3. Vérifier que tout tourne
kubectl get all -n projet-CPR
```

## Lancer le script de charge

```bash
# Récupérer l'URL du service d'inférence
minikube service inference-svc -n projet-CPR --url

# Test nominal
python scripts/load_test.py --case churn --level nominal --url http://HOST:PORT/predict

# Test charge
python scripts/load_test.py --case churn --level charge --url http://HOST:PORT/predict

# Test stress
python scripts/load_test.py --case churn --level stress --url http://HOST:PORT/predict
```

## Structure du projet

```
projet-CPR/
├── .github/workflows/ci.yml     # Pipeline CI/CD GitHub Actions
├── scripts/load_test.py         # Script de charge (fourni)
├── data/                        # Datasets (non versionnés)
├── models/                      # Artefacts entraînés (.pkl)
├── services/
│   ├── preprocessing/           # Service de preprocessing (port 8001)
│   ├── inference/               # Service d'inférence (port 8000)
│   └── monitoring/              # Service de monitoring (port 8002)
├── tests/                       # Tests unitaires (couverture ≥ 80%)
├── k8s/                         # Manifests Kubernetes
├── docker-compose.yml
└── ADR.md
```

## Services

| Service        | Port | CPU request | Memory request |
|----------------|------|-------------|----------------|
| preprocessing  | 8001 | 100m        | 128Mi          |
| inference      | 8000 | 700m        | 300Mi          |
| monitoring     | 8002 | 50m         | 64Mi           |

## Entraînement des modèles

```bash
pip install pandas scikit-learn xgboost joblib
python train_models.py
```
