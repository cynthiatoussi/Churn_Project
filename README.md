# projet-cpr — Prédiction de churn et recommandation d'offre

Pipeline ML multi-services déployé sur Kubernetes (Minikube).  
Cas 3 du TP : opérateur télécom — identification des clients à risque de résiliation.

---

## Déploiement

```bash
# 1. Démarrer Minikube avec les contraintes imposées par le TP
minikube start --cpus=4 --memory=6144 --driver=docker

# 2. Créer le namespace
kubectl create namespace projet-cpr

# 3. Déployer l'intégralité du système en une commande
kubectl apply -f k8s/ -n projet-cpr

# 4. Vérifier que les 3 pods sont en Running
kubectl get all -n projet-cpr
```

---

## Accès aux services (port-forward)

Sur Windows avec Docker driver, utiliser kubectl port-forward :

```bash
# Terminal 1 — service d'inférence (script de charge + curl)
kubectl port-forward svc/inference-svc 8000:8000 -n projet-cpr

# Terminal 2 — service de monitoring (métriques)
kubectl port-forward svc/monitoring-svc 8002:8002 -n projet-cpr
```

---

## Lancer le script de charge

```bash
# Test nominal — 10 req/min, 5 min
python scripts/load_test.py --case churn --level nominal --url http://localhost:8000/predict

# Test charge — 50 req/min, 5 min
python scripts/load_test.py --case churn --level charge --url http://localhost:8000/predict

# Test stress — 150 req/min, 5 min
python scripts/load_test.py --case churn --level stress --url http://localhost:8000/predict
```

Consulter les métriques du monitoring pendant les tests :

```bash
curl http://localhost:8002/metrics
```

---

## Tester une requête en direct

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Male","SeniorCitizen":0,"Partner":"Yes","Dependents":"No",
       "tenure":5,"Contract":"Month-to-month","PaperlessBilling":"Yes",
       "PaymentMethod":"Electronic check","PhoneService":"Yes",
       "MultipleLines":"No","InternetService":"Fiber optic",
       "OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No",
       "TechSupport":"No","StreamingTV":"No","StreamingMovies":"No",
       "MonthlyCharges":85.0,"TotalCharges":425.0}'
```

Réponse attendue :
```json
{"churn_probability": 0.793, "recommended_offer": "remise_tarifaire"}
```

---

## Structure du projet

```
projet-cpr/
├── .github/workflows/ci.yml     # Pipeline CI/CD GitHub Actions
├── scripts/load_test.py         # Script de charge (fourni par l'enseignant)
├── data/                        # Datasets (non versionnés dans .gitignore)
├── models/                      # Artefacts entraînés (.pkl + config.json)
│   └── README.md                # Fiche de validation des modèles
├── services/
│   ├── preprocessing/           # Feature engineering (port 8001)
│   ├── inference/               # Inférence ML (port 8000)
│   └── monitoring/              # Métriques (port 8002)
├── tests/                       # Tests unitaires — couverture 90%
├── k8s/                         # Manifests Kubernetes
│   ├── quota.yaml               # ResourceQuota namespace
│   ├── limitrange.yaml          # LimitRange par conteneur
│   ├── preprocessing.yaml       # Deployment + Service ClusterIP
│   ├── inference.yaml           # Deployment + Service NodePort
│   └── monitoring.yaml          # Deployment + Service ClusterIP
├── docker-compose.yml           # Déploiement local (dev)
├── train_models.py              # Script d'entraînement des modèles
├── ADR.md                       # Architecture Decision Record
└── README.md
```

---

## Services et ressources

| Service       | Port | CPU request | Mémoire request | CPU limit | Mémoire limit | Type Service |
|---------------|------|-------------|-----------------|-----------|---------------|--------------|
| preprocessing | 8001 | 100m        | 128Mi           | 200m      | 256Mi         | ClusterIP    |
| inference     | 8000 | 300m        | 400Mi           | 900m      | 512Mi         | NodePort     |
| monitoring    | 8002 | 50m         | 64Mi            | 100m      | 128Mi         | ClusterIP    |
| **TOTAL**     |      | **450m**    | **592Mi**       | **1200m** | **896Mi**     |              |
| **QUOTA**     |      | **2500m**   | **1536Mi**      | **2500m** | **1536Mi**    |              |

---

## Mesures kubectl top pods (sous charge stress — 150 req/min)

Relevé pendant le challenge de charge séance 4 :

```
NAME                             CPU(cores)   MEMORY(bytes)
inference-99685488f-x4vsz        74m          383Mi
monitoring-5b874cc5f4-dn2bh      5m           38Mi
preprocessing-5c687f6b68-zksx2   5m           38Mi
```

Les valeurs de requests sont fixées à 70-120% du pic observé, conformément à la règle du cours.

---

## Entraînement des modèles

L'entraînement se fait hors Minikube (en local ou sur Google Colab) :

```bash
pip install pandas scikit-learn xgboost joblib numpy
python train_models.py
```

Les artefacts générés (`churn_pipeline.pkl`, `offer_pipeline.pkl`, `config.json`)
sont versionnés dans `models/`.

---

## CI/CD

Le pipeline GitHub Actions se déclenche à chaque push sur `main` :

1. **Job test** : lance pytest avec couverture ≥ 80% (couverture obtenue : 90%)
2. **Job build** : construit et pousse les 3 images sur Docker Hub si les tests passent

Images disponibles sur Docker Hub :
- `cyndy155/monitoring-cpr:1.0.0`
- `cyndy155/preprocessing-cpr:1.0.0`
- `cyndy155/inference-cpr:1.0.2`