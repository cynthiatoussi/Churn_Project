# projet-cpr — Prédiction de churn et recommandation d'offre

Pipeline ML multi-services déployé sur Kubernetes (Minikube).  
Cas 3 du TP : opérateur télécom — identification des clients à risque de résiliation.

- **Étudiant** : Priscille Mawut Toussi
- **Namespace** : `projet-cpr`
- **Repo GitHub** : https://github.com/cynthiatoussi/Churn_Project
- **Docker Hub** : https://hub.docker.com/u/cyndy155

---

## Déploiement rapide (depuis un git clone)

```bash
# 1. Démarrer Minikube avec les contraintes imposées par le TP
minikube start --cpus=4 --memory=6144 --driver=docker

# 2. Créer le namespace
kubectl create namespace projet-cpr

# 3. Déployer l'intégralité du système en une commande
kubectl apply -f k8s/ -n projet-cpr

# 4. Vérifier que les 4 pods sont en Running
kubectl get all -n projet-cpr
```

> ⚠️ Sur Windows avec Docker driver, les images sont
> automatiquement téléchargées depuis Docker Hub au démarrage.

---

## Accès aux services (port-forward)

Sur Windows avec Docker driver, utiliser `kubectl port-forward` :

```bash
# Lancer les 4 port-forwards en une commande
kubectl port-forward svc/inference-svc 8000:8000 -n projet-cpr &
kubectl port-forward svc/monitoring-svc 8002:8002 -n projet-cpr &
kubectl port-forward svc/frontend-svc 3000:80 -n projet-cpr &
```

| Service | URL locale | Description |
|---|---|---|
| Frontend | http://localhost:3000 | Interface web |
| Inference | http://localhost:8000 | API prédiction |
| Monitoring | http://localhost:8002/metrics | Métriques temps réel |

> Sur Windows, tuer les port-forwards existants si besoin :
> `taskkill //F //IM kubectl.exe`

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

## Lancer le script de charge

```bash
# Test nominal — 10 req/min, 5 min
python scripts/load_test.py --case churn --level nominal --url http://localhost:8000/predict

# Test charge — 50 req/min, 5 min
python scripts/load_test.py --case churn --level charge --url http://localhost:8000/predict

# Test stress — 150 req/min, 5 min
python scripts/load_test.py --case churn --level stress --url http://localhost:8000/predict
```

Consulter les métriques pendant les tests :

```bash
curl http://localhost:8002/metrics
```

---

## Images Docker Hub

Les images sont disponibles publiquement sur Docker Hub :

| Image | Tag | Taille |
|---|---|---|
| `cyndy155/preprocessing-cpr` | `1.0.0` | ~150 Mo |
| `cyndy155/inference-cpr` | `1.0.2` | ~961 Mo |
| `cyndy155/monitoring-cpr` | `1.0.0` | ~150 Mo |
| `cyndy155/frontend-cpr` | `1.0.0` | ~30 Mo |

Télécharger les images manuellement si besoin :

```bash
docker pull cyndy155/preprocessing-cpr:1.0.0
docker pull cyndy155/inference-cpr:1.0.2
docker pull cyndy155/monitoring-cpr:1.0.0
docker pull cyndy155/frontend-cpr:1.0.0
```

---

## Structure du projet

```
projet-cpr/
├── .github/workflows/ci.yml     # Pipeline CI/CD GitHub Actions
├── scripts/load_test.py         # Script de charge
├── data/                        # Datasets (non versionnés)
├── models/                      # Artefacts entraînés (.pkl + config.json)
│   └── README.md                # Fiche de validation des modèles
├── services/
│   ├── preprocessing/           # Feature engineering (port 8001)
│   ├── inference/               # Inférence ML (port 8000)
│   ├── monitoring/              # Métriques (port 8002)
│   └── frontend/                # Interface web nginx (port 80)
├── tests/                       # Tests unitaires — couverture 90%
├── k8s/                         # Manifests Kubernetes
│   ├── quota.yaml               # ResourceQuota namespace
│   ├── limitrange.yaml          # LimitRange par conteneur
│   ├── preprocessing.yaml       # Deployment + Service ClusterIP
│   ├── inference.yaml           # Deployment + Service NodePort
│   ├── monitoring.yaml          # Deployment + Service ClusterIP
│   └── frontend.yaml            # Deployment + Service NodePort
├── docker-compose.yml           # Déploiement local (dev)
├── train_models.py              # Script d'entraînement des modèles
├── pytest.ini                   # Configuration pytest
├── ADR.md                       # Architecture Decision Record
└── README.md
```

---

## Services et ressources

Valeurs basées sur `kubectl top pods` pendant le stress test à 150 req/min.

| Service | Port | CPU request | RAM request | CPU limit | RAM limit | Type |
|---|---|---|---|---|---|---|
| preprocessing | 8001 | 100m | 128Mi | 200m | 256Mi | ClusterIP |
| inference | 8000 | 100m | 400Mi | 300m | 512Mi | NodePort |
| monitoring | 8002 | 50m | 64Mi | 100m | 128Mi | ClusterIP |
| frontend | 80 | 50m | 64Mi | 100m | 128Mi | NodePort |
| **TOTAL** | | **300m** | **656Mi** | **700m** | **1024Mi** | |
| **QUOTA** | | **2500m** | **1536Mi** | **2500m** | **1536Mi** | |
| **MARGE** | | **2200m** | **880Mi** | **1800m** | **512Mi** | |

---

## Mesures kubectl top pods (stress test — 150 req/min)

```
NAME                             CPU(cores)   MEMORY(bytes)
inference-b767754b7-x4vsz        67m          374Mi
monitoring-7bffc4756f-tfzpx      5m           38Mi
preprocessing-7d4899694b-mn2mz   5m           38Mi
frontend-858df8f78-76bg4         1m           10Mi
```

Règle appliquée : `requests = 70-120% du pic mesuré` · `limits = 120-130% du pic`

---

## CI/CD

Le pipeline GitHub Actions se déclenche à chaque push sur `main` :

1. **Job test** : pytest avec couverture ≥ 80% (couverture obtenue : **90%**, 36 tests)
2. **Job build** : construit et pousse les 4 images sur Docker Hub si les tests passent

Le pipeline est visible sur :
https://github.com/cynthiatoussi/Churn_Project/actions

---

## Entraînement des modèles

L'entraînement se fait hors Minikube (en local) :

```bash
pip install pandas scikit-learn==1.5.1 xgboost joblib numpy
python train_models.py
```

Les artefacts générés (`churn_pipeline.pkl`, `offer_pipeline.pkl`, `config.json`)
sont versionnés dans `models/`.