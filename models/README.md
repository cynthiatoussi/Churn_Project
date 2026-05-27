# Fiche de validation des modèles

## churn_model.pkl

- **Dataset** : Telco Customer Churn (IBM) — 7 043 clients, 21 variables
- **Algorithme** : XGBoost
- **AUC-ROC** : 0.842
- **F1-score** : 0.626
- **Taille de l'artefact** : 266 Ko
- **Temps d'inférence moyen** : *(à mesurer en local)*

## offer_model.pkl

- **Dataset** : Synthétique — 5 000 lignes, 5 catégories d'offres
- **Algorithme** : RandomForest
- **Accuracy** : 0.999
- **F1-score (weighted)** : 0.999
- **Taille de l'artefact** : 1396 ko
- **Temps d'inférence moyen** : *(à mesurer en local)*

## preprocessor.pkl

- **Type** : ColumnTransformer (OneHotEncoder + StandardScaler)
- **Fitté sur** : 80% du dataset Telco (train split)
- **Features catégorielles encodées** : gender, Partner, Dependents, PhoneService,
  MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
  TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod
- **Features numériques normalisées** : SeniorCitizen, tenure, MonthlyCharges, TotalCharges
