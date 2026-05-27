# Fiche de validation des modèles

## churn_model.pkl

- **Dataset** : Telco Customer Churn (IBM) — 7 043 clients, 21 variables
- **Algorithme** : RandomForest (class_weight='balanced')
- **Accuracy** : 0.71
- **AUC-ROC** : 0.831
- **F1-score** : 0.609
- **Taille de l'artefact** : 22366 Ko
- **Temps d'inférence moyen** : *(à mesurer en local)*

## offer_model.pkl

- **Dataset** : Synthétique — 5 000 lignes, 5 catégories d'offres
- **Algorithme** : XGBoost multi-classe (5 catégories)
- **Accuracy** : 0.899
- **F1-score (weighted)** : 0.897
- **Taille de l'artefact** : 1341 ko
- **Temps d'inférence moyen** : *(à mesurer en local)*

## preprocessor.pkl

- **Type** : ColumnTransformer (OneHotEncoder + StandardScaler)
- **Fitté sur** : 80% du dataset Telco (train split)
- **Features catégorielles encodées** : gender, Partner, Dependents, PhoneService,
  MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
  TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod
- **Features numériques normalisées** : SeniorCitizen, tenure, MonthlyCharges, TotalCharges
