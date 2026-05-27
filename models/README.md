# Fiche de validation des modèles

## churn_model.pkl

- **Dataset** : Telco Customer Churn (IBM) — 7 043 clients, 21 variables
- **Algorithme** : XGBoost
- **AUC-ROC** : *(à remplir après entraînement)*
- **F1-score** : *(à remplir après entraînement)*
- **Taille de l'artefact** : *(à remplir)*
- **Temps d'inférence moyen** : *(à mesurer en local)*

## offer_model.pkl

- **Dataset** : Synthétique — 5 000 lignes, 5 catégories d'offres
- **Algorithme** : RandomForest
- **Accuracy** : *(à remplir après entraînement)*
- **F1-score (weighted)** : *(à remplir)*
- **Taille de l'artefact** : *(à remplir)*
- **Temps d'inférence moyen** : *(à mesurer en local)*

## preprocessor.pkl

- **Type** : ColumnTransformer (OneHotEncoder + StandardScaler)
- **Fitté sur** : 80% du dataset Telco (train split)
- **Features catégorielles encodées** : gender, Partner, Dependents, PhoneService,
  MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
  TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod
- **Features numériques normalisées** : SeniorCitizen, tenure, MonthlyCharges, TotalCharges
