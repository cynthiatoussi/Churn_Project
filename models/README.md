# Fiche de validation des modèles

## churn_pipeline.pkl

- **Contenu** : ColumnTransformer (OneHotEncoder + StandardScaler) + RandomForest
- **Dataset** : Telco Customer Churn (IBM) — 7 043 clients, 21 variables
- **Algorithme** : RandomForest (class_weight='balanced')
- **Features engineered** : num_services, tenure_group, charge_per_tenure, no_internet
- **Accuracy** : 0.714
- **AUC-ROC** : 0.831
- **F1-score** : 0.609
- **Seuil optimal** : 0.32 (optimisé sur val set — favorise le recall)
- **Taille** : ~22 Mo
- **Temps d'inférence** : ~30-50ms (mesuré via latence moyenne 47ms au stress test)

## offer_pipeline.pkl

- **Contenu** : ColumnTransformer (OneHotEncoder + StandardScaler) + XGBoost multi-classe
- **Dataset** : Synthétique — 5 000 lignes, 5 catégories d'offres, 8% de bruit
- **Algorithme** : XGBoost (objective=multi:softmax, num_class=5)
- **Classes** : maintien_standard, offre_fidelite, option_gratuite, remise_tarifaire, upgrade_forfait
- **Accuracy** : 0.899
- **F1-score (weighted)** : 0.897
- **Taille** : ~1.3 Mo
- **Temps d'inférence** : ~10-20ms

## config.json

- **Contenu** : paramètres de configuration du service d'inférence
- **churn_threshold** : 0.32 — seuil au-delà duquel l'offre est recommandée
- **offer_classes** : liste ordonnée des 5 catégories d'offres

## Note sur l'architecture Pipeline

Le préprocesseur (ColumnTransformer) est intégré directement dans chaque Pipeline sklearn.
À l'inférence, un seul appel `pipeline.predict_proba(df)` applique automatiquement
le preprocessing puis la prédiction. Cela évite toute désynchronisation entre
le préprocesseur et le modèle lors des mises à jour.