"""
train_models.py — version finale conforme au TP
─────────────────────────────────────────────────────────────────
Modèle 1 : RandomForest + class_weight='balanced' → score de churn (0-1)
Modèle 2 : XGBoost multi-classe → recommandation d'offre (5 catégories)
Structure : 2 Pipeline pkl + config.json
─────────────────────────────────────────────────────────────────
Usage :
    pip install pandas scikit-learn xgboost joblib numpy
    python train_models.py
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# ─────────────────────────────────────────────────────────────
# Chemins & constantes
# ─────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), "data", "churn.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

OFFER_CATEGORIES = [
    "remise_tarifaire", "upgrade_forfait", "offre_fidelite",
    "option_gratuite",  "maintien_standard",
]

# ─────────────────────────────────────────────────────────────
# 1. Chargement et nettoyage
# ─────────────────────────────────────────────────────────────
print("[1/6] Chargement du dataset...")
df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df.drop(columns=["customerID"], inplace=True)
df["Churn"] = (df["Churn"] == "Yes").astype(int)

churn_rate = df["Churn"].mean()
print(f"    {len(df)} lignes — taux de churn : {churn_rate:.1%}")

# ─────────────────────────────────────────────────────────────
# 2. Feature engineering
# ─────────────────────────────────────────────────────────────
print("[2/6] Feature engineering...")

service_cols = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
df["num_services"]      = df[service_cols].apply(
    lambda row: sum(v in ["Yes", "1"] for v in row), axis=1
)
df["tenure_group"]      = pd.cut(
    df["tenure"], bins=[0, 12, 36, 72],
    labels=["new", "mid", "loyal"], right=True,
)
df["charge_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
df["no_internet"]       = (df["InternetService"] == "No").astype(int)
print("    num_services, tenure_group, charge_per_tenure, no_internet")

# ─────────────────────────────────────────────────────────────
# 3. Features & splits
# ─────────────────────────────────────────────────────────────
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "tenure_group",
]
NUMERICAL_FEATURES = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "num_services", "charge_per_tenure", "no_internet",
]
FEATURE_COLS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

X = df[FEATURE_COLS]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train_m, X_val, y_train_m, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

# ─────────────────────────────────────────────────────────────
# 4. Modèle 1 — Pipeline RandomForest churn
#    class_weight='balanced' gère le déséquilibre des classes
#    (~equivalent de scale_pos_weight pour XGBoost)
# ─────────────────────────────────────────────────────────────
print("[3/6] Entraînement pipeline churn (RandomForest)...")

preprocessor_churn = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ("num", StandardScaler(), NUMERICAL_FEATURES),
])

# Fit preprocessor sur train_m pour threshold optimization sur val
X_train_m_enc = preprocessor_churn.fit_transform(X_train_m)
X_val_enc     = preprocessor_churn.transform(X_val)
X_test_enc    = preprocessor_churn.transform(X_test)
X_train_enc   = preprocessor_churn.transform(X_train)

rf_churn = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",   # gère le déséquilibre nativement
    random_state=42,
    n_jobs=-1,
)
rf_churn.fit(X_train_m_enc, y_train_m)

# Threshold optimization sur val set
print("[4/6] Optimisation du seuil...")
y_prob_val         = rf_churn.predict_proba(X_val_enc)[:, 1]
best_threshold, best_f1 = 0.5, 0.0
for t in np.arange(0.25, 0.65, 0.01):
    y_pred_t = (y_prob_val >= t).astype(int)
    f_t      = f1_score(y_val, y_pred_t, zero_division=0)
    if f_t > best_f1:
        best_f1, best_threshold = f_t, round(t, 2)
print(f"    Seuil optimal : {best_threshold}  (F1 val = {best_f1:.3f})")

# Réentraînement final sur tout X_train
rf_final = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf_final.fit(X_train_enc, y_train)

# Évaluation sur test set
y_prob_test  = rf_final.predict_proba(X_test_enc)[:, 1]
y_pred_churn = (y_prob_test >= best_threshold).astype(int)

acc_churn = accuracy_score(y_test, y_pred_churn)
f1_churn  = f1_score(y_test, y_pred_churn)
auc_churn = roc_auc_score(y_test, y_prob_test)

print(f"    Accuracy : {acc_churn:.3f}")
print(f"    F1-score : {f1_churn:.3f}")
print(f"    AUC-ROC  : {auc_churn:.3f}")

# Construction du Pipeline final
churn_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor_churn),
    ("model",        rf_final),
])

churn_path = os.path.join(MODELS_DIR, "churn_pipeline.pkl")
joblib.dump(churn_pipeline, churn_path)
size_churn = os.path.getsize(churn_path) / 1024
print(f"    Pipeline sauvegardé → {churn_path} ({size_churn:.0f} Ko)")

# ─────────────────────────────────────────────────────────────
# 5. Modèle 2 — Pipeline XGBoost multi-classe offres
#    Classifieur multi-classes sur données synthétiques
#    comme demandé dans le TP
# ─────────────────────────────────────────────────────────────
print("[5/6] Génération données synthétiques + pipeline offres (XGBoost multi-classe)...")

np.random.seed(42)
n_synthetic   = 5000
synthetic_idx = np.random.choice(len(X), size=n_synthetic, replace=True)
X_synthetic   = X.iloc[synthetic_idx].reset_index(drop=True)

tenure_s   = X_synthetic["tenure"].values
monthly_s  = X_synthetic["MonthlyCharges"].values
contract_s = X_synthetic["Contract"].values
services_s = X_synthetic["num_services"].values

offer_labels = []
for t, m, c, s in zip(tenure_s, monthly_s, contract_s, services_s):
    if t < 12 and m > 70:
        base = "remise_tarifaire"
    elif t > 36 and c == "Month-to-month":
        base = "offre_fidelite"
    elif m < 30 or s <= 1:
        base = "option_gratuite"
    elif c == "Two year":
        base = "maintien_standard"
    else:
        base = "upgrade_forfait"
    if np.random.random() < 0.08:
        autres = [o for o in OFFER_CATEGORIES if o != base]
        offer_labels.append(np.random.choice(autres))
    else:
        offer_labels.append(base)

y_synthetic = pd.Series(offer_labels)

synth_path = os.path.join(os.path.dirname(__file__), "data", "synthetic_offers.csv")
X_synthetic.copy().assign(recommended_offer=y_synthetic).to_csv(synth_path, index=False)
print(f"    Distribution : {y_synthetic.value_counts().to_dict()}")

le          = LabelEncoder()
y_synth_enc = le.fit_transform(y_synthetic)
n_classes   = len(le.classes_)

X_synth_train, X_synth_test, y_synth_train, y_synth_test = train_test_split(
    X_synthetic, y_synth_enc,
    test_size=0.2, random_state=42, stratify=y_synth_enc,
)

preprocessor_offer = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ("num", StandardScaler(), NUMERICAL_FEATURES),
])

# XGBoost multi-classe
offer_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor_offer),
    ("model", XGBClassifier(
        objective="multi:softmax",
        num_class=n_classes,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )),
])

offer_pipeline.fit(X_synth_train, y_synth_train)

y_pred_offer = offer_pipeline.predict(X_synth_test)
acc_offer    = accuracy_score(y_synth_test, y_pred_offer)
f1_offer     = f1_score(y_synth_test, y_pred_offer, average="weighted")

print(f"    Accuracy (test set) : {acc_offer:.3f}")
print(f"    F1 (test set)       : {f1_offer:.3f}")

offer_path = os.path.join(MODELS_DIR, "offer_pipeline.pkl")
joblib.dump(offer_pipeline, offer_path)
size_offer = os.path.getsize(offer_path) / 1024
print(f"    Pipeline sauvegardé → {offer_path} ({size_offer:.0f} Ko)")

# ─────────────────────────────────────────────────────────────
# config.json
# ─────────────────────────────────────────────────────────────
config = {
    "churn_threshold": best_threshold,
    "offer_classes":   list(le.classes_),
}
config_path = os.path.join(MODELS_DIR, "config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"    Config → {config_path} : {config}")

# ─────────────────────────────────────────────────────────────
# Résumé
# ─────────────────────────────────────────────────────────────
print("\n[6/6] Résumé pour models/README.md")
print("=" * 55)
print(f"  churn_pipeline.pkl")
print(f"    Algorithme  : RandomForest (class_weight='balanced')")
print(f"    Accuracy    : {acc_churn:.3f}")
print(f"    F1-score    : {f1_churn:.3f}")
print(f"    AUC-ROC     : {auc_churn:.3f}")
print(f"    Seuil opt.  : {best_threshold}")
print(f"    Taille      : {size_churn:.0f} Ko")
print()
print(f"  offer_pipeline.pkl")
print(f"    Algorithme  : XGBoost multi-classe (5 catégories)")
print(f"    Accuracy    : {acc_offer:.3f}")
print(f"    F1-score    : {f1_offer:.3f}")
print(f"    Taille      : {size_offer:.0f} Ko")
print()
print(f"  config.json")
print(f"    threshold   : {best_threshold}")
print(f"    classes     : {list(le.classes_)}")
print("=" * 55)
print("\nTous les artefacts sont dans models/")
print("Copie ces métriques dans models/README.md !")
