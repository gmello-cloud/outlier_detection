# ==========================================
# Outlier Detection via Random Forest
# ==========================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
 
# === 1. Caricamento dati
df = pd.read_csv("financial_data.csv")
 
# Chiavi univoche
key_cols = ["OA", "RD", "Contract ID", "Instrument ID"]
 
# === 2. Creazione automatica IS_REVISION
df = df.sort_values(key_cols + ["Version"])
 
def detect_revisions(group):
    group = group.sort_values("Version").copy()
    group["IS_REVISION"] = "N"  # default
    non_key_cols = [c for c in df.columns if c not in key_cols + ["Version"]]
    for i in range(1, len(group)):
        prev = group.iloc[i - 1]
        curr = group.iloc[i]
        # Verifica differenze significative nei campi non chiave
        diff = np.any(curr[non_key_cols].values != prev[non_key_cols].values)
        if diff:
            group.loc[group.index[i - 1], "IS_REVISION"] = "N"
            group.loc[group.index[i], "IS_REVISION"] = "Y"
    return group
 
df = df.groupby(key_cols, group_keys=False).apply(detect_revisions)
 
# === 3. Feature Engineering
target_col = "IS_REVISION"
X = df.drop(columns=key_cols + ["Version", target_col])
y = df[target_col].map({"Y": 1, "N": 0})
 
# Encoding variabili categoriali
X = pd.get_dummies(X, drop_first=True)
 
# Normalizzazione numerica
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# === 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
 
# === 5. Addestramento modello Random Forest
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
 
rf_model.fit(X_train, y_train)
 
# === 6. Valutazione del modello
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]
 
print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["N", "Y"]))
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")
 
# === 7. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC={roc_auc_score(y_test, y_proba):.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.tight_layout()
plt.show()
 
# === 8. Feature importance
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(15)
 
plt.figure(figsize=(8, 5))
top_features.plot(kind="barh")
plt.title("Top 15 Feature Importances - Random Forest")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
 
# === 9. Applicazione su nuovi dati
new_data = pd.read_csv("new_reporting_data.csv")
new_data_enc = pd.get_dummies(new_data, drop_first=True)
# Allinea colonne con quelle usate nel training
new_data_enc = new_data_enc.reindex(columns=X.columns, fill_value=0)
new_data_scaled = scaler.transform(new_data_enc)
new_data["OUTLIER_SCORE"] = rf_model.predict_proba(new_data_scaled)[:, 1]
 
new_data.to_csv("rf_outlier_predictions.csv", index=False)
print("Predizioni salvate in rf_outlier_predictions.csv")
