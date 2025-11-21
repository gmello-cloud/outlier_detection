# ==========================================
# Outlier Detection via Gradient Boosting
# ==========================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
 
# === 1. Caricamento dati
df = pd.read_csv("financial_data.csv")
 
# Chiavi
key_cols = ["OA", "RD", "Contract ID", "Instrument ID"]
 
# === 2. Creazione automatica IS_REVISION
df = df.sort_values(key_cols + ["Version"])
 
def detect_revisions(group):
    group = group.sort_values("Version").copy()
    group["IS_REVISION"] = "N"  # default
    # confronta versioni successive
    for i in range(1, len(group)):
        prev = group.iloc[i - 1]
        curr = group.iloc[i]
        # confronta campi non chiave
        non_key_cols = [c for c in df.columns if c not in key_cols + ["Version"]]
        # se ci sono differenze nei campi numerici o categoriali
        diff = np.any(curr[non_key_cols].values != prev[non_key_cols].values)
        if diff:
            group.loc[group.index[i - 1], "IS_REVISION"] = "N"
            group.loc[group.index[i], "IS_REVISION"] = "Y"
    return group
 
df = df.groupby(key_cols, group_keys=False).apply(detect_revisions)
 
# === 3. Encoding e scaling
# Rimuovi colonne chiave e target
target_col = "IS_REVISION"
X = df.drop(columns=key_cols + ["Version", target_col])
y = df[target_col].map({"Y": 1, "N": 0})
 
# Gestione variabili categoriali (one-hot)
X = pd.get_dummies(X, drop_first=True)
 
# Normalizzazione
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# === 4. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
 
# === 5. Addestramento modello Gradient Boosting
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)
 
model.fit(X_train, y_train)
 
# === 6. Valutazione
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["N", "Y"]))
 
# === 7. Feature importance
import matplotlib.pyplot as plt
import xgboost as xgb
 
xgb.plot_importance(model, max_num_features=15)
plt.tight_layout()
plt.show()
 
# === 8. Applicazione per nuove osservazioni
# (es. rilevamento outlier su nuove segnalazioni)
new_records = pd.read_csv("new_reporting_data.csv")
new_records_enc = pd.get_dummies(new_records, drop_first=True)
# allinea colonne al training set
new_records_enc = new_records_enc.reindex(columns=X.columns, fill_value=0)
new_records_scaled = scaler.transform(new_records_enc)
y_pred_new = model.predict_proba(new_records_scaled)[:, 1]
new_records["OUTLIER_SCORE"] = y_pred_new
 
new_records.to_csv("outlier_predictions.csv", index=False)
print("Predizioni salvate in outlier_predictions.csv")
