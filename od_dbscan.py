# ==========================================
# Outlier Detection via DBSCAN
# ==========================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
 
# === 1. Caricamento dati
df = pd.read_csv("financial_data.csv")
 
# Chiavi identificative
key_cols = ["OA", "RD", "Contract ID", "Instrument ID"]
 
# === 2. (Facoltativo) Etichettatura automatica IS_REVISION
df = df.sort_values(key_cols + ["Version"])
 
def detect_revisions(group):
    group = group.sort_values("Version").copy()
    group["IS_REVISION"] = "N"
    non_key_cols = [c for c in df.columns if c not in key_cols + ["Version"]]
    for i in range(1, len(group)):
        prev = group.iloc[i - 1]
        curr = group.iloc[i]
        diff = np.any(curr[non_key_cols].values != prev[non_key_cols].values)
        if diff:
            group.loc[group.index[i - 1], "IS_REVISION"] = "N"
            group.loc[group.index[i], "IS_REVISION"] = "Y"
    return group
 
df = df.groupby(key_cols, group_keys=False).apply(detect_revisions)
 
# === 3. Preprocessing e Feature Engineering
target_col = "IS_REVISION"
X = df.drop(columns=key_cols + ["Version", target_col], errors="ignore")
 
# Encoding variabili categoriali
X = pd.get_dummies(X, drop_first=True)
 
# Scaling numerico
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# === 4. Applicazione DBSCAN
# Parametri di base: eps = raggio di vicinanza, min_samples = densità minima
db = DBSCAN(eps=1.5, min_samples=10, n_jobs=-1)
db.fit(X_scaled)
 
# Etichette dei cluster: -1 = rumore (outlier)
df["CLUSTER_ID"] = db.labels_
df["OUTLIER_FLAG"] = (db.labels_ == -1).astype(int)
 
# === 5. Visualizzazione PCA (2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
 
plt.figure(figsize=(7, 5))
plt.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=df["CLUSTER_ID"],
    cmap="tab10",
    s=10,
    alpha=0.7
)
plt.title("DBSCAN Clusters (Outliers = -1)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(label="Cluster ID")
plt.tight_layout()
plt.show()
 
# === 6. Valutazione (solo se IS_REVISION disponibile)
if target_col in df.columns:
    y_true = df[target_col].map({"Y": 1, "N": 0})
    print("=== Validation using IS_REVISION ===")
    print(classification_report(y_true, df["OUTLIER_FLAG"], target_names=["N", "Y"]))
    try:
        auc = roc_auc_score(y_true, df["OUTLIER_FLAG"])
        print(f"AUC-ROC: {auc:.3f}")
    except ValueError:
        print("AUC-ROC non calcolabile: classi non bilanciate o vuote.")
 
# === 7. Statistiche di output
n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
n_outliers = np.sum(df["OUTLIER_FLAG"])
print(f"Cluster trovati: {n_clusters}")
print(f"Outlier rilevati: {n_outliers} su {len(df)} ({100*n_outliers/len(df):.2f}%)")
 
# === 8. Esportazione risultati
df_out = df[key_cols + ["Version", "CLUSTER_ID", "OUTLIER_FLAG"]]
df_out.to_csv("dbscan_outlier_detection.csv", index=False)
print("File salvato: dbscan_outlier_detection.csv")
