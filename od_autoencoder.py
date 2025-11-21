# ==========================================
# Outlier Detection via Autoencoder (NN)
# ==========================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
 
# === 1. Caricamento dati
df = pd.read_csv("financial_data.csv")
 
# Chiavi del dataset
key_cols = ["OA", "RD", "Contract ID", "Instrument ID"]
 
# === 2. (Facoltativo) Etichettatura automatica IS_REVISION
# utile per validazione finale, ma non per il training dell’autoencoder
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
 
# === 3. Feature Engineering
target_col = "IS_REVISION"
X = df.drop(columns=key_cols + ["Version", target_col], errors="ignore")
 
# Encoding variabili categoriali
X = pd.get_dummies(X, drop_first=True)
 
# Scaling numerico
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# === 4. Train-test split (solo per validazione)
# Addestra l’autoencoder su dati “normali” (IS_REVISION = N)
if target_col in df.columns:
    normal_data = X_scaled[df[target_col] == "N"]
    test_data = X_scaled
    y_test = df[target_col].map({"Y": 1, "N": 0})
else:
    normal_data = X_scaled
    test_data = X_scaled
    y_test = None
 
X_train, X_val = train_test_split(normal_data, test_size=0.2, random_state=42)
 
# === 5. Costruzione dell’Autoencoder
input_dim = X_train.shape[1]
 
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation="relu")(input_layer)
encoded = Dense(32, activation="relu")(encoded)
bottleneck = Dense(8, activation="relu")(encoded)
decoded = Dense(32, activation="relu")(bottleneck)
decoded = Dense(64, activation="relu")(decoded)
output_layer = Dense(input_dim, activation="linear")(decoded)
 
autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer="adam", loss="mse")
 
autoencoder.summary()
 
# === 6. Addestramento con early stopping
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
 
history = autoencoder.fit(
    X_train, X_train,
    epochs=100,
    batch_size=64,
    shuffle=True,
    validation_data=(X_val, X_val),
    callbacks=[early_stop],
    verbose=1
)
 
# === 7. Analisi delle perdite (reconstruction error)
reconstructions = autoencoder.predict(test_data)
mse = np.mean(np.power(test_data - reconstructions, 2), axis=1)
 
# Definizione soglia di anomalia
threshold = np.percentile(mse, 95)  # top 5% come anomali
print(f"Soglia di anomalia: {threshold:.5f}")
 
df["RECON_ERROR"] = mse
df["OUTLIER_FLAG"] = (mse > threshold).astype(int)
 
# === 8. Valutazione (se disponibile IS_REVISION)
if y_test is not None:
    print("=== Validation using IS_REVISION ===")
    print(classification_report(y_test, df["OUTLIER_FLAG"], target_names=["N", "Y"]))
    auc = roc_auc_score(y_test, mse)
    print(f"AUC-ROC (reconstruction error vs IS_REVISION): {auc:.3f}")
 
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, mse)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Autoencoder Reconstruction Error")
    plt.legend()
    plt.tight_layout()
    plt.show()
 
# === 9. Esportazione dei risultati
df_outliers = df[key_cols + ["Version", "RECON_ERROR", "OUTLIER_FLAG"]]
df_outliers.to_csv("autoencoder_outlier_scores.csv", index=False)
print("File salvato: autoencoder_outlier_scores.csv")
