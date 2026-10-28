"""
Train Logistic Regression and Random Forest on the AI4I 2020 dataset.
Put ai4i2020.csv in this folder, then run:  python train_model.py
(Colab: upload ai4i2020.csv, paste this code in a cell, run it, download the output files.)

Outputs:
  prepared_predictive_maintenance.csv, logistic_regression_model.pkl,
  random_forest_model.pkl, feature_columns.pkl, metrics.json
"""
import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

FEATURES = ["Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
TARGET = "Machine failure"

# 1. Load
df = pd.read_csv("ai4i2020.csv")
print("Raw shape:", df.shape)

# 2. Preprocess: keep the 5 inputs + target, handle missing values and duplicates
df = df[FEATURES + [TARGET]]
print("Missing values:\n", df.isnull().sum())
df = df.dropna().drop_duplicates()
print("Shape after cleaning:", df.shape)
df.to_csv("prepared_predictive_maintenance.csv", index=False)

# 3. Split (stratified because failures are rare)
X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Models (scaling is inside the pipeline, so the app needs no separate scaler)
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42,
                                         class_weight="balanced"))]),
}

# 5. Train + evaluate
metrics = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics[name] = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-Score": f1_score(y_test, pred, zero_division=0),
        "Confusion Matrix": confusion_matrix(y_test, pred).tolist(),
    }
    print(name, {k: round(v, 4) for k, v in metrics[name].items()
                 if k != "Confusion Matrix"})
    print(confusion_matrix(y_test, pred))

best = max(metrics, key=lambda m: metrics[m]["F1-Score"])
metrics["_best_model"] = best
metrics["_rows"] = {"total": int(len(df)), "train": int(len(X_train)),
                    "test": int(len(X_test))}
print("Best model (by F1-Score):", best)

# 6. Save
with open("logistic_regression_model.pkl", "wb") as f:
    pickle.dump(models["Logistic Regression"], f)
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(models["Random Forest"], f)
with open("feature_columns.pkl", "wb") as f:
    pickle.dump(FEATURES, f)
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved all files.")
