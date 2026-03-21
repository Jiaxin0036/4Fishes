# src/train_anomaly.py

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from preprocess import get_feature_columns


DATA_DIR = Path("data/output")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def train_anomaly_model(fd: str = "FD001", healthy_cycle_cutoff: int = 30):
    train = pd.read_csv(DATA_DIR / f"processed_train_{fd}.csv")
    feature_cols = get_feature_columns()

    # Use early cycles as mostly healthy baseline
    baseline = train[train["cycle"] <= healthy_cycle_cutoff].copy()
    X = baseline[feature_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )
    model.fit(X_scaled)

    joblib.dump(model, MODEL_DIR / f"anomaly_model_{fd}.pkl")
    joblib.dump(scaler, MODEL_DIR / f"anomaly_scaler_{fd}.pkl")

    print(f"Saved anomaly model and scaler for {fd}.")


if __name__ == "__main__":
    train_anomaly_model("FD001")