# src/train_rul.py

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from preprocess import get_feature_columns


DATA_DIR = Path("data/output")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def train_rul_model(fd: str = "FD001"):
    train = pd.read_csv(DATA_DIR / f"processed_train_{fd}.csv")
    feature_cols = get_feature_columns()

    X = train[feature_cols]
    y = train["rul"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    mae = mean_absolute_error(y_val, preds)
    rmse = mean_squared_error(y_val, preds) ** 0.5

    print(f"Validation MAE: {mae:.3f}")
    print(f"Validation RMSE: {rmse:.3f}")

    joblib.dump(model, MODEL_DIR / f"rul_model_{fd}.pkl")
    print(f"Saved RUL model for {fd}.")


if __name__ == "__main__":
    train_rul_model("FD001")