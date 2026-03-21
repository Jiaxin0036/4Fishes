# src/train_rul.py

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from preprocess import get_feature_columns


DATA_DIR = Path("data/output")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def train_rul_model(fd: str = "FD001"):
    train = pd.read_csv(DATA_DIR / f"processed_train_{fd}.csv")
    feature_cols = get_feature_columns()

    # removes overly high rul, assumes early cycles with high RUL are mostly healthy
    train['rul'] = train['rul'].clip(upper=100)

    X = train[feature_cols]
    y = train["rul"]

    # Split based on engine  
    train_units, val_units = train_test_split(train['engine_id'].unique(), test_size=0.2, random_state=42)

    train_df = train[train['engine_id'].isin(train_units)]
    test_df = train[train['engine_id'].isin(val_units)]

    X_train = train_df[feature_cols]
    y_train = train_df['rul']

    X_val = test_df[feature_cols]
    y_val = test_df['rul']

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    mae = mean_absolute_error(y_val, preds)
    rmse = mean_squared_error(y_val, preds) ** 0.5

    print(f"Validation MAE RF: {mae:.3f}")
    print(f"Validation RMSE RF: {rmse:.3f}")

    joblib.dump(model, MODEL_DIR / f"rul_model_{fd}.pkl")
    print(f"Saved RUL model for {fd}.")


if __name__ == "__main__":
    train_rul_model("FD001")