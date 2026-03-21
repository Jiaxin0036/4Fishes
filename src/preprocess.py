from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/output")

COLUMN_NAMES = (
    ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] +
    [f"sensor_{i}" for i in range(1, 22)]
)


def load_raw_data(fd: str = "FD001"):
    train_path = DATA_DIR / f"train_{fd}.txt"
    test_path = DATA_DIR / f"test_{fd}.txt"
    rul_path = DATA_DIR / f"RUL_{fd}.txt"

    train = pd.read_csv(train_path, sep=r"\s+", header=None)
    test = pd.read_csv(test_path, sep=r"\s+", header=None)
    rul_test = pd.read_csv(rul_path, sep=r"\s+", header=None)

    # Drop empty columns if they exist
    train = train.dropna(axis=1, how="all")
    test = test.dropna(axis=1, how="all")
    rul_test = rul_test.dropna(axis=1, how="all")

    train.columns = COLUMN_NAMES
    test.columns = COLUMN_NAMES
    rul_test.columns = ["true_rul"]

    return train, test, rul_test


def add_train_rul_labels(train: pd.DataFrame) -> pd.DataFrame:
    max_cycles = train.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]

    train = train.merge(max_cycles, on="engine_id", how="left")
    train["rul"] = train["max_cycle"] - train["cycle"]
    train['cycle_norm'] = train['cycle'] / train['max_cycle']
    train = train.drop(columns=["max_cycle"])

    return train


def get_feature_columns():
    return ["op_setting_1", "op_setting_2", "op_setting_3"] + [f"sensor_{i}" for i in range(1, 22)] + ["cycle_norm"]


def get_last_cycle_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("engine_id").tail(1).copy()


def save_processed_csvs(fd: str = "FD001"):
    train, test, rul_test = load_raw_data(fd)
    train = add_train_rul_labels(train)

    train.to_csv(OUTPUT_DIR / f"processed_train_{fd}.csv", index=False)
    test.to_csv(OUTPUT_DIR / f"processed_test_{fd}.csv", index=False)
    rul_test.to_csv(OUTPUT_DIR / f"processed_rul_{fd}.csv", index=False)

    print(f"Saved processed CSV files for {fd}.")


if __name__ == "__main__":
    save_processed_csvs("FD001")