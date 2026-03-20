import pandas as pd
import os

# Paths
RAW_PATH = "../data/raw/"
OUTPUT_PATH = "../data/"

# Column names
cols = ['unit_id', 'cycle'] + \
       [f'op_setting_{i}' for i in range(1, 4)] + \
       [f'sensor_{i}' for i in range(1, 22)]

datasets = ["FD001", "FD002", "FD003", "FD004"]


def load_file(filepath, split_name, dataset_name):
    df = pd.read_csv(filepath, sep=' ', header=None)
    df = df.dropna(axis=1)
    df.columns = cols

    df['split'] = split_name
    df['dataset'] = dataset_name

    # 🔥 Fix ID collision
    df['unit_id'] = dataset_name + "_" + df['unit_id'].astype(str)

    return df


def main():
    all_data = []
    metadata_list = []

    for d in datasets:
        print(f"Processing {d}...")

        train_df = load_file(os.path.join(RAW_PATH, f"train_{d}.txt"), "train", d)
        test_df = load_file(os.path.join(RAW_PATH, f"test_{d}.txt"), "test", d)

        all_data.append(train_df)
        all_data.append(test_df)

        # Asset metadata (from training only)
        meta = train_df.groupby('unit_id')['cycle'].max().reset_index()
        meta.columns = ['unit_id', 'max_cycle']
        meta['dataset'] = d

        # Add known conditions
        if d in ["FD001", "FD003"]:
            meta['conditions'] = "Sea Level"
        else:
            meta['conditions'] = "Multiple"

        if d in ["FD001", "FD002"]:
            meta['fault_mode'] = "HPC Degradation"
        else:
            meta['fault_mode'] = "HPC + Fan Degradation"

        metadata_list.append(meta)

    # Combine everything
    df = pd.concat(all_data, ignore_index=True)
    asset_metadata = pd.concat(metadata_list, ignore_index=True)

    # Split into sensor + ops
    sensor_cols = ['unit_id', 'cycle', 'dataset', 'split'] + \
                  [f'sensor_{i}' for i in range(1, 22)]

    ops_cols = ['unit_id', 'cycle', 'dataset'] + \
               [f'op_setting_{i}' for i in range(1, 4)]

    sensor_data = df[sensor_cols]
    ops_data = df[ops_cols]

    # Save
    sensor_data.to_csv(os.path.join(OUTPUT_PATH, "sensor_data.csv"), index=False)
    ops_data.to_csv(os.path.join(OUTPUT_PATH, "ops_data.csv"), index=False)
    asset_metadata.to_csv(os.path.join(OUTPUT_PATH, "asset_metadata.csv"), index=False)

    print("\n✅ All datasets processed successfully!")


if __name__ == "__main__":
    main()