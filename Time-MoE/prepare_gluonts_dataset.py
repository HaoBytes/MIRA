import pandas as pd
from pathlib import Path
import json

def convert_to_gluonts_format(data_dir: str, output_jsonl: str):
    data_path = Path(data_dir)
    test_files = sorted(data_path.glob("*_test.csv"))
    samples = []

    for file in test_files:
        df = pd.read_csv(file)
        for channel in [0, 1]:
            signal = df.iloc[:, channel].tolist()
            sample = {
                "start": "2000-01-01 00:00:00",  # fake uniform timestamp
                "target": signal
            }
            samples.append(sample)

    # Write to JSON Lines format (GluonTS PandasDataset can read this)
    with open(output_jsonl, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Saved {len(samples)} samples to {output_jsonl}")

if __name__ == "__main__":
    convert_to_gluonts_format(
        data_dir="./mitbih_csv_split",
        output_jsonl="./mitbih_gluonts_test.jsonl"
    )
