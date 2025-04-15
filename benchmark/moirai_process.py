from pathlib import Path
import pandas as pd
import numpy as np

def load_multivariate_records(data_dir):
    records = []
    for file in sorted(Path(data_dir).glob("*_test.csv")):
        df = pd.read_csv(file)
        if df.shape[1] < 2:
            continue
        values = df.iloc[:, :2].dropna().to_numpy().T
        records.append({"target": values, "start": "2000-01-01 00:00:00"})
    print("Loaded", len(records), "multivariate sequences from", data_dir)
    return records

def load_google_covid_series(data_dir):
    records = []
    value_columns = [
        'new_confirmed', 'new_deceased', 'new_recovered', 'new_tested',
        'cumulative_confirmed', 'cumulative_deceased', 'cumulative_recovered', 'cumulative_tested'
    ]
    for file in sorted(Path(data_dir).glob("*.csv")):
        try:
            df = pd.read_csv(file)
            df = df.sort_values("date")
            start_date = pd.to_datetime(df["date"].dropna().iloc[0]) if "date" in df.columns else "2000-01-01 00:00:00"
            for col in value_columns:
                if col in df.columns:
                    series = df[col].dropna().astype(float).to_numpy()
                    if len(series) > 0:
                        records.append({"target": np.expand_dims(series, axis=0), "start": start_date})
        except Exception as e:
            print(f"Error reading {file.name}: {e}")
            continue
    print(f"Loaded {len(records)} univariate series from {data_dir}")
    return records

def load_cinc2012_test_records(data_dir):
    records = []
    for file in sorted(Path(data_dir).glob("*.txt")):
        try:
            df = pd.read_csv(file)
            if df.shape[1] < 3 or 'Value' not in df.columns:
                continue
            df = df.dropna(subset=['Value'])
            values = df['Value'].astype(float).to_numpy()
            start_time = pd.to_datetime(df['Time'].dropna().iloc[0], errors='coerce') if 'Time' in df.columns else "2000-01-01 00:00:00"
            if pd.isnull(start_time):
                start_time = "2000-01-01 00:00:00"
            records.append({"target": np.expand_dims(values, axis=0), "start": start_time})
        except Exception as e:
            print(f"Error reading {file.name}: {e}")
            continue
    print(f"Loaded {len(records)} sequences from {data_dir}")
    return records

def load_cdc_split_by_region(data_dir):
    records = []
    for file in sorted(Path(data_dir).glob("*.csv")):
        try:
            df = pd.read_csv(file)
            df = df.sort_values("Week Ending Date")
            start_date = pd.to_datetime(df["Week Ending Date"].dropna().iloc[0]) if "Week Ending Date" in df.columns else "2000-01-01 00:00:00"
            value_columns = [col for col in df.columns if col not in ["Week Ending Date", "Geographic aggregation"] and pd.api.types.is_numeric_dtype(df[col])]
            for col in value_columns:
                series = df[col].dropna().astype(float).to_numpy()
                if len(series) > 0:
                    records.append({"target": np.expand_dims(series, axis=0), "start": start_date})
        except Exception as e:
            print(f"Error reading {file.name}: {e}")
            continue
    print(f"Loaded {len(records)} CDC regional series from {data_dir}")
    return records

def load_jhu_timeseries(data_dir, is_us=False):
    records = []
    for file in sorted(Path(data_dir).glob("*.csv")):
        try:
            df = pd.read_csv(file)
            date_cols = df.columns[11:] if is_us else df.columns[4:]
            start_date = pd.to_datetime(date_cols[0])
            for i in range(df.shape[0]):
                values = df.iloc[i][date_cols].to_numpy(dtype=float)
                if not np.isnan(values).all():
                    records.append({"target": np.expand_dims(values, axis=0), "start": start_date})
        except Exception as e:
            print(f"Error reading {file.name}: {e}")
            continue
    print(f"Loaded {len(records)} JHU records from {data_dir}")
    return records

def load_dataset_by_type(data_dir, dataset_type="multivariate", jhu_is_us=False):
    if "jhcovid" in dataset_type:
        return load_jhu_timeseries(data_dir, is_us=jhu_is_us)
    elif "covid" in dataset_type:
        return load_google_covid_series(data_dir)
    elif "cinc" in dataset_type:
        return load_cinc2012_test_records(data_dir)
    elif "cdc" in dataset_type:
        return load_cdc_split_by_region(data_dir)
    else:
        return load_multivariate_records(data_dir)

__all__ = [
    "load_multivariate_records",
    "load_google_covid_series",
    "load_cinc2012_test_records",
    "load_cdc_split_by_region",
    "load_jhu_timeseries",
    "load_dataset_by_type"
]