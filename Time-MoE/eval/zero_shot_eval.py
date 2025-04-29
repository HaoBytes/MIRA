import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import argparse
import json

def load_national_weighted_ili_series(csv_path):
    import pandas as pd
    import numpy as np

    df = pd.read_csv(csv_path)

    print("Available columns:", df.columns.tolist())

    # 找到日期列
    date_col = None
    for col in df.columns:
        if "date" in col.lower():
            date_col = col
            break
    if date_col is None:
        raise ValueError("No DATE column found in the CSV file.")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # 只保留 % WEIGHTED ILI 列
    target_col = None
    for col in df.columns:
        if "weighted" in col.lower() and "ili" in col.lower():
            target_col = col
            break
    if target_col is None:
        raise ValueError("No '% WEIGHTED ILI' column found in the CSV file.")

    series_raw = df[target_col]
    mask = series_raw.notna().to_numpy()
    series = series_raw.interpolate(method="linear", limit_direction="both")
    values = series.astype(float).to_numpy()

    print(f"Using column '{target_col}': length={len(values)}, missing={np.sum(~mask)}")

    record = {
        "target": np.expand_dims(values, axis=0),  # shape [1, T]
        "start": df[date_col].iloc[0],
        "mask": np.expand_dims(mask.astype(bool), axis=0),
        "name": target_col
    }

    print(f"Loaded 1 national-level time series from {csv_path}")
    return [record]

def load_vitaldb_npy(data_dir):
    """
    Loads vitaldb time series data from .npy files in the specified directory.
    Each .npy file contains ABP, PPG, and ECG data. Each variable is treated as a separate time series.
    
    Args:
        data_dir (str): Directory containing .npy files.
    
    Returns:
        list: A list of dictionaries with keys "target" (time series) and "start" (start time).
    """
    records = []
    file_count = 0  # Counter to keep track of how many files have been loaded
    
    for file in sorted(Path(data_dir).glob("*.npy")):
        if file_count >= 10:  # Limit to the first 20 files
            break
        
        try:
            # Load the .npy file
            data = np.load(file, allow_pickle=True).item()
            
            # Extract ABP, PPG, and ECG, handle them as separate univariate series
            for variable in ['ABP', 'PPG', 'ECG']:
                if variable in data:
                    time_series = data[variable]
                    
                    # Handle NaN values by linear interpolation
                    mask = ~np.isnan(time_series)
                    values = time_series
                    # series = pd.Series(time_series).interpolate(method="linear", limit_direction="both")
                    # values = series.to_numpy()
                    
                    # Store the time series for this variable
                    records.append({
                        "target": np.expand_dims(values, axis=0),  # Make sure to store as 2D (T, 1)
                        "start": "2000-01-01 00:00:00",  # Use a fixed start time or adjust it if necessary
                        "mask": np.expand_dims(mask.astype(bool), axis=0)  # Mask for non-NaN values
                    })
            
            file_count += 1  # Increment the file counter
        
        except Exception as e:
            print(f"Error reading {file.name}: {e}")
            continue
    
    print(f"Loaded {len(records)} sequences from {data_dir}")
    return records

def load_cinc2019_jsonl_series(jsonl_path):
    """
    Load time series from a CINC2019-style JSONL file where each line contains:
    {
      "file": "xxx.psv",
      "start": "00:00",
      "target": [[...], [...], ...]  # shape [T, D]
    }

    Returns:
        list of dicts: each with keys "target": np.ndarray [T, D], "start": str
    """
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
                target_array = np.array(item["target"], dtype=np.float32)  # shape [T, D]
                start_time = item.get("start", "00:00")
                records.append({
                    "target": target_array,
                    "start": start_time,
                    "file": item.get("file", None)
                })
            except Exception as e:
                print(f"Error parsing line: {e}")
                continue

    print(f"Loaded {len(records)} multivariate sequences from {jsonl_path}")
    return records


def load_cdc_split_by_region(data_dir):
    selected_columns = [
        "Total COVID-19 Admissions",
        "Total Influenza Admissions",
        "Total RSV Admissions",
        "Number of ICU Beds Occupied"
    ]

    records = []
    for file in sorted(Path(data_dir).glob("*.csv")):
        try:
            df = pd.read_csv(file)
            df = df.sort_values("Week Ending Date")
            df["Week Ending Date"] = pd.to_datetime(df["Week Ending Date"])
            start_date = df["Week Ending Date"].iloc[0]

            for col in selected_columns:
                if col in df.columns:
                    series_raw = df[col]
                    mask = series_raw.notna().to_numpy()
                    series = series_raw.interpolate(method="linear", limit_direction="both")
                    values = series.astype(float).to_numpy()
                    if np.any(~np.isnan(values)):
                        records.append({
                            "target": np.expand_dims(values, axis=0),
                            "start": start_date,
                            "mask": np.expand_dims(mask.astype(bool), axis=0),
                        })
        except Exception as e:
            print(f"Error reading {file.name}: {e}")
            continue

    print(f"Loaded {len(records)} CDC regional series from {data_dir}")
    return records

def load_test_signal(csv_path, channel=0):
    df = pd.read_csv(csv_path)
    signal = df.iloc[:, channel].to_numpy()
    return signal.tolist()

def predict_one_window(model, input_seq, pred_len):
    device = model.device
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
    mean = input_tensor.mean(dim=-1, keepdim=True)
    std = input_tensor.std(dim=-1, keepdim=True)
    normed_input = (input_tensor - mean) / std

    with torch.no_grad():
        output = model.generate(normed_input, max_new_tokens=pred_len)

    normed_pred = output[:, -pred_len:]
    pred = normed_pred * std + mean
    return pred.squeeze(0).cpu().tolist()

def evaluate_sliding(model, signal, context_len, pred_len, step):
    total_len = len(signal)
    raw_mae, raw_mse, raw_rmse = [], [], []
    norm_mae, norm_mse, norm_rmse = [], [], []

    for start in range(0, total_len - (context_len + pred_len) + 1, step):
        context = signal[start:start+context_len]
        target = signal[start+context_len:start+context_len+pred_len]

        try:
            pred = predict_one_window(model, context, pred_len)

            # Raw
            mae_r = mean_absolute_error(target, pred)
            mse_r = mean_squared_error(target, pred)
            rmse_r = np.sqrt(mse_r)

            # Normalized
            t_mean, t_std = np.mean(target), np.std(target)
            if t_std == 0:
                continue
            target_n = (np.array(target) - t_mean) / t_std
            pred_n = (np.array(pred) - t_mean) / t_std

            mae_n = mean_absolute_error(target_n, pred_n)
            mse_n = mean_squared_error(target_n, pred_n)
            rmse_n = np.sqrt(mse_n)

            raw_mae.append(mae_r)
            raw_mse.append(mse_r)
            raw_rmse.append(rmse_r)

            norm_mae.append(mae_n)
            norm_mse.append(mse_n)
            norm_rmse.append(rmse_n)

        except Exception:
            continue

    return raw_mae, raw_mse, raw_rmse, norm_mae, norm_mse, norm_rmse

def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto"
    )

    print(f"Model '{args.model}' loaded on {model.device}")

    if args.data_type == "cdc":
        records = load_cdc_split_by_region(args.data_dir)
    elif args.data_type == "cinc2019":
        jsonl_path = Path(args.data_dir) / "test.jsonl"
        records = load_cinc2019_jsonl_series(jsonl_path)
    elif args.data_type == "national":
        csv_path = Path(args.data_dir) / "national_illness.csv"
        records = load_national_weighted_ili_series(csv_path)
    elif args.data_type == "vitaldb":
        records = load_vitaldb_npy(args.data_dir)
    else:
        raise ValueError("Unsupported data type. Use --data_type cdc, cinc2019, national or vitaldb")


    
    # records = load_cdc_split_by_region(args.data_dir) # cdc0iha
    # test_files = sorted(Path(args.data_dir).glob("*_test.csv")) mib-bh
    settings = [(512, 96), (1024, 192), (2048, 336), (3072, 720)]

    overall = {s: {"raw": [], "norm": []} for s in settings}

    # mit-bih
    # for file in tqdm(test_files, desc="Processing"):
    #     signals = [load_test_signal(file, channel=i) for i in [0, 1]]
    #     for ctx_len, pred_len in settings:
    #         step = args.step if args.step else ctx_len
    #         all_raw, all_norm = [], []

    #         for sig in signals:
    #             if len(sig) < ctx_len + pred_len:
    #                 continue
    #             rmae, rmse, rrmse, nmae, nmse, nrmse = evaluate_sliding(
    #                 model, sig, ctx_len, pred_len, step
    #             )
    #             if rmae:
    #                 all_raw.append([np.mean(rmae), np.mean(rmse), np.mean(rrmse)])
    #                 all_norm.append([np.mean(nmae), np.mean(nmse), np.mean(nrmse)])

    #         if all_raw:
    #             avg_raw = np.mean(all_raw, axis=0)
    #             avg_norm = np.mean(all_norm, axis=0)
    #             overall[(ctx_len, pred_len)]["raw"].append(avg_raw)
    #             overall[(ctx_len, pred_len)]["norm"].append(avg_norm)
    
    # cdc-iha
    # for record in tqdm(records, desc="Processing CDC series"):
    #     signal = record["target"].squeeze().tolist()

    #     for ctx_len, pred_len in settings:
    #         step = args.step if args.step else ctx_len
    #         if len(signal) < ctx_len + pred_len:
    #             continue

    #         rmae, rmse, rrmse, nmae, nmse, nrmse = evaluate_sliding(
    #             model, signal, ctx_len, pred_len, step
    #         )
    #         if rmae:
    #             avg_raw = [np.mean(rmae), np.mean(rmse), np.mean(rrmse)]
    #             avg_norm = [np.mean(nmae), np.mean(nmse), np.mean(nrmse)]
    #             overall[(ctx_len, pred_len)]["raw"].append(avg_raw)
    #             overall[(ctx_len, pred_len)]["norm"].append(avg_norm)

    #vitaldb
    for record in tqdm(records, desc="Processing Series"):
        signal = record["target"].squeeze().tolist()  # 1D list
        for ctx_len, pred_len in settings:
            step = args.step if args.step else ctx_len
            if len(signal) < ctx_len + pred_len:
                continue

            rmae, rmse, rrmse, nmae, nmse, nrmse = evaluate_sliding(
                model, signal, ctx_len, pred_len, step
            )
            if rmae:
                avg_raw = [np.mean(rmae), np.mean(rmse), np.mean(rrmse)]
                avg_norm = [np.mean(nmae), np.mean(nmse), np.mean(nrmse)]
                overall[(ctx_len, pred_len)]["raw"].append(avg_raw)
                overall[(ctx_len, pred_len)]["norm"].append(avg_norm)

                print(f"{ctx_len}->{pred_len} | Raw: MAE {avg_raw[0]:.4f} RMSE {avg_raw[2]:.4f} | "
                      f"Norm: MAE {avg_norm[0]:.4f} RMSE {avg_norm[2]:.4f}")

    # national
    # for record in tqdm(records, desc="Processing Series"):
    #     signal = record["target"].squeeze().tolist()  # 1D list
    #     name = record.get("name", "unknown")          # 变量名，比如 '% WEIGHTED ILI'

    #     for ctx_len, pred_len in settings:
    #         step = args.step if args.step else ctx_len
    #         if len(signal) < ctx_len + pred_len:
    #             continue

    #         rmae, rmse, rrmse, nmae, nmse, nrmse = evaluate_sliding(
    #             model, signal, ctx_len, pred_len, step
    #         )
    #         if rmae:  # 至少有一个有效 window
    #             avg_raw = [np.mean(rmae), np.mean(rmse), np.mean(rrmse)]
    #             avg_norm = [np.mean(nmae), np.mean(nmse), np.mean(nrmse)]

    #             overall[(ctx_len, pred_len)]["raw"].append(avg_raw)
    #             overall[(ctx_len, pred_len)]["norm"].append(avg_norm)

    #             # 打印每列结果（可选）
    #             print(f"{name} | {ctx_len}->{pred_len} | Raw: MAE {avg_raw[0]:.4f} RMSE {avg_raw[2]:.4f} | "
    #                 f"Norm: MAE {avg_norm[0]:.4f} RMSE {avg_norm[2]:.4f}")

        
    # cinc2019
    # for record in tqdm(records, desc="Processing Series"):
    #     targets = record["target"]  # shape: [T] or [T, D]
    #     if targets.ndim == 1:
    #         signals = [targets]
    #     else:
    #         signals = [targets[:, i] for i in range(targets.shape[1])]

    #     for ctx_len, pred_len in settings:
    #         step = args.step if args.step else ctx_len
    #         all_raw, all_norm = [], []

    #         for sig in signals:
    #             if len(sig) < ctx_len + pred_len:
    #                 continue
    #             rmae, rmse, rrmse, nmae, nmse, nrmse = evaluate_sliding(
    #                 model, sig.tolist(), ctx_len, pred_len, step
    #             )
    #             if rmae:
    #                 all_raw.append([np.mean(rmae), np.mean(rmse), np.mean(rrmse)])
    #                 all_norm.append([np.mean(nmae), np.mean(nmse), np.mean(nrmse)])

    #         if all_raw:
    #             avg_raw = np.mean(all_raw, axis=0)
    #             avg_norm = np.mean(all_norm, axis=0)
    #             overall[(ctx_len, pred_len)]["raw"].append(avg_raw)
    #             overall[(ctx_len, pred_len)]["norm"].append(avg_norm)

    print("\nSummary per Setting")
    raw_all, norm_all = [], []
    for (ctx, pred), results in overall.items():
        if results["raw"]:
            r = np.mean(results["raw"], axis=0)
            n = np.mean(results["norm"], axis=0)
            raw_all.append(r)
            norm_all.append(n)
            print(f"{ctx}->{pred} | Raw: MAE {r[0]:.4f} MSE {r[1]:.4f} RMSE {r[2]:.4f} | "
                  f"Norm: MAE {n[0]:.4f} MSE {n[1]:.4f} RMSE {n[2]:.4f}")
        else:
            print(f"{ctx}->{pred} | No valid results")

    if raw_all:
        r_avg = np.mean(raw_all, axis=0)
        n_avg = np.mean(norm_all, axis=0)
        print("\nOverall Average:")
        print(f"Raw:  MAE {r_avg[0]:.4f} MSE {r_avg[1]:.4f} RMSE {r_avg[2]:.4f}")
        print(f"Norm: MAE {n_avg[0]:.4f} MSE {n_avg[1]:.4f} RMSE {n_avg[2]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--data_type", type=str, required=True,
                        choices=["cdc", "cinc2019", "national", "vitaldb"],
                        help="Specify data type: cdc, cinc2019, national, or vitaldb")
    args = parser.parse_args()
    main(args)
