import os
import argparse
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from gluonts.dataset.common import ListDataset
import torch
from uni2ts.model.moirai import MoiraiModule, MoiraiForecast
from uni2ts.model.moirai_moe import MoiraiMoEModule, MoiraiMoEForecast

def load_model_and_predictor(model_name, context_length, prediction_length, patch_size, batch_size, target_dim, freq):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_lower = model_name.lower()

    prediction_length = int(prediction_length)
    context_length = int(context_length)

    if "moirai" in model_lower and "moe" not in model_lower:
        module = MoiraiModule.from_pretrained(model_name)
        model = MoiraiForecast(module, prediction_length, context_length, patch_size, 100, target_dim, 0, 0)
        predictor = model.create_predictor(batch_size=batch_size)

        def predict_fn(context, start_time=None):
            dataset = ListDataset([{
                "start": start_time or "2000-01-01 00:00:00",
                "target": context.tolist()
            }], freq=freq, one_dim_target=(target_dim == 1))
            forecast = next(predictor.predict(dataset))
            return reshape_output(np.array(forecast.mean), context.shape)

        return predict_fn

    elif "moirai-moe" in model_lower:
        module = MoiraiMoEModule.from_pretrained(model_name)
        model = MoiraiMoEForecast(module, prediction_length, context_length, patch_size, 1, target_dim, 0, 0)
        predictor = model.create_predictor(batch_size=batch_size)

        def predict_fn(context, start_time=None):
            dataset = ListDataset([{
                "start": start_time or "2000-01-01 00:00:00",
                "target": context.tolist()
            }], freq=freq, one_dim_target=(target_dim == 1))
            forecast = next(predictor.predict(dataset))
            return reshape_output(np.array(forecast.mean), context.shape)

        return predict_fn

def reshape_output(output, target_shape):
    if output.shape != target_shape and output.T.shape == target_shape:
        return output.T
    return output

def parse_settings(settings_str):
    settings = []
    for pair in settings_str.split(","):
        ctx_len, pred_len = map(int, pair.strip().split(":"))
        settings.append((ctx_len, pred_len))
    return settings

def find_jsonl_files(directory):
    return list(Path(directory).rglob("*.jsonl"))

def load_sequence_data(jsonl_file):
    records = []
    with open(jsonl_file, "r") as f:
        for line in f:
            data = json.loads(line)
            seq = np.array(data["sequence"], dtype=np.float32)
            mask = np.array(data["mask"], dtype=bool)
            time = np.array(data.get("time", list(range(len(seq)))), dtype=np.float32)  # fallback to range if not provided
            records.append((seq, mask, time))
    return records

def evaluate_sliding(predictor_fn, sequence, mask, time, ctx_len, pred_len):
    step = ctx_len
    total_len = len(sequence)
    metrics_list = []
    for start in range(0, total_len - ctx_len - pred_len + 1, step):
        context = sequence[start:start + ctx_len]
        context_time = time[start:start + ctx_len]
        target = sequence[start + ctx_len:start + ctx_len + pred_len]
        target_mask = mask[start + ctx_len:start + ctx_len + pred_len]

        try:
            pred = predictor_fn(context.reshape(1, -1), start_time=context_time[0]).squeeze()
        except Exception:
            continue

        valid_mask = target_mask
        if np.sum(valid_mask) == 0:
            continue

        target_valid = target[valid_mask]
        pred_valid = pred[valid_mask]

        mae = mean_absolute_error(target_valid, pred_valid)
        mse = mean_squared_error(target_valid, pred_valid)
        rmse = np.sqrt(mse)

        metrics_list.append((mae, mse, rmse))

    if metrics_list:
        return np.mean(metrics_list, axis=0)
    else:
        return None

def main(args):
    settings = parse_settings(args.settings)
    jsonl_files = find_jsonl_files(args.input_dir)
    print(f"Discovered {len(jsonl_files)} JSONL files.")

    for ctx_len, pred_len in settings:
        print(f"\nEvaluating Context {ctx_len} Prediction {pred_len}")
        predictor_fn = load_model_and_predictor(
            model_name=args.model,
            context_length=ctx_len,
            prediction_length=pred_len,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            target_dim=1,
            freq="1D"
        )

        raw_maes, raw_rmses = [], []

        for jsonl_file in jsonl_files:
            records = load_sequence_data(jsonl_file)
            for seq, mask, time in records:
                if len(seq) < ctx_len + pred_len:
                    continue

                result = evaluate_sliding(predictor_fn, seq, mask, time, ctx_len, pred_len)
                if result:
                    mae, mse, rmse = result
                    raw_maes.append(mae)
                    raw_rmses.append(rmse)

        if raw_maes:
            print(f"Setting {ctx_len}->{pred_len} | Raw MAE: {np.mean(raw_maes):.6f} | RMSE: {np.mean(raw_rmses):.6f}")
        else:
            print(f"Setting {ctx_len}->{pred_len} | No valid results")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing JSONL files")
    parser.add_argument("--model", type=str, required=True, help="Model name, e.g., Salesforce/moirai-1.1-R-base")
    parser.add_argument("--settings", type=str, required=True, help="Settings string, e.g., 36:24,36:36")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=16)
    args = parser.parse_args()
    main(args)
