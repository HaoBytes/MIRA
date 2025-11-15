# Copyright (c) Microsoft
# Licensed under MIT

import torch
import torch.nn.functional as F
import json
import argparse

from MIRA.mira.models.modeling_mira import MIRAForPrediction
from MIRA.mira.models.utils_time_normalization import normalize_time_for_ctrope


def load_jsonl_timeseries(jsonl_path, use_mask=False):
    sequences = []
    times = []
    masks = []

    with open(jsonl_path, "r") as f:
        for line in f:
            obj = json.loads(line)

            seq = obj["sequence"]
            tms = obj.get("time", list(range(len(seq))))
            msk = obj.get("mask", [1] * len(seq))

            sequences.append(torch.tensor(seq, dtype=torch.float32))
            times.append(torch.tensor(tms, dtype=torch.float32))
            masks.append(torch.tensor(msk, dtype=torch.float32))

    dataset_values = torch.stack(sequences, dim=0)
    dataset_times = torch.stack(times, dim=0)
    dataset_masks = torch.stack(masks, dim=0)

    if use_mask:
        return dataset_values, dataset_times, dataset_masks
    return dataset_values, dataset_times

def normalize(values, mean, std):
    return (values - mean) / std


def denormalize(values, mean, std):
    return values * std + mean

def normalize_time_once(raw_times):
    """
    Perform CT-RoPE normalization once on full sequence.
    Keeps inference consistent across all autoregressive steps.
    """
    B, T = raw_times.shape
    attn = torch.ones_like(raw_times)

    t_scaled, t_min, t_max = normalize_time_for_ctrope(
        time_values=raw_times,
        attention_mask=attn,
        seq_length=T,
        alpha=1.0,
    )
    return t_scaled, t_min, t_max


def normalize_future_t(raw_t, t_min, t_max, L):
    """Normalize future timestamps."""
    denom = (t_max - t_min).clamp(min=1e-8)
    return (raw_t - t_min) / denom * (L - 1)

def mira_predict_autoreg_norm(model, values, raw_times, context_len, pred_len, mean, std):
    device = next(model.parameters()).device

    values = values.to(device)
    raw_times = raw_times.to(device)

    mean = mean.to(device)
    std = std.to(device)

    values_norm = normalize(values, mean, std)

    full_scaled_times, t_min, t_max = normalize_time_once(raw_times)

    hist_vals = values_norm[:, :context_len]
    hist_times = full_scaled_times[:, :context_len]

    fut_raw_times = raw_times[:, context_len:context_len + pred_len]

    preds_norm_list = []

    cur_vals = hist_vals.clone()
    cur_times = hist_times.clone()

    L = raw_times.shape[1]

    for i in range(pred_len):

        inp_vals = cur_vals.unsqueeze(-1)
        inp_times = cur_times

        with torch.no_grad():
            out = model(
                input_ids=inp_vals,
                time_values=inp_times,
                next_target_time_values=None,
                return_dict=True
            )

        next_norm = out.logits[:, -1, :]
        preds_norm_list.append(next_norm.squeeze(0))

        next_raw_t = fut_raw_times[:, i:i+1]
        next_scaled_t = normalize_future_t(next_raw_t, t_min, t_max, L)

        cur_vals = torch.cat([cur_vals, next_norm], dim=1)
        cur_times = torch.cat([cur_times, next_scaled_t], dim=1)

    preds_norm = torch.stack(preds_norm_list, dim=1)
    preds = denormalize(preds_norm, mean, std)

    return preds.squeeze(0)

def evaluate_one_window(model, seq, times, C, P, mean, std):
    preds = mira_predict_autoreg_norm(model, seq.unsqueeze(0), times.unsqueeze(0), C, P, mean, std)

    gt = seq[C:C+P]

    rmse = torch.sqrt(F.mse_loss(preds, gt)).item()
    mae = F.l1_loss(preds, gt).item()

    return rmse, mae

def evaluate_one_sequence_nonoverlap(model, seq, times, C, P, mean, std):
    rmses = []
    maes = []

    T = seq.size(0)
    window_size = C + P

    for start in range(0, T, window_size):
        end = start + window_size
        if end > T:
            break

        seq_win = seq[start:end]
        tms_win = times[start:end]

        rmse, mae = evaluate_one_window(model, seq_win, tms_win, C, P, mean, std)

        rmses.append(rmse)
        maes.append(mae)

    return rmses, maes

def rolling_eval_dataset(model, dataset_values, dataset_times, settings, normalize_each=True):
    results = {}

    for (C, P) in settings:
        print(f"\n===== Evaluating setting: context={C}, pred={P} =====")
        all_rmse, all_mae = [], []

        for idx in range(dataset_values.size(0)):
            seq = dataset_values[idx]
            times = dataset_times[idx]

            if seq.size(0) < C + P:
                continue

            if normalize_each:
                mean = seq.mean()
                std = seq.std() + 1e-6
            else:
                mean = torch.tensor(0.0)
                std = torch.tensor(1.0)

            rmses, maes = evaluate_one_sequence_nonoverlap(model, seq, times, C, P, mean, std)

            all_rmse.extend(rmses)
            all_mae.extend(maes)

        rmse_avg = sum(all_rmse) / len(all_rmse)
        mae_avg = sum(all_mae) / len(all_mae)

        results[(C, P)] = dict(rmse=rmse_avg, mae=mae_avg, n=len(all_rmse))

        print(f"[{C}->{P}] N={len(all_rmse)}, RMSE={rmse_avg:.4f}, MAE={mae_avg:.4f}")

    return results

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()

    print("[INFO] Loading model:", args.model)
    model = MIRAForPrediction.from_pretrained(args.model).cuda()
    model.eval()

    print("[INFO] Loading dataset:", args.data)
    dataset_values, dataset_times = load_jsonl_timeseries(args.data)
    print("Values:", dataset_values.shape)
    print("Times:", dataset_times.shape)

    settings = [
        (48, 24),
        (72, 36),
        (96, 48),
        (128, 64),
    ]

    results = rolling_eval_dataset(model, dataset_values, dataset_times, settings)

    total_rmse = [v["rmse"] for v in results.values()]
    total_mae = [v["mae"] for v in results.values()]

    print("\n======= FINAL SUMMARY =======")
    for (C,P), info in results.items():
        print(f"{C}->{P}: RMSE={info['rmse']:.4f}, MAE={info['mae']:.4f}, N={info['n']}")

    print("\nOVERALL AVERAGE")
    print("RMSE =", sum(total_rmse) / len(total_rmse))
    print("MAE  =", sum(total_mae) / len(total_mae))


if __name__ == "__main__":
    main()
