# Copyright (c) Microsoft
# Licensed under MIT

import torch
import torch.nn.functional as F
import json
import argparse

# MIRA imports
from MIRA.mira.models.modeling_mira import MIRAForPrediction
from MIRA.mira.models.utils_time_normalization import normalize_time_for_ctrope


def load_jsonl_timeseries(jsonl_path, use_mask=False):
    """
    Load a JSONL time-series dataset where each line contains:
        {"sequence": [...], "time": [...], "mask": [...]}

    Returns:
        dataset_values: [N, T]
        dataset_times : [N, T]
    """
    sequences, times, masks = [], [], []

    with open(jsonl_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            seq = obj["sequence"]
            tms = obj["time"] if "time" in obj else list(range(len(seq)))
            msk = obj["mask"] if "mask" in obj else [1] * len(seq)

            sequences.append(torch.tensor(seq, dtype=torch.float32))
            times.append(torch.tensor(tms, dtype=torch.float32))
            masks.append(torch.tensor(msk, dtype=torch.float32))

    dataset_values = torch.stack(sequences)
    dataset_times = torch.stack(times)

    if use_mask:
        dataset_masks = torch.stack(masks)
        return dataset_values, dataset_times, dataset_masks
    else:
        return dataset_values, dataset_times


def normalize(values, mean, std):
    return (values - mean) / std

def denormalize(values, mean, std):
    return values * std + mean


def normalize_time_once(raw_times):
    """
    Apply CT-RoPE time normalization ONCE for the whole sequence
    so that inference is consistent with training.
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

    T = raw_times.shape[1]

    for i in range(pred_len):

        inp_vals = cur_vals.unsqueeze(-1)
        inp_times = cur_times

        # autoregressive
        with torch.no_grad():
            out = model(
                input_ids=inp_vals,
                time_values=inp_times,
                next_target_time_values=None,
                return_dict=True
            )

        next_norm = out.logits[:, -1, :]  # shape [1,1]
        preds_norm_list.append(next_norm.squeeze(0))

        # Normalize future time
        next_raw_t = fut_raw_times[:, i:i+1]
        next_scaled_t = normalize_future_t(next_raw_t, t_min, t_max, T)

        # Append to running window
        cur_vals = torch.cat([cur_vals, next_norm], dim=1)
        cur_times = torch.cat([cur_times, next_scaled_t], dim=1)

    preds_norm = torch.stack(preds_norm_list, dim=1)
    preds = denormalize(preds_norm, mean, std)

    return preds.squeeze(0)


def evaluate_one_window(model, seq, times, C, P, mean, std):
    seq = seq.to(model.device)
    times = times.to(model.device)

    preds = mira_predict_autoreg_norm(
        model, seq.unsqueeze(0), times.unsqueeze(0), C, P, mean, std
    )

    gt = seq[C:C+P]

    rmse = torch.sqrt(F.mse_loss(preds, gt))
    mae = F.l1_loss(preds, gt)
    return rmse.item(), mae.item()


def rolling_eval_dataset(model, dataset_values, dataset_times, settings):

    results = {}

    for (C, P) in settings:
        print(f"\n===== Evaluating: context={C}, pred={P} =====")

        rmses, maes = [], []

        for idx in range(dataset_values.size(0)):
            seq = dataset_values[idx]
            tms = dataset_times[idx]

            if seq.size(0) < C + P:
                continue

            mean = seq.mean()
            std = seq.std() + 1e-6

            rmse, mae = evaluate_one_window(model, seq, tms, C, P, mean, std)
            rmses.append(rmse)
            maes.append(mae)

        results[(C, P)] = {
            "rmse": sum(rmses) / len(rmses),
            "mae": sum(maes) / len(maes),
            "n": len(rmses),
        }

        print(f"[{C}->{P}] RMSE={results[(C,P)]['rmse']:.4f}, "
              f"MAE={results[(C,P)]['mae']:.4f}, N={results[(C,P)]['n']}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--jsonl_path", type=str, required=True)
    args = parser.parse_args()

    print(f"[INFO] Loading model: {args.model_ckpt}")
    model = MIRAForPrediction.from_pretrained(args.model_ckpt).cuda()
    model.eval()

    print("\n[INFO] Loading dataset...")
    dataset_values, dataset_times = load_jsonl_timeseries(args.jsonl_path)
    print(f"  Loaded values: {dataset_values.shape}")
    print(f"  Loaded times : {dataset_times.shape}")

    settings = [
        (48, 24),
        (72, 36),
        (96, 48),
        (120, 60),
    ]

    results = rolling_eval_dataset(
        model, dataset_values, dataset_times, settings
    )

    print("\n======= FINAL SUMMARY =======")
    for (C, P), info in results.items():
        print(f"{C}->{P}: RMSE={info['rmse']:.4f}, MAE={info['mae']:.4f}, N={info['n']}")

    rmse_avg = sum(info["rmse"] for info in results.values()) / len(results)
    mae_avg = sum(info["mae"] for info in results.values()) / len(results)

    print("\n======= OVERALL AVERAGE =======")
    print(f"Overall RMSE = {rmse_avg:.4f}")
    print(f"Overall MAE  = {mae_avg:.4f}")


if __name__ == "__main__":
    main()