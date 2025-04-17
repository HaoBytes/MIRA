import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dataset_preprocess import load_multivariate_records
from inference_wrappers import load_model_and_predictor

def main(args):
    context_list = [int(x.strip()) for x in args.context_lengths.split(",")]
    prediction_list = [int(x.strip()) for x in args.prediction_lengths.split(",")]
    step = args.step

    all_series = load_multivariate_records(args.data_dir)
    print(f"Loaded {len(all_series)} sequences from {args.data_dir}")

    setting_results = []

    for ctx, pdt in zip(context_list, prediction_list):
        print(f"\nEvaluating Context {ctx} Prediction {pdt}")
        predictor_fn = load_model_and_predictor(
            model_name=args.model,
            context_length=ctx,
            prediction_length=pdt,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            target_dim=args.target_dim,
            freq=args.freq
        )

        model_supports_multivariate = (
            "moirai" in args.model.lower() and "moe" not in args.model.lower()
        ) or "moe" in args.model.lower()

        raw_maes, raw_mses, raw_rmses = [], [], []
        norm_maes, norm_mses, norm_rmses = [], [], []

        for series_dict in tqdm(all_series, desc=f"{args.model}: {ctx}->{pdt}"):
            series = series_dict["target"]
            start_time = series_dict["start"]
            total_len = series.shape[1]
            if total_len < ctx + pdt:
                continue

            for start in range(0, total_len - ctx - pdt + 1, step or ctx):
                context = series[:, start:start+ctx]
                target = series[:, start+ctx:start+ctx+pdt]
                try:
                    if model_supports_multivariate:
                        pred = predictor_fn(context, start_time=start_time)
                        if pred.shape != target.shape:
                            print(f"[WARN] shape mismatch: pred {pred.shape} vs target {target.shape}")
                            continue
                    else:
                        pred_list, target_list = [], []
                        for ch in range(context.shape[0]):
                            ch_context = context[ch].reshape(1, -1)
                            ch_target = target[ch].reshape(1, -1)
                            try:
                                ch_pred = predictor_fn(ch_context, start_time=start_time)
                                if ch_pred.shape != ch_target.shape:
                                    print(f"[WARN] shape mismatch: pred {ch_pred.shape} vs target {ch_target.shape}")
                                    continue
                                pred_list.append(ch_pred)
                                target_list.append(ch_target)
                            except Exception as e:
                                print(f"[FAIL] Channel {ch}: {e}")
                                continue
                        if not pred_list:
                            continue
                        pred = np.concatenate(pred_list, axis=0)
                        target = np.concatenate(target_list, axis=0)

                    mae_raw = np.mean([mean_absolute_error(target[i], pred[i]) for i in range(target.shape[0])])
                    mse_raw = np.mean([mean_squared_error(target[i], pred[i]) for i in range(target.shape[0])])
                    rmse_raw = np.sqrt(mse_raw)

                    t_mean, t_std = target.mean(), target.std()
                    if t_std == 0:
                        continue
                    target_norm = (target - t_mean) / t_std
                    pred_norm = (pred - t_mean) / t_std

                    mae = np.mean([mean_absolute_error(target_norm[i], pred_norm[i]) for i in range(target.shape[0])])
                    mse = np.mean([mean_squared_error(target_norm[i], pred_norm[i]) for i in range(target.shape[0])])
                    rmse = np.sqrt(mse)

                    raw_maes.append(mae_raw)
                    raw_mses.append(mse_raw)
                    raw_rmses.append(rmse_raw)
                    norm_maes.append(mae)
                    norm_mses.append(mse)
                    norm_rmses.append(rmse)

                except Exception as e:
                    print(f"[FAIL] {e}")
                    continue

        setting_results.append({
            "context": ctx,
            "predict": pdt,
            "raw_mae": np.mean(raw_maes),
            "raw_rmse": np.mean(raw_rmses),
            "norm_mae": np.mean(norm_maes),
            "norm_rmse": np.mean(norm_rmses),
        })


    print("\nPer-Setting Results:")
    for s in setting_results:
        print(f"Setting {s['context']}->{s['predict']} | "
              f"Raw MAE: {s['raw_mae']:.6f}, RMSE: {s['raw_rmse']:.6f} | "
              f"Norm MAE: {s['norm_mae']:.6f}, RMSE: {s['norm_rmse']:.6f}")


    if setting_results:
        avg_raw_mae = np.mean([s["raw_mae"] for s in setting_results])
        avg_raw_rmse = np.mean([s["raw_rmse"] for s in setting_results])
        avg_norm_mae = np.mean([s["norm_mae"] for s in setting_results])
        avg_norm_rmse = np.mean([s["norm_rmse"] for s in setting_results])

        print("\n Overall average across all settings:")
        print(f"Raw   MAE: {avg_raw_mae:.6f} | RMSE: {avg_raw_rmse:.6f}")
        print(f"Norm  MAE: {avg_norm_mae:.6f} | RMSE: {avg_norm_rmse:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--target_dim", type=int, default=1)
    parser.add_argument("--freq", type=str, default="1D")
    parser.add_argument("--context_lengths", type=str, default="512")
    parser.add_argument("--prediction_lengths", type=str, default="96")
    args = parser.parse_args()
    main(args)
