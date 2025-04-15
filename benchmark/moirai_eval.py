import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai import MoiraiModule, MoiraiForecast
from uni2ts.model.moirai_moe import MoiraiMoEModule, MoiraiMoEForecast

from dataset_preprocess import (
    load_dataset_by_type,
    load_multivariate_records,
    load_jhu_timeseries,
    load_google_covid_series,
    load_cdc_split_by_region,
    load_cinc2012_test_records
)

def main(args):
    context_list = [int(x.strip()) for x in args.context_lengths.split(",")]
    prediction_list = [int(x.strip()) for x in args.prediction_lengths.split(",")]
    step = args.step

    all_series = load_dataset_by_type(args.data_dir, dataset_type=args.dataset_type)
    print(f"Loaded {len(all_series)} sequences from {args.data_dir} [Type: {args.dataset_type}]")

    overall_raw = [{"mae": [], "mse": [], "rmse": []} for _ in range(len(context_list))]
    overall_norm = [{"mae": [], "mse": [], "rmse": []} for _ in range(len(context_list))]

    for i, (ctx, pdt) in enumerate(zip(context_list, prediction_list)):
        print(f"\nEvaluating Context {ctx} Prediction {pdt}")
        current_step = step if step is not None else ctx

        if "moe" in args.model:
            module = MoiraiMoEModule.from_pretrained(args.model)
            model = MoiraiMoEForecast(
                module=module,
                prediction_length=pdt,
                context_length=ctx,
                patch_size=args.patch_size,
                num_samples=100,
                target_dim=args.target_dim,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
        else:
            module = MoiraiModule.from_pretrained(args.model)
            model = MoiraiForecast(
                module=module,
                prediction_length=pdt,
                context_length=ctx,
                patch_size=args.patch_size,
                num_samples=100,
                target_dim=args.target_dim,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
        predictor = model.create_predictor(batch_size=args.batch_size)

        raw_maes, raw_mses, raw_rmses = [], [], []
        norm_maes, norm_mses, norm_rmses = [], [], []

        for series_dict in tqdm(all_series, desc=f"Processing {ctx}->{pdt}"):
            series = series_dict["target"]
            start_time = series_dict["start"]
            total_len = series.shape[1]
            max_windows = (total_len - ctx - pdt) // current_step + 1
            if max_windows <= 0:
                continue

            for start in range(0, total_len - ctx - pdt + 1, current_step):
                context = series[:, start:start+ctx]
                target = series[:, start+ctx:start+ctx+pdt]

                dataset = ListDataset([{
                    "start": start_time,
                    "target": context.tolist()
                }], freq=args.freq, one_dim_target=(args.target_dim == 1))

                try:
                    forecast = next(predictor.predict(dataset))
                    pred = np.array(forecast.mean).T
                    if target.shape != pred.shape:
                        pred = pred.T
                    if target.shape != pred.shape:
                        continue

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

                    print(f"[{ctx}->{pdt}] Raw MAE: {mae_raw:.4f}, Raw RMSE: {rmse_raw:.4f}, Raw MSE: {mse_raw:.4f} | "
                          f"Norm MAE: {mae:.4f}, Norm RMSE: {rmse:.4f}, Norm MSE: {mse:.4f}")
                except Exception as e:
                    print(f"Prediction failed: {e}")
                    continue

        if raw_maes:
            overall_raw[i]["mae"].append(np.mean(raw_maes))
            overall_raw[i]["mse"].append(np.mean(raw_mses))
            overall_raw[i]["rmse"].append(np.mean(raw_rmses))

            overall_norm[i]["mae"].append(np.mean(norm_maes))
            overall_norm[i]["mse"].append(np.mean(norm_mses))
            overall_norm[i]["rmse"].append(np.mean(norm_rmses))

            print(f"\nAvg (Raw) MAE: {np.mean(raw_maes):.6f} | MSE: {np.mean(raw_mses):.6f} | RMSE: {np.mean(raw_rmses):.6f}")
            print(f"Avg (Normed) MAE: {np.mean(norm_maes):.6f} | MSE: {np.mean(norm_mses):.6f} | RMSE: {np.mean(norm_rmses):.6f}")
        else:
            print("No valid predictions for this setting")

    print("\nSummary of All Settings")
    raw_maes_all, raw_mses_all, raw_rmses_all = [], [], []
    norm_maes_all, norm_mses_all, norm_rmses_all = [], [], []

    for (ctx, pdt), raw, norm in zip(zip(context_list, prediction_list), overall_raw, overall_norm):
        if raw["mae"]:
            r_mae, r_mse, r_rmse = np.mean(raw["mae"]), np.mean(raw["mse"]), np.mean(raw["rmse"])
            n_mae, n_mse, n_rmse = np.mean(norm["mae"]), np.mean(norm["mse"]), np.mean(norm["rmse"])

            raw_maes_all.append(r_mae)
            raw_mses_all.append(r_mse)
            raw_rmses_all.append(r_rmse)

            norm_maes_all.append(n_mae)
            norm_mses_all.append(n_mse)
            norm_rmses_all.append(n_rmse)

            print(f"Setting {ctx}->{pdt} | Raw MAE: {r_mae:.6f}, Raw MSE: {r_mse:.6f}, Raw RMSE: {r_rmse:.6f} | "
                  f"Norm MAE: {n_mae:.6f}, Norm MSE: {n_mse:.6f}, Norm RMSE: {n_rmse:.6f}")
        else:
            print(f"Setting {ctx}->{pdt} | No valid predictions")

    if raw_maes_all:
        print("\nOverall Averages")
        print("Raw MAE: {:.6f}, MSE: {:.6f}, RMSE: {:.6f}".format(
            np.mean(raw_maes_all), np.mean(raw_mses_all), np.mean(raw_rmses_all)))
        print("Norm MAE: {:.6f}, MSE: {:.6f}, RMSE: {:.6f}".format(
            np.mean(norm_maes_all), np.mean(norm_mses_all), np.mean(norm_rmses_all)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, default="multivariate",
                        help="Choose from multivariate covid cinc cdc jhcovid jhcovid_us")
    parser.add_argument("--model", type=str, default="Salesforce/moirai-moe-1.0-R-base")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--target_dim", type=int, default=1)
    parser.add_argument("--freq", type=str, default="1D")
    parser.add_argument("--context_lengths", type=str, default="512,1024,2048,3072")
    parser.add_argument("--prediction_lengths", type=str, default="96,192,336,720")
    args = parser.parse_args()
    main(args)
