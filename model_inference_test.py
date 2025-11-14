# test_mira_inference.py
import argparse
import torch

from MIRA.mira.models.modeling_mira import MIRAForPrediction
from inference_core import mira_evaluate_one_window_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # -------------------------------------------------
    # Load model
    # -------------------------------------------------
    print(f"[INFO] Loading MIRA model: {args.model_ckpt}")
    model = MIRAForPrediction.from_pretrained(args.model_ckpt).to(device)
    model.eval()

    # -------------------------------------------------
    # Create dummy test data (you can replace with real)
    # -------------------------------------------------
    T = 18
    values = torch.randn(1, T) * 2.0 + 5.0    # some random signal
    times = torch.arange(T).float().unsqueeze(0)  # 0..17

    # mean/std for normalization
    mean = values.mean()
    std = values.std()

    print("[INFO] values:", values)
    print("[INFO] times:", times)

    # -------------------------------------------------
    # Run evaluation
    # -------------------------------------------------
    preds, gt, rmse, mae = mira_evaluate_one_window_norm(
        model,
        values,
        times,
        context_len=12,
        pred_len=6,
        mean=mean,
        std=std
    )

    print("=====================================")
    print("Preds:", preds)
    print("GT:", gt)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("=====================================")


if __name__ == "__main__":
    main()
