#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
from MIRA.mira.models.modeling_mira import MIRAForPrediction
from MIRA.mira.models.configuration_mira import MIRAConfig


def run_inference(model_ckpt):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Loading model from: {model_ckpt}")
    config = MIRAConfig.from_pretrained(model_ckpt)

    model = MIRAForPrediction.from_pretrained(
        model_ckpt,
        config=config
    ).to(device)
    model.eval()

    # =============================
    # 2. Create fake context
    # =============================
    B = 1
    L = 10
    P = 6

    # fake signal
    input_ids = torch.randn(B, L, 1, device=device)

    # irregular timestamps
    time_values = torch.tensor(
        [
            [0., 0.5, 1.5, 2.0, 3.0, 4.0, 6.2, 6.3, 7.0, 10.0],
            [5., 5.2, 6.0, 7.1, 7.1, 8.0, 10., 10.2, 11.0, 12.3]
        ],
        dtype=torch.float32,
        device=device
    )

    # >>> mask: 全 1（无缺失点）
    input_mask = torch.ones(B, L, dtype=torch.long, device=device)

    print("[DEBUG] input_ids:", input_ids.shape)
    print("[DEBUG] time_values:", time_values.shape)
    print("[DEBUG] input_mask:", input_mask.shape)

    # =============================
    # 3. Future timestamps
    # =============================
    future_time_values = torch.tensor(
        [
            [11., 11.5, 12., 13., 14., 18.],
            [13., 14., 15., 17., 19., 25.],
        ],
        dtype=torch.float32,
        device=device
    )

    # >>> future_mask: 也全 1
    future_mask = torch.ones(B, P, dtype=torch.long, device=device)

    print("[DEBUG] future_time_values:", future_time_values.shape)

    # =============================
    # 4. Generation
    # =============================
    print("[INFO] Running model.generate(...)")

    predicted = model.generate(
        input_ids=input_ids,
        time_values=time_values,
        mask=input_mask,
        future_time_values=future_time_values,
        future_mask=future_mask,
        max_length=L + P,
        do_sample=False,
    )

    seq = predicted.sequences

    print("\n======================")
    print("✓ Final sequences shape:", seq.shape)
    print("======================\n")

    pred_out = seq[:, -P:, :]
    print("✓ Predicted values:\n", pred_out)


def main():
    parser = argparse.ArgumentParser(description="Run MIRA inference")
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to MIRA checkpoint"
    )

    args = parser.parse_args()
    run_inference(args.ckpt)


if __name__ == "__main__":
    main()