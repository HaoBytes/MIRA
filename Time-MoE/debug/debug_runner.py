import os
import sys
from time_moe.datasets.timeawared_dataset import TimeAwareJSONLDataset
from time_moe.datasets.time_moe_window_dataset import TimeAwareWindowDataset
from time_moe.datasets.time_moe_dataset import TimeMoEDataset
from time_moe.datasets.time_moe_window_dataset import TimeMoEWindowDataset
import numpy as np

def print_array(name, arr):
    print(f"{name} (len={len(arr)}):")
    if isinstance(arr, np.ndarray):
        print(np.round(arr, 3))
    else:
        print(arr)
    print("-" * 60)

def debug_dataset(data_path, time_aware=True, context_length=20, prediction_length=20, normalization_method="zero"):
    print(f"Loading dataset from: {data_path}")
    print(f"Time-aware mode: {time_aware}")
    
    if not os.path.exists(data_path):
        print("[Error] Data path does not exist.")
        sys.exit(1)

    if time_aware:
        dataset = TimeAwareJSONLDataset(data_path, auto_quantize=True)
        window_dataset = TimeAwareWindowDataset(dataset, context_length=context_length, prediction_length=prediction_length)
    else:
        dataset = TimeMoEDataset(data_path, normalization_method=normalization_method)
        window_dataset = TimeMoEWindowDataset(dataset, context_length=context_length, prediction_length=prediction_length)

    print(f"Loaded {len(window_dataset)} windows.")

    # Print a few samples to verify
    for idx in [0, 1, 2]:
        if idx >= len(window_dataset):
            break
        print(f"\n===== Window {idx} =====")
        sample = window_dataset[idx]

        if time_aware:
            # TimeAware fields
            print_array("inputs", sample["inputs"])
            print_array("input_time", sample["input_time"])
            print_array("labels", sample["labels"])
            print_array("label_time", sample["label_time"])
            print_array("loss_masks", sample["loss_masks"])
        else:
            # TimeMoE fields
            print_array("context", sample["context"])
            print_array("context_time", sample["context_time"])
            print_array("future", sample["future"])
            print_array("future_time", sample["future_time"])
            print_array("loss_mask", sample["loss_mask"])

if __name__ == "__main__":
    # Replace the path below with your test JSONL or npy file
    data_path = "urine_test.jsonl"  # <-- 👈 Replace me!

    # Call the debug
    debug_dataset(
        data_path=data_path,
        time_aware=True,            # <-- 👈 Set True to use TimeAwareJSONLDataset
        context_length=20,
        prediction_length=20,
        normalization_method="zero"
    )
