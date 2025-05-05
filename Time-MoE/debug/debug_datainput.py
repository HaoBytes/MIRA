from time_moe.datasets.timeawared_dataset import TimeAwareJSONLDataset
from time_moe.datasets.time_moe_window_dataset import TimeAwareWindowDataset
import numpy as np

def print_array(name, arr):
    print(f"{name} (len={len(arr)}):")
    if isinstance(arr, np.ndarray):
        print(np.array2string(arr, precision=3, separator=', ', suppress_small=False))
    elif isinstance(arr, list):
        print('[', ', '.join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in arr), ']')
    else:
        print(arr)
    print("-" * 60)

if __name__ == "__main__":
    jsonl_path = "urine_test.jsonl"  # 🔁 Replace with your real path
    context_len = 20
    pred_len = 20

    print("Loading raw JSONL dataset...")
    dataset = TimeAwareJSONLDataset(jsonl_path, auto_quantize=True)

    print("Constructing Time-Aware Window Dataset...")
    window_dataset = TimeAwareWindowDataset(dataset, context_length=context_len, prediction_length=pred_len)

    print(f"Total windows: {len(window_dataset)}\n")

    # Sample 1–3 items for manual inspection
    for idx in [0, 1, 2]:
        print(f"\n===== Window {idx} =====")
        sample = window_dataset[idx]

        # === Raw unfiltered window (before applying mask)
        seq_idx, start_idx = window_dataset.valid_windows[idx]
        raw_item = dataset[seq_idx]
        raw_sequence = raw_item["sequence"][start_idx: start_idx + context_len + pred_len]
        raw_time = raw_item["time"][start_idx: start_idx + context_len + pred_len]
        raw_mask = raw_item["mask"][start_idx: start_idx + context_len + pred_len]

        print_array("RAW sequence", raw_sequence)
        print_array("RAW time", raw_time)
        print_array("RAW mask", raw_mask)

        # === Processed window after masking
        print_array("inputs", sample["inputs"])
        print_array("input_time", sample["input_time"])
        print_array("labels", sample["labels"])
        print_array("label_time", sample["label_time"])
        print_array("loss_masks", sample["loss_masks"])
