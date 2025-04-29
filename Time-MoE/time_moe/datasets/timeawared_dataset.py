#!/usr/bin/env python
# -*- coding:utf-8 _*-
import json
import numpy as np
from time_moe.datasets.time_ts_dataset import TimeAwareDataset
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from collections import Counter


def quantize_time(times, 
                               initial_resolution=1.0, 
                               min_resolution=1e-8, 
                               shrink_factor=10, 
                               jitter_eps=1e-8, 
                               max_iterations=20):
    """
    Quantize time points while ensuring uniqueness by automatically adjusting resolution.
    If maximum iterations reached, resolve duplicates manually.

    Args:
        times (array-like): Original timestamps.
        initial_resolution (float): Starting quantization resolution.
        min_resolution (float): Minimum resolution limit.
        shrink_factor (int): Factor to shrink resolution per iteration.
        jitter_eps (float): Minimum additive noise to enforce strictly increasing times.
        max_iterations (int): Max shrink attempts.

    Returns:
        np.ndarray: Quantized timestamps, guaranteed unique.
    """
    times = np.array(times, dtype=np.float64)
    resolution = initial_resolution

    for it in range(max_iterations):
        quantized = np.round(times / resolution) * resolution

        # Check if mapping is unique (more strict)
        counts = Counter(quantized)
        duplicates = [v for v, cnt in counts.items() if cnt > 1]

        if len(duplicates) == 0:
            # Success: No duplicates
            print(f"[Info] Quantization succeeded at resolution {resolution:.8f} after {it+1} iterations.")
            return quantized

        # Shrink resolution
        resolution /= shrink_factor
        if resolution < min_resolution:
            resolution = min_resolution

    # Fallback manual adjustment if still not unique
    print(f"[Warning] Maximum iterations reached. Forcing uniqueness manually at resolution {resolution:.8f}.")
    quantized = np.round(times / resolution) * resolution

    # Force resolve duplicates
    unique_quantized = []
    last_value = None
    for q in quantized:
        if last_value is None:
            unique_quantized.append(q)
        else:
            if q <= last_value:
                q = last_value + jitter_eps
            unique_quantized.append(q)
        last_value = unique_quantized[-1]

    return np.array(unique_quantized)


def read_file_by_extension(fn):
    if fn.endswith('.json'):
        with open(fn, encoding='utf-8') as file:
            data = json.load(file)
    elif fn.endswith('.jsonl'):
        data = read_jsonl_to_list(fn)
    elif fn.endswith('.yaml'):
        data = load_yaml_file(fn)
    elif fn.endswith('.npy'):
        data = np.load(fn, allow_pickle=True)
    elif fn.endswith('.npz'):
        data = np.load(fn, allow_pickle=True)
    elif fn.endswith('.npy.gz'):
        with gzip.GzipFile(fn, 'r') as file:
            data = np.load(file, allow_pickle=True)
    elif fn.endswith('.pkl') or fn.endswith('.pickle'):
        data = load_pkl_obj(fn)
    else:
        raise RuntimeError(f'Unknown file extension: {fn}')
    return data

def read_jsonl_to_list(jsonl_fn):
    with open(jsonl_fn, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file.readlines()]

def load_yaml_file(fn):
    with open(fn, 'r', encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_pkl_obj(fn):
    out_list = []
    with open(fn, 'rb') as f:
        while True:
            try:
                data = pickle.load(f)
                out_list.append(data)
            except EOFError:
                break
    if len(out_list) == 0:
        return None
    elif len(out_list) == 1:
        return out_list[0]
    else:
        return out_list

class TimeAwareJSONLDataset(TimeAwareDataset):
    def __init__(self, data_path, quantize_resolution=None, auto_quantize=False, sample_size=100):
        self.data = read_file_by_extension(data_path)
        self.num_tokens = None

        if auto_quantize and quantize_resolution is None:
            self.quantize_resolution = self._infer_quantize_resolution(sample_size=sample_size)
        else:
            self.quantize_resolution = quantize_resolution

    def _infer_quantize_resolution(self, sample_size=100):
        """auto quantize"""
        all_deltas = []

        for item in self.data[:sample_size]:
            if isinstance(item, dict) and 'time' in item:
                time = np.array(item['time'], dtype=np.float32)
                if len(time) >= 2:
                    deltas = np.diff(time)
                    deltas = deltas[deltas > 0] 
                    all_deltas.append(deltas)

        if len(all_deltas) == 0:
            print("[Warning] Cannot infer time resolution, default to 1000 ms")
            return 0.001

        all_deltas = np.concatenate(all_deltas)
        if len(all_deltas) == 0:
            print("[Warning] No positive time deltas found, default to 1.0")
            return 1.0

        estimated_resolution = np.median(all_deltas)
        print(f"[Info] Inferred quantize resolution: {estimated_resolution:.6f}")
        return estimated_resolution

    def __len__(self):
        return len(self.data)

    def __getitem__(self, seq_idx):
        seq = self.data[seq_idx]
        if isinstance(seq, dict):
            seq_len = len(seq['sequence'])
            sequence = np.array(seq['sequence'], dtype=np.float32)
            time = np.array(seq.get('time', range(seq_len)), dtype=np.float32)
            mask = np.array(seq.get('mask', [1]*seq_len), dtype=np.float32)

            if self.quantize_resolution is not None:
                time = quantize_time(time, self.quantize_resolution)

            return {
                'sequence': sequence,
                'time': time,
                'mask': mask
            }
        else:
            sequence = np.array(seq, dtype=np.float32)
            time = np.arange(len(seq), dtype=np.float32)
            mask = np.ones(len(seq), dtype=np.float32)

            if self.quantize_resolution is not None:
                time = quantize_time(time, self.quantize_resolution)

            return {
                'sequence': sequence,
                'time': time,
                'mask': mask
            }

    def get_num_tokens(self):
        if self.num_tokens is None:
            self.num_tokens = sum(len(item['sequence']) for item in self)
        return self.num_tokens

    def get_sequence_length(self, seq_idx):
        seq = self[seq_idx]
        return len(seq['sequence'])



class TimeAwareEvalDataset(Dataset):
    def __init__(self, dataset, context_length, prediction_length, normalize=False):
        self.source_dataset = dataset
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_length = context_length + prediction_length
        self.normalize = normalize

        # Precompute all valid windows
        self.valid_windows = []
        for seq_idx in range(len(dataset)):
            seq_len = dataset.get_sequence_length(seq_idx)
            if seq_len >= self.window_length:
                for start in range(seq_len - self.window_length + 1):
                    self.valid_windows.append((seq_idx, start))

    def __len__(self):
        return len(self.valid_windows)

    def __getitem__(self, idx):
        seq_idx, start = self.valid_windows[idx]
        end = start + self.window_length
        item = self.source_dataset[seq_idx]

        # Retrieve full window data
        full_sequence = item['sequence'][start:end]
        full_mask = item['mask'][start:end]
        full_time = item['time'][start:end] if item['time'] is not None else None

        # Split into context and prediction parts
        context_seq = full_sequence[:self.context_length]
        context_mask = full_mask[:self.context_length]
        pred_seq = full_sequence[self.context_length:]
        pred_mask = full_mask[self.context_length:]

        # Keep only valid (observed) values
        inputs = self._get_valid_values(context_seq, context_mask, 'inputs')
        labels = self._get_valid_values(pred_seq, pred_mask, 'labels')

        # Process time information
        if full_time is not None:
            context_time = full_time[:self.context_length]
            pred_time = full_time[self.context_length:]
            inputs['time'] = context_time[context_mask == 1]
            labels['time'] = pred_time[pred_mask == 1]

        if self.normalize:
            inputs, labels = self._normalize(inputs, labels)

        return {
            'inputs': inputs,
            'labels': labels,
            # Keep original mask information for further processing
            'input_mask': context_mask,
            'label_mask': pred_mask
        }

    def _get_valid_values(self, sequence, mask, prefix):
        """Extract valid values and record their original indices."""
        valid_mask = mask == 1
        valid_indices = np.where(valid_mask)[0]
        return {
            'sequence': sequence[valid_mask],
            'valid_indices': valid_indices,
            'original_length': len(sequence)
        }

    def _normalize(self, inputs, labels):
        """Normalize based on valid values only."""
        if len(inputs['sequence']) > 0:
            mean = inputs['sequence'].mean()
            std = inputs['sequence'].std()
            std = 1.0 if std == 0 else std

            inputs['sequence'] = (inputs['sequence'] - mean) / std
            labels['sequence'] = (labels['sequence'] - mean) / std

        return inputs, labels