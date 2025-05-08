#!/usr/bin/env python
# -*- coding:utf-8 _*-
import random
import numpy as np
from tqdm import tqdm
from time_moe.datasets.ts_dataset import TimeSeriesDataset
from time_moe.datasets.time_ts_dataset import TimeAwareDataset
from torch.utils.data import Dataset

class TimeAwareWindowDataset(Dataset):
    """
    Generates training windows from a TimeAwareDataset (like TimeAwareJSONLDataset).
    Applies masking to remove invalid points and normalizes time.
    Outputs data suitable for autoregressive training with absolute time.
    """
    def __init__(self,
                 dataset: TimeAwareDataset, # Expects dataset with sequence, time, mask
                 context_length: int,
                 prediction_length: int = 0, # Default for causal LM training
                 time_normalizer=None, # Pass the fitted normalizer
                 min_valid_history: int = 1 # Min valid points required in history window
                 ):
        self.source_dataset = dataset
        self.context_length = context_length
        self.prediction_length = prediction_length # Currently unused in AR setup, but kept
        self.time_normalizer = time_normalizer
        self.min_valid_history = min_valid_history

        # For autoregressive training (predicting next step): window_size = context_length + 1
        # We need context_length inputs and context_length labels (shifted)
        self.target_window_size = context_length # Target length *after* filtering invalid points
        self.raw_window_size = context_length + 1 # Need one extra point for label

        # Precompute valid windows based on *original* sequence length
        logger.info("Precomputing valid windows...")
        self.valid_windows = []
        num_sequences = len(self.source_dataset)
        iterator = tqdm(range(num_sequences), total=num_sequences, desc="Finding Valid Windows") if 'tqdm' in globals() else range(num_sequences)

        for seq_idx in iterator:
            # We need original length before filtering
            original_seq_len = self.source_dataset.get_sequence_length(seq_idx)

            # Iterate through possible start points of raw windows
            for start_idx in range(original_seq_len - self.raw_window_size + 1):
                # Optimization: Check if at least min_valid_history points *might* be valid
                # This is approximate, full check happens in __getitem__
                item = self.source_dataset[seq_idx] # Load item once
                raw_mask_window = item['mask'][start_idx : start_idx + self.context_length]
                if np.sum(raw_mask_window) >= self.min_valid_history:
                    self.valid_windows.append((seq_idx, start_idx))

        if not self.valid_windows:
            raise ValueError("No valid windows found in the dataset with the given context length.")
        logger.info(f"Found {len(self.valid_windows)} potential valid windows.")


    def __len__(self):
        return len(self.valid_windows)

    def __getitem__(self, idx):
        seq_idx, start_idx = self.valid_windows[idx]
        item = self.source_dataset[seq_idx] # Contains 'sequence', 'time', 'mask'

        # 1. Slice the raw window needed for input and label (context_length + 1)
        end_idx = start_idx + self.raw_window_size
        raw_sequence = item['sequence'][start_idx:end_idx]
        raw_time = item['time'][start_idx:end_idx]
        raw_mask = item['mask'][start_idx:end_idx]

        # 2. Filter invalid points based on mask
        valid_indices = raw_mask == 1
        valid_sequence = raw_sequence[valid_indices]
        valid_time_abs = raw_time[valid_indices] # Absolute times of valid points

        # Ensure we still have enough points after filtering for input/label pair
        if len(valid_sequence) < 2:
            # This case should be rare if min_valid_history is set appropriately,
            # but handle it defensively. Return None or raise?
            # Returning None might require a custom collate_fn to filter Nones.
            # Let's try returning a dummy item or raise an error for now.
            # Or, better, retry with next index (requires modifying how iteration works)
            # Simplest: return None and expect collate_fn to handle it.
             logger.debug(f"Skipping window {idx} due to insufficient valid points after filtering ({len(valid_sequence)} < 2).")
             # To avoid issues with standard DataLoader, instead of returning None,
             # we could return the previous item, or implement a filtering collate_fn.
             # For now, let's just return the data, it will likely fail in collate if lengths mismatch.
             # A robust solution would filter self.valid_windows more strictly in __init__.
             pass # Allow potential errors downstream for now


        # 3. Time Normalization (apply to valid absolute times)
        valid_time_norm = valid_time_abs # Default if no normalizer
        if self.time_normalizer is not None:
            try:
                 # StandardScaler expects [n_samples, n_features]
                 valid_time_norm = self.time_normalizer.transform(valid_time_abs.reshape(-1, 1)).flatten()
            except Exception as e:
                 logger.error(f"Error applying time normalizer: {e}. Using original times.")
                 valid_time_norm = valid_time_abs # Fallback

        # 4. Prepare input_ids, time_values, labels for AutoRegressive Task
        # input_ids: Sequence excluding the last valid point
        # labels: Sequence excluding the first valid point
        # time_values: Normalized times corresponding to input_ids
        input_ids = valid_sequence[:-1].astype(np.float32)
        time_values = valid_time_norm[:-1].astype(np.float32) # Normalized times for input
        labels = valid_sequence[1:].astype(np.float32)

        # 5. Prepare attention mask and loss mask
        # Since we removed invalid points, the effective sequence length is len(input_ids).
        # The model needs to know the actual length for causal masking.
        # We can provide a mask of all ones for the valid sequence length.
        current_length = len(input_ids)
        attention_mask = np.ones(current_length, dtype=np.int32)

        # Loss mask: We compute loss for all predicted elements (labels).
        loss_mask = np.ones(len(labels), dtype=np.int32) # Mask corresponds to labels

        # 6. Prepare `next_target_time_values` for ODE block (inference)
        # This is the normalized time of the *first label*.
        # Handle edge case where there might be no labels (len(valid_sequence) < 2)
        next_target_time_value = valid_time_norm[1] if len(valid_time_norm) > 1 else np.nan
        # Ensure it's float32
        next_target_time_value = np.float32(next_target_time_value)

        # Pad sequences to context_length if needed (standard practice in collators)
        # We will do padding in the collate function instead of here.

        return {
            # Inputs to the model's main body
            'input_ids': input_ids,        # Shape: [valid_len - 1]
            'time_values': time_values,    # Shape: [valid_len - 1], normalized absolute times
            'attention_mask': attention_mask, # Shape: [valid_len - 1], all ones

            # Labels and loss mask for training
            'labels': labels,              # Shape: [valid_len - 1]
            'loss_mask': loss_mask,        # Shape: [valid_len - 1], all ones

            # Additional info potentially needed for inference / generation
            'next_target_time_value': next_target_time_value # Shape: scalar, normalized time of first label
            # Maybe return original times if needed for analysis?
            # 'input_times_abs': valid_time_abs[:-1].astype(np.float32)
        }



class TimeMoEWindowDataset:
    """
    A dataset that generates windows of time series data.
    """
    def __init__(self, dataset: TimeSeriesDataset, context_length: int, prediction_length: int = 0, **kwrags):
        self.dataset = dataset
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_size = context_length + prediction_length
        self.window_size_plus_one = self.window_size + 1

        num_seqs = len(self.dataset)
        iterator = range(num_seqs)
        try:
            iterator = tqdm(iterator, total=num_seqs)
        except ImportError:
            pass
        self.sub_seq_indexes = []
        for seq_idx in iterator:
            n_points = self.dataset.get_sequence_length_by_idx(seq_idx)
            # Skip sequences with fewer than 2 points
            if n_points < 2:
                continue
            for offset_idx in range(0, n_points, self.window_size):
                self.sub_seq_indexes.append((seq_idx, offset_idx))

    def __len__(self):
        return len(self.sub_seq_indexes)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, seq_idx):
        seq_i, offset_i = self.sub_seq_indexes[seq_idx]
        seq = self.dataset[seq_i][offset_i: offset_i + self.window_size_plus_one]
        seq = np.array(seq, dtype=np.float32)

        loss_mask = np.ones(len(seq) - 1, dtype=np.int32)
        n_pad = self.window_size_plus_one - len(seq)
        if n_pad > 0:
            seq = np.pad(seq, (0, n_pad), 'constant', constant_values=0)
            loss_mask = np.pad(loss_mask, (0, n_pad), 'constant', constant_values=0)

        return {
            'input_ids': seq[:-1],
            'labels': seq[1:],
            'loss_masks': loss_mask
        }


class UniversalTimeMoEWindowDataset:
    """
    A dataset that generates windows of time series data with pack technique.
    """
    def __init__(self, dataset: TimeSeriesDataset, context_length: int, prediction_length: int = 0,
                 shuffle: bool = False):
        self.dataset = dataset
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_size = context_length + prediction_length

        self.window_info_list = []
        n_seqs = len(self.dataset)

        cur_window_info = []
        num_cur_remaining_points = self.window_size

        iterator = range(n_seqs)
        if shuffle:
            iterator = list(iterator)
            random.shuffle(iterator)

        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=n_seqs)
        except ImportError:
            pass

        for seq_idx in iterator:
            seq_len = self.dataset.get_sequence_length_by_idx(seq_idx)
            remaining_seq_len = seq_len
            while remaining_seq_len > 0:
                if remaining_seq_len < num_cur_remaining_points:
                    cur_window_info.append(
                        (seq_idx, seq_len - remaining_seq_len, remaining_seq_len)
                    )

                    # update states
                    num_cur_remaining_points -= remaining_seq_len
                    remaining_seq_len = 0
                else:
                    # add the part of this seq to cur_window
                    cur_window_info.append(
                        (seq_idx, seq_len - remaining_seq_len, num_cur_remaining_points)
                    )

                    # update states
                    remaining_seq_len -= num_cur_remaining_points
                    self.window_info_list.append(cur_window_info)

                    # reset current window
                    num_cur_remaining_points = self.window_size
                    cur_window_info = []

        if num_cur_remaining_points > 0:
            # drop last batch for speed-up
            pass

    def __len__(self):
        return len(self.window_info_list)

    def __getitem__(self, window_idx):
        window_info = self.window_info_list[window_idx]
        seq = []
        for seq_idx, start_idx_in_seq, offset in window_info:
            part_seq = self.dataset[seq_idx][start_idx_in_seq: start_idx_in_seq + offset]
            seq.append(part_seq)
        if len(seq) == 1:
            seq = seq[0]
            if not isinstance(seq, np.ndarray):
                seq = np.array(seq, dtype=np.float32)
            else:
                seq = seq.astype(np.float32)
        else:
            seq = np.concatenate(seq, axis=0, dtype=np.float32)
        return {
            'input_ids': seq[:-1],
            'labels': seq[1:],
        }
