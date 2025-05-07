# filename: time_aware_data_processing.py

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler # Requires scikit-learn
from tqdm import tqdm # Optional, for progress bars
import warnings # For warnings

# --- Interfaces (Assuming these exist based on uploaded filenames) ---
# From ts_dataset.py
class TimeSeriesDataset:
    def __len__(self): raise NotImplementedError
    def __getitem__(self, seq_idx): raise NotImplementedError
    def get_num_tokens(self): raise NotImplementedError
    def get_sequence_length_by_idx(self, seq_idx): raise NotImplementedError
    @staticmethod
    def is_valid_path(data_path): return True
    def __iter__(self):
        for i in range(len(self)): yield self[i]

# From time_ts_dataset.py
class TimeAwareDataset(TimeSeriesDataset):
     # Inherits methods from TimeSeriesDataset
     # __getitem__ should return dict with 'sequence', 'time', 'mask'
     def get_sequence_length(self, idx): # Renamed for consistency
          return self.get_sequence_length_by_idx(idx)


# --- Helper Function (Adapted from general_dataset.py) ---
def read_jsonl_to_list(jsonl_fn):
    """Reads a jsonl file into a list of dictionaries."""
    if not os.path.exists(jsonl_fn):
        raise FileNotFoundError(f"File not found: {jsonl_fn}")
    try:
        with open(jsonl_fn, 'r', encoding='utf-8') as file:
            return [json.loads(line) for line in file if line.strip()]
    except Exception as e:
        print(f"Error reading jsonl file {jsonl_fn}: {e}")
        raise

# --- Dataset Implementations ---

class JSONLTimeAwareDataset(TimeAwareDataset):
    """
    Loads time-aware data from a JSONL file.
    Each line should be a JSON object with "sequence", "time", and optionally "mask".
    Calculates and stores time normalization statistics.
    """
    def __init__(self, data_path: str, time_normalization: str = 'standard', sample_size_for_stats: int = 10000):
        """
        Args:
            data_path (str): Path to the .jsonl file.
            time_normalization (str or None): 'standard' for standardization, None to disable.
            sample_size_for_stats (int): Number of samples to use for calculating normalization stats.
        """
        if not (os.path.exists(data_path) and data_path.endswith('.jsonl')):
             raise ValueError(f"Invalid data path: {data_path}. Expecting a .jsonl file.")

        print(f"Loading data from {data_path}...")
        self.data = read_jsonl_to_list(data_path)
        if not self.data:
            raise ValueError(f"No data loaded from {data_path}.")
        print(f"Loaded {len(self.data)} sequences.")

        self.data_path = data_path
        self.time_normalizer = None
        self.time_normalization = time_normalization
        self._fit_time_normalizer(sample_size_for_stats)
        print(f"Finished preparing dataset wrapper for {data_path}.")

    def _fit_time_normalizer(self, sample_size):
        """Fits a StandardScaler on the time values from a sample of the data."""
        if self.time_normalization != 'standard':
            print("Time normalization disabled or unsupported method specified.")
            return

        print(f"Calculating time normalization statistics from sample (size={sample_size})...")
        all_valid_times = []
        num_items_to_sample = min(sample_size, len(self.data))
        indices_to_sample = np.random.choice(len(self.data), num_items_to_sample, replace=False)

        iterator = tqdm(indices_to_sample, desc="Sampling for Time Norm") if 'tqdm' in globals() else indices_to_sample
        items_processed = 0
        for i in iterator:
            item = self.data[i]
            try:
                if isinstance(item, dict) and 'time' in item and 'sequence' in item:
                    mask = np.array(item.get('mask', np.ones_like(item['sequence'])), dtype=int)
                    time = np.array(item['time'], dtype=np.float64)
                    sequence = item['sequence'] # Don't need value here, just length check

                    if not (len(time) == len(sequence) == len(mask)):
                         warnings.warn(f"Inconsistent lengths in item {i}, skipping for stats.", RuntimeWarning)
                         continue

                    valid_times = time[mask == 1]
                    if len(valid_times) > 0:
                        all_valid_times.append(valid_times)
                        items_processed += 1

            except Exception as e:
                warnings.warn(f"Error processing item {i} for stats: {e}. Skipping.", RuntimeWarning)

        if not all_valid_times:
             warnings.warn("No valid time data found in the sample to compute normalization statistics. Time normalization disabled.", RuntimeWarning)
             self.time_normalization = None # Disable if no data
             return

        all_times_flat = np.concatenate(all_valid_times)
        if all_times_flat.size == 0:
             warnings.warn("Concatenated time data is empty. Time normalization disabled.", RuntimeWarning)
             self.time_normalization = None # Disable if no data
             return

        print(f"Fitting StandardScaler on {len(all_times_flat)} valid time points from {items_processed} sequences...")
        self.time_normalizer = StandardScaler()
        try:
            # StandardScaler expects [n_samples, n_features]
            self.time_normalizer.fit(all_times_flat.reshape(-1, 1))
            mean_val = self.time_normalizer.mean_[0] if self.time_normalizer.mean_ is not None else float('nan')
            scale_val = self.time_normalizer.scale_[0] if self.time_normalizer.scale_ is not None else float('nan')
            print(f"Fitted StandardScaler for time: mean={mean_val:.4f}, scale={scale_val:.4f}")
            if scale_val == 0 or np.isnan(scale_val) or np.isnan(mean_val):
                warnings.warn("Time standard deviation is zero or stats are NaN. Time normalization disabled.", RuntimeWarning)
                self.time_normalizer = None
                self.time_normalization = None

        except ValueError as e:
             warnings.warn(f"Error fitting StandardScaler: {e}. Time normalization disabled.", RuntimeWarning)
             self.time_normalizer = None
             self.time_normalization = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, seq_idx):
        """Returns sequence, time, and mask for a given index."""
        if not (0 <= seq_idx < len(self.data)):
            raise IndexError(f"Index {seq_idx} out of bounds for dataset length {len(self.data)}")

        item = self.data[seq_idx]
        try:
            if isinstance(item, dict) and 'sequence' in item and 'time' in item:
                sequence = np.array(item['sequence'], dtype=np.float32)
                time = np.array(item['time'], dtype=np.float32) # Use float32
                 # Default mask to all ones if missing or None
                mask_data = item.get('mask')
                if mask_data is None:
                    mask = np.ones_like(sequence, dtype=np.int32)
                else:
                    mask = np.array(mask_data, dtype=np.int32)

                # Basic length check
                if not (len(sequence) == len(time) == len(mask)):
                    raise ValueError(f"Data inconsistency at index {seq_idx}: lengths differ.")

                return {'sequence': sequence, 'time': time, 'mask': mask}
            else:
                # Handle case where data might not be in expected dict format
                raise TypeError(f"Unexpected data format at index {seq_idx}. Expected dict with 'sequence' and 'time'. Got: {type(item)}")

        except Exception as e:
            print(f"Error processing data at index {seq_idx}: {e}")
            # Return dummy data or raise error? Raising might be better.
            raise RuntimeError(f"Failed to process item at index {seq_idx}") from e


    def get_sequence_length_by_idx(self, seq_idx):
        """Gets original sequence length without loading full data if possible."""
        # This is slow for plain jsonl, needs to load item.
        item = self.data[seq_idx]
        if isinstance(item, dict):
            return len(item['sequence'])
        else:
            # Fallback or raise error if format unexpected
            return len(item) if hasattr(item, '__len__') else 0

    def get_num_tokens(self):
        # Implement if needed, potentially slow for jsonl
        # total_tokens = sum(self.get_sequence_length_by_idx(i) for i in range(len(self)))
        # return total_tokens
        raise NotImplementedError("get_num_tokens not implemented for efficiency.")

    def get_time_normalizer(self):
        """Returns the fitted time normalizer."""
        return self.time_normalizer

# --- Windowing Dataset ---
class MaskedTimeAwareWindowDataset(Dataset):
    """
    Generates training windows from a TimeAwareDataset.
    Applies masking to remove invalid points and normalizes time.
    """
    def __init__(self,
                 dataset: TimeAwareDataset,
                 context_length: int,
                 prediction_length: int = 0, # Default for causal LM (predict next step)
                 time_normalizer=None,
                 min_valid_history: int = 1,
                 min_valid_window: int = 2 # Need at least 2 points for input/label pair
                 ):
        if not hasattr(dataset, 'get_time_normalizer'):
             warnings.warn("Input dataset does not have 'get_time_normalizer' method. Time normalization might not work as expected.", RuntimeWarning)

        self.source_dataset = dataset
        self.context_length = context_length
        # Prediction length currently unused in output formatting for AR, but keep if needed later
        self.prediction_length = prediction_length
        self.time_normalizer = time_normalizer # The fitted StandardScaler object
        self.min_valid_history = max(1, min_valid_history) # Ensure at least 1
        self.min_valid_window = max(2, min_valid_window) # Need at least input and label

        # Target window size for input/label pair in AR task
        self.raw_window_size = context_length + 1 # Need one extra point to form labels

        # Precompute valid windows based on *original* sequence length
        print("Precomputing valid windows...")
        self.valid_windows = []
        num_sequences = len(self.source_dataset)
        iterator = tqdm(range(num_sequences), total=num_sequences, desc="Finding Valid Windows") if 'tqdm' in globals() else range(num_sequences)

        skipped_count = 0
        for seq_idx in iterator:
            original_seq_len = self.source_dataset.get_sequence_length(seq_idx)

            # Iterate through possible start points of raw windows
            for start_idx in range(original_seq_len - self.raw_window_size + 1):
                # --- Strict Pre-filtering ---
                # Load the mask for the potential window to do a more accurate check
                # This loads item multiple times but ensures validity better
                try:
                    item = self.source_dataset[seq_idx]
                    raw_mask_window = item['mask'][start_idx : start_idx + self.raw_window_size]
                    # Check if *enough valid points* exist in the context part AND the *whole window*
                    if (np.sum(raw_mask_window[:self.context_length]) >= self.min_valid_history and
                        np.sum(raw_mask_window) >= self.min_valid_window):
                         self.valid_windows.append((seq_idx, start_idx))
                    else:
                         skipped_count += 1
                except Exception as e:
                     warnings.warn(f"Error pre-checking window (seq={seq_idx}, start={start_idx}): {e}. Skipping.", RuntimeWarning)
                     skipped_count += 1


        if not self.valid_windows:
            raise ValueError("No valid windows found in the dataset with the given context length and minimum valid point requirements.")
        print(f"Found {len(self.valid_windows)} valid windows (skipped {skipped_count} potential windows during pre-filtering).")

    def __len__(self):
        return len(self.valid_windows)

    def __getitem__(self, idx):
        if not (0 <= idx < len(self.valid_windows)):
            raise IndexError(f"Index {idx} out of bounds for valid windows length {len(self.valid_windows)}")

        seq_idx, start_idx = self.valid_windows[idx]

        try:
            item = self.source_dataset[seq_idx]

            # 1. Slice the raw window
            end_idx = start_idx + self.raw_window_size
            raw_sequence = item['sequence'][start_idx:end_idx]
            raw_time = item['time'][start_idx:end_idx]
            raw_mask = item['mask'][start_idx:end_idx]

            # 2. Filter invalid points
            valid_indices_bool = (raw_mask == 1)
            valid_sequence = raw_sequence[valid_indices_bool]
            valid_time_abs = raw_time[valid_indices_bool] # Absolute times

            # Ensure minimum points condition met (should be guaranteed by __init__)
            if len(valid_sequence) < self.min_valid_window:
                 # This indicates an issue with pre-filtering or data
                 warnings.warn(f"Window {idx} (Seq {seq_idx}, Start {start_idx}) has < {self.min_valid_window} valid points after filtering ({len(valid_sequence)}). Returning dummy data.", RuntimeWarning)
                 # Return dummy data matching expected structure but potentially zero length
                 # Note: Collate function needs to handle potential zero-length tensors robustly
                 dummy_float = np.array([], dtype=np.float32)
                 dummy_int = np.array([], dtype=np.int32)
                 return {
                     'input_ids': dummy_float, 'time_values': dummy_float,
                     'attention_mask': dummy_int, 'labels': dummy_float,
                     'loss_mask': dummy_int, 'next_target_time_value': np.float32(np.nan)
                 }


            # 3. Time Normalization
            valid_time_norm = valid_time_abs # Default if no normalizer
            if self.time_normalizer is not None:
                try:
                    valid_time_norm = self.time_normalizer.transform(valid_time_abs.reshape(-1, 1)).flatten()
                except Exception as e:
                    warnings.warn(f"Error applying time normalizer at index {idx}: {e}. Using original times.", RuntimeWarning)
                    valid_time_norm = valid_time_abs # Fallback

            valid_time_norm = valid_time_norm.astype(np.float32) # Ensure float32

            # 4. Prepare input_ids, time_values, labels
            input_ids = valid_sequence[:-1].astype(np.float32)
            time_values = valid_time_norm[:-1] # Normalized times for input
            labels = valid_sequence[1:].astype(np.float32)

            # 5. Prepare attention mask and loss mask
            current_length = len(input_ids) # Length of the processed input sequence
            attention_mask = np.ones(current_length, dtype=np.int32)
            loss_mask = np.ones(len(labels), dtype=np.int32) # Corresponds to labels

            # 6. Prepare `next_target_time_values` (normalized time of the first label)
            next_target_time_value = valid_time_norm[1] if len(valid_time_norm) > 1 else np.float32(np.nan)

            return {
                'input_ids': input_ids,           # Shape: [valid_len_in_window - 1]
                'time_values': time_values,       # Shape: [valid_len_in_window - 1], normalized absolute times
                'attention_mask': attention_mask, # Shape: [valid_len_in_window - 1], all ones
                'labels': labels,                 # Shape: [valid_len_in_window - 1]
                'loss_mask': loss_mask,           # Shape: [valid_len_in_window - 1], all ones
                'next_target_time_value': next_target_time_value # Shape: scalar, normalized
            }
        except Exception as e:
             print(f"Error getting item at index {idx} (Seq {seq_idx}, Start {start_idx}): {e}")
             # Propagate error or return dummy? Propagating is usually better for debugging.
             raise RuntimeError(f"Failed to get item at index {idx}") from e


# --- Collate Function ---
def masked_time_aware_collate_fn(batch, pad_value=0.0, pad_time_value=0.0):
    """
    Collates data from MaskedTimeAwareWindowDataset, padding sequences to max length in batch.
    Handles potentially empty sequences by returning an empty batch dictionary.
    """
    # Filter out problematic items (e.g., those with zero length after filtering)
    batch = [item for item in batch if item and item['input_ids'] is not None and len(item['input_ids']) > 0]

    if not batch:
        # Return an empty dictionary or dictionary with empty tensors if the entire batch was filtered
        return {
            'input_ids': torch.empty(0, 0).float(),
            'time_values': torch.empty(0, 0).float(),
            'attention_mask': torch.empty(0, 0).long(),
            'labels': torch.empty(0, 0).float(),
            'loss_mask': torch.empty(0, 0).bool(),
            'next_target_time_value': torch.empty(0).float()
        }

    # Find max length in the filtered batch
    max_len = 0
    for item in batch:
        max_len = max(max_len, len(item['input_ids']))

    # Pad each item in the filtered batch
    padded_batch = {key: [] for key in batch[0].keys()} # Initialize with keys from first item

    for item in batch:
        current_len = len(item['input_ids'])
        padding_length = max_len - current_len

        if padding_length < 0:
            raise ValueError("Negative padding length detected in collate_fn. This should not happen.")

        for key, value in item.items():
            if key == 'next_target_time_value':
                padded_batch[key].append(value) # Scalar, just append
            elif isinstance(value, np.ndarray):
                if value.ndim == 0: # Handle 0-dim arrays (shouldn't happen for sequences)
                     padded_value = value # No padding needed
                elif padding_length == 0:
                     padded_value = value # No padding needed
                else:
                     pad_width = (0, padding_length) # Pad only at the end
                     constant_values = pad_time_value if 'time' in key else pad_value
                     padded_value = np.pad(value, pad_width, 'constant', constant_values=constant_values)
                padded_batch[key].append(padded_value)
            else: # Should not happen if __getitem__ returns numpy arrays
                padded_batch[key].append(value)


    # Stack arrays into tensors
    collated_batch = {}
    try:
        collated_batch['input_ids'] = torch.from_numpy(np.stack(padded_batch['input_ids'])).float() # Model expects float
        collated_batch['time_values'] = torch.from_numpy(np.stack(padded_batch['time_values'])).float()
        collated_batch['attention_mask'] = torch.from_numpy(np.stack(padded_batch['attention_mask'])).long() # Use Long for HF models
        collated_batch['labels'] = torch.from_numpy(np.stack(padded_batch['labels'])).float()
        collated_batch['loss_mask'] = torch.from_numpy(np.stack(padded_batch['loss_mask'])).bool() # Use Bool for masking
        collated_batch['next_target_time_value'] = torch.tensor(padded_batch['next_target_time_value'], dtype=torch.float32)
    except ValueError as e:
         print("Error during tensor stacking in collate_fn. Check shapes and padding.")
         # Print shapes for debugging
         for key, val_list in padded_batch.items():
              if isinstance(val_list, list) and val_list:
                   shapes = [np.array(v).shape for v in val_list]
                   print(f"  Shapes in batch for '{key}': {shapes}")
              else:
                   print(f"  Value for '{key}': {val_list}")
         raise e
    except Exception as e:
         print(f"Unexpected error during tensor conversion: {e}")
         raise e

    return collated_batch