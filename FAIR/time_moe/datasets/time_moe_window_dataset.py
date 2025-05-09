#!/usr/bin/env python
# -*- coding:utf-8 _*-
import random
import numpy as np
from tqdm import tqdm
from time_moe.datasets.ts_dataset import TimeSeriesDataset
from torch.utils.data import Dataset
from transformers.utils import logging

logger = logging.get_logger(__name__)

class TimeAwareWindowDataset(Dataset):
    """
    Generates training windows from a TimeSeriesDataset (like TimeAwareJSONLDataset).
    Applies masking to remove invalid points and normalizes time.
    Outputs data SUITABLE FOR AUTOREGRESSIVE TRAINING with absolute time.
    """
    def __init__(self,
                 dataset: TimeSeriesDataset, # Expects dataset providing sequence, time, mask via __getitem__
                 context_length: int,
                 # prediction_length is not strictly needed for AR, but keep for potential signature compatibility? 
                 # Or remove if it causes confusion. Let's keep it but default to 0 and ensure it's not used wrongly.
                 prediction_length: int = 0, 
                 time_normalizer=None, 
                 min_valid_history: int = 1 # Use the original parameter name
                 ):
        
        # --- 恢复检查逻辑 ---
        # prediction_length should typically be 0 for autoregressive training.
        # We don't need the strict check that caused the error.
        # if prediction_length <= 0:
        #     raise ValueError("prediction_length must be positive for forecasting setup.") #<-- 移除或注释掉这个检查

        self.source_dataset = dataset
        self.context_length = context_length
        # self.prediction_length = prediction_length # Not used in AR logic below
        self.time_normalizer = time_normalizer
        self.min_valid_history = min_valid_history # Use the correct attribute name
        
        # --- 恢复原始的 raw_window_size 计算 ---
        # For autoregressive training (predicting next step): window_size = context_length + 1
        self.raw_window_size = context_length + 1 # Need one extra point for label

        # --- Precompute valid windows (逻辑保持不变，但使用 self.min_valid_history) ---
        logger.info("Precomputing valid windows for Autoregressive training...")
        self.valid_windows = []
        num_sequences = len(self.source_dataset)
        iterator = tqdm(range(num_sequences), total=num_sequences, desc="Finding Valid AR Windows") if 'tqdm' in globals() else range(num_sequences)

        for seq_idx in iterator:
            try:
                original_seq_len = self.source_dataset.get_sequence_length_by_idx(seq_idx) 
            except Exception as e:
                # --- 打印更详细的错误信息 ---
                import traceback
                logger.error(f"EXCEPTION caught while getting sequence length for seq_idx {seq_idx}: {type(e).__name__} - {e}")
                logger.error(traceback.format_exc()) # 打印完整的 traceback
                # --- 结束修改 ---
                original_seq_len = None # 保持设为 None

            if original_seq_len is None:
                logger.warning(f"Sequence length for seq_idx {seq_idx} ended up as None (likely due to error above). Skipping this sequence.")
                continue 

            if not isinstance(original_seq_len, int) or original_seq_len < 0:
                 logger.warning(f"Invalid sequence length for seq_idx {seq_idx}: {original_seq_len}. Skipping.")
                 continue # 跳过无效长度
            
            # Iterate through possible start points of raw windows
            for start_idx in range(original_seq_len - self.raw_window_size + 1):
                try:
                    item = self.source_dataset[seq_idx] # Load item once
                    if not isinstance(item, dict) or 'mask' not in item:
                         logger.warning(f"Item at seq_idx {seq_idx} is not a dict or missing 'mask'. Assuming all valid for this window check.", once=True)
                         raw_context_mask = np.ones(self.context_length, dtype=int) 
                    else:
                        if start_idx + self.context_length > len(item['mask']): continue
                        raw_context_mask = item['mask'][start_idx : start_idx + self.context_length] # Check context part

                    # 使用正确的属性名 self.min_valid_history
                    if np.sum(raw_context_mask) >= self.min_valid_history:
                        self.valid_windows.append((seq_idx, start_idx))
                except Exception as e:
                     logger.error(f"Error processing seq_idx {seq_idx} in __init__: {e}")


        if not self.valid_windows:
            # Update error message if needed
            raise ValueError(f"No valid AR windows found with context_length={context_length}, "
                             f"and min_valid_history={self.min_valid_history}. "
                             f"Check sequence lengths ({num_sequences} sequences processed) and data validity (masks).")
        logger.info(f"Found {len(self.valid_windows)} potential valid AR windows.")


    def __len__(self):
        return len(self.valid_windows)

    def __getitem__(self, idx):
        seq_idx, start_idx = self.valid_windows[idx]
        
        try:
            item = self.source_dataset[seq_idx]
            if not isinstance(item, dict) or not all(k in item for k in ['sequence', 'time', 'mask']):
                 if isinstance(item, (np.ndarray, list)): 
                     raw_sequence_full = np.array(item, dtype=np.float32)
                     raw_time_full = np.arange(len(raw_sequence_full), dtype=np.float32)
                     raw_mask_full = np.ones(len(raw_sequence_full), dtype=np.int32)
                     # Slice the generated full arrays
                     end_idx = start_idx + self.raw_window_size # Use AR window size
                     if end_idx > len(raw_sequence_full): 
                         raise IndexError(f"Calculated end_idx {end_idx} exceeds generated sequence length {len(raw_sequence_full)}")
                     raw_sequence_window = raw_sequence_full[start_idx:end_idx]
                     raw_time_window = raw_time_full[start_idx:end_idx]
                     raw_mask_window = raw_mask_full[start_idx:end_idx]
                 else:
                      raise TypeError(f"Item from source_dataset at index {seq_idx} is not a dict or sequence array.")
            else:
                # Item is a dict, slice using AR window size
                end_idx = start_idx + self.raw_window_size 
                raw_sequence_window = item['sequence'][start_idx:end_idx]
                raw_time_window = item['time'][start_idx:end_idx]
                raw_mask_window = item['mask'][start_idx:end_idx]

        except IndexError as e:
             logger.error(f"IndexError accessing data for window idx {idx} (seq_idx={seq_idx}, start_idx={start_idx}): {e}")
             raise e 
        except Exception as e:
             logger.error(f"Unexpected error accessing data for window idx {idx} (seq_idx={seq_idx}, start_idx={start_idx}): {e}")
             raise e

        # --- 恢复自回归逻辑 ---
        # 1. Filter invalid points based on mask
        valid_indices = raw_mask_window == 1
        valid_sequence = raw_sequence_window[valid_indices]
        valid_time_abs = raw_time_window[valid_indices] # Absolute times of valid points

        # Ensure we still have enough points after filtering for input/label pair
        if len(valid_sequence) < 2:
             logger.debug(f"Skipping window {idx} (seq={seq_idx}, start={start_idx}) due to insufficient valid points ({len(valid_sequence)} < 2).")
             # Returning None requires collate_fn to handle it.
             # For simplicity during debugging, let's return a dummy dict that collate_fn might filter.
             # A truly robust solution filters self.valid_windows more strictly in __init__
             # or uses a filtering collate_fn.
             return None # Let time_aware_collate_fn handle this

        # 2. Time Normalization (apply to valid absolute times)
        valid_time_norm = valid_time_abs # Default if no normalizer
        if self.time_normalizer is not None:
            try:
                 if len(valid_time_abs) > 0:
                    valid_time_norm = self.time_normalizer.transform(valid_time_abs.reshape(-1, 1)).flatten()
            except Exception as e:
                 logger.error(f"Error applying time normalizer for window idx {idx}: {e}. Using original times.")
                 valid_time_norm = valid_time_abs # Fallback

        # 3. Prepare input_ids, time_values, labels for AutoRegressive Task
        input_ids = valid_sequence[:-1].astype(np.float32)
        time_values = valid_time_norm[:-1].astype(np.float32) # Normalized times for input
        labels = valid_sequence[1:].astype(np.float32)

        # 4. Prepare attention mask and loss mask (for valid sequence length)
        current_length = len(input_ids)
        attention_mask = np.ones(current_length, dtype=np.int32)
        loss_mask = np.ones(len(labels), dtype=np.int32) # Original code used int32, collate converts to bool

        # 5. Prepare `next_target_time_values` (Optional, but original code had it)
        next_target_time_value = valid_time_norm[1] if len(valid_time_norm) > 1 else np.nan
        next_target_time_value = np.float32(next_target_time_value)
        # 6. Return dictionary suitable for time_aware_collate_fn
        return {
            'input_ids': input_ids,        
            'time_values': time_values,    
            'attention_mask': attention_mask, 
            'labels': labels,              
            'loss_masks': loss_mask,        
<<<<<<< HEAD
            'next_target_time_value': next_target_time_value 
=======
            'next_target_time_values': next_target_time_value 
>>>>>>> 7e18418 ([DEBUG] CT-RoPE)
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
