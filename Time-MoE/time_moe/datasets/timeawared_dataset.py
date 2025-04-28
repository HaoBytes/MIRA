#!/usr/bin/env python
# -*- coding:utf-8 _*-
import json
import numpy as np
from time_ts_dataset import TimeAwareDataset
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class TimeAwareJSONLDataset(TimeAwareDataset):
    def __init__(self, data_path):
        self.data = read_file_by_extension(data_path)
        self.num_tokens = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, seq_idx):
        seq = self.data[seq_idx]
        if isinstance(seq, dict):
            # 确保sequence、time和mask长度一致
            seq_len = len(seq['sequence'])
            return {
                'sequence': np.array(seq['sequence'], dtype=np.float32),
                'time': np.array(seq.get('time', range(seq_len)), dtype=np.float32),
                'mask': np.array(seq.get('mask', [1]*seq_len), dtype=np.float32)
            }
        return {
            'sequence': np.array(seq, dtype=np.float32),
            'time': np.arange(len(seq), dtype=np.float32),
            'mask': np.ones(len(seq), dtype=np.float32)
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