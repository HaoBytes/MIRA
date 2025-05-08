#!/usr/bin/env python
# -*- coding:utf-8 _*-
from abc import abstractmethod


class TimeSeriesDataset:
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, seq_idx):
        """应返回包含以下键的字典：
        - 'sequence': 数值序列
        - 'time': 时间信息(可选)
        - 'mask': 有效值掩码
        """
        pass

    @abstractmethod
    def get_num_tokens(self):
        pass

    @abstractmethod
    def get_sequence_length_by_idx(self, seq_idx):
        pass

    @staticmethod
    def is_valid_path(data_path):
        return True

    def __iter__(self):
        n_seqs = len(self)
        for i in range(n_seqs):
            yield self[i]