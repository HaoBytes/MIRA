#!/usr/bin/env python
# -*- coding:utf-8 _*-
from abc import abstractmethod
import numpy as np

class TimeAwareDataset:
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
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
    def get_sequence_length(self, idx):
        pass

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]