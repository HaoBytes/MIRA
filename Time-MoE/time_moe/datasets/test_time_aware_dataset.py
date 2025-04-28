#!/usr/bin/env python
# -*- coding:utf-8 _*-
import unittest
import tempfile
import os
import json
import numpy as np
from timeawared_dataset import TimeAwareJSONLDataset, TimeAwareEvalDataset  # 替换为实际模块路径

class TestTimeAwareDatasets(unittest.TestCase):
    def setUp(self):
        # 创建临时测试JSONL文件
        self.test_data = [
            {"sequence": [1.0, 2.0, np.nan, 4.0], "time": [0, 1, 2, 3], "mask": [1, 1, 0, 1]},
            {"sequence": [np.nan, 5.0, 6.0, 7.0, np.nan], "time": [10, 11, 12, 13, 14], "mask": [0, 1, 1, 1, 0]},
            {"sequence": [8.0, 9.0, 10.0], "mask": [1, 1, 1]}  # 测试缺少time字段的情况
        ]
        
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.jsonl', delete=False)
        for item in self.test_data:
            self.temp_file.write(json.dumps(item) + '\n')
        self.temp_file.close()

    def tearDown(self):
        os.unlink(self.temp_file.name)

    def test_time_aware_jsonl_dataset(self):
        """测试基础数据集加载功能"""
        dataset = TimeAwareJSONLDataset(self.temp_file.name)
        
        # 测试长度
        self.assertEqual(len(dataset), 3)
        
        # 测试第一项
        item0 = dataset[0]
        np.testing.assert_array_equal(item0['sequence'], np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float32))
        np.testing.assert_array_equal(item0['time'], np.array([0, 1, 2, 3], dtype=np.float32))
        np.testing.assert_array_equal(item0['mask'], np.array([1, 1, 0, 1], dtype=np.float32))
        
        # 测试自动填充time
        item2 = dataset[2]
        np.testing.assert_array_equal(item2['time'], np.array([0, 1, 2], dtype=np.float32))
        
        # 测试token计数
        self.assertEqual(dataset.get_num_tokens(), 12)  # 4+5+3

    def test_time_aware_eval_dataset(self):
        """测试评估数据集功能"""
        base_dataset = TimeAwareJSONLDataset(self.temp_file.name)
        eval_dataset = TimeAwareEvalDataset(
            dataset=base_dataset,
            context_length=2,
            prediction_length=1,
            normalize=False
        )
        
        # 测试窗口数量 (第一个序列有2个有效窗口，第二个有3个，第三个有1个)
        self.assertEqual(len(eval_dataset), 6)
        
        # 测试第一个窗口 (来自第一个序列)
        sample0 = eval_dataset[0]
        self.assertEqual(sample0['inputs']['original_length'], 2)
        self.assertEqual(sample0['labels']['original_length'], 1)
        
        # 验证输入的有效值
        np.testing.assert_array_equal(sample0['inputs']['sequence'], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(sample0['inputs']['valid_indices'], np.array([0, 1]))
        np.testing.assert_array_equal(sample0['input_mask'], np.array([1, 1]))
        
        # 验证标签处理方式
        if len(sample0['labels']['sequence']) == 0:
            # 如果实现过滤了NaN值
            self.assertEqual(len(sample0['labels']['valid_indices']), 0)
        else:
            # 如果实现保留了NaN值
            np.testing.assert_array_equal(sample0['labels']['sequence'], np.array([np.nan]))
            self.assertEqual(len(sample0['labels']['valid_indices']), 0)
        
        # 测试时间信息
        np.testing.assert_array_equal(sample0['inputs']['time'], np.array([0, 1]))
        if len(sample0['labels']['sequence']) > 0:
            np.testing.assert_array_equal(sample0['labels']['time'], np.array([2]))

    def test_normalization(self):
        """测试归一化功能"""
        base_dataset = TimeAwareJSONLDataset(self.temp_file.name)
        eval_dataset = TimeAwareEvalDataset(
            dataset=base_dataset,
            context_length=2,
            prediction_length=1,
            normalize=True
        )
        
        # 获取第三个样本 (来自第二个序列的第二个窗口)
        sample = eval_dataset[3]
        
        # 计算输入窗口的统计量 (应该是[6.0,7.0])
        input_window = np.array([6.0, 7.0])
        expected_mean = input_window.mean()
        expected_std = input_window.std()
        
        # 验证归一化是否正确
        normalized_input = sample['inputs']['sequence']
        np.testing.assert_almost_equal(normalized_input.mean(), 0, decimal=5)
        np.testing.assert_almost_equal(normalized_input.std(), 1, decimal=5)
        
        # 验证标签是否使用相同的归一化参数
        if len(sample['labels']['sequence']) > 0:
            original_label = np.array([7.0])
            normalized_label = (original_label - expected_mean) / expected_std
            np.testing.assert_almost_equal(
                sample['labels']['sequence'], 
                normalized_label, 
                decimal=5
            )

    def test_reconstruction(self):
        """测试序列重建功能"""
        base_dataset = TimeAwareJSONLDataset(self.temp_file.name)
        eval_dataset = TimeAwareEvalDataset(
            dataset=base_dataset,
            context_length=2,
            prediction_length=1,
            normalize=False
        )
        
        def reconstruct(valid_data):
            full_seq = np.full(valid_data['original_length'], np.nan)
            full_seq[valid_data['valid_indices']] = valid_data['sequence']
            return full_seq
        
        # 测试重建第一个样本
        sample = eval_dataset[0]
        reconstructed_input = reconstruct(sample['inputs'])
        np.testing.assert_array_equal(reconstructed_input, np.array([1.0, 2.0]))
        
        reconstructed_label = reconstruct(sample['labels'])
        np.testing.assert_array_equal(reconstructed_label, np.array([np.nan]))

if __name__ == '__main__':
    unittest.main()