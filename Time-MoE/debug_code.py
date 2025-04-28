import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from typing import Optional
from time_moe.models.configuration_time_moe import TimeMoeConfig

# 假设这些是你项目中已有的定义
class TimeMoeConfig:
    def __init__(self, **kwargs):
        self.input_size = kwargs.get('input_size', 1)
        self.hidden_size = kwargs.get('hidden_size', 64)
        self.hidden_act = kwargs.get('hidden_act', 'silu')
        self.rms_norm_eps = kwargs.get('rms_norm_eps', 1e-6)
        self.max_position_embeddings = kwargs.get('max_position_embeddings', 1024)
        self.rope_theta = kwargs.get('rope_theta', 10000.0)

class TimeMoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

ACT2FN = {"silu": nn.SiLU(), "gelu": nn.GELU()}

# 1. 测试数据加载器
class TimeSeriesTestDataset(Dataset):
    def __init__(self, jsonl_path, max_length=1024):
        with open(jsonl_path, 'r') as f:
            self.data = [json.loads(line) for line in f][:10]  # 只加载前10个样本测试
        
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 处理序列数据
        seq = torch.tensor(item['sequence'], dtype=torch.float32)
        time = torch.tensor(item['time'], dtype=torch.float32)
        mask = torch.tensor(item['mask'], dtype=torch.float32)
        
        # 截断或填充到固定长度
        if len(seq) > self.max_length:
            seq = seq[:self.max_length]
            time = time[:self.max_length]
            mask = mask[:self.max_length]
        else:
            pad_len = self.max_length - len(seq)
            seq = torch.cat([seq, torch.zeros(pad_len)], dim=0)
            time = torch.cat([time, torch.zeros(pad_len)], dim=0)
            mask = torch.cat([mask, torch.zeros(pad_len)], dim=0)
            
        return {
            'values': seq.unsqueeze(-1),  # [max_len, 1]
            'times': time.unsqueeze(-1),  # [max_len, 1]
            'mask': mask.unsqueeze(-1)   # [max_len, 1]
        }

# 2. 测试函数
def test_embeddings():
    # 初始化配置
    config = TimeMoeConfig(
        input_size=1,
        hidden_size=64,
        hidden_act="silu",
        rms_norm_eps=1e-6
    )
    
    # 初始化嵌入层
    embedding_layer = OptimizedTimeSeriesInputEmbedding(config)
    
    # 测试数据
    test_data = {
        'values': torch.randn(2, 10, 1),  # [batch, seq_len, input_size]
        'times': torch.arange(10).float().unsqueeze(0).unsqueeze(-1).expand(2, 10, 1),
        'mask': torch.ones(2, 10, 1)
    }
    
    # 测试前向传播
    print("\n=== Testing OptimizedTimeSeriesInputEmbedding ===")
    print("Input shape:", test_data['values'].shape)
    print("Time shape:", test_data['times'].shape)
    
    # 测试带时间戳的情况
    embeddings_with_time = embedding_layer(test_data['values'], test_data['times'])
    print("\nWith time values:")
    print("Output shape:", embeddings_with_time.shape)
    print("Sample output:", embeddings_with_time[0, 0, :5])  # 打印第一个样本的第一个时间步的前5个维度
    
    # 测试不带时间戳的情况
    embeddings_no_time = embedding_layer(test_data['values'])
    print("\nWithout time values:")
    print("Output shape:", embeddings_no_time.shape)
    print("Sample output:", embeddings_no_time[0, 0, :5])
    
    # 测试梯度
    loss = embeddings_with_time.sum() + embeddings_no_time.sum()
    loss.backward()
    print("\nGradient checks:")
    print("Time scale grad:", embedding_layer.time_proj[0].weight.grad.norm())
    print("Value embedding grad:", embedding_layer.value_emb.weight.grad.norm())

def test_rotary_embedding():
    # 初始化配置
    dim = 32
    rotary_emb = TimeAwareRotaryEmbedding(dim)
    
    # 测试数据
    batch_size = 2
    seq_len = 10
    query = torch.randn(batch_size, seq_len, dim)
    key = torch.randn(batch_size, seq_len, dim)
    times = torch.arange(seq_len).float().unsqueeze(0).expand(batch_size, seq_len)
    
    print("\n=== Testing TimeAwareRotaryEmbedding ===")
    print("Query shape:", query.shape)
    print("Time shape:", times.shape)
    
    # 测试带时间戳的情况
    cos, sin = rotary_emb(query, time_values=times)
    print("\nWith time values:")
    print("Cos shape:", cos.shape)
    print("Sin shape:", sin.shape)
    
    # 应用旋转位置编码
    q_embed = query * cos + rotate_half(query) * sin
    print("\nAfter applying rotary embedding:")
    print("Rotated query shape:", q_embed.shape)
    print("Sample rotated query:", q_embed[0, 0, :5])
    
    # 测试不带时间戳的情况
    cos, sin = rotary_emb(query)
    print("\nWithout time values (fallback to standard RoPE):")
    print("Cos shape:", cos.shape)
    print("Sin shape:", sin.shape)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

# 3. 主测试函数
def main():
    # 测试嵌入层
    test_embeddings()
    
    # 测试旋转位置编码
    test_rotary_embedding()
    
    # 测试数据加载
    print("\n=== Testing Data Loading ===")
    dataset = TimeSeriesTestDataset("test_data.jsonl")  # 替换为你的测试数据路径
    loader = DataLoader(dataset, batch_size=2)
    
    sample = next(iter(loader))
    print("Batch shapes:")
    print("Values:", sample['values'].shape)
    print("Times:", sample['times'].shape)
    print("Mask:", sample['mask'].shape)

if __name__ == "__main__":
    main()