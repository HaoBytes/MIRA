import pytest
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
from time_moe.models.modeling_time_moe import TimeMoeForPrediction
from time_moe.models.configuration_time_moe import TimeMoeConfig
from time_moe.models.modeling_time_moe import TimeMoeForPrediction
from time_moe.datasets.time_moe_window_dataset import TimeAwareWindowDataset

DEBUG = True  # 全局调试开关

# 增强的测试配置
TEST_CONFIG = {
    "input_size": 1,
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "max_position_embeddings": 256,
    "time_aware": True,
    "temporal_embed_dim": 16
}

# 模拟时间序列数据集
class MockTimeSeriesDataset(Dataset):
    def __init__(self, num_samples=100, seq_length=24):
        self.data = []
        for _ in range(num_samples):
            seq_len = np.random.randint(seq_length//2, seq_length*2)
            
            # 生成有效序列（约80%有效值）
            values = np.random.randn(seq_len).astype(np.float32)
            mask = np.random.choice([0,1], size=seq_len, p=[0.2, 0.8]).astype(np.float32)
            time = np.cumsum(np.random.exponential(1.0, size=seq_len)).astype(np.float32)
            
            # 添加随机缺失
            values[mask==0] = 0.0  # 用0表示缺失值
            
            self.data.append({
                "sequence": values,
                "time": time,
                "mask": mask
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        if DEBUG and idx % 20 == 0:  # 抽样打印原始数据
            print(f"\n📊 Raw Sample {idx}:")
            print(f"   Sequence Len: {len(item['sequence'])}")
            print(f"   Missing Rate: {1 - item['mask'].mean():.1%}")
            print(f"   Time Range: {item['time'][0]:.2f} - {item['time'][-1]:.2f}")
        return item

# 测试数据加载
def test_data_pipeline():
    dataset = MockTimeSeriesDataset(num_samples=10)
    
    # 验证数据格式
    sample = dataset[0]
    assert "sequence" in sample
    assert "time" in sample
    assert "mask" in sample
    assert len(sample["sequence"]) == len(sample["time"]) == len(sample["mask"])
    
    # 验证数据加载器
    loader = DataLoader(dataset, batch_size=2, collate_fn=lambda x: x)
    batch = next(iter(loader))
    assert len(batch) == 2

    if DEBUG:
        # 数据分布分析
        seq_lens = [len(sample["sequence"]) for sample in dataset]
        print(f"\n📈 Data Distribution:")
        print(f"   Avg Sequence Length: {np.mean(seq_lens):.1f} ± {np.std(seq_lens):.1f}")
        print(f"   Missing Rate Distribution: {np.mean([1 - s['mask'].mean() for s in dataset]):.1%}")
    print("Data pipeline test passed!")

# 测试模型初始化
def test_model_initialization():

    
    config = TimeMoeConfig(**TEST_CONFIG)
    if DEBUG:
        print("\n🔍 Model Configuration:")
        for k, v in config.__dict__.items():
            if not k.startswith('_'):
                print(f"   {k:25}: {v}")
    
    def param_hook(module, input, output):
        if DEBUG and isinstance(module, torch.nn.Linear):
            print(f"   Layer {module.__class__.__name__}:")
            print(f"      Weight Norm: {module.weight.norm().item():.4f}")
            if module.bias is not None:
                print(f"      Bias Mean: {module.bias.mean().item():.4f}")
                      
    model = TimeMoeForPrediction(config)

    if DEBUG:
        # 注册前向钩子
        for name, layer in model.named_children():
            layer.register_forward_hook(param_hook)
    
    # 验证参数初始化
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0
    print(f"Model parameters: {total_params}")
    
    # 验证前向传播
    test_input = {
        "input_ids": torch.randn(2, 16, 1),  # [batch, seq_len, input_size]
        "time_values": torch.randn(2, 16, 1), 
        "attention_mask": torch.ones(2, 16).bool()
    }
    outputs = model(**test_input)
    
    assert outputs.logits.shape == (2, 16, 1)
    print("Model initialization test passed!")

# 训练循环测试
def test_training_loop():
    
    # 准备数据
    dataset = MockTimeSeriesDataset(num_samples=100)
    window_dataset = TimeAwareWindowDataset(
        dataset, 
        context_length=24,
        prediction_length=12,
        normalize=True
    )
    
    # 初始化模型
    config = TimeMoeConfig(**TEST_CONFIG)
    model = TimeMoeForPrediction(config)

    class DebugTrainer(TestTrainer):
        def training_step(self, model, inputs):
            if DEBUG:
                print(f"\n🔥 Training Step:")
                print(f"   Input Shape: {inputs['input_ids'].shape}")
                print(f"   Time Values Range: {inputs['time_values'].min():.2f} - {inputs['time_values'].max():.2f}")
                print(f"   Valid Tokens: {(inputs['attention_mask'] == 1).sum()/inputs['attention_mask'].numel():.1%}")
            
            loss = super().training_step(model, inputs)
            
            if DEBUG:
                print(f"   Current Loss: {loss.item():.4f}")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"   {name:30} Grad Norm: {param.grad.norm().item():.4f}")
                    else:
                        print(f"   {name:30} No Gradient")
            return loss
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./test_output",
        evaluation_strategy="no",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        learning_rate=1e-4,
        logging_steps=1,
        remove_unused_columns=False
    )

    trainer = DebugTrainer(
        model=model,
        args=training_args,
        train_dataset=window_dataset,
        data_collator=collate_fn,
    )

    # 自定义collate函数
    def collate_fn(batch):
        processed = {
            "input_ids": [],
            "time_values": [],
            "labels": [],
            "attention_mask": []
        }
        
        for item in batch:
            seq_len = len(item["inputs"]["sequence"])
            processed["input_ids"].append(torch.tensor(item["inputs"]["sequence"]))
            processed["time_values"].append(torch.tensor(item["inputs"]["time"]))
            processed["labels"].append(torch.tensor(item["labels"]["sequence"]))
            processed["attention_mask"].append(torch.ones(seq_len))
        
        # Padding处理
        processed["input_ids"] = torch.nn.utils.rnn.pad_sequence(
            processed["input_ids"], batch_first=True, padding_value=0
        )
        processed["labels"] = torch.nn.utils.rnn.pad_sequence(
            processed["labels"], batch_first=True, padding_value=-100
        )
        processed["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
            processed["attention_mask"], batch_first=True, padding_value=0
        )
        processed["time_values"] = torch.nn.utils.rnn.pad_sequence(
            processed["time_values"], batch_first=True, padding_value=0
        )
        
        return processed
    
    # 自定义Trainer
    class TestTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(
                input_ids=inputs["input_ids"],
                time_values=inputs["time_values"],
                attention_mask=inputs["attention_mask"]
            )
            loss = torch.nn.functional.mse_loss(
                outputs.logits, 
                inputs["labels"],
                reduction='none'
            )
            mask = (inputs["labels"] != -100).float()
            loss = (loss * mask).sum() / mask.sum()
            return (loss, outputs) if return_outputs else loss
    
    # 训练测试
    trainer = TestTrainer(
        model=model,
        args=training_args,
        train_dataset=window_dataset,
        data_collator=collate_fn,
    )
    
    # 执行训练
    train_result = trainer.train()
    assert train_result.training_loss < float("inf")
    print(f"Training loss: {train_result.training_loss}")
    print("Training loop test passed!")

# 推理测试
def test_inference():
    from modeling_time_moe import TimeMoeForPrediction
    from configuration_time_moe import TimeMoeConfig
    
    # 初始化模型
    config = TimeMoeConfig(**TEST_CONFIG)
    model = TimeMoeForPrediction(config)
    model.eval()
    
    # 生成测试输入
    test_input = {
        "input_ids": torch.randn(1, 24, 1),
        "time_values": torch.arange(24).float().unsqueeze(0).unsqueeze(-1),
        "attention_mask": torch.ones(1, 24).bool()
    }

    # 推理过程跟踪
    def print_layer_output(module, input, output):
        if DEBUG:
            print(f"   {module.__class__.__name__:20} Output Shape: {output.shape}")

    # 注册前向钩子
    handles = []
    for name, layer in model.named_children():
        handles.append(layer.register_forward_hook(print_layer_output))
    
    # 前向传播
    with torch.no_grad():
        outputs = model(**test_input)
    
    # 验证输出
    assert outputs.logits.shape == (1, 24, 1)
    print("Inference test passed!")

    for h in handles:
        h.remove()

# 执行所有测试
if __name__ == "__main__":
    test_data_pipeline()
    test_model_initialization()
    test_training_loop()
    test_inference()
    print("All tests passed!")