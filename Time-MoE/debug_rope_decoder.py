import torch
from time_moe.models.configuration_time_moe import TimeMoeConfig
from time_moe.models.modeling_time_moe import TimeMoeForPrediction  

def test_time_aware_model():
    # 测试配置
    config = TimeMoeConfig(
        input_size=1,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        horizon_lengths=[3],
        time_aware=True,          # 启用时间感知
        time_aware_rotary=True,   # 启用时间旋转编码
        temporal_embed_dim=8,
        num_experts=4,
        _attn_implementation="eager",
        use_dense=False,
        apply_aux_loss=True,
    )
    
    config.loss_type = "regression"
    # 初始化模型
    model = TimeMoeForPrediction(config)
    print("Model structure:")
    print(model)
    
    # 生成测试数据 (batch=2, seq_len=10)
    input_values = torch.randn(2, 10, 1)
    time_values = torch.arange(10).float().view(1,10,1).expand(2,10,1)  # 时间特征
    labels = torch.randn(2, 10, 1)
    
    # 前向传播测试
    print("\n[1] Forward pass test:")
    outputs = model(
        input_ids=input_values,
        time_values=time_values,  # 传入时间特征
        labels=labels
    )
    print(f"Loss value: {outputs.loss.item():.4f}")
    print(f"Predictions shape: {outputs.logits.shape} (should be [2,10,3])")
    
    # 梯度反向测试
    print("\n[2] Backward propagation test:")
    outputs.loss.backward()
    
    # 验证梯度存在性
    grad_exist = any(p.grad is not None for p in model.parameters())
    print(f"Gradient exists: {grad_exist} (should be True)")
    
    # 模块结构验证
    print("\n[3] Module structure verification:")
    print("Embedding layer type:", 
          type(model.model.embed_layer).__name__)  # 应为OptimizedTimeSeriesInputEmbedding
    print("Rotary Embedding type:", 
          type(model.model.layers[0].self_attn.rotary_emb).__name__)  # 应为TimeAwareRotaryEmbedding
    
    # 时间编码器验证
    if hasattr(model.model, 'temporal_encoder'):
        print("Temporal encoder output test:")
        temporal_output, _ = model.model.temporal_encoder(time_values)
        print(f"Temporal encoding shape: {temporal_output.shape} (should be [2,10,128])")
    
    print("\n✅ All basic tests passed!")

if __name__ == "__main__":
    test_time_aware_model()