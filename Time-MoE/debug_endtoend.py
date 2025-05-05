import json
import numpy as np
import random
import os
from tqdm import tqdm

def generate_pseudo_time_series(
    num_series: int = 1000,          # 要生成的序列数量
    min_len: int = 50,             # 序列最小长度
    max_len: int = 512,            # 序列最大长度 (可以根据需要调整)
    nan_prob: float = 0.05,        # 每个点为 NaN 的概率
    time_irregularity: float = 0.3 # 时间戳不规则性因子 (0 到 1，越大越不规则)
) -> list:
    """
    生成伪时间序列数据。

    Args:
        num_series: 生成的序列数量。
        min_len: 序列的最小长度。
        max_len: 序列的最大长度。
        nan_prob: 序列中每个点为 NaN 的概率。
        time_irregularity: 时间戳不规则性的程度 (0 表示完全规则，1 表示最大不规则)。

    Returns:
        一个包含字典的列表，每个字典代表一条时间序列。
    """
    dataset = []
    for _ in tqdm(range(num_series), desc="Generating pseudo dataset"):
        seq_len = random.randint(min_len, max_len)

        # 1. 生成基础序列 (例如，带有一些模式的正弦波)
        time_points = np.arange(seq_len)
        frequency = random.uniform(0.05, 0.5)
        amplitude = random.uniform(0.5, 2.0)
        phase = random.uniform(0, np.pi * 2)
        noise_level = random.uniform(0.05, 0.2)

        sequence = amplitude * np.sin(2 * np.pi * frequency * time_points + phase)
        sequence += np.random.normal(0, noise_level, seq_len) # 添加噪声
        sequence = sequence.astype(np.float32) # 使用 float32 存储

        # 2. 生成不规则时间戳 (必须是严格递增的)
        times = np.zeros(seq_len, dtype=np.float64) # 使用 float64 保证精度
        current_time = 0.0
        for i in range(seq_len):
            # 基础步长 + 不规则随机步长
            base_step = 0.1 # 可以调整基础时间间隔
            random_step = random.uniform(-base_step * time_irregularity, base_step * time_irregularity)
            step = base_step + random_step
            # 确保步长为正，避免时间倒流
            step = max(step, 1e-6) # 保证最小步长，防止时间戳过于接近甚至相等
            current_time += step
            times[i] = current_time

        # 3. 生成掩码并引入 NaN
        mask = np.ones(seq_len, dtype=np.int32)
        nan_indices = np.random.rand(seq_len) < nan_prob
        sequence[nan_indices] = np.nan
        mask[nan_indices] = 0

        # 确保序列至少包含一个有效点 (避免全为 NaN)
        if np.sum(mask) == 0:
            valid_idx = random.randint(0, seq_len - 1)
            # 重新生成一个非 NaN 的值
            sequence[valid_idx] = amplitude * np.sin(2 * np.pi * frequency * time_points[valid_idx] + phase) + np.random.normal(0, noise_level)
            mask[valid_idx] = 1


        dataset.append({
            "sequence": sequence.tolist(), # 转换为列表以便 JSON 序列化
            "time": times.tolist(),        # 转换为列表
            "mask": mask.tolist()          # 转换为列表
        })

    return dataset

# --- 使用示例 ---
if __name__ == "__main__":
    # 定义参数
    output_dir = "pseudo_dataset"
    output_filename = "pseudo_time_series.jsonl"
    num_sequences_to_generate = 5000  # 生成更多数据用于训练
    max_sequence_length = 1024       # 匹配 main.py 的默认 max_length

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    # 生成数据集
    pseudo_data = generate_pseudo_time_series(
        num_series=num_sequences_to_generate,
        max_len=max_sequence_length,
        nan_prob=0.05,
        time_irregularity=0.4 # 可以调整不规则程度
    )

    # 将数据集写入 JSON Lines 文件
    print(f"\nWriting dataset to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(pseudo_data, desc="Writing JSON Lines"):
            f.write(json.dumps(item) + '\n')

    print(f"\nSuccessfully generated pseudo dataset with {num_sequences_to_generate} sequences at {output_path}")

    # --- 如何使用 main.py 运行 ---
    print("\n--- Example command to run main.py ---")
    # 假设你的 main.py 在当前目录的 time_moe 子目录中
    # 需要根据你的项目结构调整 python -m time_moe.main
    command = f"""
    python -m time_moe.main \\
        --data_path {output_path} \\
        --model_path Maple728/TimeMoE-50M \\
        --output_path ./logs/pseudo_run \\
        --max_length {max_sequence_length} \\
        --learning_rate 5e-5 \\
        --num_train_epochs 1.0 \\
        --normalization_method zero \\
        --global_batch_size 32 \\
        --micro_batch_size 8 \\
        --logging_steps 10 \\
        --save_strategy no \\
        --evaluation_strategy no \\
        --time_aware \\
        --from_scratch
        --precision bf16 if using compatible hardware
    """
    # 注意：上面的命令是 Linux/macOS 格式，Windows 下需要将 `\` 替换为 `^` 或写在一行
    print(command)