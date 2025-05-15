import os
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import argparse
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap, os  


@torch.no_grad()                            
def collect_gating_scores(model, records, ctx_len=128, device="cuda"):  

    model.eval().to(device)

    batch = []
    for r in records:
        seq = r["sequence"][:ctx_len]          # (ctx_len,)
        batch.append(seq)
    inp = torch.tensor(batch, dtype=torch.float32, device=device)  # [B, ctx_len]

    out = model.model(
        input_ids=inp,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True
    )
    router_logits = out.router_logits          # list[L] each [B, ctx_len, E]
    num_layers = len(router_logits)
    num_experts = router_logits[0].shape[-1]
    mats = []
    for l in router_logits:
        probs = torch.softmax(l, -1)
        if probs.dim() == 3:                    # [B, seq, E]
            mean_probs = probs.mean(dim=(0, 1))
        else:                                   # [tokens, E]
            mean_probs = probs.mean(dim=0)
        mats.append(mean_probs)
    return torch.stack(mats).cpu()   


def plot_gating_heatmap(mat, title="Gating Scores", vmax=0.5, save_png=None):  
    """
    mat: Tensor/ndarray  [num_layers, num_experts]
    """
    plt.figure(figsize=(3, 5))
    sns.heatmap(mat, vmin=0, vmax=vmax, cmap="YlGnBu",
                cbar_kws={"label": "avg p(expert)"},
                yticklabels=np.arange(mat.shape[0]),
                xticklabels=np.arange(mat.shape[1]))
    plt.ylabel("Layers"); plt.xlabel("Experts"); plt.title(title)
    plt.tight_layout()
    if save_png:
        plt.savefig(save_png, dpi=300, bbox_inches="tight")
        print(f"[Info] heatmap saved to {save_png}")
    else:
        plt.show()


def parse_settings(settings_str):
    settings = []
    for s in settings_str.split(","):
        ctx, pred = map(int, s.strip().split(":"))
        settings.append((ctx, pred))
    return settings


def load_jsonl_records(jsonl_path):
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            item = json.loads(line)
            seq = np.array(item["sequence"], dtype=np.float32)
            mask = np.array(item["mask"], dtype=bool)
            if np.any(~mask):
                valid = np.where(mask)[0]
                seq_interp = np.interp(np.arange(len(seq)), valid, seq[valid])
            else:
                seq_interp = seq.copy()
            mean, std = seq_interp.mean(), seq_interp.std()
            norm_seq = (seq_interp - mean) / (std + 1e-8)
            records.append({"sequence": norm_seq, "mask": mask, "mean": mean, "std": std})
    return records


def batched_prediction(model, batch_contexts, pred_len):
    device = model.device
    input_tensor = torch.tensor(batch_contexts, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model.generate(input_tensor, max_new_tokens=pred_len)
    return outputs[:, -pred_len:].cpu().numpy()


def masked_metrics(y_true, y_pred, mask):
    y_true = np.array(y_true)[mask]
    y_pred = np.array(y_pred)[mask]
    if len(y_true) == 0:
        return None
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse


def evaluate_sequence(model, sequence, mask, context_len, pred_len, batch_size, mean, std):
    total_len = len(sequence)
    batch_contexts, batch_targets, batch_masks, results = [], [], [], []
    for start in range(0, total_len - (context_len + pred_len) + 1, context_len):
        context = sequence[start:start + context_len]
        target = sequence[start + context_len:start + context_len + pred_len]
        target_mask = mask[start + context_len:start + context_len + pred_len]
        batch_contexts.append(context)
        batch_targets.append(target)
        batch_masks.append(target_mask)
        if len(batch_contexts) == batch_size:
            preds = batched_prediction(model, batch_contexts, pred_len)
            for i in range(len(preds)):
                # Denormalize predictions and targets
                denorm_preds = preds[i] * std + mean
                denorm_target = batch_targets[i] * std + mean
                metrics = masked_metrics(denorm_target, denorm_preds, batch_masks[i])
                if metrics:
                    results.append(metrics)
            batch_contexts, batch_targets, batch_masks = [], [], []
    if batch_contexts:
        preds = batched_prediction(model, batch_contexts, pred_len)
        for i in range(len(preds)):
            denorm_preds = preds[i] * std + mean
            denorm_target = batch_targets[i] * std + mean
            metrics = masked_metrics(denorm_target, denorm_preds, batch_masks[i])
            if metrics:
                results.append(metrics)
    if results:
        return np.mean(results, axis=0)
    return None


def process_file(model, jsonl_path, settings, batch_size):
    records = load_jsonl_records(jsonl_path)
    summary = []
    for context_len, pred_len in settings:
        maes, mses, rmses = [], [], []
        for record in records:
            result = evaluate_sequence(
                model,
                record["sequence"],
                record["mask"],
                context_len,
                pred_len,
                batch_size,
                record["mean"],
                record["std"]
            )
            if result is not None and len(result) > 0:
                maes.append(result[0])
                mses.append(result[1])
                rmses.append(result[2])
        if maes:
            summary.append({
                "context_len": context_len,
                "pred_len": pred_len,
                "mae": np.mean(maes),
                "mse": np.mean(mses),
                "rmse": np.mean(rmses)
            })
    return summary

def plot_multi_gating_mats(gating_dict, cmap="viridis", vmax=0.5, save_png=None):
    """
    gating_dict: Dict[str, ndarray[L,E]]
                 key = json 文件名 (或任意 title)
    """
    n = len(gating_dict)
    if n == 0:
        return
    layers, experts = next(iter(gating_dict.values())).shape

    # 每个子图宽 3 英寸，高 5 英寸
    fig, axes = plt.subplots(
        1, n, figsize=(3 * n, 5), sharey=True,
        gridspec_kw={"wspace": 0.15}
    )
    if n == 1:
        axes = [axes]

    for ax, (name, mat) in zip(axes, gating_dict.items()):
        sns.heatmap(
            mat, vmin=0, vmax=vmax, cmap=cmap,
            ax=ax, cbar=False if ax is not axes[-1] else True,
        )
        # 如果文件名太长，简短显示
        title = textwrap.shorten(os.path.basename(name), width=20, placeholder="…")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Experts")
        if ax is axes[0]:
            ax.set_ylabel("Layers")
        else:
            ax.set_ylabel("")

    plt.tight_layout()
    if save_png:
        plt.savefig(save_png, dpi=300, bbox_inches="tight")
        print(f"[Info] multi-heatmap saved → {save_png}")
    else:
        plt.show()

def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, device_map="auto"
    )
    print(f"Loaded model: {args.model} on {model.device}")
    gating_mats = {}                                             # <<< 新增
    jsonl_files = (
        list(Path(args.input_dir).rglob("*.jsonl"))
        if os.path.isdir(args.input_dir) else
        [Path(args.input_dir)]
    )

    for jf in jsonl_files:
        records = load_jsonl_records(jf)
        mat = collect_gating_scores(
            model, records, ctx_len=args.ctx_len,
            device=str(model.device)
        )
        gating_mats[str(jf)] = mat                               # key 用文件路径
        # 也可以顺手存 .npy
        if args.gating_npy:
            npy_path = Path(args.gating_npy) / f"{jf.stem}.npy"
            np.save(npy_path, mat.numpy())

    # 统一画图
    if args.gating_png:
        plot_multi_gating_mats(
            gating_mats,
            cmap=args.cmap,            # 见下一节 CLI
            vmax=args.vmax,
            save_png=args.gating_png
        )

    # if args.gating_png or args.gating_npy:                   
    #     all_records = []
    #     for jf in jsonl_files:
    #         all_records.extend(load_jsonl_records(jf))
    #     gating_mat = collect_gating_scores(
    #         model, all_records, ctx_len=args.ctx_len,
    #         device=str(model.device)
    #     )                                                       # Tensor[L,E]
    #     if args.gating_npy:
    #         np.save(args.gating_npy, gating_mat.numpy())
    #         print(f"[Info] gating matrix saved to {args.gating_npy}")
    #     plot_gating_heatmap(gating_mat, save_png=args.gating_png)

    settings = parse_settings(args.settings)
    overall_results = {f"{c}:{p}": {"mae": [], "mse": [], "rmse": []}
                       for c, p in settings}

    for jsonl_file in jsonl_files:
        print(f"\nProcessing {jsonl_file}...")
        summaries = process_file(model, jsonl_file, settings, args.batch_size)
        for s in summaries:
            key = f"{s['context_len']}:{s['pred_len']}"
            print(f"{key} | MAE: {s['mae']:.4f}, MSE: {s['mse']:.4f}, "
                  f"RMSE: {s['rmse']:.4f}")
            for m in ["mae", "mse", "rmse"]:
                overall_results[key][m].append(s[m])

    print("\n==== Overall Averages Across All Files ====")
    for key, metrics in overall_results.items():
        if metrics["mae"]:
            print(f"{key} | MAE: {np.mean(metrics['mae']):.4f}, "
                  f"MSE: {np.mean(metrics['mse']):.4f}, "
                  f"RMSE: {np.mean(metrics['rmse']):.4f}")

# ------------------ 4. CLI 参数 ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--settings", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    # <<< 新增：控制采集 / 保存
    parser.add_argument("--gating_png", type=str,
                        help="输出多子图热力图的 png 路径")
    parser.add_argument("--gating_npy", type=str,
                        help="若指定为目录，则把每个 gating mat 存 *.npy 到该目录")
    parser.add_argument("--ctx_len", type=int, default=128)
    parser.add_argument("--cmap", type=str, default="viridis",
                        help="Matplotlib colormap 名称 (默认 viridis)")
    parser.add_argument("--vmax", type=float, default=0.5,
                        help="热力图上限 (默认为 0.5，与论文一致)")
    args = parser.parse_args()
    main(args)
