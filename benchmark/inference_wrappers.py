from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai import MoiraiModule, MoiraiForecast
from uni2ts.model.moirai_moe import MoiraiMoEModule, MoiraiMoEForecast
import timesfm
from chronos import BaseChronosPipeline

def reshape_output(output, target_shape):
    if output.shape != target_shape and output.T.shape == target_shape:
        return output.T
    return output

def load_model_and_predictor(model_name, context_length, prediction_length, patch_size, batch_size, target_dim, freq):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_lower = model_name.lower()

    # Moirai
    if "moirai" in model_lower and "moe" not in model_lower:
        module = MoiraiModule.from_pretrained(model_name)
        model = MoiraiForecast(module, prediction_length, context_length, patch_size, 100, target_dim, 0, 0)
        predictor = model.create_predictor(batch_size=batch_size)

        def predict_fn(context, start_time=None):
            dataset = ListDataset([{
                "start": start_time or "2000-01-01 00:00:00",
                "target": context.tolist()
            }], freq=freq, one_dim_target=(target_dim == 1))
            forecast = next(predictor.predict(dataset))
            return reshape_output(np.array(forecast.mean), context.shape)

        return predict_fn

    # Moirai-MoE
    elif "moirai-moe" in model_lower:
        module = MoiraiMoEModule.from_pretrained(model_name)
        model = MoiraiMoEForecast(module, prediction_length, context_length, patch_size, 1, target_dim, 0, 0)
        predictor = model.create_predictor(batch_size=batch_size)

        def predict_fn(context, start_time=None):
            dataset = ListDataset([{
                "start": start_time or "2000-01-01 00:00:00",
                "target": context.tolist()
            }], freq=freq, one_dim_target=(target_dim == 1))
            forecast = next(predictor.predict(dataset))
            return reshape_output(np.array(forecast.mean), context.shape)

        return predict_fn

    # Time-MoE
    elif "timemoe" in model_lower:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto"
        ).to(device)
        model.eval()

        def predict_fn(context, start_time=None):
            if context.ndim == 2 and context.shape[0] == 1:
                input_tensor = torch.tensor(context, dtype=torch.float32).to(model.device)
            elif context.ndim == 1:
                input_tensor = torch.tensor(context[None, :], dtype=torch.float32).to(model.device)
            else:
                raise ValueError(f"[TimeMoE] Unexpected input shape: {context.shape}")


            mean = input_tensor.mean(dim=-1, keepdim=True)
            std = input_tensor.std(dim=-1, keepdim=True)
            normed_input = (input_tensor - mean) / (std + 1e-6)

            with torch.no_grad():
                output = model.generate(normed_input, max_new_tokens=prediction_length)

            normed_pred = output[:, -prediction_length:]
            pred = normed_pred * std + mean
            return pred[0].cpu().numpy().reshape(1, -1)

        return predict_fn

    # TimesFM
    elif "timesfm" in model_lower:
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu" if torch.cuda.is_available() else "cpu",
                per_core_batch_size=batch_size,
                context_len=context_length,
                horizon_len=prediction_length,
                use_positional_embedding=False
            ),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=model_name)
        )

        def predict_fn(context, start_time=None):
            # context: shape (C, L) or (L,)
            if context.ndim == 2:
                series = context[0]  # 单变量
            elif context.ndim == 1:
                series = context
            else:
                raise ValueError("Invalid context shape for TimesFM")

            # 需要提供一个 list[1D array]
            forecast_input = [series.astype(np.float32)]
            frequency_input = [0]  # dummy freq

            point_forecast, _ = tfm.forecast(forecast_input, freq=frequency_input)
            pred = point_forecast[0]  # shape: (prediction_length,)
            return pred.reshape(1, -1)  # reshape to (C, L)

        return predict_fn

    # Chronos
    elif "chronos" in model_lower:
        pipe = BaseChronosPipeline.from_pretrained(
            model_name,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
        )

        def predict_fn(context, start_time=None):
            if context.ndim == 2:
                context_tensor = torch.tensor(context[0], dtype=torch.float32)
            else:
                context_tensor = torch.tensor(context, dtype=torch.float32)

            quantiles, mean = pipe.predict_quantiles(
                context=context_tensor,
                prediction_length=prediction_length,
                quantile_levels=[0.5]
            )
            return mean[0].detach().cpu().numpy().reshape(1, -1)

        return predict_fn

    # TiMER
    elif "timer" in model_lower:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True
        ).to(device)
        model.eval()

        def predict_fn(context, start_time=None):
            
            if context.ndim == 2 and context.shape[0] == 1:
                seq = context[0]
            elif context.ndim == 1:
                seq = context
            else:
                raise ValueError(f"[TiMER] Unexpected input shape: {context.shape}")


            max_ctx_len = min(getattr(model.config, "max_position_embeddings", 2048), 2048)
            if len(seq) > max_ctx_len:
                print(f"[TiMER] Truncating input from {len(seq)} to {max_ctx_len}")
                seq = seq[-max_ctx_len:]

            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(model.device)  # [1, L]

            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            normed_x = (x - mean) / (std + 1e-6)

            with torch.no_grad():
                output = model.generate(
                    normed_x,
                    max_new_tokens=prediction_length,
                    do_sample=False,
                    use_cache=False
                )

            pred = output[:, -prediction_length:] * std + mean
            return pred[0].cpu().numpy().reshape(1, -1)

        return predict_fn


    else:
        raise ValueError(f"Unsupported or unknown model: {model_name}")
