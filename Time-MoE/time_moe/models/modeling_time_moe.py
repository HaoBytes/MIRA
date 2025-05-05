# filename: modeling_time_moe.py
import math
from typing import Optional, Tuple, List, Union
import warnings

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel, Cache, DynamicCache, StaticCache
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import MoeModelOutputWithPast, MoeCausalLMOutputWithPast
from transformers.utils import logging, is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10

# Assuming TimeMoeConfig is defined in this relative path
from .configuration_time_moe import TimeMoeConfig
# Assuming TSGenerationMixin is defined in this relative path
from .ts_generation_mixin import TSGenerationMixin

logger = logging.get_logger(__name__)

# Flash Attention Optional Import
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    if not is_flash_attn_2_available():
        logger.info("Flash Attention imported but version is incompatible or check failed.")
        _flash_attn_available = False
    else:
        c = True
except ImportError:
    logger.info("Flash Attention not available.")
    _flash_attn_available = False
    flash_attn_func = lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError("Flash Attention not available"))
    flash_attn_varlen_func = lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError("Flash Attention not available"))



def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def load_balancing_loss_func(
        gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
        top_k: int,
        num_experts: int = None,
        attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor], List[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        top_k (`int`)
            Selected Top k over the experts.
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, (tuple, list)) or gate_logits[0] is None:
        return 0.0

    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each expert
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, 2, num_experts))
            .reshape(-1, 2, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(dim=0))

    return overall_loss * num_experts


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # The gather operation requires loading the whole embedding matrix, so we rotate first.
    # sin shape is [1, 1, seq_len, dim] or [seq_len, dim]
    # position_ids shape is [bsz, seq_len]
    # Gather sin/cos based on position_ids. Output shape [bsz, seq_len, dim]
    # Need to handle potential difference in seq_len between pos_ids and cache length
    if cos.dim() == 4: 
        cos = cos.squeeze(1).squeeze(0) # Remove batch and head dimensions [seq_len, dim]
    if sin.dim() == 4: 
        sin = sin.squeeze(1).squeeze(0) # Remove batch and head dimensions [seq_len, dim]


    # Ensure position_ids are within bounds of the cache
    position_ids = position_ids.clamp(0, cos.shape[0] - 1)

    cos = cos[position_ids].unsqueeze(unsqueeze_dim) # Shape: [bsz, 1, seq_len, dim] or [bsz, seq_len, 1, dim]
    sin = sin[position_ids].unsqueeze(unsqueeze_dim) # Shape: [bsz, 1, seq_len, dim] or [bsz, seq_len, 1, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# # Modified RoPE applying rotation based on absolute positions/times
# def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
#     """Applies Rotary Position Embedding to the input tensor `x`.

#     Args:
#         x (`torch.Tensor`): The input tensor (query or key). Shape [batch, num_heads, seq_len, head_dim].
#         cos (`torch.Tensor`): The cosine part of the rotary embedding. Shape [batch, 1, seq_len, head_dim].
#         sin (`torch.Tensor`): The sine part of the rotary embedding. Shape [batch, 1, seq_len, head_dim].

#     Returns:
#         `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
#     """
#     # cos = cos.unsqueeze(1) # Handled in RoPE forward
#     # sin = sin.unsqueeze(1) # Handled in RoPE forward
#     return (x * cos) + (rotate_half(x) * sin)


class TimeMoeInputEmbedding(nn.Module):
    """
    Use a mlp layer to embedding the time-series.
    """

    def __init__(self, config: TimeMoeConfig):
        super().__init__()
        self.config = config
        self.input_size = config.input_size  # default 1
        self.hidden_size = config.hidden_size
        # Original implementation used separate gate and emb layers
        self.emb_layer = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.gate_layer = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # x shape: [batch, seq_len, input_size]
        emb = self.act_fn(self.gate_layer(x)) * self.emb_layer(x)
        return emb


class OptimizedTimeSeriesInputEmbedding(nn.Module):
    """
    Enhanced embedding layer with temporal feature fusion.
    (Based on user's previous code - keep if desired, though not in paper snippet)
    """
    def __init__(self, config: TimeMoeConfig):
        super().__init__()
        self.config = config
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size

        # Value embedding path
        self.value_emb = nn.Linear(config.input_size, config.hidden_size)
        self.value_gate = nn.Linear(config.input_size, config.hidden_size)
        self.value_act = ACT2FN[config.hidden_act]

        # Temporal embedding path (if time values are used)
        self.time_emb = nn.Sequential(
            nn.Linear(1, config.hidden_size // 2), # Embedding time delta or value
            nn.SiLU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size)
        )

        # Adaptive fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size), # Takes concatenated value and time embeddings
            nn.Sigmoid()
        )

        # Threshold for regularity check
        self.time_variance_threshold = nn.Parameter(torch.tensor(0.1)) # Learnable threshold
        self.register_buffer('epsilon', torch.tensor(1e-6))

        # Layer normalization
        self.norm = TimeMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def is_regular_timing(self, time_values: torch.Tensor) -> torch.Tensor:
        """
        Check if time intervals are regular based on variance.
        Input shape: [batch_size, seq_len] or [batch_size, seq_len, 1]
        Returns: [batch_size, 1, 1] Boolean tensor (True if regular)
        """
        if time_values.dim() == 3:
            time_values = time_values.squeeze(-1) # [batch_size, seq_len]
        if time_values.shape[1] < 2: # Cannot compute variance for sequence length < 2
            return torch.ones((time_values.shape[0], 1, 1), dtype=torch.bool, device=time_values.device)

        deltas = time_values[:, 1:] - time_values[:, :-1] # [batch_size, seq_len - 1]
        # Use unbiased variance estimator? `unbiased=True`? Default is False.
        variances = torch.var(deltas, dim=1, keepdim=True, unbiased=False) # [batch_size, 1]

        # Use softplus on threshold to ensure it's positive
        threshold = F.softplus(self.time_variance_threshold) + self.epsilon
        is_regular = variances < threshold
        return is_regular.unsqueeze(-1) # Shape: [batch_size, 1, 1]

    def forward(self, x: torch.Tensor, time_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x shape: [batch, seq_len, input_size]
        # time_values shape: [batch, seq_len] or [batch, seq_len, 1]

        # Value processing
        value_gated = self.value_act(self.value_gate(x))
        value_emb = value_gated * self.value_emb(x) # [batch, seq_len, hidden_size]

        # If no time values provided, return only value embedding
        if time_values is None:
            return self.norm(value_emb)

        if time_values.dim() == 2:
            time_values = time_values.unsqueeze(-1) # Ensure shape [batch, seq_len, 1]

        # Check regularity per batch item
        is_regular = self.is_regular_timing(time_values) # [batch, 1, 1]

        # Compute time embedding only for irregular sequences if needed
        # Use relative time differences (deltas) for time embedding? Or absolute time?
        # Using absolute time for embedding here:
        time_emb = self.time_emb(time_values) # [batch, seq_len, hidden_size]

        # Fusion logic:
        # Combine based on regularity. If regular, use value_emb. If irregular, fuse.
        gate_input = torch.cat([value_emb, time_emb], dim=-1) # [batch, seq_len, hidden_size * 2]
        gate = 0.5 + 0.5 * self.fusion_gate(gate_input) # Scale sigmoid output to [0.5, 1.0]

        # Fuse based on gate for irregular sequences
        fused_emb_irregular = gate * value_emb + (1 - gate) * time_emb

        # Select final embedding based on regularity mask
        # is_regular shape is [B, 1, 1], needs expansion
        final_emb = torch.where(is_regular.expand_as(value_emb), value_emb, fused_emb_irregular)

        return self.norm(final_emb)

class ContinuousTimeRotaryEmbedding(nn.Module):
    """Continuous-Time Rotary Positional Encoding (CT-RoPE)"""
    def __init__(self, dim, base=10000.0, device=None):
        super().__init__()
        self.dim = dim; self.base = float(base)

        # Inverse frequency: lambda^(-2m/d) term = (base^(-2/d))^m
        # Original RoPE inv_freq: 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False) # Shape: [dim / 2]

        # Learnable frequency modulation parameter w_m, initialized to 1
        self.w = nn.Parameter(torch.ones(self.dim // 2, device=device)) # Shape: [dim / 2]
        self._cos_cached = None
        self._sin_cached = None
        self._last_time_values_hash = None

    def _compute_cos_sin(self, time_values: torch.Tensor, device: torch.device, dtype: torch.dtype):
        """
        Compute cosine and sine embeddings based on absolute time values.

        Args:
            time_values (`torch.Tensor`): Absolute timestamps. Shape [batch, seq_len].
            device (`torch.device`): Device for computation.
            dtype (`torch.dtype`): Data type for computation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Cosine and Sine Tensors.
                                            Shape [batch, 1, seq_len, dim]
        """
        current_hash = (time_values.sum().item(), tuple(time_values.shape), device, dtype) # More robust hash
        if current_hash == self._last_time_values_hash and self._cos_cached is not None:
             return self._cos_cached, self._sin_cached
        freqs = torch.einsum('b s, d -> b s d', time_values.to(device=device, dtype=torch.float32), self.inv_freq)
        modulated_freqs = freqs * self.w.to(device=device, dtype=torch.float32)
        emb = torch.cat((modulated_freqs, modulated_freqs), dim=-1)
        emb = emb.unsqueeze(1) # Add head dim for broadcasting: [B, 1, L, Dh]
        cos = emb.cos().to(dtype); sin = emb.sin().to(dtype)
        self._cos_cached = cos
        self._sin_cached = sin
        self._last_time_values_hash = current_hash
        return cos, sin

    def forward(self, q: torch.Tensor, k: torch.Tensor, time_values: torch.Tensor):
        """
        Apply CT-RoPE to Query and Key tensors.

        Args:
            q (`torch.Tensor`): Query tensor. Shape [batch, num_heads, seq_len, head_dim].
            k (`torch.Tensor`): Key tensor. Shape [batch, num_heads, seq_len, head_dim].
            time_values (`torch.Tensor`): Absolute timestamps for the sequence. Shape [batch, seq_len].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Rotated q and k tensors.
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        if time_values.shape[0] != batch_size or time_values.shape[1] != seq_len:
            raise ValueError(f"CT-RoPE input time_values shape mismatch: Expected ({batch_size}, {seq_len}), got {time_values.shape}")
        assert head_dim == self.dim, f"Head dimension {head_dim} != RoPE dimension {self.dim}"

        cos, sin = self._compute_cos_sin(time_values, q.device, q.dtype) # Shape [B, 1, L, Dh]

        # ===apply_rotary_pos_emb ===
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        # ==========================================================
        return q_embed, k_embed

# Fallback standard RoPE (if needed)
class TimeMoeRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # Return cos/sin cache slices suitable for apply_rotary_pos_emb
        # Shape: [1, 1, seq_len, dim] for broadcasting
        cos = self.cos_cached[:seq_len].to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        return cos, sin


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->TimeMOE
class TimeMoeRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# FeedForward block used by Experts
class TimeMoeTemporalBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


# Standard MLP (used if use_dense=True)
class TimeMoeMLP(TimeMoeTemporalBlock):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__(hidden_size, intermediate_size, hidden_act)

    def forward(self, hidden_state):
        # Return None for router_logits compatibility
        return super().forward(hidden_state), None


# Sparse MoE Layer
class TimeMoeSparseExpertsLayer(nn.Module):
    def __init__(self, config: TimeMoeConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob # Add this config flag if desired

        # Adjust intermediate size for experts if scaling by top_k
        # moe_intermediate_size = self.config.intermediate_size // self.config.num_experts # Or maybe // top_k ? check MoE papers
        # Let's assume intermediate_size in config is PER EXPERT for simplicity, or shared_expert size
        expert_intermediate_size = self.config.moe_intermediate_size if hasattr(config, 'moe_intermediate_size') else self.config.intermediate_size

        # gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [TimeMoeTemporalBlock(
                hidden_size=self.config.hidden_size,
                intermediate_size=expert_intermediate_size,
                hidden_act=self.config.hidden_act,
            ) for _ in range(self.num_experts)]
        )

        # Shared expert (using standard intermediate size)
        self.shared_expert = TimeMoeTemporalBlock(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            hidden_act=self.config.hidden_act,
        )
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim) # [batch * sequence_length, hidden_dim]

        # router_logits -> (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states_flat)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.norm_topk_prob:
             routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6) # Add clamp for stability

        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype) # Shape: [tokens, top_k]

        final_hidden_states = torch.zeros_like(hidden_states_flat)

        # One hot encode the selected experts to create an expert mask
        # expert_mask -> [tokens, top_k, num_experts]
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            # Find which tokens routed to this expert (in any of the top_k positions)
            # idx: index within top_k (0, 1, ...), top_x: token index in flattened batch
            token_indices, top_k_indices = torch.where(expert_mask[..., expert_idx])

            if token_indices.numel() > 0:
                # Get the corresponding routing weights for these tokens/expert pairs
                current_routing_weights = routing_weights[token_indices, top_k_indices, None] # Shape: [num_tokens_for_expert, 1]

                # Index the correct hidden states
                current_state = hidden_states_flat[token_indices] # Shape: [num_tokens_for_expert, hidden_dim]

                # Compute expert output and scale by routing weight
                current_hidden_states = expert_layer(current_state) * current_routing_weights

                # Add to final hidden states (scatter add)
                final_hidden_states.index_add_(0, token_indices, current_hidden_states.to(hidden_states.dtype))


        # Shared expert computation
        shared_expert_output = self.shared_expert(hidden_states_flat)
        shared_gate_weights = F.sigmoid(self.shared_expert_gate(hidden_states_flat)) # Shape: [tokens, 1]
        shared_expert_output = shared_gate_weights * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


def compute_delta_t(time_values):
    """ Computes pairwise time differences: delta_t[i, j] = time[i] - time[j] """
    # time_values shape: [batch, seq_len]
    if time_values is None:
        return None
    delta = time_values.unsqueeze(2) - time_values.unsqueeze(1) # [batch, seq_len, seq_len]
    return delta

# Attention Layer with Time Decay and Gating
class TimeMoeAttention(nn.Module):
    """
    Multi-headed attention modified with optional time decay.
    """
    def __init__(self, config: TimeMoeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "lead to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self._time_aware_rotary_flag = getattr(config, "time_aware_rotary", False) # Read flag BEFORE use

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta # Base for RoPE
        self.is_causal = True # Assuming causal attention
        self.attention_dropout = config.attention_dropout

        # Time decay parameters (optional)
        self.time_decay = getattr(config, "time_decay", False) # Default to False if not in config
        self.use_gated_time_decay = getattr(config, "use_gated_time_decay", False)
        self.use_nonlinear_time_decay = getattr(config, "use_nonlinear_time_decay", False)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)


        if self._time_aware_rotary_flag:
            if 'ContinuousTimeRotaryEmbedding' in globals():
                 logger.info(f"Layer {layer_idx}: Using ContinuousTimeRotaryEmbedding.")
                 self.rotary_emb = ContinuousTimeRotaryEmbedding(
                     self.head_dim,
                     base=self.rope_theta
                 )
            else:
                 logger.error(f"Layer {layer_idx}: time_aware_rotary=True but ContinuousTimeRotaryEmbedding class not found! Falling back to standard RoPE.")
                 self.rotary_emb = TimeMoeRotaryEmbedding(
                     self.head_dim,
                     max_position_embeddings=config.max_position_embeddings,
                     base=config.rope_theta
                 )
                 self._time_aware_rotary_flag = False 
        else:
            logger.info(f"Layer {layer_idx}: Using standard TimeMoeRotaryEmbedding.")
            self.rotary_emb = TimeMoeRotaryEmbedding(
                 self.head_dim,
                 max_position_embeddings=config.max_position_embeddings,
                 base=config.rope_theta
             )


        # if self.time_aware_rotary:
        #     if 'ContinuousTimeRotaryEmbedding' in globals():
        #          logger.info(f"Layer {layer_idx}: Using ContinuousTimeRotaryEmbedding.")
        #          self.rotary_emb = ContinuousTimeRotaryEmbedding(
        #              self.head_dim,
        #              base=self.rope_theta
        #          )
        #     else:
        #          logger.error(f"Layer {layer_idx}: time_aware_rotary=True but ContinuousTimeRotaryEmbedding class not found! Falling back to standard RoPE.")
        #          self.rotary_emb = TimeMoeRotaryEmbedding(
        #              self.head_dim,
        #              max_position_embeddings=config.max_position_embeddings,
        #              base=config.rope_theta
        #          )
        #          self.time_aware_rotary = False 
        # else:
        #      logger.info(f"Layer {layer_idx}: Using standard TimeMoeRotaryEmbedding.")
        #      self.rotary_emb = TimeMoeRotaryEmbedding(
        #          self.head_dim,
        #          max_position_embeddings=config.max_position_embeddings,
        #          base=config.rope_theta 
        #         )


        # Temporal difference encoding layer (Not explicitly used in current logic, maybe for future use?)
        # self.time_proj = nn.Linear(1, config.hidden_size) if config.time_aware else None

        # Parameters for Time Decay
        if self.time_decay:
            # Learnable scale per head for decay: lambda_h
            self.lambda_decay = nn.Parameter(
                 torch.rand(self.num_heads) * 0.1 + 1e-6
            )
            if self.use_gated_time_decay:
                 # MLP for gating: Linear(head_dim) -> SiLU -> Linear(1) -> Sigmoid
                 # Projects query head_dim to scalar gate gamma_h
                 self.gate_mlp = nn.Sequential(
                     nn.Linear(self.head_dim, self.head_dim // 2, bias=False),
                     nn.SiLU(),
                     nn.Linear(self.head_dim // 2, 1, bias=False),
                     nn.Sigmoid()
                 )
            if self.use_nonlinear_time_decay:
                 # MLP for non-linear decay: f_theta
                 # Input is scaled delta_t, output is value before final sigmoid sigma
                 self.nonlinear_decay_mlp = nn.Sequential(
                     nn.Linear(1, 16, bias=False),
                     nn.SiLU(),
                     nn.Linear(16, 1, bias=False)
                 )
            # Define the monotonic decay function sigma (e.g., sigmoid or softplus based)
            # Using sigmoid for [0, 1] range as per paper Eq(6)
            self.decay_activation = nn.Sigmoid()

    def forward(
            self,
            hidden_states: torch.Tensor,
            # time_deltas: Optional[torch.Tensor] = None, # Use time_values instead
            time_values: Optional[torch.Tensor] = None, # Absolute time values: [batch, seq_len]
            attention_mask: Optional[torch.Tensor] = None, # Shape [batch, 1, q_len, kv_len]
            position_ids: Optional[torch.LongTensor] = None, # Needed for standard RoPE
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: Optional[bool] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, L, Dh]
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # [B, Hkv, L, Dh]
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # [B, Hkv, L, Dh]

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            # kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            # Simpler way for basic Cache object:
            kv_seq_len = past_key_value.get_seq_length(self.layer_idx)

        # Apply Rotary Positional Embedding
        if self._time_aware_rotary_flag and isinstance(self.rotary_emb, ContinuousTimeRotaryEmbedding):
            if time_values is None: 
                raise ValueError("`time_values` needed for CT-RoPE.")
            
            current_time_values = time_values

            # if past_key_value is not None and current_time_values.shape[1] != kv_seq_len:
            #      raise ValueError(f"CT-RoPE time_values length mismatch. Expected {kv_seq_len}, got {current_time_values.shape[1]}.")
            
            # Pass only current q/k and corresponding time_values to CT RoPE forward
            time_values_for_rope = current_time_values[:, -q_len:] if past_key_value else current_time_values
            query_states, key_states = self.rotary_emb(query_states, key_states, time_values_for_rope)

        elif isinstance(self.rotary_emb, TimeMoeRotaryEmbedding): # Standard RoPE
            if position_ids is None: 
                raise ValueError("`position_ids` needed for standard RoPE.")
            
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
  
            # Ensure position_ids corresponds to current query tokens if caching
            current_position_ids = position_ids[:, -q_len:] if past_key_value else position_ids
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, current_position_ids)
            # ========================================
        else: 
            raise TypeError(f"Unexpected rotary embedding type: {type(self.rotary_emb)}")


        # KV Cache Update
        if past_key_value is not None:
            # Key and Value states at this point are *only the new ones*
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        # GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Matmul
        # print("Query States Shape:", query_states.shape)
        # print("Key States Shape (before transpose):", key_states.shape)
        # # Check if head_dim might be zero
        # print("Head Dim:", self.head_dim)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        # attn_weights shape: [B, H, L, Lkv]


        # --- Apply Time Decay (if enabled) ---
        gated_decay = torch.ones_like(attn_weights)
        if self.time_decay and time_values is not None:
            # Ensure time_values covers the full kv_seq_len for decay calculation
            if time_values.shape[1] != kv_seq_len:
                 warnings.warn(f"Time values length ({time_values.shape[1]}) doesn't match KV sequence length ({kv_seq_len}) for time decay. Skipping decay.")
            else:
                 delta_t = compute_delta_t(time_values) # [B, L_q, L_kv] - Needs slicing if L_q != L_kv
                 # Slice delta_t for current queries vs all keys
                 if q_len != kv_seq_len:
                      delta_t = delta_t[:, -q_len:, :] # Get rows for current queries

                 abs_delta_t = delta_t.abs().unsqueeze(1) # [B, 1, L_q, L_kv]
                 lambda_h = F.softplus(self.lambda_decay).view(1, -1, 1, 1)

                 phi_decay = None
                 if self.use_nonlinear_time_decay:
                    # Ensure calculations match [B, H, L_q, L_kv] shape
                    scaled_delta_t = lambda_h * abs_delta_t # Broadcasts lambda_h
                    bsz_h, num_h, q_len_h, kv_len_h = scaled_delta_t.shape
                    reshaped_scaled_delta_t = scaled_delta_t.permute(0, 2, 3, 1).reshape(-1, 1)
                    mlp_input_dtype = self.nonlinear_decay_mlp[0].weight.dtype
                    decay_mlp_output = self.nonlinear_decay_mlp(reshaped_scaled_delta_t.to(mlp_input_dtype))
                    decay_mlp_output = decay_mlp_output.view(bsz_h, q_len_h, kv_len_h, num_h).permute(0, 3, 1, 2)
                    phi_decay = self.decay_activation(decay_mlp_output) # Shape [B, H, L_q, L_kv]
                 else:
                    phi_decay = torch.sigmoid(-lambda_h * abs_delta_t) # Shape [B, H, L_q, L_kv]

                 if self.use_gated_time_decay:
                     gate_mlp_input_dtype = self.gate_mlp[0].weight.dtype
                     # Apply gate only to current query states
                     gamma_h = self.gate_mlp(query_states.to(gate_mlp_input_dtype)) # [B, H, L_q, 1]
                     gated_decay = gamma_h * phi_decay + (1.0 - gamma_h) # Shape [B, H, L_q, L_kv]
                 else:
                     gated_decay = phi_decay

                 # Apply decay factor to attention weights
                 # Need to handle potential broadcasting if gated_decay doesn't have head dim
                 # Ensure gated_decay is [B, H, L, Lkv]
                 attn_weights = attn_weights * gated_decay.to(attn_weights.dtype)


        # --- Apply Attention Mask ---
        # if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        #     raise ValueError(
        #         f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
        #         f" {attn_weights.size()}"
        #     )

        if attention_mask is not None:
            # expected_mask_shape = (bsz, 1, q_len, kv_seq_len)
            # if attention_mask.size() != expected_mask_shape:
            #     # Adapt mask shape if possible (e.g., from [B, Lkv])
            #     if attention_mask.dim() == 2 and attention_mask.shape == (bsz, kv_seq_len):
            #         warnings.warn("Expanding 2D attention mask to 4D.")
            #         attention_mask = attention_mask[:, None, None, :].expand(bsz, 1, q_len, kv_seq_len)
            #     else:
            #         raise ValueError(f"Attention mask shape mismatch: {attention_mask.size()} vs {expected_mask_shape}")

            # Apply mask (additive, large negative value for masked positions)
            attn_weights = attn_weights + attention_mask # Broadcasting [B,1,L,Lkv] to [B,H,L,Lkv]


        # --- Softmax and Dropout ---
        # upcast attention to fp32 for stability
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)


        # --- Compute Attention Output ---
        attn_output = torch.matmul(attn_weights, value_states) # [B, H, L, Dh]

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # Reshape and Project Output
        attn_output = attn_output.transpose(1, 2).contiguous() # [B, L, H, Dh]
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size) # [B, L, D]

        attn_output = self.o_proj(attn_output) # [B, L, D]

        attn_weights_out = attn_weights if output_attentions else None
        # Return Cache object if use_cache is True
        present_key_value = past_key_value if use_cache else None

        return attn_output, attn_weights_out, present_key_value


class TimeMoeFlashAttention2(TimeMoeAttention):
    """ Flash Attention 2 implementation - Adapts TimeMoeAttention """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Check if Flash Attention is actually available
        if not _flash_attn_available():
             raise ImportError("Flash Attention 2 is not installed or available. Use `attn_implementation='eager'`.")
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()


    def forward(
            self,
            hidden_states: torch.Tensor,
            time_values: Optional[torch.Tensor] = None, # Absolute time values: [batch, seq_len]
            attention_mask: Optional[torch.LongTensor] = None, # Needs careful handling for FA unpad/pad
            position_ids: Optional[torch.LongTensor] = None, # Needed for standard RoPE
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None, # Specific to static cache in FA
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        # Flash Attention V2 does not support outputting attention weights
        output_attentions = False
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "in standard Hugging Face setups. Use `sdpa` or `eager`."
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for RoPE/Processing: [B, H, L, Dh]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            # Note: FA2 typically requires recomputing RoPE for the full sequence if using cache,
            # unless specific optimizations are applied.
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # --- Apply RoPE ---
        if isinstance(self.rotary_emb, ContinuousTimeRotaryEmbedding):
            if time_values is None: raise ValueError("`time_values` needed for CT-RoPE.")
            # Assume time_values passed covers the full sequence length (cache + current)
            # prepare_inputs_for_generation should handle this concatenation
            if past_key_value is not None and time_values.shape[1] != kv_seq_len:
                 raise ValueError(f"CT-RoPE time_values length mismatch. Expected {kv_seq_len}, got {time_values.shape[1]}. Check prepare_inputs_for_generation.")
            # Apply RoPE based on the potentially full time_values
            # Note: The RoPE forward itself might need adjustment if q_len != kv_seq_len
            # But typically Q and K are rotated based on their respective absolute times/positions
            query_states, key_states = self.rotary_emb(query_states, key_states, time_values)

        elif isinstance(self.rotary_emb, TimeMoeRotaryEmbedding): # Standard RoPE
            if position_ids is None: raise ValueError("`position_ids` needed for standard RoPE.")
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            # Ensure position_ids corresponds to current query tokens if caching
            current_position_ids = position_ids[:, -q_len:] if past_key_value else position_ids
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, current_position_ids)
        

        else: 
            raise TypeError(f"Unexpected rotary embedding type: {type(self.rotary_emb)}")
        
        # --- KV Cache Update ---
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        # --- Prepare inputs for Flash Attention ---
        # Transpose QKV to Flash Attention format: [B, L, H, Dh]
        # query_states = query_states.transpose(1, 2)
        # key_states = key_states.transpose(1, 2)
        # value_states = value_states.transpose(1, 2)

        # GQA: Repeat K/V heads AFTER transpose for FA format
        # FA expects [B, L, Hkv, Dh] for K/V before potential GQA/MQA expansion inside FA function
        # We need to expand K/V to [B, L, H, Dh] BEFORE passing to flash_attn_func if using GQA.
        # However, flash_attn_func can often handle MHA/GQA/MQA internally if shapes are correct.
        # Let's pass [B, L, H, Dh] for Q and [B, L, Hkv, Dh] for K/V.
        # If flash_attn library requires explicit repeating, do it here.
        # Assuming flash_attn_func handles GQA based on num_heads/num_kv_heads mismatch.
        # key_states_fa = repeat_kv(key_states.transpose(1,2), self.num_key_value_groups).transpose(1,2) # If needed
        # value_states_fa = repeat_kv(value_states.transpose(1,2), self.num_key_value_groups).transpose(1,2) # If needed
        
         # GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Matmul
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)


        dropout_rate = self.attention_dropout if self.training else 0.0

        # --- Handle Time Decay for Flash Attention (Not Natively Supported) ---
        # Time decay (gated or not) modifies the attention scores *before* softmax.
        # Flash Attention computes the full attention (including softmax) internally.
        # Applying time decay with Flash Attention usually requires modifying the FA kernel itself
        # or accepting that the decay cannot be perfectly replicated.
        if self.time_decay:
             warnings.warn("Time decay is not natively supported by Flash Attention 2. "
                           "The decay effect will be ignored when using attn_implementation='flash_attention_2'. "
                           "Use 'eager' for time decay.")



        # Cast to appropriate dtype if needed
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            target_dtype = torch.float16 # Default target for FA
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            logger.warning_once(f"Flash Attention inputs cast to {target_dtype}.")
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)


        # --- Call Flash Attention ---
        # `causal` argument determines masking inside FA
        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        # --- Reshape and Project Output ---
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        # Cast back to original dtype if necessary
        if input_dtype == torch.float32:
             attn_output = attn_output.to(input_dtype)

        attn_weights = None # FA doesn't return weights

        return attn_output, attn_weights, past_key_value


    def _flash_attention_forward(
            self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention. Handles unpadding/padding if attention_mask indicates padding.
        """
        # Check for padding tokens in attention_mask if it's a boolean mask
        # HuggingFace attention_mask is additive (-inf/0), not boolean.
        # Standard FA implementations often take sequence lengths (cu_seqlens) for padding.
        # Or they take a boolean mask. Additive mask needs conversion or separate handling.

        # For simplicity, assume no padding or handle it outside _flash_attention_forward if possible.
        # If attention_mask is the standard HF causal mask (additive), FA causal=True handles it.
        # If attention_mask includes padding (e.g., from tokenizer), FA needs sequence lengths.

        use_padding = False
        if attention_mask is not None:
             # Check if the mask indicates padding (values other than 0 or -inf/large negative)
             # This check is heuristic. A better way is to know the mask's origin.
             if not torch.all((attention_mask == 0) | (attention_mask < -1e3)):
                  warnings.warn("Flash Attention V2 received a non-standard additive mask. Padding might not be handled correctly. Trying unpad/pad logic.")
                  # Requires attention_mask to be [B, Lkv] boolean/int mask for unpad/pad
                  # The passed attention_mask is likely [B, 1, L, Lkv] additive. Conversion needed.
                  # Simplification: Ignore padding handling with FA for now.
                  # use_padding = True # Enable if unpad/pad is implemented below
                  pass


        if use_padding:
             # This requires flash_attn.bert_padding utilities and a compatible mask
             raise NotImplementedError("Padding handling with Flash Attention V2 and additive mask is complex and not fully implemented here.")
             # (query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens) = \
             #     self._upad_input(query_states, key_states, value_states, compatible_boolean_mask, query_length)
             # attn_output_unpad = flash_attn_varlen_func(...)
             # attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)

        else:
            # --- Standard Flash Attention call without padding ---
            causal = self.is_causal # Use causal flag

            # Note: RoCm flash attention might have issues with q_len=1 causal mask before v2.1
            if self._flash_attn_uses_top_left_mask and query_length == 1:
                 causal = False # Workaround for older FA versions

            attn_output = flash_attn_func(
                query_states, # [B, L, H, Dh]
                key_states,   # [B, Lkv, Hkv, Dh]
                value_states, # [B, Lkv, Hkv, Dh]
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
                # window_size=(-1,-1), # For full attention
            )

        return attn_output

    # _upad_input helper (if padding is implemented) - requires boolean mask
    # def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
    #     # ... (implementation from original HF code) ... requires boolean mask


# Mapping for attention implementations
TIME_MOE_ATTENTION_CLASSES = {
    "eager": TimeMoeAttention,
    'flash_attention_2': TimeMoeFlashAttention2 if _flash_attn_available else TimeMoeAttention,

}


class TimeMoeDecoderLayer(nn.Module):
    def __init__(self, config: TimeMoeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx # Store layer index

        # Determine Attention Implementation
        attn_implementation = config._attn_implementation
        if attn_implementation == "flash_attention_2" and not _flash_attn_available():
            logger.warning(
                "Flash Attention 2 requested but not available. Falling back to eager attention."
            )
            attn_implementation = "eager"
        if attn_implementation == "flash_attention_2" and config.time_decay:
             logger.warning(
                 "Time decay is not supported with Flash Attention 2. Falling back to eager attention for time decay compatibility."
             )
             attn_implementation = "eager"


        self.self_attn = TIME_MOE_ATTENTION_CLASSES[attn_implementation](config, layer_idx)

        # MoE or Dense MLP layer
        self.use_dense = getattr(config, 'use_dense', False) # Check if using dense MLP instead of MoE
        if self.use_dense:
            self.ffn_layer = TimeMoeMLP(
                hidden_size=self.config.hidden_size,
                intermediate_size=self.config.intermediate_size,
                hidden_act=self.config.hidden_act,
            )
        else:
            self.ffn_layer = TimeMoeSparseExpertsLayer(config)

        # Layer Norms
        self.input_layernorm = TimeMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = TimeMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            time_values: Optional[torch.Tensor] = None,    # Pass time values down
            attention_mask: Optional[torch.Tensor] = None, # Pass attention mask down
            position_ids: Optional[torch.LongTensor] = None, # Pass position ids down
            past_key_value: Optional[Tuple[torch.Tensor]] = None, # Pass cache down
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Tuple[torch.Tensor]], Optional[torch.Tensor]]: # Added router_logits to output tuple type hint
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )
        """ Forward pass for the decoder layer. """

        residual = hidden_states

        hidden_states_norm = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states=hidden_states_norm,
            time_values=time_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs # Pass other potential args like cache_position
        )
        attn_output = attn_outputs[0]         # The attention output tensor
        attn_weights = attn_outputs[1]      # Optional attention weights
        present_key_value = attn_outputs[2] # Optional updated KV cache

        hidden_states = residual + attn_output # Apply residual connection

        # Fully Connected (MoE or MLP)
        residual = hidden_states
        hidden_states_norm = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.ffn_layer(hidden_states_norm) # Get router logits as well
        hidden_states = residual + hidden_states # Apply residual connection

        outputs = (hidden_states, attn_weights, present_key_value, router_logits)

        return outputs


class TimeMoePreTrainedModel(PreTrainedModel):
    config_class = TimeMoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TimeMoeDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = False # Explicitly state SDPA support if applicable
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        # elif isinstance(module, torch.nn.Embedding): # No standard token embeddings here
        #     module.weight.data.normal_(mean=0.0, std=std)
        #     if module.padding_idx is not None:
        #         module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, TimeMoeRMSNorm):
             # Typically initialized to ones, handled in RMSNorm init
             pass
        elif isinstance(module, ContinuousTimeRotaryEmbedding):
             # Initialize w close to 1 (already done in init)
             pass


class TimeMoeModel(TimeMoePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`TimeMoeDecoderLayer`]

    Args:
        config: TimeMoeConfig
    """

    def __init__(self, config: TimeMoeConfig):
        super().__init__(config)
        self.padding_idx = -1 # Define padding_idx if needed elsewhere, e.g., for embeddings
        self.vocab_size = config.input_size # Treat input dimension as vocab size for compatibility

        # Input Embedding Layer
        self.time_aware_embedding = getattr(config, "time_aware_embedding", False) # Check config
        if self.time_aware_embedding:
            logger.info("Using OptimizedTimeSeriesInputEmbedding.")
            self.embed_layer = OptimizedTimeSeriesInputEmbedding(config)
        else:
            logger.info("Using standard TimeMoeInputEmbedding.")
            self.embed_layer = TimeMoeInputEmbedding(config)

        # Temporal Encoder (Not used in current main path, from user's original code)
        # self.temporal_encoder = nn.LSTM(...)

        # Decoder Layers
        self.layers = nn.ModuleList(
            [TimeMoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = TimeMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps) # Final LayerNorm

        self.gradient_checkpointing = False # Set dynamically via .gradient_checkpointing_enable()

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        # Return the module that computes the initial embeddings
        return self.embed_layer

    def set_input_embeddings(self, value):
        # Set the module for initial embeddings (e.g., for tying weights)
        self.embed_layer = value

    def forward(
            self,
            input_ids: Optional[torch.FloatTensor] = None, # Input time series values [B, L, Din]
            time_values: Optional[torch.FloatTensor] = None, # Absolute time values [B, L]
            attention_mask: Optional[torch.Tensor] = None,   # Padding mask [B, L]
            position_ids: Optional[torch.LongTensor] = None, # Position indices [B, L] (for standard RoPE)
            past_key_values: Optional[Cache] = None, # KV cache
            inputs_embeds: Optional[torch.FloatTensor] = None, # Option to provide embeddings directly
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # --- Input Handling ---
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # Ensure input_ids has feature dimension if it's missing
            if input_ids.dim() == 2: # Assume shape [B, L] -> [B, L, 1]
                input_ids = input_ids.unsqueeze(-1)
            batch_size, seq_length, input_dim = input_ids.shape
            device = input_ids.device
        elif inputs_embeds is not None:
            batch_size, seq_length, hidden_dim = inputs_embeds.shape
            device = inputs_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # --- Cache Handling ---
        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        # --- Prepare position_ids (for standard RoPE) ---
        if position_ids is None:
             # Standard sequential position ids
             position_ids = torch.arange(
                 past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
             )
             position_ids = position_ids.unsqueeze(0) # Shape [1, seq_length]
             # No need to expand to batch size, broadcasting usually handles this.
             # position_ids = position_ids.expand(batch_size, seq_length) # If needed by RoPE implementation
        # else:
             # position_ids = position_ids.view(-1, seq_length).long() # Ensure correct shape/type

        # --- Get Input Embeddings ---
        if inputs_embeds is None:
             if self.time_aware_embedding:
                  inputs_embeds = self.embed_layer(input_ids, time_values=time_values)
             else:
                  inputs_embeds = self.embed_layer(input_ids)

        # --- Prepare Attention Mask ---
        # Input attention_mask shape: [batch_size, seq_length] (pad=0, non-pad=1)
        # Output attention_mask shape: [batch_size, 1, q_len, kv_len] (additive causal mask)
        if self._attn_implementation == "flash_attention_2":
             # Flash Attention V2 requires special mask handling (or causal flag)
             # Let's rely on the causal flag in FA forward and pass None or minimal mask if no padding
             # If there IS padding, FA needs cu_seqlens or boolean mask - complex to derive here
             if attention_mask is not None and torch.any(attention_mask == 0):
                 # If padding exists, FA2 needs careful handling not fully implemented here
                 warnings.warn("Passing attention mask with padding to Flash Attention 2 requires specific handling (cu_seqlens or boolean mask). Causal flag will be used, padding ignored.")
                 # Set mask to None to rely solely on causal flag if padding exists
                 attention_mask = None # Or derive boolean mask / cu_seqlens if possible
             else:
                 # No padding, only causal mask needed, handled by FA's causal=True
                 attention_mask = None
        else: # Eager attention
             # Prepare the standard 4D additive causal mask
             attention_mask = _prepare_4d_causal_attention_mask(
                 attention_mask, # Input mask [B, L]
                 (batch_size, seq_length),
                 inputs_embeds,
                 past_key_values_length,
                 sliding_window=None, # Add if using sliding window attention
             )

        hidden_states = inputs_embeds

        # --- Gradient Checkpointing ---
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # --- Decoder Layers ---
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if not self.config.use_dense else None # Collect router logits if using MoE
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = None
            if self.gradient_checkpointing and self.training:
                # Define function for checkpointing
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        # inputs order: hidden_states, time_values, attention_mask, position_ids, None, output_attentions, use_cache
                        # Match the non-kwargs signature of TimeMoeDecoderLayer.forward carefully
                        # The signature is: (self, hidden_states, time_values=None, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs)
                        # We need to pass args matching the signature up to the first kwarg-only arg or **kwargs
                        return module(*inputs[:4], past_key_value=None, output_attentions=inputs[5], use_cache=inputs[6])

                    return custom_forward

                # Prepare inputs for checkpointing
                checkpoint_inputs = (
                    hidden_states,
                    time_values,
                    attention_mask,
                    position_ids,
                    None, # past_key_value (not supported with checkpointing)
                    output_attentions,
                    use_cache # Should be False here
                )
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    *checkpoint_inputs,
                    use_reentrant=not self.config.gradient_checkpointing_kwargs.get("use_reentrant", True) # from HF checkpointing logic
                )

            else:
                # Standard forward pass
                layer_outputs = decoder_layer(
                    hidden_states,
                    time_values=time_values,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            # Unpack outputs
            hidden_states = layer_outputs[0]
            attn_weights = layer_outputs[1]
            present_key_value = layer_outputs[2]
            router_logits = layer_outputs[3] # May be None if dense

            if use_cache:
                # next_decoder_cache collects the cache from the *last* layer
                next_decoder_cache = present_key_value

            if output_attentions:
                all_self_attns += (attn_weights,)

            if all_router_logits is not None and router_logits is not None:
                all_router_logits += (router_logits,)


        hidden_states = self.norm(hidden_states) # Apply final layer norm

        # Add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # --- Prepare Output ---
        next_cache = None
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache) # Keep this to handle input
            if use_legacy_cache and past_key_values is not None: # Convert legacy input if needed
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            
            next_cache = next_decoder_cache

        if not return_dict:
            # ensure next_cache is in the tuple if not None
            outputs = [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if all_router_logits is not None:
                    outputs.append(all_router_logits)
            return tuple(v for v in outputs if v is not None)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits # Will be None if dense=True
        )

class TimeMoeOutputLayer(nn.Module):
    """ Output layer projecting hidden state to time series predictions. """
    def __init__(self, hidden_size: int, horizon_length: int, input_size: int = 1):
        super().__init__()
        self.output_size = input_size * horizon_length
        self.out_layer = nn.Linear(
            hidden_size,
            self.output_size,
            bias=False,
        )

    def forward(self, x):
        """
        Args:
            x (torch.FloatTensor): Hidden states shape [B, seq_len, hidden_size]
        Returns:
            torch.FloatTensor: Predictions shape [B, seq_len, input_size * horizon_length]
        """
        return self.out_layer(x)


class TimeMoeForPrediction(TimeMoePreTrainedModel, TSGenerationMixin):
    """ TimeMoE model with output heads for time series forecasting. """
    _supports_cache_class = True

    # Add _tied_weights_keys if needed, e.g., for tying input/output embeddings if applicable
    # _tied_weights_keys = ["lm_heads.out_layer.weight"] # Example if tying output head

    def __init__(self, config: TimeMoeConfig):
        super().__init__(config)
        self.config = config
        self.model = TimeMoeModel(config) # The main TimeMoE model

        # Output layers for different prediction horizons
        self.horizon_lengths = config.horizon_lengths
        self.input_size = config.input_size
        lm_head_list = []
        self.horizon_length_map = {}
        for i, horizon_length in enumerate(self.horizon_lengths):
            lm_head_list.append(
                TimeMoeOutputLayer(
                    hidden_size=self.config.hidden_size,
                    input_size=self.config.input_size,
                    horizon_length=horizon_length,
                )
            )
            self.horizon_length_map[horizon_length] = i
        self.lm_heads = nn.ModuleList(lm_head_list)

        # Loss related parameters
        # Check if aux loss should be applied (only if MoE is used)
        self.apply_aux_loss = config.apply_aux_loss and not getattr(config, 'use_dense', False)
        self.num_experts_per_tok = config.num_experts_per_tok
        self.router_aux_loss_factor = config.router_aux_loss_factor
        self.num_experts = config.num_experts

        # Regularization coefficients for CT-RoPE (read from config)
        self.rope_w_reg_alpha = getattr(config, "rope_w_reg_alpha", 0.0) # Default to 0 if not in config
        self.rope_smoothness_reg_beta = getattr(config, "rope_smoothness_reg_beta", 0.0) # Default to 0 if not in config

        # Loss function (Huber Loss)
        # Allow delta to be configured
        huber_delta = getattr(config, "huber_delta", 1.0) # Default delta = 1.0
        self.loss_function = torch.nn.HuberLoss(reduction='none', delta=huber_delta)

        # Initialize weights and apply final processing
        self.post_init()
    
    def _tie_weights(self): 
        pass

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        # Return the output layer(s). Return the first one as a proxy if needed for tying.
        # Note: Having multiple heads makes direct weight tying complex.
        return self.lm_heads[0].out_layer

    def set_output_embeddings(self, new_encoder):
        # Typically used for weight tying - careful with multiple heads.
        # This might require custom logic if tying weights across heads or to input embeddings.
        # Example: Set the first head's layer
        self.lm_heads[0].out_layer = new_encoder


    def set_decoder(self, decoder):
        # Set the main TimeMoeModel instance
        self.model = decoder

    def get_decoder(self):
        # Get the main TimeMoeModel instance
        return self.model


    def forward(
            self,
            input_ids: Optional[torch.FloatTensor] = None,    # Input time series values [B, L, Din]
            time_values: Optional[torch.FloatTensor] = None,  # Absolute time values [B, L]
            attention_mask: Optional[torch.Tensor] = None,    # Padding mask [B, L]
            position_ids: Optional[torch.LongTensor] = None,  # Position indices [B, L] (for standard RoPE)
            past_key_values: Optional[List[torch.FloatTensor]] = None, # KV cache
            inputs_embeds: Optional[torch.FloatTensor] = None,# Option to provide embeddings directly
            labels: Optional[torch.FloatTensor] = None,       # Target values [B, L, Din] or [B, L]
            loss_masks: Optional[torch.FloatTensor] = None,   # Mask for loss calculation [B, L, Din] or [B, L]
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            max_horizon_length: Optional[int] = None, # For inference: predict up to this horizon
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache


        # Pass inputs through the base TimeMoeModel
        outputs = self.model(
            input_ids=input_ids,
            time_values=time_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        predictions = None
        loss = None
        aux_loss = None # Placeholder for aux loss calculation

        # --- Loss Calculation (if labels are provided) ---
        if labels is not None:
            total_forecast_loss = 0.0
            num_active_heads = 0

            # Ensure labels and masks have feature dimension
            if labels.dim() == 2: labels = labels.unsqueeze(-1)
            if loss_masks is not None and loss_masks.dim() == 2: loss_masks = loss_masks.unsqueeze(-1)


            for i, horizon_length in enumerate(self.horizon_lengths):
                # Calculate predictions for this horizon
                lm_head = self.lm_heads[i]
                one_predictions = lm_head(hidden_states) # [B, L, Din * H]

                # Calculate autoregressive loss for this horizon
                one_loss = self.calc_ar_loss(one_predictions, labels, loss_masks, horizon_length)
                if one_loss is not None:
                     total_forecast_loss += one_loss
                     num_active_heads += 1

                # Store predictions from the first head (or a specific one if needed)
                if i == 0:
                    predictions = one_predictions

            # Average forecast loss across heads
            if num_active_heads > 0:
                loss = total_forecast_loss / num_active_heads
            else:
                # Handle case where no loss could be computed (e.g., all masked out)
                # Return 0 loss? Or handle upstream?
                loss = torch.tensor(0.0, device=hidden_states.device, requires_grad=True)


            # --- Auxiliary MoE Load Balancing Loss ---
            if self.apply_aux_loss:
                router_logits = outputs.router_logits if return_dict else outputs[-1] # Get router logits

                # Need attention mask in [B, L] format for aux loss function
                # The model forward might receive 4D mask or None.
                # Reconstruct [B, L] mask if possible.
                input_mask_for_aux = None
                if input_ids is not None: # Use input_ids shape
                    bsz, seq_len, _ = input_ids.shape
                    # If original attention_mask was passed, use it. Assume it was [B, L].
                    # This requires passing the original [B, L] mask down, not the 4D one.
                    # Let's assume `attention_mask` passed to *this* forward method is [B, L].
                    if attention_mask is not None and attention_mask.dim() == 2:
                          input_mask_for_aux = attention_mask
                elif inputs_embeds is not None: # Use inputs_embeds shape
                    bsz, seq_len, _ = inputs_embeds.shape
                    if attention_mask is not None and attention_mask.dim() == 2:
                          input_mask_for_aux = attention_mask
                else: # Cannot determine shape
                    pass

                if router_logits is not None:
                    aux_loss = load_balancing_loss_func(
                        router_logits,
                        top_k=self.num_experts_per_tok,
                        num_experts=self.config.num_experts,
                        attention_mask=input_mask_for_aux # Pass the [B, L] mask
                    )
                    # Add aux loss to main loss
                    loss += self.router_aux_loss_factor * aux_loss.to(loss.device)
                else:
                     logger.warning("apply_aux_loss is True, but router_logits were not found.")


            # --- CT-RoPE Regularization Loss ---
            if self.config.time_aware_rotary and (self.rope_w_reg_alpha > 0 or self.rope_smoothness_reg_beta > 0):
                rope_w_reg_loss = 0.0
                rope_smoothness_reg_loss = 0.0
                num_rope_layers = 0

                for module in self.model.modules():
                    if isinstance(module, ContinuousTimeRotaryEmbedding):
                        num_rope_layers += 1
                        w = module.w
                        inv_freq = module.inv_freq.to(w.device) # Ensure device match

                        # ||w - 1||^2 term
                        w_reg = torch.sum((w - 1.0)**2)
                        rope_w_reg_loss += w_reg

                        # Smoothness: sum( (d(theta_m)/dt)^2 ) = sum( (w_m * inv_freq_m)^2 )
                        deriv_sq = (w * inv_freq)**2
                        rope_smoothness_reg_loss += torch.sum(deriv_sq)

                # Average over layers? Or sum? Paper implies sum over m, but maybe average over layers.
                # Let's average over layers found.
                if num_rope_layers > 0:
                     rope_w_reg_loss /= num_rope_layers
                     rope_smoothness_reg_loss /= num_rope_layers

                ct_rope_reg_loss = (
                    self.rope_w_reg_alpha * rope_w_reg_loss +
                    self.rope_smoothness_reg_beta * rope_smoothness_reg_loss
                )
                loss += ct_rope_reg_loss.to(loss.device)


        # --- Inference Case (no labels) ---
        else:
            # Select appropriate head based on max_horizon_length
            if max_horizon_length is None:
                # Default to shortest horizon if not specified
                selected_horizon = self.horizon_lengths[0]
            else:
                # Find the largest configured horizon <= max_horizon_length
                selected_horizon = self.horizon_lengths[0] # Start with shortest
                for h in self.horizon_lengths[1:]:
                    if h <= max_horizon_length:
                        selected_horizon = h
                    else:
                        break # Stop once configured horizon exceeds max

            # Get predictions from the selected head
            selected_head_idx = self.horizon_length_map[selected_horizon]
            lm_head = self.lm_heads[selected_head_idx]
            predictions = lm_head(hidden_states) # [B, L, Din * H_selected]

            # If the selected head's horizon > max_horizon, truncate output
            output_dims = self.input_size * selected_horizon
            required_dims = self.input_size * max_horizon_length if max_horizon_length is not None else output_dims

            if output_dims > required_dims:
                predictions = predictions[..., :required_dims]


        # --- Prepare Output ---
        if not return_dict:
            output = (predictions,) + outputs[1:] # Combine predictions with base model outputs
            # Return loss and aux_loss first if calculated
            if loss is not None:
                return (loss, aux_loss) + output if aux_loss is not None else (loss,) + output
            else:
                return output


        # Use MoeCausalLMOutputWithPast for consistency, even if MoE isn't used
        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss, # Will be None if not calculated
            logits=predictions,
            past_key_values=outputs.past_key_values if return_dict else outputs[1],
            hidden_states=outputs.hidden_states if return_dict else outputs[2],
            attentions=outputs.attentions if return_dict else outputs[3],
            router_logits=outputs.router_logits if return_dict and not self.config.use_dense else None # Return router logits if MoE
        )


    def calc_ar_loss(self, predictions, labels, loss_masks, horizon_length):
        """
        Calculates the autoregressive Huber loss for a given prediction horizon.

        Args:
            predictions (torch.Tensor): Model predictions. Shape [B, L, Din * H].
            labels (torch.Tensor): Ground truth labels. Shape [B, L, Din].
            loss_masks (torch.Tensor, optional): Loss mask. Shape [B, L, Din].
            horizon_length (int): The prediction horizon length (H).

        Returns:
            torch.Tensor: Scalar loss value for this horizon, or None if horizon is invalid.
        """
        if horizon_length <= 0:
            return None # Cannot compute loss for non-positive horizon

        batch_size, seq_len, _ = predictions.shape
        input_size = labels.shape[-1] # Din

        # Reshape predictions to [B, L, H, Din]
        try:
            shift_predictions = predictions.view(batch_size, seq_len, horizon_length, input_size)
        except RuntimeError as e:
            logger.error(f"Error reshaping predictions with shape {predictions.shape} for horizon {horizon_length} and input_size {input_size}: {e}")
            return None # Cannot calculate loss if reshape fails

        # Prepare shifted labels [B, L, H, Din]
        # Pad labels: [B, Din, L] -> [B, Din, L + H - 1]
        labels_padded = F.pad(labels.permute(0, 2, 1), (0, horizon_length - 1), mode='constant', value=0)
        # Unfold to get sliding windows: [B, Din, L, H]
        shift_labels = labels_padded.unfold(dimension=-1, size=horizon_length, step=1)
        # Permute to [B, L, H, Din]
        shift_labels = shift_labels.permute(0, 2, 3, 1).to(predictions.device) # Ensure device match

        # Prepare shifted masks [B, L, H, Din] (if mask provided)
        shift_masks = None
        if loss_masks is not None:
            # Pad masks: [B, Din, L] -> [B, Din, L + H - 1]
            masks_padded = F.pad(loss_masks.permute(0, 2, 1), (0, horizon_length - 1), mode='constant', value=0)
             # Unfold: [B, Din, L, H]
            shift_masks = masks_padded.unfold(dimension=-1, size=horizon_length, step=1)
            # Permute to [B, L, H, Din]
            shift_masks = shift_masks.permute(0, 2, 3, 1).to(predictions.device) # Ensure device match
            # Ensure mask is float for multiplication
            shift_masks = shift_masks.float()


        # Calculate Huber loss (element-wise)
        losses = self.loss_function(shift_predictions, shift_labels) # Shape [B, L, H, Din]

        # Apply mask
        if shift_masks is not None:
            # Make sure shapes match for broadcasting/multiplication
            if losses.shape != shift_masks.shape:
                 logger.error(f"Loss shape {losses.shape} mismatch with mask shape {shift_masks.shape}")
                 # Fallback: don't use mask if shapes mismatch
                 loss_value = torch.mean(losses) # Average over all elements
            else:
                 masked_losses = losses * shift_masks
                 # Normalize by the sum of the mask (number of valid elements)
                 mask_sum = shift_masks.sum()
                 if mask_sum > 0:
                      loss_value = masked_losses.sum() / mask_sum
                 else:
                      # No valid elements to compute loss, return 0?
                      loss_value = torch.tensor(0.0, device=losses.device, requires_grad=True)
        else:
            # No mask, just average the loss
            loss_value = torch.mean(losses)

        return loss_value


    def prepare_inputs_for_generation(
            self,
            input_ids: Optional[torch.FloatTensor] = None,
            time_values: Optional[torch.FloatTensor] = None, # Need to handle time values cache
            past_key_values: Optional[List[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            **kwargs
    ):
        """ Prepares inputs for autoregressive generation. """

        # --- Standard KV Cache Handling ---
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else: # Legacy cache format
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only unprocessed tokens: (Logic assumes input_ids contains full sequence initially)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                # Input embeddings are likely passed, adjust input_ids based on mask
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
                if time_values is not None:
                     time_values = time_values[:, -(attention_mask.shape[1] - past_length):]
            elif past_length < input_ids.shape[1]:
                 # Standard case: input_ids has full sequence, take only new tokens
                 input_ids = input_ids[:, past_length:]
                 if time_values is not None:
                      time_values = time_values[:, past_length:]
            # Else: input_ids likely only contains new tokens already

            # Crop attention mask if exceeding max cache length
            if (max_cache_length is not None and attention_mask is not None
                    and cache_length + input_ids.shape[1] > max_cache_length):
                attention_mask = attention_mask[:, -max_cache_length:]


        # --- Position IDs ---
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            # Requires attention_mask to be [B, L]
            # if attention_mask.dim() == 4: # If 4D mask was passed somehow, reduce it
            #      attention_mask_2d = attention_mask[:,0,0,:] # Heuristic, might be wrong
            # else:
            #      attention_mask_2d = attention_mask

            # position_ids = attention_mask_2d.long().cumsum(-1) - 1
            # position_ids.masked_fill_(attention_mask_2d == 0, 1) # Fill padding with 1? Or 0? HF uses 1.
            # if past_key_values:
            #     # Adjust position_ids for the new tokens based on cache length
            #     position_ids = position_ids[:, -input_ids.shape[1]:] + cache_length # Increment by cache length

            # Determine the sequence length of the new inputs
            if inputs_embeds is not None:
                seq_length = inputs_embeds.shape[1]
            elif input_ids is not None:
                seq_length = input_ids.shape[1]
            else:
                # Should not happen if either input_ids or inputs_embeds is present
                seq_length = 0
            
            # Determine the cache length
            cache_length = 0
            if past_key_values is not None:
                if isinstance(past_key_values, Cache):
                    cache_length = past_key_values.get_seq_length()
                else: # Legacy format (should not happen if input was Cache)
                    try: # Add error handling just in case
                        cache_length = past_key_values[0][0].shape[2]
                    except:
                        cache_length = 0 # Fallback

            # Create position_ids for the new tokens, starting after the cache
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                cache_length, cache_length + seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0) # Shape [1, seq_length]

        # --- Inputs: Embeddings or IDs ---
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            logger.info('Using provided inputs_embeds for first step.')
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # Use input_ids (potentially sliced)
            model_inputs = {"input_ids": input_ids}


        # --- Time Values Handling (Crucial for CT-RoPE) ---
        # CT-RoPE needs the absolute time values for the *entire* sequence (past + current).
        # Standard HF generation loop passes only the *new* input_ids.
        # We need to maintain and pass the full time_values history.
        cached_time_values = kwargs.get("cached_time_values", None) # Get from kwargs

        if self.config.time_aware_rotary:
             if past_key_values is None: # First step
                  # Assume time_values passed initially covers the first input_ids sequence
                  model_inputs["time_values"] = time_values
                  # Store it for the next step
                  # model_inputs["cached_time_values"] = time_values
             else: # Subsequent steps
                  if cached_time_values is None:
                      raise ValueError("`cached_time_values` must be provided for generation with CT-RoPE after the first step.")
                  if time_values is None:
                       raise ValueError("`time_values` for the new token(s) must be provided for generation with CT-RoPE.")

                  # Concatenate cached times with new times
                  full_time_values = torch.cat([cached_time_values, time_values], dim=1)
                  model_inputs["time_values"] = full_time_values
                  # Pass concatenated times for caching in the next step
                  # model_inputs["cached_time_values"] = full_time_values
        else:
             # If not using CT-RoPE, standard RoPE uses position_ids, no time_values needed.
             model_inputs["time_values"] = None # Or pass original time_values if TimeAwareEmbedding needs it


        # --- Final Model Inputs Dict ---
        model_inputs.update({
                "position_ids": position_ids, # Pass adjusted position_ids
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask, # Pass potentially modified mask
            })

        # Remove None values from model_inputs if they are optional in the forward signature
        model_inputs = {k: v for k, v in model_inputs.items() if v is not None}

        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """ Reorders the KV cache for beam search/sampling. """
        # Adapts based on whether past_key_values is Cache object or tuple
        if isinstance(past_key_values, Cache):
             return past_key_values.reorder_cache(beam_idx)
        else: # Handle legacy tuple format
             reordered_past = ()
             for layer_past in past_key_values:
                  reordered_past += (
                       tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
                  )
             return reordered_past