# filename: test_modeling_unit.py
import unittest
import torch
from torch import nn

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel, Cache, DynamicCache, StaticCache
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import MoeModelOutputWithPast, MoeCausalLMOutputWithPast
from transformers.utils import logging, is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10

# Assuming the model files are in the same directory or accessible via path
# Adjust import paths if necessary
from time_moe.models.configuration_time_moe import TimeMoeConfig
from time_moe.models.modeling_time_moe import (
    ContinuousTimeRotaryEmbedding,
    TimeMoeAttention,
    TimeMoeForPrediction,
    apply_rotary_pos_emb,
    rotate_half,
    compute_delta_t
)

# Helper to create a basic config for testing
def create_test_config(**kwargs):
    # Minimal config for component testing
    config_dict = {
        "input_size": 1,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "head_dim": 8, # hidden_size / num_attention_heads
        "num_key_value_heads": 4,
        "horizon_lengths": [1],
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "initializer_range": 0.02,
        "rope_theta": 1000.0,
        "attention_dropout": 0.0,
        "use_cache": False,
        "_attn_implementation": "eager", # Ensure eager for testing attention internals
        "use_dense": True, # Use dense MLP for simpler testing unless MoE specifically tested
        "apply_aux_loss": False, # Consistent with use_dense=True
        # Time related defaults (can be overridden by kwargs)
        "time_aware_rotary": True,
        "time_decay": False,
        "use_gated_time_decay": False,
        "use_nonlinear_time_decay": False,
        "rope_w_reg_alpha": 0.0,
        "rope_smoothness_reg_beta": 0.0,
        "huber_delta": 1.0,
    }
    config_dict.update(kwargs)
    # Recalculate head_dim if hidden_size/num_heads changed
    config_dict["head_dim"] = config_dict["hidden_size"] // config_dict["num_attention_heads"]
    if config_dict["num_key_value_heads"] is None:
        config_dict["num_key_value_heads"] = config_dict["num_attention_heads"]

    # Ensure MoE/Dense consistency
    config_dict["apply_aux_loss"] = False if config_dict["use_dense"] else config_dict.get("apply_aux_loss", True)

    return TimeMoeConfig(**config_dict)

class TestContinuousTimeRoPE(unittest.TestCase):

    def setUp(self):
        self.config = create_test_config(time_aware_rotary=True)
        self.head_dim = self.config.head_dim
        self.rope = ContinuousTimeRotaryEmbedding(dim=self.head_dim, base=self.config.rope_theta)
        self.batch_size = 2
        self.seq_len = 10
        self.num_heads = self.config.num_attention_heads
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rope.to(self.device)

    def test_w_parameter(self):
        """Check if learnable 'w' parameter exists and has correct shape."""
        self.assertTrue(hasattr(self.rope, 'w'))
        self.assertIsInstance(self.rope.w, nn.Parameter)
        self.assertEqual(self.rope.w.shape, (self.head_dim // 2,))
        self.assertTrue(self.rope.w.requires_grad)

    def test_forward_pass_shape(self):
        """Test the forward pass output shapes."""
        q = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim, device=self.device)
        k = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim, device=self.device)
        # Absolute, strictly increasing time values
        time_values = torch.linspace(0, self.seq_len - 1, self.seq_len, device=self.device).unsqueeze(0).repeat(self.batch_size, 1) * 0.1
        time_values = time_values.to(torch.float32) # Use float64 for time precision

        q_rot, k_rot = self.rope(q, k, time_values.to(self.device))

        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)
        self.assertEqual(q_rot.dtype, q.dtype)
        self.assertEqual(k_rot.dtype, k.dtype)

    def test_gradient_flow(self):
        """Check if gradients flow back to the 'w' parameter."""
        q = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim, device=self.device, requires_grad=True)
        k = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim, device=self.device)
        time_values = torch.linspace(0, self.seq_len - 1, self.seq_len, device=self.device).unsqueeze(0).repeat(self.batch_size, 1) * 0.1
        time_values = time_values.to(torch.float32)

        q_rot, _ = self.rope(q, k, time_values.to(self.device))
        dummy_loss = q_rot.sum()
        dummy_loss.backward()

        self.assertIsNotNone(self.rope.w.grad)
        self.assertGreater(torch.abs(self.rope.w.grad).sum(), 0) # Check grad is non-zero

    def test_cache(self):
        """Test if cos/sin cache is used."""
        q = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim, device=self.device)
        k = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim, device=self.device)
        time_values = torch.linspace(0, self.seq_len - 1, self.seq_len, device=self.device).unsqueeze(0).repeat(self.batch_size, 1) * 0.1
        time_values = time_values.to(torch.float32)

        # First pass - compute and cache
        self.rope._last_time_values_hash = None # Reset cache state
        self.rope(q, k, time_values.to(self.device))
        first_hash = self.rope._last_time_values_hash
        self.assertIsNotNone(first_hash)

        # Second pass with same time values - should use cache
        self.rope(q, k, time_values.to(self.device))
        second_hash = self.rope._last_time_values_hash
        # Note: The simple hash might not be perfectly robust, but this checks basic caching behavior
        self.assertEqual(first_hash, second_hash)

        # Third pass with different time values - should recompute
        time_values_new = time_values + 1.0
        self.rope(q, k, time_values_new.to(self.device))
        third_hash = self.rope._last_time_values_hash
        self.assertNotEqual(first_hash, third_hash)


class TestTimeDecayAttention(unittest.TestCase):

    def setUp(self):
        # Enable all time decay features
        self.config = create_test_config(
            time_aware_rotary=True, # CT-RoPE needs to be active to pass time_values
            time_decay=True,
            use_gated_time_decay=True,
            use_nonlinear_time_decay=True
        )
        self.attention = TimeMoeAttention(config=self.config, layer_idx=0)
        self.batch_size = 2
        self.seq_len = 10
        self.hidden_size = self.config.hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention.to(self.device)

    def test_forward_pass_shape(self):
        """Test attention forward pass with time decay enabled."""
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
        time_values = torch.sort(torch.rand(self.batch_size, self.seq_len, device=self.device) * self.seq_len, dim=1)[0]
        time_values = time_values.to(torch.float32)
        # Create a causal mask
        attention_mask = _prepare_4d_causal_attention_mask(None, (self.batch_size, self.seq_len), hidden_states, 0)

        attn_output, attn_weights, _ = self.attention(
            hidden_states=hidden_states,
            time_values=time_values.to(self.device),
            attention_mask=attention_mask,
            position_ids=None # Not needed for CT-RoPE
        )

        self.assertEqual(attn_output.shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertIsNone(attn_weights) # output_attentions=False by default

    def test_gradient_flow(self):
        """Check gradients for time decay parameters."""
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device, requires_grad=True)
        time_values = torch.sort(torch.rand(self.batch_size, self.seq_len, device=self.device) * self.seq_len, dim=1)[0]
        time_values = time_values.to(torch.float32)
        attention_mask = _prepare_4d_causal_attention_mask(None, (self.batch_size, self.seq_len), hidden_states, 0)

        attn_output, _, _ = self.attention(
            hidden_states=hidden_states,
            time_values=time_values.to(self.device),
            attention_mask=attention_mask,
            position_ids=None
        )
        dummy_loss = attn_output.sum()
        dummy_loss.backward()

        self.assertTrue(hasattr(self.attention, 'lambda_decay'))
        self.assertIsNotNone(self.attention.lambda_decay.grad)
        self.assertTrue(hasattr(self.attention, 'gate_mlp'))
        self.assertTrue(any(p.grad is not None for p in self.attention.gate_mlp.parameters()))
        self.assertTrue(hasattr(self.attention, 'nonlinear_decay_mlp'))
        self.assertTrue(any(p.grad is not None for p in self.attention.nonlinear_decay_mlp.parameters()))

if __name__ == '__main__':
    unittest.main()