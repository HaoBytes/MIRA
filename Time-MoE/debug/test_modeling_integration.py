# filename: test_modeling_integration.py
import unittest
import torch

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
from time_moe.models.modeling_time_moe import TimeMoeForPrediction
from test_modeling_unit import create_test_config # Reuse helper from unit tests

class TestModelIntegration(unittest.TestCase):

    def _create_dummy_batch(self, batch_size, seq_len, input_size, device, use_time=True):
        """Helper to create a dummy data batch."""
        input_data = torch.randn(batch_size, seq_len, input_size, device=device)
        labels = torch.randn(batch_size, seq_len, input_size, device=device) # Simple shifted labels for testing
        loss_masks = torch.ones_like(labels, device=device)
        time_values = None
        if use_time:
            # Generate sorted, unique-ish absolute time values
            time_values = torch.sort(torch.rand(batch_size, seq_len, device=device) * seq_len * 0.5, dim=1)[0]
            # Add small increments to ensure strict monotonicity for testing
            time_values = time_values + torch.linspace(0, 1e-4, seq_len, device=device).unsqueeze(0)
            time_values = time_values.to(torch.float64) # Use float64 for time

        return {
            "input_ids": input_data,
            "labels": labels,
            "loss_masks": loss_masks,
            "time_values": time_values.to(device) if time_values is not None else None,
            "return_dict": True,
        }

    def test_forward_pass_standard_rope(self):
        """Test end-to-end forward pass with standard RoPE."""
        config = create_test_config(
            num_hidden_layers=2, # A bit deeper
            time_aware_rotary=False,
            use_dense=False, # Use MoE
            num_experts=4,
            num_experts_per_tok=2,
            apply_aux_loss=True, # Test aux loss
            router_aux_loss_factor=0.01,
            hidden_size=16, # Even smaller for speed
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=32,
        )
        model = TimeMoeForPrediction(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train() # Ensure training mode for aux loss etc.

        batch = self._create_dummy_batch(batch_size=2, seq_len=20, input_size=config.input_size, device=device, use_time=False)
        # Need position_ids for standard RoPE
        batch["position_ids"] = torch.arange(0, 20, device=device).unsqueeze(0).repeat(2, 1)

        outputs = model(**batch)

        self.assertTrue("logits" in outputs)
        self.assertTrue("loss" in outputs)
        self.assertTrue("aux_loss" in outputs)
        self.assertTrue("router_logits" in outputs)

        expected_logit_shape = (2, 20, config.input_size * config.horizon_lengths[0])
        self.assertEqual(outputs.logits.shape, expected_logit_shape)
        self.assertTrue(torch.is_tensor(outputs.loss))
        self.assertEqual(outputs.loss.ndim, 0) # Scalar loss
        self.assertTrue(torch.is_tensor(outputs.aux_loss))
        self.assertEqual(outputs.aux_loss.ndim, 0) # Scalar loss
        self.assertIsNotNone(outputs.router_logits) # Should exist for MoE
        # Check backward pass
        outputs.loss.backward()

    def test_forward_pass_ct_rope_and_decay(self):
        """Test end-to-end forward pass with CT-RoPE, Time Decay, and Regularization."""
        config = create_test_config(
            num_hidden_layers=2,
            hidden_size=16,
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=32,
            time_aware_rotary=True, # <<< Enable CT-RoPE
            time_decay=True, # <<< Enable Time Decay
            use_gated_time_decay=True,
            use_nonlinear_time_decay=True,
            rope_w_reg_alpha=0.01, # <<< Enable Reg Loss
            rope_smoothness_reg_beta=0.001, # <<< Enable Reg Loss
            use_dense=True, # Use Dense to isolate from MoE aux loss
            apply_aux_loss=False,
        )
        model = TimeMoeForPrediction(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()

        batch = self._create_dummy_batch(batch_size=2, seq_len=20, input_size=config.input_size, device=device, use_time=True)
        # position_ids not strictly needed by model forward if CT-RoPE is used, but pass None explicitly
        batch["position_ids"] = None

        outputs = model(**batch)

        self.assertTrue("logits" in outputs)
        self.assertTrue("loss" in outputs)
        self.assertTrue("aux_loss" not in outputs or outputs.aux_loss is None) # No aux loss for dense
        self.assertTrue("router_logits" not in outputs or outputs.router_logits is None) # No router logits for dense

        expected_logit_shape = (2, 20, config.input_size * config.horizon_lengths[0])
        self.assertEqual(outputs.logits.shape, expected_logit_shape)
        self.assertTrue(torch.is_tensor(outputs.loss))
        self.assertEqual(outputs.loss.ndim, 0) # Scalar loss

        # Check backward pass
        initial_loss = outputs.loss.item()
        outputs.loss.backward()

        # Check if 'w' parameter got grads
        found_w_grad = False
        for name, param in model.named_parameters():
            if "rotary_emb.w" in name:
                self.assertIsNotNone(param.grad, f"Gradient for {name} is None")
                self.assertGreater(torch.abs(param.grad).sum(), 0, f"Gradient for {name} is zero")
                found_w_grad = True
            # Check decay grads too
            if "lambda_decay" in name:
                 self.assertIsNotNone(param.grad, f"Gradient for {name} is None")
                 self.assertGreater(torch.abs(param.grad).sum(), 0, f"Gradient for {name} is zero")
            if "gate_mlp" in name:
                 self.assertIsNotNone(param.grad, f"Gradient for {name} is None")
                 # Grad might be small, just check it exists
            if "nonlinear_decay_mlp" in name:
                 self.assertIsNotNone(param.grad, f"Gradient for {name} is None")

        self.assertTrue(found_w_grad, "Did not find gradient for CT-RoPE 'w' parameter")

        # Optional: Check if reg loss had an effect (difficult to isolate precisely)
        # Zero out alpha/beta, run again, check if loss changes slightly
        config.rope_w_reg_alpha = 0.0
        config.rope_smoothness_reg_beta = 0.0
        model_no_reg = TimeMoeForPrediction(config).to(device).train()
        # Copy weights to ensure fair comparison (simple way)
        model_no_reg.load_state_dict(model.state_dict())
        with torch.no_grad(): # No need for backward pass here
            outputs_no_reg = model_no_reg(**batch)
        # Loss might change slightly due to removing reg terms
        self.assertNotAlmostEqual(initial_loss, outputs_no_reg.loss.item(), delta=1e-5, msg="Loss did not change when removing regularization")


    def test_forward_with_cache(self):
        """Test forward pass with use_cache=True."""
        config = create_test_config(
            num_hidden_layers=1,
            time_aware_rotary=True, # Test caching with CT-RoPE
            time_decay=True,
            rope_w_reg_alpha=0.0, # Disable reg for simplicity
            rope_smoothness_reg_beta=0.0,
            use_cache=True, # <<< Enable Cache
        )
        model = TimeMoeForPrediction(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval() # Use eval mode for generation/caching tests

        seq_len_step1 = 10
        seq_len_step2 = 1

        # --- Step 1 ---
        batch1 = self._create_dummy_batch(batch_size=1, seq_len=seq_len_step1, input_size=config.input_size, device=device, use_time=True)
        batch1['use_cache'] = True
        batch1['position_ids'] = None
        batch1.pop('labels') # No labels needed for inference with cache test
        batch1.pop('loss_masks')

        with torch.no_grad():
             outputs1 = model(**batch1)

        self.assertTrue("logits" in outputs1)
        self.assertTrue("past_key_values" in outputs1)
        self.assertIsNotNone(outputs1.past_key_values)
        # Check cache length
        self.assertEqual(outputs1.past_key_values.get_seq_length(), seq_len_step1)

        # --- Step 2 ---
        # Prepare inputs for the next step using prepare_inputs_for_generation
        batch2_inputs = self._create_dummy_batch(batch_size=1, seq_len=seq_len_step2, input_size=config.input_size, device=device, use_time=True)

        # Simulate prepare_inputs_for_generation logic for time_values
        # Requires passing cached time values from step 1
        model_inputs = model.prepare_inputs_for_generation(
             input_ids=batch2_inputs['input_ids'],
             past_key_values=outputs1.past_key_values,
             attention_mask=torch.ones(1, seq_len_step1 + seq_len_step2, device=device), # Full mask including past
             use_cache=True,
             # Crucial for CT-RoPE: pass new time value AND cached time values
             time_values=batch2_inputs['time_values'], # New time value(s)
             cached_time_values=batch1['time_values'] # Time values from step 1
        )

        # Check prepared inputs
        self.assertEqual(model_inputs['input_ids'].shape[1], seq_len_step2) # Should only contain new input ID
        self.assertIsNotNone(model_inputs['past_key_values'])
        self.assertEqual(model_inputs['time_values'].shape[1], seq_len_step1 + seq_len_step2) # Should have full time history
        # self.assertIsNotNone(model_inputs['cached_time_values']) # Should be passed for next step
        self.assertEqual(model_inputs['attention_mask'].shape[1], seq_len_step1 + seq_len_step2)
        self.assertEqual(model_inputs['position_ids'].shape[1], seq_len_step2) # Position IDs for new tokens
        self.assertEqual(model_inputs['position_ids'][0, -1], seq_len_step1 + seq_len_step2 - 1) # Check last position ID


        # Run forward pass for step 2
        with torch.no_grad():
             outputs2 = model(**model_inputs)

        self.assertTrue("logits" in outputs2)
        self.assertTrue("past_key_values" in outputs2)
        self.assertIsNotNone(outputs2.past_key_values)
        # Check cache length updated correctly
        self.assertEqual(outputs2.past_key_values.get_seq_length(), seq_len_step1 + seq_len_step2)
        # Check output logit shape corresponds to the single new input token
        expected_logit_shape_step2 = (1, seq_len_step2, config.input_size * config.horizon_lengths[0])
        self.assertEqual(outputs2.logits.shape, expected_logit_shape_step2)


if __name__ == '__main__':
    unittest.main()