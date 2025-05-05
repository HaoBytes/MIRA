from typing import List
from transformers import PretrainedConfig


class TimeMoeConfig(PretrainedConfig):
    model_type = "time_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            input_size: int = 1,
            hidden_size: int = 4096,
            intermediate_size: int = 22016,
            horizon_lengths: List[int] = [1], # Changed default from 1 to [1] to match List hint
            num_hidden_layers: int = 32,
            num_attention_heads: int = 32,
            num_key_value_heads: int = None,
            hidden_act: str = "silu",
            num_experts_per_tok: int = 2,
            num_experts: int = 1,
            max_position_embeddings: int = 32768,
            initializer_range: float = 0.02,
            rms_norm_eps: float = 1e-6,
            use_cache: bool = True,
            use_dense: bool = False,

            rope_theta: float = 10000.0, # Changed to float from int
            learnable_rope: bool = False, # Seems unused, keeping for now

            attention_dropout: float = 0.0,
            apply_aux_loss: bool = True,
            router_aux_loss_factor: float = 0.02,
            # tie_word_embeddings: bool = False, # Controlled via super init now
            use_temporal_fusion: bool = True, # Seems related to OptimizedTimeSeriesInputEmbedding

            # --- Flags for Time Awareness (Based on original + required for new features) ---
            time_aware: bool = False,  # Keep original flag if needed by other parts
            time_aware_embedding: bool = False, # Flag for OptimizedTimeSeriesInputEmbedding (duplicate of use_temporal_fusion?)
            time_aware_rotary: bool = False,  # Flag to enable ContinuousTimeRotaryEmbedding (CT-RoPE)
            temporal_embed_dim: int = 64, # Used by OptimizedTimeSeriesInputEmbedding
            time_decay=True, # Keep original default? Setting True for example
            lambda_type="head-wise", # Currently hardcoded in Attention, keep for info
            use_gated_time_decay: bool = True, # Keep original default? Setting True for example
            use_nonlinear_time_decay: bool = True, # Keep original default? Setting True for example

            # --- Regularization Coefficients (NEW/Corrected based on paper) ---
            rope_w_reg_alpha: float = 0.0, # Regularization for CT-RoPE 'w' vector
            rope_smoothness_reg_beta: float = 0.0, # Regularization for CT-RoPE smoothness
            huber_delta: float = 1.0, # Delta for Huber loss (Adding this as it was in model code)

            # --- MoE Specific Config (Optional, added from my generated code) ---
            moe_intermediate_size: int = None, # Can specify different intermediate size for experts
            norm_topk_prob: bool = False, # Option to normalize top-k routing probabilities

            # --- Other (Added from my generated code) ---
            gradient_checkpointing_kwargs: dict = None,

            # --- Original Regularization Coeffs (Marking as deprecated/unused) ---
            position_reg_coeff=0.01,      # Deprecated if using CT-RoPE reg
            temporal_decay_reg_coeff=0.01, # Deprecated (no explicit reg mentioned for lambda_decay/gate_mlp in paper)

            **kwargs,
    ):
        # Assign attributes first
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
        # Ensure horizon_lengths is a list
        if isinstance(horizon_lengths, int):
            horizon_lengths = [horizon_lengths]
        self.horizon_lengths = horizon_lengths
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache

        self.rope_theta = float(rope_theta) # Ensure float
        self.learnable_rope = learnable_rope # Keep original flag

        self.attention_dropout = attention_dropout
        # Ensure apply_aux_loss matches MoE usage
        self.use_dense = use_dense
        self.apply_aux_loss = False if self.use_dense else apply_aux_loss # Don't apply if dense
        self.router_aux_loss_factor = router_aux_loss_factor

        # Time features from original + new
        self.time_aware = time_aware # Keep original
        self.time_aware_embedding = time_aware_embedding # Keep added flag (maybe redundant with use_temporal_fusion?)
        self.use_temporal_fusion = use_temporal_fusion # Keep original flag
        self.temporal_embed_dim = temporal_embed_dim # Keep original flag

        self.time_aware_rotary = time_aware_rotary
        self.time_decay = time_decay
        self.use_gated_time_decay = use_gated_time_decay
        self.use_nonlinear_time_decay = use_nonlinear_time_decay
        self.lambda_type = lambda_type # Keep original info

        # Regularization
        # Using new names based on paper
        self.rope_w_reg_alpha = rope_w_reg_alpha
        self.rope_smoothness_reg_beta = rope_smoothness_reg_beta
        # Keep original names if needed by other code, but mark clearly
        self.position_reg_coeff = position_reg_coeff # Potentially deprecated
        self.temporal_decay_reg_coeff = temporal_decay_reg_coeff # Potentially deprecated
        self.huber_delta = huber_delta

        # MoE optional config
        self.moe_intermediate_size = moe_intermediate_size if moe_intermediate_size is not None else intermediate_size
        self.norm_topk_prob = norm_topk_prob

        # Default _attn_implementation if not provided in kwargs
        if "_attn_implementation" not in kwargs:
             kwargs["_attn_implementation"] = "eager"

        # Ensure gradient_checkpointing_kwargs is a dict
        self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs if gradient_checkpointing_kwargs is not None else {"use_reentrant": True}

        # Check MoE/Dense consistency
        if self.use_dense and self.apply_aux_loss:
            self.apply_aux_loss = False
            print("Warning: `use_dense` is True, setting `apply_aux_loss` to False.")
        if not self.use_dense and self.num_experts <= 0:
             print("Warning: MoE enabled (use_dense=False) but num_experts <= 0. Defaulting num_experts to 1.")
             self.num_experts = 1

        # Handle attn_implementation potentially passed in kwargs or set default
        self._attn_implementation = kwargs.pop('_attn_implementation', 'eager')

        # Original assert relating use_dense and apply_aux_loss
        # Note: The logic `apply_aux_loss = False if self.use_dense else apply_aux_loss` already handles this.
        # The assert might be redundant or slightly different. Keep original if required.
        assert self.use_dense ^ self.apply_aux_loss, 'Both use_dense and apply_aux_loss cannot be set to True or False at the same time.'


        # === MODIFICATION START ===
        # Explicitly set tie_word_embeddings=False and pass remaining kwargs to superclass init
        # This prevents the PreTrainedModel base class from attempting weight tying.
        super().__init__(
            tie_word_embeddings=False,
            **kwargs,
        )
        # === MODIFICATION END ===

        # Store hidden size / heads relation for convenience elsewhere (can be done after super init)
        self.head_dim = self.hidden_size // self.num_attention_heads

        # Remove the old commented out super().__init__ call if it exists
        # # kwargs.pop('tie_word_embeddings', None)
        # # super().__init__(
        # #     tie_word_embeddings=tie_word_embeddings,
        # #     **kwargs,
        # # )