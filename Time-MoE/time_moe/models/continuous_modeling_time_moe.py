class EnhancedTimeMoeAttention(nn.Module):
    """
    Enhanced Multi-headed attention with continuous time encoding mechanisms:
    - Multi-scale temporal decay
    - Gated time modulation
    - Relative position biasing
    - Nonlinear temporal decay
    """
    def __init__(self, config: TimeMoeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        # Projection layers
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Rotary embeddings
        if config.use_enhanced_attention:
            self.rotary_emb = TimeStepAwareRotaryEmbedding(
                dim=self.head_dim,
                max_timescale=config.max_position_embeddings,
                base=config.rope_theta,
                device=device  
            )
        else:
            self.rotary_emb = TimeMoeRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
                device=device
            )

        # Continuous time encoding components
        self.time_encoder = ContinuousTimeEncoding(config, self.num_heads, self.head_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            time_values: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        
        # Original projections
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for attention computation
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Rotary embeddings
        kv_seq_len = key_states.shape[-2]

        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(
            value_states, 
            seq_len=kv_seq_len,
            time_values=time_values  \
        )

        query_states, key_states = apply_enhanced_rotary(
            query_states, 
            key_states, 
            cos, 
            sin,
            position_ids=position_ids,
            time_values=time_values 
        )

        # Continuous time encoding
        time_features = self.time_encoder(
            hidden_states=hidden_states,
            time_values=time_values
            attention_mask=attention_mask,
            query_states=query_states,
            key_states=key_states
        )

        # Update key/value states with temporal gates
        key_states = key_states * time_features['key_gates']
        value_states = value_states * time_features['value_gates']

        # Update attention with temporal biases
        temporal_bias = time_features['temporal_bias']
        decay_mask = time_features['decay_mask']

        # KV cache handling
        if past_key_value is not None:
            current_key = key_states * time_features['key_gates'][..., -1:, :]  
            current_value = value_states * time_features['value_gates'][..., -1:, :]

            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(
                current_key, 
                current_value, 
                self.layer_idx, 
                cache_kwargs
            )
        
        else:   
            key_states = key_states * time_features['key_gates']
            value_states = value_states * time_features['value_gates']
             
        # Repeat KV heads if needed
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Attention computation
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Apply temporal decay mask
        attn_weights = attn_weights * decay_mask

        # Add temporal bias
        attn_weights = attn_weights + temporal_bias

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax and dropout
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        # Attention output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class ContinuousTimeEncoding(nn.Module):
    """Continuous time encoding module with multiple temporal mechanisms"""
    def __init__(self, config: TimeMoeConfig, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Multi-scale temporal decay parameters
        self.decay_factors = nn.Parameter(torch.randn(num_heads))
        self.temporal_scales = nn.Parameter(torch.linspace(1, 10, num_heads))
        
        # Gated time modulation
        self.gate_network = nn.Sequential(
            nn.Linear(config.hidden_size, 4*config.hidden_size),
            nn.SiLU(),
            nn.Linear(4*config.hidden_size, 2*num_heads*head_dim)
        )
        
        # Relative temporal bias
        self.rel_temp_bias = nn.Embedding(512, num_heads)
        
        # Nonlinear decay parameters
        self.decay_gamma = nn.Parameter(torch.ones(num_heads))
        self.decay_beta = nn.Parameter(torch.zeros(num_heads))

    def _temporal_decay(self, time_diff: torch.Tensor) -> torch.Tensor:
        """Nonlinear decay function with learnable parameters"""
        scaled_diff = time_diff * self.temporal_scales.view(1,1,-1,1)
        return torch.exp(-F.softplus(self.decay_gamma) * torch.relu(scaled_diff) + self.decay_beta)

    def forward(self, 
               hidden_states: torch.Tensor,
               position_ids: torch.LongTensor,
               time_values: torch.FloatTensor,
               attention_mask: Optional[torch.Tensor],
               query_states: torch.Tensor,
               key_states: torch.Tensor) -> dict:
        
        # Time difference matrix
        seq_len = position_ids.shape[1]
        time_diff = position_ids.unsqueeze(-1) - position_ids.unsqueeze(-2)  # [B, S, S]
        
        # Multi-head decay mask
        decay_mask = self._temporal_decay(time_diff.unsqueeze(-1))  # [B, S, S, H]
        decay_mask = decay_mask.permute(0, 3, 1, 2)  # [B, H, S, S]
        
        # Relative temporal bias
        rel_pos = (time_diff.clamp(-256, 256) + 256).long()  # [B, S, S]
        rel_bias = self.rel_temp_bias(rel_pos)  # [B, S, S, H]
        rel_bias = rel_bias.permute(0, 3, 1, 2) * 0.1  # [B, H, S, S]
        
        # Gated modulation
        gate_params = self.gate_network(hidden_states)  # [B, S, 2*H*D]
        gate_params = gate_params.view(
            *gate_params.shape[:2], 
            self.num_heads, 
            2,  # query/key 
            self.head_dim
        )
        query_gates = gate_params[..., 0, :].sigmoid()  # [B, S, H, D]
        key_gates = gate_params[..., 1, :].sigmoid()    # [B, S, H, D]
        
        # Apply gates to query/key states
        return {
            'key_gates': key_gates.transpose(1, 2),      # [B, H, S, D]
            'value_gates': key_gates.transpose(1, 2),     # Share gates for K/V
            'temporal_bias': rel_bias,                    # [B, H, S, S]
            'decay_mask': decay_mask                      # [B, H, S, S]
        }

class TimeStepAwareRotaryEmbedding(TimeMoeRotaryEmbedding):
    """An improved Rotary Embedding that supports real-valued time steps."""
    def __init__(self, dim, max_timescale=10000, base=10000, device=None):
        super().__init__(
            dim, 
            max_position_embeddings=max_timescale, 
            base=base, 
            device = config._device if hasattr(config, '_device') else torch.device('cpu')
        )
        # Extend to support continuous time values
        self.time_scale = nn.Parameter(torch.tensor(1.0, device=device))
        # self.register_buffer("time_scale", torch.tensor(1.0, device=device))


    def _set_cos_sin_cache(self, time_values, device, dtype):
        """Generate frequencies based on real time values."""
        if isinstance(time_values, torch.Tensor):
            scaled_time = time_values * self.time_scale.view(-1, 1)
            freqs = torch.einsum('bs,d->bsd', scaled_time, self.inv_freq)
        else:
            freqs = torch.outer(time_values * self.time_scale, self.inv_freq)
        
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)

    def forward(self, x, seq_len=None, time_values=None):       
        if isinstance(time_values, torch.Tensor):
            if time_values.dim() == 1:
                time_values = time_values.unsqueeze(0)  
            batch_size, seq_len = time_values.shape
            time_values = time_values.view(-1)  
        else:
            batch_size = 1  
        
        self._set_cos_sin_cache(time_values, x.device, x.dtype)
        
        # [B, S, D]
        cos = self.cos_cached.view(batch_size, seq_len, -1)
        sin = self.sin_cached.view(batch_size, seq_len, -1)
        return cos.to(x.dtype), sin.to(x.dtype)

def apply_enhanced_rotary(q, k, cos, sin, position_ids=None, time_values=None, unsqueeze_dim=1):
    """Enhanced Rotary Embedding supporting continuous time values."""
    if time_values is not None:
       # When using real-valued time steps, broadcast without indexing
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
    elif position_ids is not None:
        # Traditional discrete position indexing mode
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    else:
        raise ValueError("need proide position_ids or time_values")
        
    # Apply rotation (numerically stable implementation)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed