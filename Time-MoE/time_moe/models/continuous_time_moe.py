######################### 
#######time rope
######################### 

class OptimizedTimeSeriesEmbedding(nn.Module):
    """
    Optimized time series embedding module with multi-scale feature extraction
    and adaptive time modulation.
    Combines the efficiency of standard value embedding with enhanced temporal features.
    """

    def __init__(self, config: TimeMoeConfig):
        super().__init__()
        self.config = config
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        
        # Value embedding path (preserving original GLU-like style for efficiency)
        self.value_emb = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.value_gate = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.value_act = ACT2FN[config.hidden_act]
        
        # Time feature path (enhanced, multi-scale processing)
        self.time_proj = nn.Sequential(
            nn.Linear(1, self.hidden_size // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size),
            TimeMoeRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        )
        
        # Adaptive fusion gate to combine value and time embeddings
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Sigmoid()
        )
        
        # Output normalization with residual connection
        self.out_norm = TimeMoeRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, x: torch.Tensor, time_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input time series tensor of shape (batch_size, seq_len, input_size)
            time_values: Optional time stamps tensor of shape (batch_size, seq_len, 1)
        
        Returns:
            Embedded tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Compute value embeddings (GLU-style: gated linear unit)
        value_emb = self.value_act(self.value_gate(x)) * self.value_emb(x)
        
        # If no time values are provided, fallback to pure value embedding
        if time_values is None:
            logger.info("Time values not provided, using only value embeddings.")
            return value_emb
            
        # Ensure time_values has the correct shape
        if time_values.dim() == 2:
            time_values = time_values.unsqueeze(-1)
        
        # Extract multi-scale time features
        time_emb = self.time_proj(time_values)
        
        # Adaptive gating between value and time features
        gate_input = torch.cat([value_emb, time_emb], dim=-1)
        fusion_gate = self.fusion_gate(gate_input)
        
        # Mix embeddings using the fusion gate (similar to Highway networks)
        mixed_emb = fusion_gate * value_emb + (1.0 - fusion_gate) * time_emb
        
        # Apply residual connection and normalization
        output = self.out_norm(value_emb + mixed_emb)
        
        return output


######################### 
#######time rope
######################### 

class TimeAwareRotaryEmbedding(TimeMoeRotaryEmbedding):
    def __init__(self, dim, max_t=1e5, base=10000, device=None):
        super().__init__(dim, max_position_embeddings=1024, base=base, device=device)
        self.time_scale = nn.Parameter(torch.tensor(1.0))  # Learnable time scaling factor
        
    def _set_cos_sin_cache(self, time_values, device, dtype):
        """Generate rotation matrices based on real time values."""
        scaled_time = time_values * self.time_scale
        freqs = torch.outer(scaled_time, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)

    def forward(self, x, time_values=None, seq_len=None):
        if time_values is None:  # Fallback to standard RoPE
            return super().forward(x, seq_len=seq_len)
        
        # Ensure time values match the sequence length
        if time_values.dim() == 1:
            time_values = time_values.unsqueeze(0)
        
        # Dynamically generate time-aware rotation matrices
        self._set_cos_sin_cache(time_values, x.device, x.dtype)
        return self.cos_cached.to(x.dtype), self.sin_cached.to(x.dtype)