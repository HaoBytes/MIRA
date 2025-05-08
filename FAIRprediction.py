# --- Main Prediction Model ---
class TimeMoeForPrediction(TimeMoePreTrainedModel, TSGenerationMixin):
    _supports_cache_class = True

    def __init__(self, config: TimeMoeConfig):
        super().__init__(config)
        self.config = config
        self.model = TimeMoeModel(config)

        # Initialize ODE Block (if enabled)
        self.use_terminal_ode = getattr(config, "use_terminal_ode", False)
        if self.use_terminal_ode:
             if not is_torchdiffeq_available:
                  logger.error("use_terminal_ode=True but torchdiffeq is not installed. ODE block disabled.")
                  self.ode_extrapolation_block = None
                  self.use_terminal_ode = False # Disable flag
             else:
                  self.ode_extrapolation_block = TerminalODEBlock(config)
        else:
             self.ode_extrapolation_block = None

        # Initialize LM Heads
        self.horizon_lengths = config.horizon_lengths
        self.input_size = config.input_size
        lm_head_list = []
        self.horizon_length_map = {}
        for i, horizon_length in enumerate(self.horizon_lengths):
            lm_head_list.append(TimeMoeOutputLayer(config.hidden_size, horizon_length, config.input_size))
            self.horizon_length_map[horizon_length] = i
        self.lm_heads = nn.ModuleList(lm_head_list)

        # Loss Function
        self.huber_delta = config.huber_delta
        self.loss_function = torch.nn.HuberLoss(reduction='none', delta=self.huber_delta)

        # MoE Loss Params
        self.apply_aux_loss = config.apply_aux_loss and not config.use_dense
        self.num_experts_per_tok = config.num_experts_per_tok
        self.router_aux_loss_factor = config.router_aux_loss_factor
        self.num_experts = config.num_experts

        self.post_init()

    # ... (getters/setters, _tie_weights as before) ...
    def get_input_embeddings(self): return self.model.get_input_embeddings()
    def set_input_embeddings(self, value): self.model.set_input_embeddings(value)
    def get_output_embeddings(self): return self.lm_heads[0].out_layer
    def set_output_embeddings(self, new_layer): self.lm_heads[0].out_layer = new_layer
    def _tie_weights(self): pass

    def forward(
            self,
            input_ids: Optional[torch.FloatTensor] = None,
            time_values: Optional[torch.FloatTensor] = None, # Abs times [B, L] for backbone
            next_target_time_values: Optional[torch.FloatTensor] = None, # Abs times [B] for ODE target
            attention_mask: Optional[torch.Tensor] = None,    # Padding mask [B, L]
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,          # Changed type hint
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.FloatTensor] = None,       # Target values [B, L, Din] or [B, L]
            loss_masks: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            max_horizon_length: Optional[int] = None,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # --- Backbone Forward Pass ---
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

        # --- Extract Last Hidden State and Time ---
        hidden_states_all = outputs.last_hidden_state if return_dict else outputs[0]
        # State corresponding to the *last token processed in this pass*
        hidden_states_last = hidden_states_all[:, -1, :] # Shape [B, D]

        time_values_last = None
        if time_values is not None:
             # Time corresponding to the last token processed
             time_values_last = time_values[:, -1] # Shape [B]

        # --- Apply Terminal ODE Extrapolation ---
        hidden_states_for_head = hidden_states_last # Default state
        if self.use_terminal_ode and self.ode_extrapolation_block is not None:
            # Requires the target time for the *next* prediction step
            if next_target_time_values is None:
                if self.training:
                     warnings.warn("`next_target_time_values` not provided during training. Skipping terminal ODE.")
                else: # Crucial for inference
                     raise ValueError("`next_target_time_values` must be provided for inference when use_terminal_ode=True.")
            elif time_values_last is None:
                 raise ValueError("`time_values` must be provided for the last token when use_terminal_ode=True.")
            else:
                 # Ensure correct shapes (scalar or [B])
                 if time_values_last.dim() == 0: time_values_last = time_values_last.expand(hidden_states_last.shape[0])
                 if next_target_time_values.dim() == 0: next_target_time_values = next_target_time_values.expand(hidden_states_last.shape[0])

                 hidden_states_for_head = self.ode_extrapolation_block(
                     h_N=hidden_states_last,
                     t_N=time_values_last,
                     t_Nplus1=next_target_time_values
                 )

        # --- Prepare State for Prediction Heads ---
        # Unsqueeze sequence dimension (length 1)
        hidden_states_for_head = hidden_states_for_head.unsqueeze(1) # Shape [B, 1, D]

        # --- Prediction and Loss Calculation ---
        loss = None
        aux_loss = None
        predictions = None

        if labels is not None:
             # Standard AR loss: predict step N+1 from state after processing step N.
             # Compare prediction from hidden_states_for_head with label at step N.
             total_forecast_loss = 0.0
             num_active_heads = 0

             if labels.dim() == 2: labels = labels.unsqueeze(-1)
             if loss_masks is not None and loss_masks.dim() == 2: loss_masks = loss_masks.unsqueeze(-1)

             # Target label is the one corresponding to the step *after* the last input step processed.
             # If input seq len processed was `L_proc`, target is `label[:, L_proc-1, :]`.
             input_seq_len = hidden_states_all.shape[1] # Length processed in this fwd pass
             label_seq_len = labels.shape[1]

             if label_seq_len >= input_seq_len:
                 labels_target = labels[:, input_seq_len-1:input_seq_len, :] # Shape [B, 1, Din]
                 masks_target = loss_masks[:, input_seq_len-1:input_seq_len, :] if loss_masks is not None else None
             else:
                 logger.warning(f"Label seq len ({label_seq_len}) < input seq len ({input_seq_len}). Cannot compute loss for last step.")
                 labels_target = None

             if labels_target is not None:
                 for i, horizon_length in enumerate(self.horizon_lengths):
                      lm_head = self.lm_heads[i]
                      one_predictions = lm_head(hidden_states_for_head) # [B, 1, Din*H]
                      one_loss = self.calc_prediction_loss(one_predictions, labels_target, masks_target, horizon_length)
                      if one_loss is not None:
                           total_forecast_loss += one_loss
                           num_active_heads += 1
                      if i == 0: predictions = one_predictions # Store first head's prediction

                 if num_active_heads > 0: loss = total_forecast_loss / num_active_heads
                 else: loss = torch.tensor(0.0, device=hidden_states_for_head.device, requires_grad=True)

             # --- Auxiliary MoE Loss ---
             if self.apply_aux_loss:
                 # Aux loss calculated based on the *entire sequence processed* by the backbone
                 router_logits_tuple = outputs.router_logits if return_dict else outputs[-1]
                 input_mask_for_aux = attention_mask # Pass original [B, L] mask if available
                 if input_mask_for_aux is not None and input_mask_for_aux.dim() == 4:
                     # Attempt to recover [B, L] mask
                     input_mask_for_aux = (input_mask_for_aux[:,0,0,:] > -1e8).long() if input_mask_for_aux.shape[1]==1 and input_mask_for_aux.shape[2]==1 else None
                     if input_mask_for_aux is None: logger.warning("Could not recover 2D mask for aux loss.")

                 if router_logits_tuple is not None:
                     aux_loss = load_balancing_loss_func(router_logits_tuple, self.num_experts_per_tok, self.num_experts, input_mask_for_aux)
                     if loss is not None: loss += self.router_aux_loss_factor * aux_loss.to(loss.device)
                     else: loss = self.router_aux_loss_factor * aux_loss

        else: # Inference
             # --- Prediction using selected head ---
            if max_horizon_length is None: selected_horizon = self.horizon_lengths[0]
            else:
                selected_horizon = self.horizon_lengths[0]
                for h in self.horizon_lengths[1:]:
                    if h <= max_horizon_length: selected_horizon = h
                    else: break
            selected_head_idx = self.horizon_length_map[selected_horizon]
            lm_head = self.lm_heads[selected_head_idx]
            predictions = lm_head(hidden_states_for_head) # [B, 1, Din * H_selected]
            # Truncate if selected horizon > max_horizon (should not happen with logic above)
            # required_dims = self.input_size * (max_horizon_length if max_horizon_length is not None else selected_horizon)
            # if predictions.shape[-1] > required_dims: predictions = predictions[..., :required_dims]


        # --- Prepare Output ---
        if not return_dict:
             base_outputs_tuple = outputs[1:] if not isinstance(outputs, MoeModelOutputWithPast) else tuple(getattr(outputs, k) for k in outputs.keys() if k != 'last_hidden_state')
             output_tuple = (predictions,) + base_outputs_tuple
             final_output_list = []
             if loss is not None: final_output_list.append(loss)
             if aux_loss is not None: final_output_list.append(aux_loss)
             final_output_list.extend(list(output_tuple))
             return tuple(final_output_list)

        # Use MoeCausalLMOutputWithPast for consistency
        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=predictions, # Shape [B, 1, PredictionSize]
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits # Will be None if dense=True
        )

    def calc_prediction_loss(self, predictions, labels_target, masks_target, horizon_length):
        """ Calculates loss for predictions made from the single final state. """
        # ... (Implementation from previous response - comparing pred[:,0,h,:] with label_target[:,0,h,:]) ...
        if labels_target is None: return None
        batch_size, seq_len, _ = predictions.shape # seq_len is 1
        input_size = labels_target.shape[-1] # Din

        try:
            pred_reshaped = predictions.view(batch_size, seq_len, horizon_length, input_size) # [B, 1, H, Din]
        except Exception as e:
            logger.error(f"Loss Reshape Error: {e}. Pred: {predictions.shape}, H: {horizon_length}, Din: {input_size}")
            return None

        # Compare prediction[:, 0, h, :] with label_{N+1+h}
        # Requires labels_target to contain future labels [B, 1, H, Din]
        # If labels_target is just [B, 1, Din] (for N+1), only compute loss for h=0
        if labels_target.shape[1] != pred_reshaped.shape[1] : # Should both be 1
             logger.error(f"Label target seq len {labels_target.shape[1]} != Prediction seq len {pred_reshaped.shape[1]}")
             return None

        if labels_target.dim() == 3 and horizon_length > 1: # labels_target is [B, 1, Din], need [B, 1, H, Din]
             logger.warning_once(f"Multi-horizon loss (H={horizon_length}) requires labels for future steps. "
                              f"Only calculating loss on first step prediction vs provided label.")
             targets_reshaped = labels_target # [B, 1, Din]
             pred_reshaped = pred_reshaped[:, :, 0:1, :] # [B, 1, 1, Din]
             masks_reshaped = masks_target # [B, 1, Din] or None
             if masks_reshaped is not None: masks_reshaped = masks_reshaped.unsqueeze(2) # [B, 1, 1, Din]
             horizon_length = 1 # Force single step comparison
        elif labels_target.dim() == 3 and horizon_length == 1: # Labels [B, 1, Din], H=1
             targets_reshaped = labels_target # [B, 1, Din]
             pred_reshaped = pred_reshaped.squeeze(2) # [B, 1, Din]
             masks_reshaped = masks_target # [B, 1, Din] or None
        elif labels_target.dim() == 4 and labels_target.shape[2] == horizon_length: # Labels [B, 1, H, Din]
             targets_reshaped = labels_target
             masks_reshaped = masks_target # Assumed [B, 1, H, Din] or None
        else: # Shape mismatch
             logger.error(f"Label target shape {labels_target.shape} incompatible with horizon {horizon_length}")
             return None

        losses = self.loss_function(pred_reshaped, targets_reshaped) # [B, 1, H?, Din]

        if masks_reshaped is not None:
            if losses.shape != masks_reshaped.shape:
                 logger.error(f"Loss shape {losses.shape} != Mask shape {masks_reshaped.shape}")
                 loss_value = torch.mean(losses) # Fallback
            else:
                 masked_losses = losses * masks_reshaped.float()
                 mask_sum = masks_reshaped.sum()
                 loss_value = masked_losses.sum() / mask_sum if mask_sum > 0 else torch.tensor(0.0).to(losses)
        else:
            loss_value = torch.mean(losses)

        return loss_value