# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3 model with confidence head for FSDP training."""

import torch
import torch.nn as nn
from transformers import Qwen3ForCausalLM

from .dense_common import CausalLMOutputWithConfidence


class Qwen3ForCausalLMWithConfidence(Qwen3ForCausalLM):
    """
    Qwen3 model with dual heads: traditional LM head and confidence regression head.
    This version is designed for FSDP training using transformers models.
    """

    def __init__(self, config):
        super().__init__(config)

        # Initialize confidence regression head (outputs scalar confidence)
        self.confidence_head = nn.Linear(in_features=config.hidden_size, out_features=1, bias=False)

        # Initialize weights for confidence head
        nn.init.normal_(self.confidence_head.weight, mean=0.0, std=config.initializer_range)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        """
        Forward pass with dual heads (LM head + confidence head).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get base model outputs
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = transformer_outputs[0]

        # Standard LM head forward
        logits = self.lm_head(hidden_states)

        # Confidence head forward - outputs per-token confidence scores
        confidence_logits = self.confidence_head(hidden_states)  # (batch_size, seq_len, 1)
        confidence_scores = torch.sigmoid(confidence_logits.squeeze(-1)).to(torch.float32)  # (batch_size, seq_len)
        last_confidence_scores = confidence_scores[:, -1] 
        loss = None
        if labels is not None:
            # Compute standard language modeling loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # Return custom output with confidence scores
        return CausalLMOutputWithConfidence(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            confidence_scores=last_confidence_scores,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        """
        Prepare inputs for generation (inference mode).
        """
        return super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
