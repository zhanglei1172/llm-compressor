import torch

from llmcompressor.utils.dev import skip_weights_initialize



class Qwen3OmniMoeThinkerTextExperts(torch.nn.ModuleList):
    def __init__(self, config, original, calibrate_all_experts):
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeThinkerTextMLP,
        )
        self.calibrate_all_experts = calibrate_all_experts
        self.num_experts = original.num_experts
        with skip_weights_initialize():
            super().__init__(
                [Qwen3OmniMoeThinkerTextMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
            )

        for i in range(self.num_experts):
            gate_proj = original[i].gate_proj.weight.data
            up_proj = original[i].up_proj.weight.data
            down = original[i].down_proj.weight.data

            self[i].gate_proj.weight.data = gate_proj.t().clone().contiguous()
            self[i].up_proj.weight.data = up_proj.t().clone().contiguous()
            self[i].down_proj.weight.data = down.t().clone().contiguous()

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size * sequence_length, hidden_dim)
            selected_experts: (batch_size * sequence_length, top_k)
            routing_weights: (batch_size * sequence_length, top_k)
        Returns:
            (batch_size * sequence_length, hidden_dim)
        """
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)

        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            if self.calibrate_all_experts:
                current_hidden_states = self[expert_idx](hidden_states)[None, top_x].reshape(-1, hidden_states.shape[-1]) * top_k_weights[top_x, idx, None]
            else:
                current_state = hidden_states[None, top_x].reshape(-1, hidden_states.shape[-1])
                current_hidden_states = self[expert_idx](current_state) * top_k_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        return final_hidden_states


def replace(config, module, calibrate_all_experts):
    return Qwen3OmniMoeThinkerTextExperts(
        config=config.get_text_config(),
        original=module,
        calibrate_all_experts=calibrate_all_experts,
    )
