import torch



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
        current_hidden_states = self[expert_idx](hidden_states)[None, top_x].reshape(-1, hidden_states.shape[-1]) * top_k_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    return final_hidden_states


def replace():
    return forward
