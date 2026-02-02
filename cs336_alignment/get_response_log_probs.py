import torch
from transformers import PreTrainedModel

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Get per-token conditional log-probabilities from a causal language model.

    Args:
        model: PreTrainedModel HuggingFace model for scoring
        input_ids: shape (batch_size, sequence_length), concatenated prompt + response tokens
        labels: shape (batch_size, sequence_length), labels (-100 for ignored positions)
        return_token_entropy: If True, also return per-token entropy

    Returns:
        dict with:
            "log_probs": shape (batch_size, sequence_length), conditional log p(x_t | x_{<t})
            "token_entropy": optional, shape (batch_size, sequence_length), per-token entropy
    """
    # Get logits from model
    outputs = model(input_ids)
    logits = outputs.logits  # (batch_size, sequence_length, vocab_size)

    # NOTE: input_ids and labels are already shifted in tokenize_prompt_and_output:
    # input_ids = [t0, t1, t2, t3] (without final token)
    # labels = [t1, t2, t3, t4] (without first token)
    # So logits[:, i] already predicts labels[:, i] - no additional shift needed!

    # Compute log probabilities for all vocabulary positions
    log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)

    # Gather log probs for the specific tokens in labels
    # Handle -100 labels by clamping to 0 for gathering, then masking
    labels_for_gather = labels.clamp(min=0).unsqueeze(-1)  # (batch_size, seq_len, 1)
    gathered_log_probs = log_probs_all.gather(dim=-1, index=labels_for_gather).squeeze(-1)

    # Mask positions where labels were -100 (set to 0)
    mask = labels == -100
    gathered_log_probs = gathered_log_probs.masked_fill(mask, 0.0)

    result = {"log_probs": gathered_log_probs}

    # Optionally compute entropy
    if return_token_entropy:
        from cs336_alignment.compute_entropy import compute_entropy
        entropy = compute_entropy(logits)  # (batch_size, seq_len)
        result["token_entropy"] = entropy

    return result