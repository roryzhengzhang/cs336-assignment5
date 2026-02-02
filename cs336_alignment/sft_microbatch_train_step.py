import torch
from cs336_alignment.masked_normalize import masked_normalize


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.

    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
            SFT policy being trained.
        response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
            prompt/padding.
        gradient_accumulation_steps Number of microbatches per optimizer step.
        normalize_constant The constant by which to divide the sum. It is fine to leave this as 1.0.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
                this so we can log it.
            metadata Dict with metadata from the underlying loss call, and any other statistics you
                might want to log.
    """
    # Compute negative log-likelihood (cross-entropy loss)
    # For SFT, we want to maximize log likelihood, which is equivalent to minimizing negative log likelihood
    negative_log_probs = -policy_log_probs

    # Sum over sequence dimension and normalize, but only for response tokens
    # This gives us the average negative log likelihood per normalize_constant tokens
    per_example_loss = masked_normalize(
        tensor=negative_log_probs,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=1  # Sum over sequence length dimension
    )

    # Average over batch to get the microbatch loss
    loss = per_example_loss.mean()

    # Adjust for gradient accumulation
    # Divide by gradient_accumulation_steps so that when gradients accumulate,
    # they average correctly
    adjusted_loss = loss / gradient_accumulation_steps

    # Backward pass on adjusted loss
    adjusted_loss.backward()

    # Prepare metadata
    metadata = {}

    # Return the adjusted loss for logging
    return adjusted_loss, metadata
