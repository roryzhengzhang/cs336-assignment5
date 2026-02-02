import torch

def compute_entropy(logits: torch.Tensor):
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).
    Args:
        logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
        containing unnormalized logits.
        Returns:
        torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token
        prediction.
    Note: we should use a numerically stable method (e.g., using logsumexp) to avoid overflow
    """
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -torch.sum(probs * log_probs, dim=-1)

    