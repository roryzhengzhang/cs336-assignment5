import torch

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Sum over a dimension and normalize by a constant, considering only those elements where mask == 1.

    Args:
        tensor: torch.Tensor The tensor to sum and normalize.
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the sum.
        normalize_constant: float the constant to divide by for normalization, e.g. sequence length.
        dim: int | None the dimension to sum along before normalization. If None, sum over all dimensions.

    Returns:
        torch.Tensor the normalized sum, where masked elements (mask == 0) don't contribute to the sum.
    """
    # Apply mask: zero out elements where mask == 0
    masked_tensor = tensor * mask

    # Sum along the specified dimension (or all if dim is None)
    summed = torch.sum(masked_tensor, dim=dim)

    # Normalize by the constant
    # The constant makes the metric length-invariant and interpretable as an average,
    normalized = summed / normalize_constant

    return normalized