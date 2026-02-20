import torch
import torch.nn.functional as F

def get_probs(logits):
    """Convert raw model logits to a probability distribution via softmax.

    Args:
        logits: Raw output scores from the model (list, array, or tensor) with
            shape (batch_size, num_classes).

    Returns:
        torch.Tensor: Probability scores with shape (batch_size, num_classes),
            where each row sums to 1.0.
    """
    convert_tensor = torch.tensor(logits)
    probs = F.softmax(convert_tensor, dim=1)
    return probs