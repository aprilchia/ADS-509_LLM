import torch
import torch.nn.functional as F

def get_probs(logits):
    convert_tensor = torch.tensor(logits)
    probs = F.softmax(convert_tensor, dim=1)
    return probs