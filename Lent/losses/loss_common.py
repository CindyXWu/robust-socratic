import torch
import torch.nn as nn
import torch.nn.functional as F
from config_setup import LossType


# Needs to be global scope
kl_loss = nn.KLDivLoss(log_target=False, reduction='batchmean')
ce_loss = nn.CrossEntropyLoss(reduction='mean')
mse_loss = nn.MSELoss(reduction='mean')


def base_distill_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    loss_type: LossType,
    temp: float) -> torch.Tensor:
    """
    Args:
        scores: Student logits.
        targets: Teacher logits.
    """
    # Only compute softmax for KL divergence loss
    if loss_type == LossType.KL:
        return kl_loss(
            F.log_softmax(scores/temp, dim=-1),
            F.softmax(targets/temp, dim=-1)
            )
    elif loss_type == LossType.MSE:
        return mse_loss(
            F.softmax(scores/temp, dim=-1),
            F.softmax(targets/temp, dim=-1)
        )
    else:
        raise ValueError("Loss function not supported.")