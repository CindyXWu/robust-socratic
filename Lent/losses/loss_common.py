import torch
import torch.nn as nn
import torch.nn.functional as F
from config_setup import LossType
from typing import Callable


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
    

def mixup_loss(loss_fn: Callable, logits: torch.Tensor, label: torch.Tensor, label_2: torch.Tensor, lam: float):
    """Update existing loss function for mixup.
    This function is expected to only be relevant with hard label distillation.
    Loss function should already have loss type and temperature set with functools partial.
    """
    return lam*loss_fn(logits, label) + (1-lam)*loss_fn(logits, label_2)