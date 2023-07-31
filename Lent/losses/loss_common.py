import torch
import torch.nn as nn
import torch.nn.functional as F


kl_loss = nn.KLDivLoss(reduction='batchmean')
ce_loss = nn.CrossEntropyLoss(reduction='mean')
mse_loss = nn.MSELoss(reduction='mean')


# def base_distill_loss(
#     scores: torch.Tensor,
#     targets: torch.Tensor,
#     loss_type: LossType,
#     temp: float) -> torch.Tensor:
#     """Distillation loss function."""
#     # Only compute softmax for KL divergence loss
#     if isinstance(loss_type, LossType.KL):
#         soft_scores = F.log_softmax(scores/temp)
#         soft_targets = F.softmax(targets/temp)
#     elif isinstance(loss_type, LossType.MSE):
#         soft_scores = F.softmax(scores/temp)
#         soft_targets = F.softmax(targets/temp)
#     else:
#         raise ValueError("Loss function not supported.")
#     distill_loss = (temp**2)*loss_type(soft_scores, soft_targets)
#     return distill_loss


# Only difference between these two functions is dim=1 setting in the function below
def base_distill_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    loss_type,
    temp: float) -> torch.Tensor:
    """Distillation loss function."""
    # Only compute softmax for KL divergence loss
    if isinstance(loss_type, LossType.KL):
        soft_scores = F.log_softmax(scores/temp, dim=1)
        soft_targets = F.softmax(targets/temp, dim=1)
    elif isinstance(loss_type, LossType.MSE):
        soft_scores = F.softmax(scores/temp, dim=1)
        soft_targets = F.softmax(targets/temp, dim=1)
    else:
        raise ValueError("Loss function not supported.")
    distill_loss = (temp**2)*loss_type(soft_scores, soft_targets)
    return distill_loss