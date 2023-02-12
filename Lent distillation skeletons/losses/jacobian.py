import torch.nn as nn
import numpy as np

# Define loss functions
bceloss_fn = nn.BCELoss()
sigmoid = nn.Sigmoid()
kldivloss = nn.KLDivLoss(reduction='batchmean')

def jacobian(scores, targets, T, s_jac, t_jac, loss_fn):
    """Distillation loss (all soft target) with Jacobian penalty.
    Args:
        scores: torch tensor, output of the student model
        targets: torch tensor, output of the teacher model
        T: float, temperature
        s_jac: torch tensor, Jacobian matrix of the student model
        t_jac: torch tensor, Jacobian matrix of the teacher model
        loss_fn: base loss function - MSE, BCE, KLDiv
    """
    soft_pred = sigmoid(scores/T)
    soft_targets = sigmoid(targets/T)
    jacobian_term = s_jac @ t_jac/(np.linalg.norm(s_jac)*np.linalg.norm(t_jac))
    loss = T**2 * loss_fn(soft_pred, soft_targets) + jacobian_term
    return loss

def jacobian_mix(scores, targets, labels, T, alpha, jacobian, loss_fn):
    """Distillation loss (weighted average of soft and hard target) with Jacobian penalty."""
    soft_pred = sigmoid(scores/T)
    soft_targets = sigmoid(targets/T)
    loss = T**2 * (alpha * loss_fn(soft_pred, soft_targets) + (1-alpha) * loss_fn(scores, labels))
    return loss
    