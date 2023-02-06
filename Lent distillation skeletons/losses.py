import torch.nn as nn
import numpy as np

# Define loss functions
bceloss_fn = nn.BCELoss()
sigmoid = nn.Sigmoid()
kldivloss_fn = nn.KLDivLoss(reduction='batchmean')

def jacobian_loss(scores, targets, T, s_jac, t_jac, loss_fn):
    """Custom distillation loss (all soft target) with Jacobian penalty.
    :param scores: torch tensor, output of the student model
    :param targets: torch tensor, output of the teacher model
    :param T: float, temperature
    :param s_jac: torch tensor, Jacobian matrix of the student model
    :param t_jac: torch tensor, Jacobian matrix of the teacher model
    :param loss_fn: torch loss function, loss function to use
    """
    soft_pred = sigmoid(scores/T)
    soft_targets = sigmoid(targets/T)
    jacobian_term = s_jac @ t_jac/(np.linalg.norm(s_jac)*np.linalg.norm(t_jac))
    loss = T**2 * loss_fn(soft_pred, soft_targets) + jacobian_term
    return loss

def jacobian_loss_mix(scores, targets, labels, T, alpha, jacobian, loss_fn):
    """Custom distillation loss (weighted average of soft and hard target) with Jacobian penalty."""
    soft_pred = sigmoid(scores/T)
    soft_targets = sigmoid(targets/T)
    loss = T**2 * (alpha * loss_fn(soft_pred, soft_targets) + (1-alpha) * loss_fn(scores, labels))
    return loss