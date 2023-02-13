"""
Distillation with Jacobian penalty.
>>>'Knowledge Distillation with Jacobian Matching' (Srinivas, Flueret 2018).
"""
import torch.nn as nn
import numpy as np
import torch
from feature_match import activation_jac

# Define loss functions
bceloss_fn = nn.BCELoss()
sigmoid = nn.Sigmoid()
kldivloss = nn.KLDivLoss(reduction='batchmean')

def jacobian(scores, targets, T, s_jac, t_jac, alpha, loss_fn):
    """Eq 10 adapted for input-output Jacobian matrix, not vector of output wrt largest pixel in attention map.

    See function below for adapation with attention maps.
    No hard targets used, purely distillation loss.
    Args:
        scores: torch tensor, output of the student model
        targets: torch tensor, output of the teacher model
        T: float, temperature
        s_jac: torch tensor, Jacobian matrix of the student model
        t_jac: torch tensor, Jacobian matrix of the teacher model
        alpha: float, weight of the jacobian penalty
        loss_fn: base loss function - MSE, BCE, KLDiv
    """
    soft_pred = sigmoid(scores/T)
    soft_targets = sigmoid(targets/T)
    diff = s_jac/np.linalg.norm(s_jac, ord=2) - t_jac/np.linalg.norm(t_jac, ord=2)
    jacobian_loss = np.linalg.norm(diff, ord=2)**2
    loss = (1-alpha) * T**2 * loss_fn(soft_pred, soft_targets) + alpha * jacobian_loss
    return loss

def jacobian_attention(inputs, scores, targets, T, student, s_layer, teacher, t_layer, alpha, loss_fn):
    """Eq 10, attention map Jacobian vector.
    
    J = dZ/dX where X is all inputs and Z is channel-wise square of attention maps.
    No hard targets used, purely distillation loss.
    Args:
        inputs: torch tensor, input to the model
        scores: torch tensor, output of the student model
        targets: torch tensor, output of the teacher model
        T: float, temperature
        student: torch model
        layer: pytorch module, layer of the model
        teacher: torch model
        alpha: float, weight of the jacobian penalty
        loss_fn: base loss function - MSE, BCE, KLDiv
    """
    s_jac = activation_jac(student, inputs, s_layer)
    t_jac = activation_jac(teacher, inputs, t_layer)
    soft_pred = sigmoid(scores/T)
    soft_targets = sigmoid(targets/T)
    diff = s_jac/np.linalg.norm(s_jac, ord=2) - t_jac/np.linalg.norm(t_jac, ord=2)
    jacobian_loss = np.linalg.norm(diff, ord=2)**2
    loss = (1-alpha) * T**2 * loss_fn(soft_pred, soft_targets) + alpha * jacobian_loss
    return loss

# Code to be used in trainig for obtaining the correct derivatives
# Get gradient of activation wrt input
def get_jacobian(model, x):
    """Get Jacobian matrix of the model output wrt input.
    
    Args:
        model: torch model
        x: torch tensor, input to the model
    """
    x.requires_grad = True
    y = model(x)
    y.backward(torch.ones_like(y))
    # Or can do torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
    return x.grad