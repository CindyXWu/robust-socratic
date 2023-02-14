"""
Distillation with Jacobian penalty.
>>>'Knowledge Distillation with Jacobian Matching' (Srinivas, Flueret 2018).
"""
import torch.nn as nn
import numpy as np
import torch
from feature_match import activation_jac
from utils import *

# Define loss functions
bceloss_fn = nn.BCELoss()
sigmoid = nn.Sigmoid()
kldivloss = nn.KLDivLoss(reduction='batchmean')

def jacobian_loss(scores, targets, inputs, T, student, teacher, t_jac, alpha, loss_fn):
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
    s_jac = get_jacobian(student, inputs)
    t_jac = get_jacobian(teacher, inputs)
    diff = s_jac/np.linalg.norm(s_jac, ord=2) - t_jac/np.linalg.norm(t_jac, ord=2)
    jacobian_loss = np.linalg.norm(diff, ord=2)**2
    loss = (1-alpha) * T**2 * loss_fn(soft_pred, soft_targets) + alpha * jacobian_loss
    return loss

def jacobian_attention_loss(inputs, scores, targets, T, student, s_layer, teacher, t_name, t_layer, alpha, loss_fn):
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
        t_name: str, saved name of teacher model
        alpha: float, weight of the jacobian penalty
        loss_fn: base loss function - MSE, BCE, KLDiv
    """
    s_jac = get_activation_jacobian(student, inputs, s_layer, True)
    t_jac = get_activation_jacobian(student, inputs, s_layer, False, t_name)
    soft_pred = sigmoid(scores/T)
    soft_targets = sigmoid(targets/T)
    diff = s_jac/np.linalg.norm(s_jac, ord=2) - t_jac/np.linalg.norm(t_jac, ord=2)
    jacobian_loss = np.linalg.norm(diff, ord=2)**2
    loss = (1-alpha) * T**2 * loss_fn(soft_pred, soft_targets) + alpha * jacobian_loss
    return loss

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