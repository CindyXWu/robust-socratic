"""
Distillation with Jacobian penalty.
>>>'Knowledge Distillation with Jacobian Matching' (Srinivas, Flueret 2018).
"""
import torch.nn as nn
import numpy as np
import torch
from tqdm import tqdm

# Define loss functions
bceloss_fn = nn.BCELoss()
sigmoid = nn.Sigmoid()
kldivloss = nn.KLDivLoss(reduction='batchmean')

def jacobian_loss(scores, targets, inputs, T, student, teacher, alpha, loss_fn):
    """Eq 10 adapted for input-output Jacobian matrix, not vector of output wrt largest pixel in attention map.

    See function below for adapation with attention maps.
    No hard targets used, purely distillation loss.
    Args:
        scores: torch tensor, output of the student model
        targets: torch tensor, output of the teacher model
        inputs: torch tensor, input to the model
        T: float, temperature
        s_jac: torch tensor, Jacobian matrix of the student model
        t_jac: torch tensor, Jacobian matrix of the teacher model
        alpha: float, weight of the jacobian penalty
        loss_fn: base loss function to be used for the input-output distillation loss - MSE, BCE, KLDiv
    """
    soft_pred = nn.functional.softmax(scores/T, dim=1)
    soft_targets = nn.functional.softmax(targets/T, dim=1)
    # Change these two lines of code depending on which Jacobian you want to use
    s_jac = get_approx_jacobian(student, inputs)
    t_jac = get_approx_jacobian(teacher, inputs)
    diff = s_jac/torch.norm(s_jac, 2) - t_jac/torch.norm(t_jac, 2)
    jacobian_loss = torch.norm(diff, 2)**2
    loss = (1-alpha) * T**2 * loss_fn(soft_pred, soft_targets) + alpha * jacobian_loss
    return loss

# TODO: DEBUG
def jacobian_attention_loss(scores, targets, inputs, T, student, s_layer, teacher, t_name, t_layer, alpha, loss_fn):
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
    t_jac = get_activation_jacobian(teacher, inputs, t_layer, False, t_name)
    soft_pred = sigmoid(scores/T)
    soft_targets = sigmoid(targets/T)
    diff = s_jac/torch.norm(s_jac, 2) - t_jac/torch.norm(t_jac, 2)
    jacobian_loss = torch.norm(diff, 2)**2
    loss = (1-alpha) * T**2 * loss_fn(soft_pred, soft_targets) + alpha * jacobian_loss
    return loss

# DEBUGGED
def get_jacobian(model, x):
    """Get Jacobian matrix of the model output wrt input.
    
    Args:
        model: torch model
        x: torch tensor, input to the model
    Returns:
        jacobian: torch tensor, Jacobian matrix of the model output wrt input
    """
    batch_size = x.size(0)
    input_dim = x.numel() // batch_size
    x = x.requires_grad_()
    y = model(x)
    output_dim = y.numel() // batch_size
    jacobian = torch.zeros(batch_size, output_dim, input_dim, device=x.device)
    for i in range(output_dim):
        grad_output = torch.zeros(batch_size, output_dim, device=x.device)
        grad_output[:, i] = 1
        grad_input, = torch.autograd.grad(y, x, grad_output, retain_graph=True)
        jacobian[:, i, :] = grad_input.view(batch_size, input_dim)
    return jacobian

# TODO: DEBUG
def get_activation_jacobian(model, x, layer, dynamic, load_name=None, aggregate_chan=True):
    """Get Jacobian matrix of activation to input for a saved model or model during training.
    
    Args:
        model: torch model
        layer: layer name of module
        x: torch tensor, input to the model
        dynamic: bool, whether to compute for saved model or model during training
        load_name: file name of saved model which contains state dict
        aggregate_chan: bool, whether to aggregate the channels of the feature activation
    Returns:
        jacobian: torch tensor, Jacobian matrix of activation to input
    """
    batch_size = x.size(0)
    model = model()
    if not dynamic:
        # Load saved model
        checkpoint = torch.load(load_name)
        model.load_state_dict(checkpoint)
    layer_output = model._modules[layer](x)

    # Aggregate the channels of the feature activation using root squared absolute value of channels to create activation map
    if aggregate_chan:
        layer_output = torch.sqrt(torch.sum(torch.abs(layer_output)**2, dim=1))

    # Compute the Jacobian matrix
    jacobian = torch.zeros(batch_size, layer_output.numel(), x.numel())
    for i in tqdm(range(layer_output.numel())):
        grad_output = torch.zeros(batch_size, layer_output.size())
        grad_output.view(-1)[:, i] = 1
        # Compute each row of Jacobian at a time
        # layer_output.view(-1) is function to differentiate - flattened to 1D tensor for compatibility with PyTorch's gradient computation
        jacobian[:, i, :] = torch.autograd.grad(layer_output.view(-1), x, grad_outputs=grad_output.view(-1), retain_graph=True)[0].view(-1)
    
    return jacobian

# DEBUGGED
def get_approx_jacobian(model, x):
    """Rather than computing Jacobian for all output classes, compute for most probable class.
    
    Required due to computational constraints for Jacobian computation.
    Args:
        model: torch model
        x: torch tensor, input to the model
    Returns:
        jacobian: 1D torch tensor, vector of derivative of most probable class wrt input
    """
    batch_size = x.size(0)
    # numel returns the number of elements in the tensor
    input_dim = x.numel() // batch_size
    x = x.requires_grad_()
    y = model(x)
    jacobian = torch.zeros(batch_size, input_dim, device=x.device)
    output_dim = y.numel() // batch_size
    grad_output = torch.zeros(batch_size, output_dim, device=x.device)
    # Index of most likely class
    i = torch.argmax(y, dim=1)
    grad_output[:, i] = 1
    grad_input, = torch.autograd.grad(y, x, grad_output, retain_graph=True)
    jacobian = grad_input.view(batch_size, input_dim)
    return jacobian