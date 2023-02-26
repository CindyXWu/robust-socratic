"""Distillation with Jacobian penalty.
>>>'Knowledge Distillation with Jacobian Matching' (Srinivas, Flueret 2018).
"""
import torch.nn.functional as F
import torch
import wandb
from image_models import *

def jacobian_loss(scores, targets, inputs, T, alpha, batch_size, loss_fn, input_dim, output_dim):
    """Eq 10, no hard targets used.
    Args:
        scores: logits of student [batch_size, num_classes]
        targets: logits of teacher [batch_size, num_classes]
        inputs: [batch_size, input_dim, input_dim, channels]
        T: float, temperature
        alpha: float, weight of jacobian penalty
        loss_fn: for classical distill loss - MSE, BCE, KLDiv
    """
    # Only compute softmax for KL divergence loss
    if loss_fn == nn.KLDivLoss:
        soft_pred = F.softmax(scores/T, dim=1)
        soft_targets = F.softmax(targets/T, dim=1)
    else:
        soft_pred = scores/T
        soft_targets = targets/T
        
    s_jac = get_approx_jacobian(scores, inputs, batch_size, input_dim, output_dim)
    t_jac = get_approx_jacobian(targets, inputs, batch_size, input_dim, output_dim)

    # Don't add batch size to norm calculation
    s_norm = s_jac / torch.norm(s_jac, 2, dim=-1).unsqueeze(1)
    t_norm = t_jac / torch.norm(t_jac, 2, dim=-1).unsqueeze(1)
    jacobian_loss = torch.norm(s_norm - t_norm, 2, dim=1)**2
    jacobian_loss = torch.mean(jacobian_loss)   # Batchwise mean
    jacobian_loss.requires_grad = True
    distill_loss = T**2 * loss_fn(soft_pred, soft_targets)
    loss = (1-alpha) * distill_loss + alpha * jacobian_loss
    return  loss

def get_approx_jacobian(output, x, batch_size, input_dim, output_dim):
    """Rather than computing Jacobian for all output classes, compute for most probable class.
    Args:
        output: [batch_size, output_dim=num_classes]
        x: input with requres_grad() True [batch_size, input_dim]
    """
    grad_output = torch.zeros(batch_size, output_dim, device=x.device)
    # Index of most likely class
    i = torch.argmax(output, dim=1)
    grad_output[:, i] = 1
    jacobian = torch.autograd.grad(output, x, grad_output, retain_graph=True)[0]
    return jacobian.view(batch_size, input_dim)

# TODO: LAST DEBUG
def get_jacobian(output, x, batch_size, input_dim, output_dim):
    jacobian = torch.zeros(batch_size, output_dim, input_dim, device=x.device)
    for i in range(output_dim):
        grad_output = torch.zeros(batch_size, output_dim, device=x.device)
        grad_output[:, i] = 1
        grad_input, = torch.autograd.grad(output, x, grad_output, retain_graph=True)[0]
        jacobian[:, i, :] = grad_input.view(batch_size, input_dim)
    return jacobian

#===================================================================================================

def jacobian_attention_loss(student, teacher, scores, targets, inputs, batch_size, T, alpha, loss_fn):
    """Eq 10, attention map Jacobian vector.
    J = dZ/dX where X is all inputs and Z is channel-wise square of attention maps.
    Args:
        s_map: torch tensor, student attention map [num_channels] with height and width squeezed out
        t_map: torch tensor, teacher attention map [num_channels] with height and width squeezed out
        T: float, temperature
        alpha: float, weight of the jacobian penalty
        loss_fn: base loss function - MSE, CE, KL
    """
    s_jac = get_grads(student, inputs, batch_size, 3)
    t_jac = get_grads(teacher, inputs, batch_size, 3)

    if loss_fn == nn.KLDivLoss:
        soft_pred = F.softmax(scores/T, dim=1)
        soft_targets = F.softmax(targets/T, dim=1)
    else:
        soft_pred = scores/T
        soft_targets = targets/T

    s_norm = s_jac / torch.norm(s_jac, 2, dim=-1).unsqueeze(1) 
    t_norm = t_jac / torch.norm(t_jac, 2, dim=-1).unsqueeze(1)
    jacobian_loss = torch.norm(s_norm - t_norm, 2)
    jacobian_loss = torch.mean(jacobian_loss)   # Batchwise mean
    distill_loss = T**2 * loss_fn(soft_pred, soft_targets)
    loss = (1 - alpha) * distill_loss + alpha * jacobian_loss
    return loss

def get_grads(model, inputs, batch_size, layer_num):
    """Extract feature maps from model and call backward() to get input grad.
    Args:
        model: torch model, model to extract feature maps from
        inputs: torch tensor, input to model
        layer_num: int, layer number to extract feature maps from
    """
    inputs.requires_grad = True
    def get_activations():
        def forward_hook(model, input, output):
            features.append(output)
        return forward_hook

    features = []
    layer = list(model.children())[0][layer_num] # Length 2 list so get 0th index to access layers
    layer.register_forward_hook(get_activations())  # Register hook
    model(inputs) # Forward pass
    
    map = features[0].view(batch_size, -1)
    i = torch.argmax(map, dim=1)
    gradient = torch.ones_like(map[:,i])
    gradient.requires_grad = True
    map[:,i].backward(gradient, retain_graph=True)
    grads = inputs.grad.view(batch_size, -1)

    # Clear grads and set requires_grad to True
    inputs.grad.detach().zero_()
    model.zero_grad()
    grads.requires_grad = True

    return grads
    
# TODO: implement
def projection(s_map, t_map):
    """Project feature maps from two models onto the same dimension for comparison."""
    return