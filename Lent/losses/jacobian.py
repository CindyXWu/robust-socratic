"""Distillation with Jacobian penalty.
>>>'Knowledge Distillation with Jacobian Matching' (Srinivas, Flueret 2018).
"""
import torch.nn.functional as F
import torch
from losses.loss_common import *
from config_setup import MainConfig


def get_jacobian_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    inputs: torch.Tensor,
    config: MainConfig,
    input_dim: int,
    approx=True) -> torch.Tensor:
    """Eq 10, no hard targets used.
    
    Args:
        scores: logits of student [batch_size, num_classes].
        targets: logits of teacher [batch_size, num_classes].
        inputs: [batch_size, input_dim, input_dim, channels].
        approx: Whether to approximate full Jacobian with top-k.
    """
    # Number of classes to use for Jacobian approximation
    k = 20
    output_dim = config.dataset.output_size
    batch_size = config.dataloader.train_bs
    
    # If the number of classes is less than k, use exact Jacobian regardless of approx flag
    if not approx or output_dim <= k:
        t_jac = get_jacobian(targets, inputs, batch_size, input_dim, output_dim)
        s_jac = get_jacobian(scores, inputs, batch_size, input_dim, output_dim)
    else:
        i = torch.topk(targets, k, dim=1)[1]
        t_jac = get_approx_jacobian(targets, inputs, batch_size, output_dim, i)
        s_jac = get_approx_jacobian(scores, inputs, batch_size, output_dim, i)

    s_jac= torch.div(s_jac, torch.norm(s_jac, 2, dim=-1).unsqueeze(1))
    t_jac = torch.div(t_jac, torch.norm(t_jac, 2, dim=-1).unsqueeze(1))
    loss = torch.mean(torch.norm(t_jac-s_jac, 2, dim=1)) # Batchwise reduction
    
    return  loss


def get_approx_jacobian(
    output: torch.Tensor,
    x: torch.Tensor,
    batch_size: int,
    output_dim: int,
    top_k_idx: int) -> torch.Tensor:
    """Compute Jacobian for the top-k most probable classes instead of just the most probable class.
    Args:
        output: [batch_size, output_dim=num_classes]
        x: input with requres_grad() True [batch_size, input_dim]
        top_k_idx: indices of top-k classes. If None, calculate (assume for teacher). If not None, assume for calculating for student (needs to match teacher).
    """
    assert x.requires_grad
    jacobian = torch.zeros(batch_size, output_dim, x.numel() // batch_size, device=x.device)
    
    for idx in range(top_k_idx.shape[1]):
        grad_output = torch.zeros(batch_size, output_dim, device=x.device)
        grad_output.scatter_(1, top_k_idx[:, idx].unsqueeze(1), 1)
        # Need retain graph for further grad calcs in loss.backward()
        grad_input = torch.autograd.grad(output, x, grad_output, create_graph=True, retain_graph=True)[0]
        jacobian[:, top_k_idx[:, idx], :] = grad_input.view(batch_size, -1)
        
    return jacobian.view(batch_size, -1)


def get_jacobian(output: torch.Tensor, 
                 x: torch.Tensor,
                 batch_size: int,
                 input_dim: int,
                 output_dim:int) -> torch.Tensor:
    """Autograd method. Need to keep grads, so set create_graph to True."""
    assert output.requires_grad
    jacobian = torch.zeros(batch_size, output_dim, input_dim, device=x.device)
    
    for i in range(output_dim):
        grad_output = torch.zeros(batch_size, output_dim, device=x.device)
        grad_output[:, i] = 1
        grad_input = torch.autograd.grad(output, x, grad_output, create_graph=True, retain_graph=True)[0]
        jacobian[:, i, :] = grad_input.view(batch_size, input_dim)
        
    return jacobian.view(batch_size, -1)


# def get_approx_jacobian(output, x, batch_size, output_dim, i=None):
#     """Rather than computing Jacobian for all output classes, compute for most probable class.
#     Args:
#         output: [batch_size, output_dim=num_classes]
#         x: input with requres_grad() True [batch_size, input_dim]
#         i: index of top class
#     """
#     assert x.requires_grad
#     grad_output = torch.zeros(batch_size, output_dim, device=x.device)
#     # Needed to keep class same between teacher and student
#     # In this case, anchor teacher and compare student against top-1 class
#     if i == None:
#         i = torch.argmax(output, dim=1) # Index of most likely class
#     grad_output[:, i] = 1
#     # create_graph=True to keep grads
#     output.backward(grad_output, create_graph=True)
#     jacobian = x.grad.view(batch_size, -1)
#     return jacobian


# def get_jacobian(output, x, batch_size, input_dim, output_dim):
#     """In order to keep grads, need to set create_graph to True, but computation very slow."""
#     assert x.requires_grad
#     jacobian = torch.zeros(batch_size, output_dim, input_dim, device=x.device)
#     for i in range(output_dim):
#         grad_output = torch.zeros(batch_size, output_dim, device=x.device)
#         grad_output[:, i] = 1
#         # create_graph = True to keep grads
#         output.backward(grad_output, create_graph=True)
#         jacobian[:, i, :] = x.grad.view(batch_size, -1)
#         x.grad.zero_()
#     return jacobian.view(batch_size, -1) # Flatten
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
    distill_loss = get_distill_loss(scores, targets, T, loss_fn)

    s_jac = get_grads(student, inputs, batch_size, 3)
    t_jac = get_grads(teacher, inputs, batch_size, 3)
    s_norm = s_jac / torch.norm(s_jac, 2, dim=-1).unsqueeze(1) 
    t_norm = t_jac / torch.norm(t_jac, 2, dim=-1).unsqueeze(1)
    jacobian_loss = torch.norm(s_norm - t_norm, 2)
    jacobian_loss = torch.mean(jacobian_loss)   # Batchwise mean
    loss = (1 - alpha) * distill_loss + alpha * jacobian_loss
    return loss


def get_grads(model, inputs, batch_size, layer):
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
    layer.register_forward_hook(get_activations())  # Register hook
    model(inputs) # Forward pass
    
    map = features[0].view(batch_size, -1)
    i = torch.argmax(map, dim=1)
    gradient = torch.ones_like(map[:,i])
    gradient.requires_grad = True
    map[:,i].backward(gradient, retain_graph=True)
    grads = inputs.grad.view(batch_size, -1)

    # Clear grads and set requires_grad to True
    inputs.grad.zero_()
    model.zero_grad()
    grads.requires_grad = True

    return grads