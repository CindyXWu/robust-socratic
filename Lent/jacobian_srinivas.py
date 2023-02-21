"""Distillation with Jacobian penalty.
>>>'Knowledge Distillation with Jacobian Matching' (Srinivas, Flueret 2018).
"""
import torch.nn.functional as F
import torch

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
    soft_pred = F.softmax(scores/T, dim=1)
    soft_targets = F.softmax(targets/T, dim=1)
    # Change these two lines of code depending on which Jacobian you want to use
    s_jac = get_approx_jacobian(scores, inputs, batch_size, input_dim, output_dim)
    t_jac = get_approx_jacobian(targets, inputs, batch_size, input_dim, output_dim)

    # Don't add batch size to norm calculation
    s_norm = s_jac / torch.norm(s_jac, 2, dim=-1).unsqueeze(1)
    t_norm = t_jac / torch.norm(t_jac, 2, dim=-1).unsqueeze(1)
    diff = s_norm - t_norm
    
    jacobian_loss = torch.norm(diff, 2, dim=1)**2
    # distill_loss = T**2 * loss_fn(soft_pred, soft_targets)
    # loss = (1-alpha) * distill_loss + alpha * jacobian_loss
    jacobian_loss.requires_grad = True
    return jacobian_loss

# def get_approx_jacobian(output, x, batch_size):
#     """Rather than computing Jacobian for all output classes, compute for most probable class.
#     Args:
#         output: [batch_size, output_dim=num_classes]
#         x: input with requres_grad() True - [batch_size, input_dim, input_dim, channels]
#     Returns:
#         jacobian: 1D torch tensor, vector of derivative of most probable class wrt input
#     """
#     output_dim = output.shape[1] # num of classes
#     input_dim = x.view(batch_size, -1).shape[1]
#     grad_output = torch.zeros(batch_size, output_dim, device=x.device)
#     # Index of most likely class, tensor of batch size shape
#     i = torch.argmax(output, dim=1) 
#     jacobian = torch.zeros(batch_size, input_dim, device=x.device)
#     grad_output.view(-1)[i] = 1
#     jacobian = torch.autograd.grad(output, x, grad_output)[0]
#     return jacobian.view(batch_size, -1)

def get_approx_jacobian(output, x, batch_size, input_dim, output_dim):
    """Rather than computing Jacobian for all output classes, compute for most probable class.
    Args:
        output: [batch_size, output_dim=num_classes]
        x: input with requres_grad() True [batch_size, input_dim]
    Returns:
        jacobian: 1D torch tensor, vector of derivative of most probable class wrt input
    """
    jacobian = torch.zeros(batch_size, input_dim, device=x.device)
    grad_output = torch.zeros(batch_size, output_dim, device=x.device)
    # Index of most likely class
    i = torch.argmax(output, dim=1)
    grad_output.view(-1)[i] = 1
    grad_input, = torch.autograd.grad(output, x, grad_output, retain_graph=True)
    jacobian = grad_input.view(batch_size, input_dim)
    return jacobian

def get_jacobian(output, x, batch_size):
    input_dim = x.shape(1)
    output_dim = output.shape(1)
    jacobian = torch.zeros(batch_size, output_dim, input_dim, device=x.device)
    for i in range(output_dim):
        grad_output = torch.zeros(batch_size, output_dim, device=x.device)
        grad_output[:, i] = 1
        grad_input, = torch.autograd.grad(output, x, grad_output, allow_unused=True)
        jacobian[:, i, :] = grad_input.view(batch_size, input_dim)
    return jacobian

#===================================================================================================

def jacobian_attention_loss(scores, targets, s_map, t_map, batch_size, T, alpha, loss_fn):
    """Eq 10, attention map Jacobian vector.
    J = dZ/dX where X is all inputs and Z is channel-wise square of attention maps.
    Args:
        s_map: torch tensor, student attention map [num_channels] with height and width squeezed out
        t_map: torch tensor, teacher attention map [num_channels] with height and width squeezed out
        T: float, temperature
        alpha: float, weight of the jacobian penalty
        loss_fn: base loss function - MSE, CE, KL
    """
    s_jac = get_activation_jacobian(scores, s_map, batch_size)
    print(s_jac.shape)
    t_jac = get_activation_jacobian(targets, t_map, batch_size)
    dims = [d for d in range(1, s_jac.dims+1)]    # Don't add batch dimension to norm calculation
    soft_pred = F.softmax(scores/T)
    soft_targets = F.softmax(targets/T)
    diff = s_jac/torch.norm(s_jac, 2, dim=dims) - t_jac/torch.norm(t_jac, 2, dim=dims)
    jacobian_loss = torch.norm(diff, 2)**2
    loss = (1-alpha) * T**2 * loss_fn(soft_pred, soft_targets) + alpha * jacobian_loss
    return loss

def get_activation_jacobian(x, feature_map, batch_size):
    """Rather than computing Jacobian for entire activation map, compute for largest activation.
    Args:
        output: [batch_size, output_dim=num_classes]
        x: input with requres_grad() True - [batch_size, input_dim, input_dim, channels]
    """
    output_dim = feature_map.numel() // batch_size
    # Largest pixel in feature map
    i = torch.argmax(feature_map.view(batch_size, -1).detach(), dim=1)
    grad_output = torch.zeros(batch_size, output_dim, device=x.device)
    grad_output[:, i] = 1
    # Compute each row of Jacobian at a time
    # layer_output.view(-1) is function to differentiate - flattened for autograd
    jacobian = torch.autograd.grad(feature_map.view(batch_size, -1), x, grad_output, retain_graph=True)[0]
    return jacobian

# def get_features(x, batch_size):
#     features = []
#     def get_activations():
#         def hook(model, input, output):
#             features.append(output)
#         return hook
    
# TODO: implement
def projection(s_map, t_map):
    """Project feature maps from two models onto the same dimension for comparison."""
    return