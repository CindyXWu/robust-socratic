"""Shared utility methods for losses."""
import torch.nn.functional as F
import torch
from basic1_models import *

device = "cuda" if torch.cuda.is_available() else "cpu"

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

def get_activation_jacobian(model, x, layer, dynamic, load_name=None, aggregate_chan=True):
    """Get Jacobian matrix of activation to input for a saved model or model during training.
    
    Args:
        model: torch model
        layer: layer name of module
        x: torch tensor, input to the model
        dynamic: bool, whether to compute for saved model or model during training
        load_name: file name of saved model which contains state dictaggregate_chan: bool, whether to aggregate the channels of the feature activation
    """
    model = model()
    if not dynamic:
        # Load saved model
        checkpoint = torch.load(load_name, map_location=device)
        model.load_state_dict(checkpoint)
    layer_output = model._modules[layer](x)

    # Aggregate the channels of the feature activation using root squared absolute value of channels to create activation map
    if aggregate_chan:
        layer_output = torch.sqrt(torch.sum(torch.abs(layer_output)**2, dim=1))

    # Compute the Jacobian matrix
    J = torch.zeros(layer_output.numel(), x.numel())
    for i in range(layer_output.numel()):
        grad_output = torch.zeros(layer_output.size())
        grad_output.view(-1)[i] = 1
        # Compute each row of Jacobian at a time
        # layer_output.view(-1) is function to differentiate - flattened to 1D tensor for compatibility with PyTorch's gradient computation
        J[i, :] = torch.autograd.grad(layer_output.view(-1), x, grad_outputs=grad_output.view(-1), retain_graph=True)[0].view(-1)
    
    return J