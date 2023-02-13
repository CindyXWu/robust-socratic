import torch.nn as nn
import numpy as np
import torch

# Define loss functions
bceloss_fn = nn.BCELoss()
sigmoid = nn.Sigmoid()
kldivloss = nn.KLDivLoss(reduction='batchmean')

def activation_jac(my_model, x, layer):
    """Get the channel-wise activations (Eq 8) of a specific layer, using forward hook.
    Then compute the derivative wrt input and return as Jacobian matrix.
    
    Args:
        my_model: torch model
        x: torch tensor, input to the model
        layer: pytorch module, layer of the model
    """
    # Compute activations
    activations = {}
    model = my_model()
    x.requires_grad = True
    output = model(x)

    def get_hook(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    model.layer.register_forward_hook(get_hook(str(layer)))
    
    # Now compute derivatives of activation wrt input
    # ??? help

    return