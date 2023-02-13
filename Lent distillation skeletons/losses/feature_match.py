import torch.nn as nn
import numpy as np
import torch

# Define loss functions
bceloss_fn = nn.BCELoss()
sigmoid = nn.Sigmoid()
kldivloss = nn.KLDivLoss(reduction='batchmean')

device = "cuda" if torch.cuda.is_available() else "cpu"

def activation_jac(my_model, x, layer):
    """Get the channel-wise activations (Eq 8) of a specific layer, using forward hook.
    Then compute the derivative wrt input and return as Jacobian matrix.
    
    Args:
        my_model: torch model
        x: torch tensor, input to the model
        layer: pytorch module, layer of the model
    """
    model = my_model()
    x.requires_grad = True
    output = model(x)
    activations = []
    def get_hook(name):
        def hook(model, input, output):
            activations.append(output.detach())
        return hook
    
    model.layer.register_forward_hook(get_hook(str(layer)))
    
    # Now compute derivatives of activation wrt input
    gradients = torch.autograd.grad(outputs=activations, inputs=x,
                          grad_outputs=torch.ones(activations.size()).to(device),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]

    return