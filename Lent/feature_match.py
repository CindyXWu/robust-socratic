"""Contains utility code for extracting feature maps from models."""
import torch
from basic1_models import *

def feature_map_diff(student_output, teacher_output, aggregate_chan):
    """Compute the difference between the feature maps of the student and teacher models.
    
    Args:
        student_output: torch tensor, activation map of teacher model [batch_size, num_channels]
        teacher_output: torch tensor, output of teacher model [batch_size, num_channels]
        aggregate_chan: bool, whether to aggregate the channels of the feature activation
    """
    # Aggregate the channels of the feature activation using root squared absolute value of channels to create activation map
    if aggregate_chan:
        student_output = torch.sqrt(torch.sum(torch.abs(student_output)**2, dim=1))
        teacher_output = torch.sqrt(torch.sum(torch.abs(teacher_output)**2, dim=1))

    # Compute the difference between the feature maps
    diff = torch.norm( (student_output/torch.norm(student_output, p=2) - teacher_output/torch.norm(teacher_output, p=2) ), p=2)
    
    return diff

def feature_extractor(model, inputs, layer_num):
    """Extract feature maps from model.
    
    Args:
        model: torch model, model to extract feature maps from
        inputs: torch tensor, input to model
        layer_num: int, layer number to extract feature maps from
    """
    def get_activations():
        def forward_hook(model, input, output):
            features.append(output)
        return forward_hook
    
    def retain_grads(x, batch_size):
        def backward_hook(module, grad_input, grad_output):
            grad_input.retain_grads()
            features = module(x)
            output_dim = features.shape[-1]
            i = torch.argmax(features.view(batch_size, -1).detach(), dim=1)
            grad_output = torch.zeros(batch_size, output_dim, device=x.device)
            grad_output[:, i] = 1
            jac = torch.autograd.grad(outputs=features, inputs=x, grad_outputs=grad_output, retain_graph=True, create_graph=True, only_inputs=True)
            print(jac)
            grads.append(jac)
        return backward_hook
    
    batch_size = inputs.shape[0]
    grads = []
    features = []
    layer = list(model.children())[0][layer_num] # Length 2 list so get 0th index to access layers
    layer.register_forward_hook(get_activations())  # Register hook
    layer.register_backward_hook(retain_grads(batch_size, inputs))
    model(inputs) # Forward pass
    loss.backward()
    return features[0], grads[0]