"""Contains utility code for extracting feature maps from models."""
import torch
from basic1_models import *

def feature_map_diff(s_map, t_map, aggregate_chan):
    """Compute the difference between the feature maps of the student and teacher models.
    Args:
        s_map: torch tensor, activation map of teacher model [batch_size, num_channels]
        t_map: torch tensor, output of teacher model [batch_size, num_channels]
        aggregate_chan: bool, whether to aggregate the channels of the feature activation
    """
    # Aggregate the channels of the feature activation using root squared absolute value of channels to create activation map
    if aggregate_chan:
        s_map = torch.sqrt(torch.sum(torch.abs(s_map)**2, dim=1))
        t_map = torch.sqrt(torch.sum(torch.abs(t_map)**2, dim=1))
    # Compute the difference between the feature maps
    diff = torch.mean(s_map/torch.norm(s_map, p=2, dim=-1).unsqueeze(-1) - t_map/torch.norm(t_map, p=2, dim=-1).unsqueeze(-1) )
    return diff

def feature_extractor(model, inputs, batch_size, layer_num):
    """Extract feature maps from model.
    Args:
        model: torch model, model to extract feature maps from
        inputs: torch tensor, input to model
        layer_num: int, layer number to extract feature maps from
    """
    def get_activations(batch_size):
        def forward_hook(model, input, output):
            output = output.view(batch_size, -1)
            output = torch.autograd.Variable(output.data, requires_grad=True)
            output.retain_grad()
            features.append(output)
        return forward_hook

    batch_size = inputs.shape[0]
    features = []
    layer = list(model.children())[0][layer_num] # Length 2 list so get 0th index to access layers
    layer.register_forward_hook(get_activations(batch_size))  # Register hook
    model(inputs) # Forward pass
    return features[0]
    