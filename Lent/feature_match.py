"""Contains utility code for extracting feature maps from models."""
import torch
from basic1_models import *
import torch.nn.functional as F
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGet

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
    assert s_map.requires_grad
    # Compute the difference between the feature maps
    loss = F.mse_loss(s_map, t_map, reduction='mean')
    assert loss.requires_grad
    return loss

# def feature_extractor(model, inputs, batch_size, layer):
#     """DEPRECTAED: cannot retain grad on output, so not useful for loss function.
#     Extract feature maps from model.
#     Args:
#         model: torch model, model to extract feature maps from
#         inputs: torch tensor, input to model
#         layer_num: int, layer number to extract feature maps from
#     """
#     def get_activations(batch_size):
#         def forward_hook(model, input, output):
#             assert input.requires_grad
#             assert output.requires_grad
#             output.retain_grad()
#             features.append(output.view(batch_size, -1))
#         return forward_hook
#     assert inputs.requires_grad
#     batch_size = inputs.shape[0]
#     features = []
#     # Length 2 list so get 0th index to access layers
#     layer.register_forward_hook(get_activations(batch_size))  # Register hook
#     model(inputs) # Forward pass
#     assert features[0].requires_grad
#     features[0].retain_grad()
#     return features[0]

def feature_extractor(model, inputs, return_layers, grad=True):
    """Extract feature maps from model.
    Args:
        model: torch model, model to extract feature maps from
        inputs: torch tensor, input to model
        layer_num: int, layer number to extract feature maps from
    """
    assert inputs.requires_grad
    mid_getter = MidGet(model, return_layers, True)
    mid_outputs, model_outputs = mid_getter(inputs)
    features =  list(mid_outputs.items())[0][1]
    if grad == False:
        features = features.detach()
        return features
    assert features.requires_grad
    return features