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
        def hook(model, input, output):
            features.append(output)
        return hook
    features = []
    layer = list(model.children())[0][layer_num] # Length 2 list so get 0th index to access layers
    layer.register_forward_hook(get_activations())  # Register hook
    model(inputs) # Foeward pass
    return features[0]

# if __name__ == "__main__":
#     def feature_extractor(model, inputs, layer_num):
#         layer = list(model.children())[0][layer_num] # Length 2 list so get 0th index to access layers
#         activations = layer(inputs)
#         grads = torch.autograd.grad(activations, x, )