import torch.nn as nn
import torch
from basic1_models import *

# Define loss functions
bceloss_fn = nn.BCELoss()
sigmoid = nn.Sigmoid()
kldivloss = nn.KLDivLoss(reduction='batchmean')

device = "cuda" if torch.cuda.is_available() else "cpu"

def feature_map_diff(student, teacher, x, layer, aggregate_chan):
    """Compute the difference between the feature maps of the student and teacher models.
    
    Args:
        student: torch model
        teacher: torch model
        x: torch tensor, input to the model
        layer: layer name of module
        aggregate_chan: bool, whether to aggregate the channels of the feature activation
    """
    # Load and extract models
    student = student()
    teacher = teacher()
    student_output = student._modules[layer](x)
    teacher_output = teacher._modules[layer](x)

    # Aggregate the channels of the feature activation using root squared absolute value of channels to create activation map
    if aggregate_chan:
        student_output = torch.sqrt(torch.sum(torch.abs(student_output)**2, dim=1)).view(-1)
        teacher_output = torch.sqrt(torch.sum(torch.abs(teacher_output)**2, dim=1)).view(-1)

    # Compute the difference between the feature maps
    diff = torch.norm( (student_output/torch.norm(student_output, p=2) - teacher_output/torch.norm(teacher_output, p=2) ), p=2)

    return diff

if __name__ == "__main__":
    model = small_linear_net()
    model.eval()