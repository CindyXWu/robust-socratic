import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Callable

from config_setup import DistillConfig, LossType

# Needs to be global scope
kl_loss = nn.KLDivLoss(log_target=False, reduction='batchmean')
ce_loss = nn.CrossEntropyLoss(reduction='mean')
mse_loss = nn.MSELoss(reduction='mean')


def base_distill_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    loss_type: LossType,
    temp: float) -> torch.Tensor:
    """
    Args:
        scores: Student logits.
        targets: Teacher logits.
    """
    # Only compute softmax for KL divergence loss
    if loss_type == LossType.KL:
        return kl_loss(
            F.log_softmax(scores/temp, dim=-1),
            F.softmax(targets/temp, dim=-1)
            )
    elif loss_type == LossType.MSE:
        return mse_loss(
            F.softmax(scores/temp, dim=-1),
            F.softmax(targets/temp, dim=-1)
        )
    else:
        raise ValueError("Loss function not supported.")
    

def mixup_loss(loss_fn: Callable, logits: torch.Tensor, label: torch.Tensor, label_2: torch.Tensor, lam: float):
    """Update existing loss function for mixup.
    This function is expected to only be relevant with hard label distillation.
    Loss function should already have loss type and temperature set with functools partial.
    """
    return lam*loss_fn(logits, label) + (1-lam)*loss_fn(logits, label_2)


@torch.no_grad()
def get_distill_test_loss(
    teacher: nn.Module, 
    student: nn.Module, 
    test_loader: DataLoader, 
    config: DistillConfig,
    num_eval_batches: int,
    device: torch.device = torch.device("cuda")) -> float:
    
    teacher.eval()
    student.eval()
    total_loss = 0.0
    
    for i, (inputs, _) in test_loader:
        if num_eval_batches != 0 and i >= num_eval_batches:
            break
        inputs = inputs.to(device)
        scores, targets = student(inputs), teacher(inputs)
        loss = base_distill_loss(
            scores=scores, 
            targets=targets, 
            loss_type=config.base_distill_loss_type,
            temp=config.dist_temp
        )
        total_loss += loss.item()

    avg_test_loss = total_loss / num_eval_batches
    
    return avg_test_loss


@torch.no_grad()
def get_teacher_test_loss(
    model: nn.Module, 
    test_loader: DataLoader, 
    num_eval_batches: int,
    device: torch.device = torch.device("cuda")) -> float:
    
    model.eval()
    total_loss = 0.0
    
    for i, (inputs, labels) in test_loader:
        if num_eval_batches != 0 and i >= num_eval_batches:
            break
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = ce_loss(outputs, labels)
        total_loss += loss.item()
    avg_test_loss = total_loss / num_eval_batches
    
    return avg_test_loss
