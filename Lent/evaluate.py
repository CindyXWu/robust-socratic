import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Callable

from losses.loss_common import *


@torch.no_grad()
def evaluate(model: nn.Module, 
             dataloader: DataLoader, 
             batch_size: int, 
             num_eval_batches: int, 
             device: torch.device = torch.device("cuda")) -> float:
    """Accuracy for num_eval_batches batches."""
    model.eval()
    acc = 0.0
    
    for i, (features, labels) in enumerate(dataloader):
        labels = labels.to(device)
        features = features.to(device)
        scores = model(features)
        _, pred = torch.max(scores, 1)
        acc += torch.sum(torch.eq(pred, labels)).item()
        if num_eval_batches != 0 and i >= num_eval_batches:
            break
        
    model.train()
    
    # Avg acc - frac data points correctly classified
    return (acc*100 / ((i+1)*batch_size))


@torch.no_grad()
def counterfactual_evaluate(teacher: nn.Module, 
                     student: nn.Module, 
                     dataloader: DataLoader, 
                     batch_size: int, 
                     num_eval_batches: int, 
                     device: torch.device = torch.device("cuda")) -> float:
    """Student test accuracy, T-S KL and T-S top-1 accuracy for num_eval_batches batches."""
    acc, KL, top_1 = 0.0, 0.0, 0.0
    student.eval()
    
    for i, (features, labels) in enumerate(dataloader):
        if num_eval_batches != 0 and i >= num_eval_batches:
            break
        
        labels, features = labels.to(device), features.to(device)
        targets, scores = teacher(features), student(features)
        s_pred, t_pred = torch.argmax(scores, dim=1), torch.argmax(targets, dim=1)
        
        acc += torch.sum(torch.eq(s_pred, labels)).item() # Total accurate samples in batch
        KL += kl_loss(F.log_softmax(scores, dim=1), F.softmax(targets, dim=1)) # Batchwise mean KL
        top_1 += torch.eq(s_pred, t_pred).float().mean() # Batchwise mean top-1 accuracy
        
    avg_acc = acc*100/(i*batch_size)
    avg_KL = KL/i
    avg_top_1 = top_1*100/i
    
    student.train()
    
    return avg_acc, avg_KL, avg_top_1


def mixup_accuracy(metric_fn: Callable, logits, label, label_2, lam):
    """Update existing loss function for mixup.
    This function is expected to only be relevant with hard label distillation.
    Loss function should already have loss type and temperature set with functools partial.
    """
    return lam*metric_fn(logits, label) + (1-lam)*metric_fn(logits, label_2)