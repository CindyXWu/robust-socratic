import torch
from torch import nn, optim
from torch.nn import Module
from torch.utils.data import DataLoader

import numpy as np
import wandb
from tqdm import tqdm
import logging
from collections import defaultdict

from losses.loss_common import *
from losses.jacobian import get_jacobian_loss
from losses.contrastive import CRDLoss
from losses.feature_match import feature_extractor

from models.resnet_ap import CustomResNet18, CustomResNet50
from config_setup import MainConfig, DistillLossType, DistillConfig
from constructors import get_counterfactual_dataloaders, create_dataloaders
from plotting_exhaustive import plot_images

from typing import Optional, List, Dict
    

@torch.no_grad()
def evaluate(model: nn.Module, 
             dataloader: DataLoader, 
             batch_size: int, 
             num_eval_batches: int, 
             title: Optional[str] = None,
             device: torch.device = torch.device("cuda")) -> float:
    """Accuracy for num_eval_batches batches."""
    model.eval()
    acc = 0
    
    for i, (features, labels) in enumerate(dataloader):
        labels = labels.to(device)
        features = features.to(device)
        scores = model(features)
        _, pred = torch.max(scores, 1)
        acc += torch.sum(torch.eq(pred, labels)).item()
        if num_eval_batches != 0 and i >= num_eval_batches:
            break
        
    if title:
        plot_images(dataloader, num_images=batch_size, title=title)
        
    model.train()
    
    # Avg acc - frac data points correctly classified
    return (acc*100 / ((i+1)*batch_size))


@torch.no_grad()
def counterfactual_evaluate(teacher: nn.Module, 
                     student: nn.Module, 
                     dataloader: DataLoader, 
                     batch_size: int, 
                     num_eval_batches: int, 
                     title: Optional[str] = None,
                     device: torch.device = torch.device("cuda")) -> float:
    """Student test accuracy, T-S KL and T-S top-1 accuracy for num_eval_batches batches."""
    acc, KL, top_1 = 0, 0, 0
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
        
    if title:
        plot_images(dataloader, num_images=batch_size, title=title)
        
    avg_acc = acc*100/(i*batch_size)
    avg_KL = KL/i
    avg_top_1 = top_1*100/i
    
    student.train()
    
    return avg_acc, avg_KL, avg_top_1


class LRScheduler(object):
    def __init__(self, optimizer, num_epochs, base_lr, final_lr, iter_per_epoch):
        self.base_lr = base_lr
        # Iterations per epoch
        decay_iter = iter_per_epoch * num_epochs
        self.lr_schedule = final_lr+0.5*(base_lr-final_lr)*(1+np.cos(np.pi*np.arange(decay_iter)/decay_iter))        
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr'] = self.lr_schedule[self.iter]
        self.iter += 1
        self.current_lr = lr

    def get_lr(self):
        return self.current_lr
    

def weight_reset(model: nn.Module):
    """Reset weights of model at start of training."""
    for module in model.modules():
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")


def train_teacher(teacher: nn.Module, 
                  train_loader: DataLoader, 
                  test_loader: DataLoader, 
                  optimizer: optim.Optimizer,
                  scheduler: LRScheduler,
                  config: MainConfig,
                  device: torch.device = torch.device("cuda")) -> None:
    """
    Currently hard coding in evaluating teachers with exhaustive datasets, but can add this to config file later.
    
    Things pulled from config:
        - epochs: Number of epochs to train for. Calculated from num_iters.
        - dataloader.test_bs: Batch size for evaluation.
        - eval_frequency: How many iterations between evaluations. If None, assumed to be 1 epoch, if the dataset is not Iterable.
        - num_eval_batches: How many batches to evaluate on.
        - early_stop_patience: Number of epochs with no accuracy improvement before training stops
        - teacher_save_path: Where to save the teacher model.
        - wandb_project_name: Name of wandb project.
    """
    # Really important: reset model weights compared to default initialisation
    if isinstance(teacher, CustomResNet18) or isinstance(teacher, CustomResNet50):
        teacher.weight_reset()
    train_acc_list, test_acc_list = [], []
    it = 0
    no_improve_count = 0
    best_test_acc = 0.0
    
    cf_dataloaders = get_counterfactual_dataloaders(config, config_groupname="exhaustive_configs")
    
    for epoch in range(config.epochs):
        print("Epoch: ", epoch)
        teacher.train()
        train_loss = []

        for inputs, labels in tqdm(train_loader, desc=f"Training iterations within epoch {epoch}"):
            inputs, labels = inputs.to(device), labels.to(device)
            scores = teacher(inputs)

            optimizer.zero_grad()
            loss = ce_loss(scores, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            lr = scheduler.get_lr()
            train_loss.append(loss.detach().cpu().numpy())
            
            if it % config.eval_frequency == 0:
                train_acc = evaluate(teacher, train_loader, batch_size=config.dataloader.test_bs, num_eval_batches=config.num_eval_batches, device=device)
                test_acc = evaluate(teacher, test_loader, batch_size=config.dataloader.test_bs, num_eval_batches=config.num_eval_batches, device=device)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)

                # Counterfactual evaluations for teacher
                cf_accs = defaultdict(float)
                for name in cf_dataloaders:
                    cf_accs[name] = evaluate(
                        model=teacher,
                        dataloader=cf_dataloaders[name],
                        batch_size=config.dataloader.test_bs,
                        num_eval_batches=config.num_eval_batches,
                        device=device)
                
                print(f'Project {config.wandb_project_name}, Epoch: {epoch}, Train accuracy: {train_acc}, Test accuracy: {test_acc}, LR {lr}')
                
                results_dict = {**cf_accs, **{
                    "Train Acc": train_acc, 
                    "Test Acc": test_acc, 
                    "Loss": np.mean(train_loss), 
                    "LR": lr}}
                
                wandb.log(results_dict) # Can also optionally add step here
        
                # Early stopping logic
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    no_improve_count = 0
                    save_model(f"{config.teacher_save_path}_best", epoch, teacher, optimizer, train_loss, train_acc_list, test_acc_list, [train_acc, test_acc])
                else:
                    no_improve_count += 1
                if no_improve_count >= config.early_stop_patience:
                    print("Early stopping due to no improvement in test accuracy.")
                    return
            
            it += 1
    
    # Hope is that by separating end of run saving with working, a working test run won't overwrite the best model
    save_model(f"{config.teacher_save_path}_best", epoch, teacher, optimizer, train_loss, train_acc_list, test_acc_list, [train_acc, test_acc])

    
def train_distill(
    teacher: Module, 
    student: Module, 
    train_loader: DataLoader, 
    test_loader: DataLoader, 
    optimizer: optim.Optimizer,
    scheduler: LRScheduler,
    config: DistillConfig,
    device: torch.device = torch.device("cuda")) -> None:
    """
    Args:
        base_dataset: Which dataset: CIFAR10, CIFAR100, Dominoes or Shapes.
        contrast_temp: Contrastive loss temperature.
        temp: Base distillation loss temperature.
        nonbase_loss_frac: Contrastive and Jacobian loss weight compared to base.
        s_layer, t_layer: Layer to extract features from for contrastive loss
        loss_num: Index of loss dictionary (in info_dicts.py) describing which loss fn to use.
        N_its: Number of iterations to train for.
        its_per_log: Number of iterations between logging.
    """
    # Really important: reset model weights compared to default initialisation
    if isinstance(student, CustomResNet18) or isinstance(student, CustomResNet50):
        student.weight_reset()
    teacher.eval()
    student.train()
    train_acc_list, test_acc_list = [], []
    it = 0
    no_improve_count = 0
    best_test_acc = 0.0

    sample = next(iter(train_loader))
    batch_size, c, w, h = sample[0].shape
    assert batch_size == config.dataloader.train_bs
    input_dim = c*w*h

    cf_dataloaders = get_counterfactual_dataloaders(config, config_groupname="counterfactual_configs")

    for epoch in range(config.epochs):
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs.requires_grad = True
            scores, targets = student(inputs), teacher(inputs) 

            # for param in student.parameters():
            #     assert param.requires_grad
            loss = base_distill_loss(
                        scores=scores, 
                        targets=targets, 
                        loss_type=config.base_distill_loss_type,
                        temp=config.dist_temp
            )
            jacobian_loss = None
            contrastive_loss = None
            
            match config.distill_loss_type:
                case DistillLossType.BASE:
                    pass
                
                case DistillLossType.JACOBIAN:
                    jacobian_loss = get_jacobian_loss(
                        scores=scores,
                        targets=targets, 
                        inputs=inputs,
                        config=config,
                        input_dim=input_dim
                    )
                    loss = (1-config.nonbase_loss_frac)*loss + config.nonbase_loss_frac*jacobian_loss
                    
                case DistillLossType.CONTRASTIVE:
                    s_map = feature_extractor(student, inputs, config.s_layer).view(batch_size, -1)
                    t_map = feature_extractor(teacher, inputs, config.t_layer).view(batch_size, -1).detach()
                    contrastive_loss = CRDLoss(
                        s_dim=s_map.shape[1],
                        t_dim=t_map.shape[1],
                        T=config.contrast_temp
                    )
                    loss = (1-config.nonbase_loss_frac)*loss + config.nonbase_loss_frac*contrastive_loss(s_map, t_map, labels)
                    
                # case DistillLossType.FEATURE_MAP: # Currently only for self-distillation
                #     layer = 'feature_extractor.10'
                # Only works if model has a feature extractor method
                #     s_map = student.attention_map(inputs, layer)
                #     t_map = teacher.attention_map(inputs, layer).detach() 
                    # loss = feature_map_diff(scores, targets, s_map, t_map, T=1, alpha=0.2, loss_fn=mse_loss, aggregate_chan=False)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            lr = scheduler.get_lr()
            train_loss = loss.detach().cpu().numpy()

            # if it == 0: check_grads(student)
            if it % config.eval_frequency == 0:
                train_acc = evaluate(student, train_loader, batch_size=config.dataloader.test_bs, num_eval_batches=config.num_eval_batches, device=device)
                test_acc, test_KL, test_top1 = counterfactual_evaluate(teacher, student, test_loader, batch_size=config.dataloader.test_bs, num_eval_batches=config.num_eval_batches, title=None, device=device)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                
                # Dictionary holds counterfactual acc, KL and top 1 fidelity for each dataset
                cf_evals = defaultdict(float)
                for name in cf_dataloaders:
                    title = f'Dominoes_{name}'
                    # Currently not plotting datasets
                    cf_evals[name], cf_evals[f"{name} T-S KL"], cf_evals[f"{name} T-S Top 1 Fidelity"] = counterfactual_evaluate(teacher, student, cf_dataloaders[name], batch_size=config.dataloader.test_bs, num_eval_batches=config.num_eval_batches, title=None)

                print(f"Project: {config.wandb_run_name}, Iteration: {it}, Epoch: {epoch}, Loss: {train_loss}, LR: {lr}, Base Temperature: {config.dist_temp}, Jacobian Temperature: {config.jac_temp}, Contrastive Temperature: {config.contrast_temp}, Nonbase Loss Frac: {config.nonbase_loss_frac}, ")

                results_dict = {**cf_evals, **{
                    "T-S KL": test_KL, 
                    "T-S Top 1 Fidelity": test_top1, 
                    "S Train": train_acc, 
                    "S Test": test_acc, 
                    "S Loss": train_loss, 
                    "S LR": lr, 
                    "Jacobian Loss": jacobian_loss, 
                    "Contrastive Loss": contrastive_loss
                }}
                
                wandb.log(results_dict)
                
                # Early stopping logic
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    no_improve_count = 0
                    save_model(f"{config.teacher_save_path}_best", epoch, student, optimizer, train_loss, train_acc_list, test_acc_list, [train_acc, test_acc])
                else:
                    no_improve_count += 1
                if no_improve_count >= config.early_stop_patience:
                    print("Early stopping due to no improvement in test accuracy.")
                    return
                
            it += 1

        ## Visualise 3d at end of each epoch
        # if loss_num == 2:
        #     visualise_features_3d(s_map, t_map, title=run_name+"_"+str(it))
        
    # By separating end of run saving with working, working test run won't overwrite best model
    save_model(f"{config.teacher_save_path}_best", epoch, teacher, optimizer, train_loss, train_acc_list, test_acc_list, [train_acc, test_acc])


def check_grads(model: nn.Module):
    for param in model.parameters():
        assert param.grad is not None


def save_model(
    path: str, 
    epoch: int, 
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    train_loss: List[float], 
    train_acc: List[float], 
    test_acc: List[float],
    final_acc: float) -> None:
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_hist': train_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'final_acc': final_acc
                },
                path)


def print_saved_model(checkpoint: Dict) -> None:
    for key, value in checkpoint.items():
        print(f"Key: {key}")
        print(f"Value: {value}")

