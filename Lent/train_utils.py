import torch
from torch import nn, optim
from torch.nn import Module
from torch.utils.data import DataLoader

import numpy as np
import wandb
from einops import rearrange
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import random
from omegaconf import OmegaConf
from collections import defaultdict
from typing import List, Dict

from losses.loss_common import *
from losses.jacobian import get_jacobian_loss
from losses.contrastive import CRDLoss
from losses.feature_match import feature_extractor

from models.resnet_ap import CustomResNet18, CustomResNet50
from config_setup import MainConfig, DistillLossType, DistillConfig
from constructors import get_counterfactual_dataloaders
from plotting_common import plot_PIL_batch
from evaluate import evaluate, counterfactual_evaluate, mixup_accuracy


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
    
    if config.config_type == "TARGETED":
        cf_groupname = "targeted_cf"
    elif config.config_type == "EXHAUSTIVE":
        cf_groupname = "exhaustive"
    cf_dataloaders = get_counterfactual_dataloaders(config, cf_groupname)
    
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
                    save_model(f"{config.teacher_save_path}", epoch, teacher, optimizer, train_loss, train_acc_list, test_acc_list, [train_acc, test_acc])
                else:
                    no_improve_count += 1
                if no_improve_count >= config.early_stop_patience:
                    print("Early stopping due to no improvement in test accuracy.")
                    return
            
            it += 1

        # Get saliency map
        single_image = inputs[0].detach().clone().unsqueeze(0).requires_grad_()
        saliency_t = get_saliency_map(teacher, single_image).squeeze().detach().cpu().numpy()
        t_prob = F.softmax(teacher(single_image), dim=1).squeeze().detach().cpu().numpy()
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # Create a figure with 2 subplots side by side
        single_image = rearrange(single_image.squeeze(0).detach().cpu().numpy(), 'c h w -> h w c')
        axs[0].imshow(single_image)
        axs[0].set_title("Original Image")
        axs[1].imshow(saliency_t)
        axs[1].set_title("Teacher Saliency Map")
        
        t_prob_str = ["Teacher - " + f"Class {i}: {p:.4f}" for i, p in enumerate(t_prob)]
        plt.figtext(0.92, 0.5, t_prob_str, fontsize=10, ha="center", va="center", bbox={'boxstyle': "round", 'facecolor': "white"})
        plt.tight_layout()

        wandb.log({f"Epoch {epoch}": [wandb.Image(fig)]}, step=it)
        
    save_model(f"{config.teacher_save_path}", epoch, teacher, optimizer, train_loss, train_acc_list, test_acc_list, [train_acc, test_acc])


def get_mixup_data(inputs: torch.Tensor, labels: torch.Tensor, augmentation_params: dict):
    """Mixup within a batch of data, inputs.
    
    Returns:
        mixed_inputs: Mixed inputs based on in-batch shuffling.
        random_labels: Shuffled labels in the same order as the randomly shuffled inputs.
        lam: Mixing coefficient for mixup, drawn from beta distribution with parameters alpha and beta set by augmentation_params.
    """
    bsize = inputs.size()[0]
    shuffled_idx = torch.randperm(bsize)
    lam = random.betavariate(augmentation_params['alpha'], augmentation_params['beta'])
    mixed_inputs = lam*inputs + (1-lam)*inputs[shuffled_idx, :]
    random_labels = labels[shuffled_idx]
    
    return mixed_inputs, random_labels, lam


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
    aug = config.dataset.use_augmentation
    augmentation_params = OmegaConf.to_container(config.augmentation, resolve=True)
    # Some checks for model saving
    if config.is_sweep:
        assert config.save_model is False or config.save_model_as_artifact is False, "Don't save model during sweep"
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

    if config.config_type == "TARGETED":
        cf_groupname = "targeted_cf"
    elif config.config_type == "EXHAUSTIVE":
        cf_groupname = "exhaustive"
    
    cf_dataloaders = get_counterfactual_dataloaders(config, cf_groupname)
    
    for epoch in range(config.epochs):
        for inputs, labels in tqdm(train_loader):
            labels = labels.to(device)
            if aug:
                inputs, labels_2, lam = get_mixup_data(inputs, labels, augmentation_params)
                labels_2 = labels_2.to(device)
            inputs = inputs.to(device)
            inputs.requires_grad_()
            
            scores, targets = student(inputs), teacher(inputs)
            loss = base_distill_loss(
                        scores=scores, 
                        targets=targets, 
                        loss_type=config.base_distill_loss_type,
                        temp=config.dist_temp)
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
                    """Haven't edited this loss function for mixup yet."""
                    s_map = feature_extractor(student, inputs, config.s_layer).view(batch_size, -1).cuda()
                    t_map = feature_extractor(teacher, inputs, config.t_layer).view(batch_size, -1).cuda()
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

            if it == 0:  # Debugging
                check_grads(student)
                for name in cf_dataloaders:
                    batch_image = plot_PIL_batch(dataloader=cf_dataloaders[name], num_images=(batch_size//4))
                    wandb.log({name.split(":")[-1].strip().replace(' ', '_'): [wandb.Image(batch_image)]}, step=it)
                    
            if it % config.eval_frequency == 0:
                train_acc = evaluate(student, train_loader, batch_size=config.dataloader.test_bs, num_eval_batches=config.num_eval_batches, device=device)
                test_acc, test_KL, test_top1 = counterfactual_evaluate(teacher, student, test_loader, batch_size=config.dataloader.test_bs, num_eval_batches=config.num_eval_batches, device=device)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                
                # Dictionary holds counterfactual acc, KL and top 1 fidelity for each dataset
                cf_evals = defaultdict(float)
                for name in cf_dataloaders:
                    # Currently not plotting datasets
                    cf_evals[name], cf_evals[f"{name} T-S KL"], cf_evals[f"{name} T-S Top 1 Fidelity"] = counterfactual_evaluate(
                        teacher=teacher,
                        student=student,
                        dataloader=cf_dataloaders[name],
                        batch_size=config.dataloader.test_bs,
                        num_eval_batches=config.num_eval_batches
                        )
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
                
                # Get saliency map
                single_image = inputs[0].detach().clone().unsqueeze(0).requires_grad_()
                saliency_t = get_saliency_map(teacher, single_image).squeeze().detach().cpu().numpy()
                saliency_s = get_saliency_map(student, single_image).squeeze().detach().cpu().numpy()
                
                # Plot saliency map
                s_prob = F.softmax(student(single_image), dim=1).squeeze().detach().cpu().numpy()
                t_prob = F.softmax(teacher(single_image), dim=1).squeeze().detach().cpu().numpy()
                
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Create a figure with 3 subplots side by side
                single_image = rearrange(single_image.squeeze(0).detach().cpu().numpy(), 'c h w -> h w c')
                axs[0].imshow(single_image)
                axs[0].set_title("Original Image")
                axs[1].imshow(saliency_s)
                axs[1].set_title("Student Saliency Map")
                axs[2].imshow(saliency_t)
                axs[2].set_title("Teacher Saliency Map")
                
                # Display probabilities
                s_prob_str = ["Student - " + f"Class {i}: {p:.4f}" for i, p in enumerate(s_prob)]
                t_prob_str = ["Teacher - " + f"Class {i}: {p:.4f}" for i, p in enumerate(t_prob)]
                all_probs_str = "\n".join(s_prob_str + [""] + t_prob_str)
                plt.figtext(0.92, 0.5, all_probs_str, fontsize=10, ha="center", va="center", bbox={'boxstyle': "round", 'facecolor': "white"})
                plt.tight_layout()  # Adjust the layout so that plots do not overlap

                wandb.log({f"Iteration {it}": [wandb.Image(fig)]}, step=it)

                if it > config.min_iters and config.use_early_stop: # Only consider early stopping beyond certain threshold, and if we set the model to train with early-stop
                    # Early stopping logic
                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                        no_improve_count = 0
                        if config.save_model: # May run sweeps where you don't want to save model
                            save_model(f"{config.student_save_path}", epoch, student, optimizer, train_loss, train_acc_list, test_acc_list, [train_acc, test_acc])
                    else:
                        no_improve_count += 1
                    if no_improve_count >= config.early_stop_patience:
                        print("Early stopping due to no improvement in test accuracy.")
                        return
             
            it += 1 

    if config.save_model:
        save_model(f"{config.teacher_save_path}", epoch, teacher, optimizer, train_loss, train_acc_list, test_acc_list, [train_acc, test_acc])


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
    

def get_saliency_map(
    model: nn.Module,
    input: torch.Tensor) -> torch.Tensor:
    """
    Compute saliency map of an input.

    Args:
        model: PyTorch model.
        input_tensor: PyTorch tensor, input for which saliency map is to be computed.
        target_class: Class for which saliency map is computed.

    Returns:
        Saliency map as a PyTorch tensor.
    """
    model.eval()  # Switch the model to evaluation mode
    assert input.requires_grad

    scores = model(input)
    score_max_class = scores.argmax()
    # Zero out all other classes apart from target_class
    score = scores[0, score_max_class]
    score.backward()
    """Saliency would be the gradient with respect to the input image now. But note that the input image has 3 channels,
    R, G and B. To derive a single class saliency value for each pixel (i, j),  we take the maximum magnitude
    across all colour channels."""
    saliency, _ = torch.max(input.grad.data.abs(), dim=1)

    model.train()  # Switch the model back to training mode
    
    return saliency
    

def weight_reset(model: nn.Module):
    """Reset weights of model at start of training."""
    for module in model.modules():
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")