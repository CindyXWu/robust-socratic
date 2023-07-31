import torch
from torch import nn, optim
import wandb
from tqdm import tqdm
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Tuple

from models.resnet import wide_resnet_constructor
from models.resnet_ap import CustomResNet18
from models.lenet import LeNet5
from models.mlp import mlp_constructor

from plotting_exhaustive import *
from losses.jacobian import *
from losses.contrastive import *
from losses.feature_match import *
from datasets.utils_ekdeep import *
from datasets.shapes_3D import *
from info_dicts import *
from config_setup import MainConfig, DatasetType, DistillLossType
from constructors import get_counterfactual_dataloaders


kl_loss = nn.KLDivLoss(reduction='batchmean')
ce_loss = nn.CrossEntropyLoss(reduction='mean')
mse_loss = nn.MSELoss(reduction='mean')


@torch.no_grad()
def evaluate(model: nn.Module, 
             dataloader: DataLoader, 
             batch_size: int, 
             max_ex: int, 
             title: Optional[str] = None,
             device: torch.device = torch.device("cuda")) -> float:
    """Accuracy for max_ex batches."""
    acc = 0
    for i, (features, labels) in enumerate(dataloader):
        labels = labels.to(device)
        features = features.to(device)
        scores = model(features)
        _, pred = torch.max(scores, 1)
        acc += torch.sum(torch.eq(pred, labels)).item()
        if max_ex != 0 and i >= max_ex:
            break
    if title:
        plot_images(dataloader, num_images=batch_size, title=title)
    # Avg acc - frac data points correctly classified
    return (acc*100 / ((i+1)*batch_size))


@torch.no_grad()
def counterfactual_evaluate(teacher: nn.Module, 
                     student: nn.Module, 
                     dataloader: DataLoader, 
                     batch_size: int, 
                     max_ex: int, 
                     title: Optional[str] = None) -> float:
    """Student test accuracy, T-S KL and T-S top-1 accuracy for max_ex batches."""
    acc = 0
    KL = 0
    top_1 = 0
    for i, (features, labels) in enumerate(dataloader):
        if max_ex != 0 and i >= max_ex:
            break
        labels, features = labels.to(device), features.to(device)
        targets, scores = teacher(features), student(features)
        s_pred, t_pred = torch.argmax(scores, 1), torch.argmax(targets, 1)
        acc += torch.sum(torch.eq(s_pred, labels)).item() # Total accurate samples in batch
        KL += kl_loss(F.log_softmax(scores, dim=1), F.softmax(targets, dim=1)) # Batchwise mean KL
        top_1 += torch.eq(s_pred, t_pred).float().mean() # Batchwise mean top-1 accuracy
    if title:
        plot_images(dataloader, num_images=batch_size, title=title)
    avg_acc = acc*100/(i*batch_size)
    avg_KL = KL/i
    avg_top_1 = top_1*100/i
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


def train_teacher(model: nn.Module, 
                  train_loader: DataLoader, 
                  test_loader: DataLoader, 
                  optimizer: optim.Optimizer,
                  scheduler: LRScheduler,
                  config: MainConfig,
                  epochs: int,
                  device: torch.device = torch.device("cuda")) -> None:
    """
    Args:
    """
    weight_reset(model) # Really important: reset weights compared to default initialisation
    train_acc_list, test_acc_list = [], []
    it = 0
    
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        train_loss = []
        model.train()
        
        for inputs, labels in tqdm(train_loader, desc=f"Training iterations within epoch {epoch}"):
            inputs, labels = inputs.to(device), labels.to(device)
            scores = model(inputs)

            optimizer.zero_grad()
            loss = ce_loss(scores, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            lr = scheduler.get_lr()
            train_loss.append(loss.detach().cpu().numpy())
            
            if it % config.eval_frequency == 0:
                batch_size = inputs.shape[0]
                train_acc = evaluate(model, train_loader, batch_size, max_ex=20)
                test_acc = evaluate(model, test_loader, batch_size, max_ex=10)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)

                print(f'Project {config.wandb_project_name}, Epoch: {epoch}, Train accuracy: {train_acc}, Test accuracy: {test_acc}, LR {lr}')
                wandb.log({"Train Acc": train_acc, "Test Acc": test_acc, "Loss": np.mean(train_loss), "LR": lr}, step=it)

            it += 1

        # Checkpoint model at end of epoch
        save_model(f"{config.teacher_save_path}_working", epoch, model, optimizer, train_loss, train_acc_list, test_acc_list, [train_acc, test_acc])
    
    save_model(f"{config.teacher_save_path}_final", epoch, model, optimizer, train_loss, train_acc_list, test_acc_list, [train_acc, test_acc])

    
def train_distill(
    teacher: Module, 
    student: Module, 
    train_loader: DataLoader, 
    test_loader: DataLoader, 
    optimizer: optim.Optimizer,
    scheduler: LRScheduler,
    config: MainConfig,
    epochs: int,
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
    teacher.eval()
    weight_reset(student) # Really important: reset weights compared to default initialisation
    student.train()

    it = 0

    sample = next(iter(train_loader))
    batch_size, c, w, h = sample[0].shape
    assert batch_size == config.dataloader.train_bs
    input_dim = c*w*h

    cf_dataloaders = get_counterfactual_dataloaders(config.dataset_type, batch_size)

    for epoch in range(epochs):
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs.requires_grad = True
            scores, targets = student(inputs), teacher(inputs) 

            # for param in student.parameters():
            #     assert param.requires_grad
            match config.distill_loss_type:
                case DistillLossType.BASE:
                    loss = base_distill_loss(scores, targets, config.dist_temp)
                # TODO: RENAME PARAMS
                case DistillLossType.JACOBIAN:
                    loss = jacobian_loss(
                        scores=scores,
                        targets=targets, 
                        inputs=inputs,
                        T=config.jac_temp,
                        alpha=config.nonbase_loss_frac,
                        batch_size=batch_size,
                        loss_fn=config.jacobian_loss_type,
                        input_dim=input_dim,
                        output_dim=config.dataset.output_size)
                case DistillLossType.CONTRASTIVE:
                    s_map = feature_extractor(student, inputs, config.s_layer).view(batch_size, -1)
                    t_map = feature_extractor(teacher, inputs, config.t_layer).view(batch_size, -1).detach()
                    contrastive_loss = CRDLoss(s_map.shape[1], t_map.shape[1], T=config.contrast_temp)
                    loss = config.nonbase_loss_frac*contrastive_loss(s_map, t_map, labels)+(1-config.nonbase_loss_frac)*base_distill_loss(scores, targets, config.dist_temp)
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
                train_acc = evaluate(student, train_loader, batch_size, max_ex=config.num_eval_batches)
                test_acc, test_KL, test_top1 = counterfactual_evaluate(teacher, student, test_loader, batch_size, max_ex=config.num_eval_batches, title=None)
                # Dictionary holds counterfactual acc, KL and top 1 fidelity for each dataset
                cf_evals = defaultdict(float)

                for name in cf_dataloaders:
                    title = f'Dominoes_{name}'
                    # Currently not plotting datasets
                    cf_evals[name], cf_evals[f"{name} T-S KL"], cf_evals[f"{name} T-S Top 1 Fidelity"] = counterfactual_evaluate(teacher, student, cf_dataloaders[name], batch_size, max_ex=config.num_eval_batches, title=None)

                print(cf_evals)
                print(f"Project: {config.wandb_run_name}, Iteration: {it}, Epoch: {epoch}, Loss: {train_loss}, LR: {lr}, Base Temperature: {config.dist_temp}, Jacobina Temperature: {config.jac_temp}, Contrastive Temperature: {config.contrast_temp}, Nonbase Loss Frac: {config.nonbase_loss_frac}, ")

                results_dict = {**cf_evals, **{
                    "T-S KL": test_KL, 
                    "T-S Top 1 Fidelity": test_top1, 
                    "S Train": train_acc, 
                    "S Test": test_acc, 
                    "S Loss": train_loss, 
                    "S LR": lr, } 
                }

                wandb.log(results_dict, step=it)

            it += 1

        ## Visualise 3d at end of each epoch
        # if loss_num == 2:
        #     visualise_features_3d(s_map, t_map, title=run_name+"_"+str(it))


def check_grads(model: nn.Module):
    for param in model.parameters():
        assert param.grad is not None


def base_distill_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    temp: float) -> torch.Tensor:
    scores = F.log_softmax(scores/temp)
    targets = F.softmax(targets/temp)
    return kl_loss(scores, targets)


def save_model(
    path: str, 
    epoch: int, 
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    train_loss: List[float], 
    train_acc: List[float], 
    test_acc: List[float],
    final_acc: float,
    ) -> None:
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_hist': train_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'final_acc': final_acc
                },
                path)
    
