"""Construct models, get class numbers, get alpha, construct dataloaders."""
import torch
import torch.nn as nn
from typing import Tuple
from dataclasses import asdict
from config_setup import *
from datasets.utils_ekdeep import *
from datasets.shapes_3D import *


class Constructor():
    def __init__(self, config: MainConfig):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.class_num = self.get_class_num()
        self.distill_loss_type = config.distill_loss_type
        self.dataset_type = config.dataset_type
        self.model_type = config.model_type
        
    def make_model(self) -> nn.Module:
        """Factory method for creating student and teacher models."""   
        models = {
                ModelType.LENET5_3CHAN: (LeNet5(self.class_num).to(self.device), {'feature_extractor.10': 'feature_extractor.10'}),
                ModelType.RESNET18_ADAPTIVE_POOLING: (CustomResNet18(self.class_num).to(self.device), {'layer4.1.bn2': 'bn_bn2'}),
                ModelType.RESNE20_WIDE: (wide_resnet_constructor(3, self.class_num).to(self.device), {"11.path2.5": "final_features"}),
            }
        return models.get(self.model_type)
    
    def get_class_num(self):
        match self.dataset_type:
            case DatasetType.DOMINOES:
                return 10
            case DatasetType.SHAPES:
                return 8
            case DatasetType.CIFAR100:
                return 100
            case DatasetType.CIFAR10:
                return 10
            case _:
                raise ValueError(f"Unknown dataset type {self.dataset_type}")
    
    def get_alpha(self):
        """Weighting of this loss term with respect to base soft label matching distillation.
        Currently hard-coded, but probably better as hyperparameter."""
        match self.loss_type:
            case DistillLossType.BASE:
                return 1
            case DistillLossType.JACOBIAN:
                return 0.5
            case DistillLossType.CONTRASTIVE:
                return 0.01
            case _:
                raise ValueError(f"Unknown loss type {self.distill_loss_type}")


def create_dataloader(exp_num: int,
                      main_config: MainConfig,
                      counterfactual: bool = False,
                      ) -> Tuple[DataLoader, DataLoader]:
    """Set train and test loaders based on dataset and experiment.
    For generating dominoes: box is cue 1, MNIST is cue 2.
    For CIFAR100 box: box is cue 1.
    
    Args:
        exp_num: index in the ExperimentConfig tuple.
        counterfactual: whether this dataloader being generated is a counterfactual dataloader. If true, then we load experiments based on the counterfactual configs group inside ConfigGroups class.
    Returns:
        train_loader: same as test_loader, just split differently.
        test_loader: "".
    """
    base_dataset = main_config.dataset_type
    train_bsize = main_config.train_bsize
    config_group: Tuple[ExpConfig, ...] = ConfigGroups.counterfactual_configs if counterfactual else ConfigGroups.targeted_exp_configs
    config: ExperimentConfig = config_group[exp_num].config if exp_config is None else exp_config
    im_frac, m1_frac, m2_frac, rand_im, rand_m1, rand_m2 = asdict(config).values()

    if base_dataset == DatasetType.DOMINOES: # BOX: MECH 1, MNIST: MECH 2
        train_loader = get_box_dataloader(load_type='train', base_dataset='Dominoes', batch_size=train_bsize, rand_im=rand_im, box_frac=m1_frac, mnist_frac=m2_frac, im_frac=im_frac, randomize_box=rand_m1, randomize_mnist=rand_m2)
        test_loader = get_box_dataloader(load_type='test', base_dataset='Dominoes', batch_size=train_bsize, rand_im=rand_im, box_frac=m1_frac, mnist_frac=m2_frac, im_frac=im_frac, randomize_box=rand_m1, randomize_mnist=rand_m2)

    elif base_dataset == DatasetType.SHAPES: # FLOOR: MECH 1, SCALE: MECH 2
        train_loader = dataloader_3D_shapes('train', batch_size=batch_size, randomise=rand_im, floor_frac=m1_frac, scale_frac=m2_frac)
        test_loader = dataloader_3D_shapes('test', batch_size=batch_size, randomise=rand_im, floor_frac=m1_frac, scale_frac=m2_frac)
    
    elif base_dataset in [DatasetType.CIFAR100, DatasetType.CIFAR10]: # Image frac isn't relevant - always 100 for these exps so don't pass in
        cue_type='box' if m1_frac != 0 else 'nocue'
        train_loader = get_box_dataloader(load_type='train', base_dataset=base_dataset, cue_type=cue_type, cue_proportion=m1_frac, randomize_cue=rand_m1, rand_im = rand_im, batch_size=train_bsize)
        test_loader = get_box_dataloader(load_type ='test', base_dataset=base_dataset, cue_type=cue_type, cue_proportion=m1_frac, randomize_cue=rand_m1, rand_im = rand_im, batch_size=train_bsize)
 
    return train_loader, test_loader


def get_counterfactual_dataloaders(main_config: MainConfig) -> dict[str, DataLoader]:
    """Get a dictionary of dataloaders for counterfactual evaluation. Key of dictionary tells us which settings for counterfactual evals are used."""
    dataloaders = {}
    for exp_num, (name, config) in enumerate(ConfigGroups.counterfactual_configs.items()):
        _, dataloaders[name] = create_dataloader(exp_num, main_config, config, counterfactual=True)
        

class LRScheduler(object):
    def __init__(self, optimizer, num_epochs, base_lr, final_lr, iter_per_epoch):
        self.base_lr = base_lr
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