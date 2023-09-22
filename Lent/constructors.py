"""Construct models, get class numbers, get alpha, construct dataloaders."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from typing import Tuple, Optional
from dataclasses import asdict
import numpy as np
import logging
import omegaconf
from omegaconf import OmegaConf
from functools import partial

import os

from config_setup import MainConfig, DistillConfig, ModelType, DatasetType, DistillLossType, DatasetConfig, ConfigGroups, ExperimentConfig, OptimizerType, ConfigType
from datasets.dominoes_box import get_box_dataloader
from datasets.shapes_3D import dataloader_3D_shapes
from models.resnet import wide_resnet_constructor
from models.resnet_ap import CustomResNet18
from models.lenet import LeNet5
from models.mlp import mlp_constructor


def get_nonbase_loss_frac(distill_config: DistillConfig):
    """To be called by main function. Contains all necessary dictionaries.
    Jacobian currently fixed via empirical experimental results. Contrastive not yet fixed.
    """
    nonbase_loss_frac_dict = {DistillLossType.JACOBIAN: 0.15, DistillLossType.CONTRASTIVE: 0.03}
    return nonbase_loss_frac_dict.get(distill_config.distill_loss_type)


def get_dataset_output_size(config: MainConfig) -> int:
    dataset_output_sizes = {
    "DOMINOES": 10,
    "SHAPES": 8,
    "CIFAR100": 100,
    "CIFAR10": 10
    }
    return dataset_output_sizes.get(config.dataset_type)


def model_constructor(config: MainConfig, is_student: bool) -> nn.Module:
    """Constructs a model based on a specified model type."""
    model_type = config.student_model_type if is_student else config.model_type
    if model_type == ModelType.MLP:
        mlp_config = config.student_mlp_config if is_student else config.mlp_config
        model = mlp_constructor(
            input_size=config.dataset.input_length,
            hidden_sizes=mlp_config.hidden_sizes,
            output_size=mlp_config.output_size,
            bias=mlp_config.add_bias,
        )
    elif model_type == ModelType.RESNET_WIDE:
        wrn_config = config.student_wrn_config if is_student else config.wrn_config
        model = wide_resnet_constructor(
            blocks_per_stage=wrn_config.blocks_per_stage,
            width_factor=wrn_config.width_factor,
            output_size=config.dataset.output_size,
        )
    elif model_type == ModelType.RESNET18_ADAPTIVE_POOLING:
        model = CustomResNet18(config.dataset.output_size)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    return model


def get_model_intermediate_layer(config: MainConfig) -> str:
    """For feature difference losses.
    TODO: edit ResNet wide to check for model width first
    """
    models = {
            ModelType.LENET5_3CHAN: {'feature_extractor.10': 'feature_extractor.10'},
            ModelType.RESNET18_ADAPTIVE_POOLING: {'layer4.1.bn2': 'bn_bn2'},
            ModelType.RESNET_WIDE: {"11.path2.5": "final_features"},
        }
    return models.get(config.model_type)


def create_dataloaders(config: MainConfig,
                      exp_config: Optional[ExperimentConfig] = None
                      ) -> Tuple[DataLoader, DataLoader]:
    """Set train and test loaders based on dataset and experiment.
    For generating dominoes: box is cue 1, MNIST is cue 2.
    For CIFAR100 box: box is cue 1.

    Two ways to run this function:
    1. If used to generate counterfactual datasets, index into ConfigGroup.counterfactual_configs dictionary. exp_config must be specified.
    2. Otherwise assume generating main training and testing dataloaders for experiment - values are in MainConfig object 'config' due to Hydra handling this.

    Form of exp_config exampled in configs/targeted_distillation folder.
    
    Args:
        exp_config: Optional experiment config specified separately if and only if generating counterfactual dataloaders.
        Handling of this is expected to be external to this function.
    Returns:
        train_loader: Same as test_loader, just split differently.
        test_loader: "".
    """
    base_dataset = config.dataset_type
    
    if exp_config is None: # TRAINING DATALOADERS
        batch_size = config.dataloader.train_bs
        if 'student_save_path' in config and config.student_save_path is not None: # Distillation
            experiment = config.experiment_s
        else: experiment = config.experiment # Teacher training
        im_frac, m1_frac, m2_frac, rand_im, rand_m1, rand_m2 = OmegaConf.to_container(experiment.experiment_config).values()
    else: # COUNTERFACTUAL DATALOADERS - TEST
        batch_size = config.dataloader.test_bs
        im_frac, m1_frac, m2_frac, rand_im, rand_m1, rand_m2 = asdict(exp_config).values()

    box_cue_size = config.dataset.box_cue_size
    box_pattern = config.dataset.box_cue_pattern
    
    if base_dataset == DatasetType.DOMINOES: # BOX: MECH 1, MNIST: MECH 2
        partial_get_box_dataloader = partial(get_box_dataloader, base_dataset='Dominoes', batch_size=batch_size, randomize_img=rand_im, box_frac=m1_frac, mnist_frac=m2_frac, image_frac=im_frac, randomize_box=rand_m1, randomize_mnist=rand_m2, box_cue_size=box_cue_size, box_pattern=box_pattern)
        
        train_loader = partial_get_box_dataloader(load_type='train')
        test_loader = partial_get_box_dataloader(load_type='test')

    elif base_dataset == DatasetType.SHAPES: # FLOOR: MECH 1, SCALE: MECH 2
        train_loader = dataloader_3D_shapes('train', batch_size=batch_size, randomise=rand_im, floor_frac=m1_frac, scale_frac=m2_frac)
        test_loader = dataloader_3D_shapes('test', batch_size=batch_size, randomise=rand_im, floor_frac=m1_frac, scale_frac=m2_frac)
    
    elif base_dataset in [DatasetType.CIFAR100, DatasetType.CIFAR10]: # Image frac isn't relevant - always 100 for these exps so don't pass in
        cue_type='box' if m1_frac != 0 else 'nocue'
        partial_get_box_dataloader = partial(get_box_dataloader, base_dataset=base_dataset, cue_type=cue_type, batch_size=batch_size,  cue_proportion=m1_frac, randomize_cue=rand_m1, randomize_img = rand_im, box_cue_size=box_cue_size, box_pattern=box_pattern)
        
        train_loader = partial_get_box_dataloader(load_type='train')
        test_loader = partial_get_box_dataloader(load_type='test')
 
    return train_loader, test_loader


def get_counterfactual_dataloaders(config: MainConfig) -> dict[str, DataLoader]:
    """Get a dictionary of dataloaders for counterfactual evaluation. Key is string describing counterfactual experiment settings.
    """
    if config.config_type == "TARGETED":
        cf_groupname = "targeted_cf"
    elif config.config_type == "EXHAUSTIVE":
        cf_groupname = "exhaustive"
    elif config.config_type == "FRAC":
        cf_groupname = "frac_cf"
    else:
        raise ValueError("groupname in config YAML file incorrect")
        
    dataloaders = {}
    config_group = getattr(ConfigGroups, cf_groupname)
    for idx, exp_config in enumerate(config_group):
        name, exp_config = exp_config.name, exp_config.experiment_config
        _, dataloaders[name] = create_dataloaders(config, exp_config)
    return dataloaders


def create_or_load_dataset(dataset_type: str, dataset_config: DatasetConfig) -> Dataset:
    """Create or load an existing dataset based on a specified filepath and dataset type."""
    filepath = f'{dataset_config.data_folder}/{dataset_config.filename}.pt'
    if os.path.exists(filepath):
        dataset = torch.load(filepath)
    else:
        dataset_type = globals()[dataset_type]
        dataset = dataset_type(dataset_config)
        torch.save(dataset, filepath)
    return dataset
        

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


def optimizer_constructor(
    config: MainConfig,
    model: nn.Module,
    train_loader: DataLoader) -> optim.Optimizer:
    match config.optimization.optimizer_type:
        case OptimizerType.SGD:
            optim_constructor = torch.optim.SGD
        case OptimizerType.ADAM:
            optim_constructor = torch.optim.Adam
        case _:
            raise ValueError(f"Unknown optimizer type: {config.optimization.optimizer_type}")
    optim = optim_constructor(
        params=model.parameters(),
        lr=config.optimization.base_lr,
        **config.optimization.optimizer_kwargs,
    )
    
    if config.optimization.cosine_lr_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optim,
            T_max=config.num_iters,
        )
        scheduler = LRScheduler(optim, config.epochs, base_lr=config.optimization.base_lr, final_lr=config.optimization.final_lr, iter_per_epoch=len(train_loader))
        
    return optim, scheduler


def change_frac_filename(config: MainConfig, exp_idx: int, is_student: bool = False) -> str:
    match (exp_idx, is_student):
        case ("0", False) | ("2", False):
            name = config.experiment.name.replace("x", str(config.experiment_s.experiment_config.m1_frac))
        case ("0", True) | ("2", True):
            name = config.experiment_s.name.replace("x", str(config.experiment_s.experiment_config.m1_frac))
        case ("1", False) | ("3", False):
            name = config.experiment.name.replace("x", str(config.experiment_s.experiment_config.m2_frac))
        case ("1", True) | ("3", True):
            name = config.experiment_s.name.replace("x", str(config.experiment_s.experiment_config.m2_frac))
        case _:
            raise ValueError(f"Unsupported values for exp_idx ({exp_idx}) and is_student ({is_student})")
    return name