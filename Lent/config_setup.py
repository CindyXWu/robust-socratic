"""Bunch of dataclasses and enums to group immutable variables into classes."""
from dataclasses import dataclass, field
from typing import Any, Tuple
from collections import namedtuple
import numpy as np
from enum import Enum

from models.image_models import *


# Only doing self-distillation for now, so specify singular architecture type
class ModelType(str, Enum):
    RN18AP = "ResNet18AdaptivePooling"
    RN20W = "ResNet20Wide"
    LENET = "LeNet5"


class DistillLossType(str, Enum):
    BASE = "Soft loss KL"
    JAC = "Jacobian matching plus base"
    CONTRAST = "Contrastive plus base"


class DatasetType(str, Enum):
    DOMINOES = "Dominoes"
    SHAPES = "Shapes"
    PARITY = "Parity"
    CIFAR100 = "CIFAR100"
    CIFAR10 = "CIFAR10"


class AugType(str, Enum):
    NONE = "None"
    MIXUP = "Mixup"
    UNION = "Teacher/distillation union"


class OptimizerType(str, Enum):
    SGD = "SGD"
    ADAM = "ADAM"
    ADAMW = "ADAMW"


@dataclass
class ExperimentConfig:
    """If there is only one mechanism, values of mech 2 frac and random mech 2 do not matter."""
    im_frac: float = field(default=1.0)
    m1_frac: float = field(default=0.0, metadata=['BOX', 'FLOOR'])
    m2_frac: float = field(default=0.0, metadata=['MNIST', 'SCALE'])
    rand_im: bool
    rand_m1: bool
    rand_m2: bool

    def __post_init__(self):
        for attr in ['im_frac', 'm1_frac', 'm2_frac']:
            value = getattr(self, attr)
            if not 0 <= value <= 1:
                raise ValueError(f"{attr} must be between 0 and 1")
            
ExpConfig = namedtuple('ExpConfig', ['name', 'config'])


@dataclass
class ConfigGroups:
    """Immutable ordered tuples of experimental configurations. 
    Ordered tuples support direct indexing.
    E.g. exhaustive_configs[0].config should give ExperimentConfig(im_frac=1, m1_frac=0, m2_frac=0, rand_im=False, rand_m1=False, rand_m2=False)
    """
    exhaustive_configs = (
        ExpConfig("C", ExperimentConfig(im_frac=1, m1_frac=0, m2_frac=0, rand_im=False, rand_m1=False, rand_m2=False)), 
        ExpConfig("B", ExperimentConfig(im_frac=1, m1_frac=1, m2_frac=0, rand_im=True, rand_m1=False, rand_m2=False)), 
        ExpConfig("M", ExperimentConfig(im_frac=1, m1_frac=0, m2_frac=1, rand_im=True, rand_m1=False, rand_m2=False)), 
        ExpConfig("MB", ExperimentConfig(im_frac=1, m1_frac=1, m2_frac=1, rand_im=True, rand_m1=False, rand_m2=False)), 
        ExpConfig("CM", ExperimentConfig(im_frac=1, m1_frac=0, m2_frac=1, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("CB", ExperimentConfig(im_frac=1, m1_frac=1, m2_frac=0, rand_im=False, rand_m1=False, rand_m2=False)), 
        ExpConfig("CMB", ExperimentConfig(im_frac=1, m1_frac=1, m2_frac=1, rand_im=False, rand_m1=False, rand_m2=False))
    )


    # Numbers relate to M, S1 and S2 respectively in the string below
    targeted_exp_configs = (
        ExpConfig("No mechanisms (baseline): 100 0 0", ExperimentConfig(im_frac=1, m1_frac=0, m2_frac=0, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("Teacher one spurious: 100 0 60", ExperimentConfig(im_frac=1, m1_frac=0, m2_frac=0.6, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("Teacher both spurious: 100 30 60", ExperimentConfig(im_frac=1, m1_frac=0.3, m2_frac=0.6, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("Student both spurious: 100 60 90", ExperimentConfig(im_frac=1, m1_frac=0.6, m2_frac=0.9, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("Student both spurious: 100 90 60", ExperimentConfig(im_frac=1, m1_frac=0.9, m2_frac=0.6, rand_im=False, rand_m1=False, rand_m2=False))
    )


    counterfactual_configs = (
        ExpConfig("All mechanisms: 100 100 100", ExperimentConfig(im_frac=1, m1_frac=1, m2_frac=1, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("Only spurious: 0 100 100", ExperimentConfig(im_frac=0, m1_frac=1, m2_frac=1, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("Randomize S1: 100 R 100", ExperimentConfig(im_frac=1, m1_frac=1, m2_frac=1, rand_im=False, rand_m1=True, rand_m2=False)),
        ExpConfig("Randomize S2: 100 100 R", ExperimentConfig(im_frac=1, m1_frac=1, m2_frac=1, rand_im=False, rand_m1=False, rand_m2=True)),
        ExpConfig("Randomize image, both spurious: R 100 100", ExperimentConfig(im_frac=1, m1_frac=1, m2_frac=1, rand_im=True, rand_m1=False, rand_m2=False))
    )


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
    

@dataclass
class OptimizerConfig:
    optimizer_type: OptimizerType = OptimizerType.SGD
    base_lr = 1e-3 # Base LR for SGD
    weight_decay: float = 0.0
    momentum: float = 0.0
    # For cosine LR scheduler
    cosine_lr: bool = False
    initial_lr: int = 1e-1
    final_lr: int = 1e-5


@dataclass
class MainConfig:
    model_type: ModelType
    dataset_type: DatasetType
    config_group: Tuple[ExpConfig, ...] # Group of specified configs - allows separate experiment types to be run for each dataset
    distill_loss_type: DistillLossType
    aug_type: AugType
    teach_data_num: int = 0
    dist_data_num: int = 0 # Which counterfactual dataset
    
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    
    distill_epochs: int
    epochs: int
    train_bsize: int = 64
    eval_bsize: int = 16
    
    wandb_run: bool = False # Whether to log to wandb
    is_sweep: bool = False


    def __post_init__(self):
        t_exp_name = self.config_group[self.teach_data_num].name.split(":")[-1].strip()
        s_exp_name = self.config_group[self.dist_data_num].name.split(":")[-1].strip()
        

@dataclass
class HyperparamConfig(MainConfig):
    """May move a few of these things to hydra."""
    N_its: int = None # Standardise training length where dataset size can vary between runs.
    t: int = 30 # KL soft base distillation temperature
    tau: float = 0.1 # Contrastive loss temperature