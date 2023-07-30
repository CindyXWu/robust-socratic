"""Bunch of dataclasses and enums to group immutable variables into classes."""
import numpy as np
from enum import Enum
import yaml
from dataclasses import dataclass, field
from typing import Any, Tuple
from collections import namedtuple

from models.image_models import *


class ModelType(str, Enum):
    RESNET18_ADAPTIVE_POOLING = "RN18AP"
    RESNE20_WIDE = "RN20W"
    LENET5_3CHAN = "LENET5"


class DistillLossType(str, Enum):
    BASE = "BASE"
    JACOBIAN = "JACOBIAN"
    CONTRASTIVE = "CONTRASTIVE"


class DatasetType(str, Enum):
    DOMINOES = "DOMINOES"
    SHAPES = "SHAPES"
    PARITY = "PARITY"
    CIFAR100 = "CIFAR100"
    CIFAR10 = "CIFAR10"


class AugType(str, Enum):
    NONE = "NONE"
    MIXUP = "MIXUP"
    UNION_DATASET = "UNION"


class OptimizerType(str, Enum):
    SGD = "SGD"
    ADAM = "ADAM"
    ADAMW = "ADAMW"
    
    
@dataclass
class ExperimentConfig:
    """If there is only one mechanism, values of mech 2 frac and random mech 2 do not matter."""
    im_frac: float = 1.0
    m1_frac: float = field(default=0.0, metadata={'tags': ['BOX', 'FLOOR']})
    m2_frac: float = field(default=0.0, metadata={'tags': ['MNIST', 'SCALE']})
    rand_im: bool = False
    rand_m1: bool = False
    rand_m2: bool = False


ExpConfig = namedtuple('ExpConfig', ['name', 'config'])


@dataclass
class ConfigGroups:
    """Immutable ordered tuples of experimental configurations. 
    Ordered tuples support direct indexing.
    E.g. exhaustive_configs[0].config should give ExperimentConfig(im_frac=1, m1_frac=0, m2_frac=0, rand_im=False, rand_m1=False, rand_m2=False).
    
    This is used mostly as a storage format - now deprecated for use in code with Hydra.
    But creating new configs should be done by changing values here first.
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
    targeted_configs = (
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


@dataclass
class DataLoaderConfig:
    """
    train_fraction: Fraction of dataset to be set aside for training.
    batch_size: For both train and test.
    seed: Random seed for reproducibility, ensuring fn returns same split with same args.
    """
    train_bs: int = 64
    test_bs: int = 32
    num_workers: int = 1
    train_fraction: float = 0.95
    shuffle_train: bool = True
    seed: int = 42
    

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


# TODO: Check Hydra config grouping.
@dataclass
class MainConfig:
    """Does not include which config group to load experiment from. This is specified from command line via counterfactual_config=counterfactual_0 [citation needed]."""
    model_type: ModelType
    dataset_type: DatasetType
    num_training_iter: int
    distill_epochs: int
    teacher_epochs: int
    
    distill_loss_type: DistillLossType
    aug_type: AugType
    teach_data_num: int = 0
    dist_data_num: int = 0 # Which counterfactual dataset
    dist_temp: float = 30 # Base distillation temperature
    contrast_temp: float = 0.1 # Contrastive loss temperature
    
    optimization: OptimizerConfig = field(default_factory=OptimizerConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    
    log_to_wandb: bool = False # Whether to log to wandb
    save_model_as_artifact: bool = True
    wandb_project_name: str = "iib-fcnn" # Serves as base project name - model type and dataset also included
    model_save_path: str = "trained_models"
    is_sweep: bool = False

    # TODO: Change this
    def __post_init__(self):
        t_exp_name = self.config_group[self.teach_data_num].name.split(":")[-1].strip()
        s_exp_name = self.config_group[self.dist_data_num].name.split(":")[-1].strip()
        

def config_to_yaml(configs, filename_prefix):
    for i, config in enumerate(configs):
        filename = f"{filename_prefix}_{i}.yaml"
        with open(filename, 'w') as file:
            yaml.dump({'name': config.name, 'config': vars(config.config)}, file)


def create_new_configs():
    config_to_yaml(ConfigGroups.targeted_configs, 'lent/configs/targeted_configs/targeted')
    config_to_yaml(ConfigGroups.exhaustive_configs, 'lent/configs/exhaustive_configs/exhaustive')
    config_to_yaml(ConfigGroups.counterfactual_configs, 'lent/configs/counterfactual_configs/counterfactual')
        
        
if __name__ == "__main__":
    pass