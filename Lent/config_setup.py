"""Bunch of dataclasses and enums to group immutable variables into classes."""
from enum import Enum
import yaml
from dataclasses import dataclass, field
from typing import Optional
from collections import namedtuple


class ModelType(str, Enum):
    MLP = "MLP"
    RESNET18_ADAPTIVE_POOLING = "RN18AP"
    RESNE20_WIDE = "RN20W"
    LENET5_3CHAN = "LENET5"


class DistillLossType(str, Enum):
    BASE = "BASE"
    JACOBIAN = "JACOBIAN"
    CONTRASTIVE = "CONTRASTIVE"


class LossType(str, Enum):
    KL = "KL"
    MSE = "MSE"
    CE = "CE"


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
class ResNetConfig:
    blocks_per_stage: int = 3 # Gives ResNet20: 3 blocks * 3 stages * 2 layers per block + input layer + output layer
    _width_factor: int = field(default=1, repr=False) # Hide this field for repr

    @property
    def width_factor(self):
        return self._width_factor


@dataclass
class DatasetConfig:
    dataset_filename = "filename.csv"
    data_folder: str = "data"
    output_size: int = 10


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
    base_lr = 3e-1 # Base LR for SGD
    weight_decay: float = 0.0
    momentum: float = 0.0
    # For cosine LR scheduler
    final_lr: int = 1e-3
    cosine_lr_schedule: bool = True


# TODO: Consider making this a MainConfig and creating a separate DistillConfig
@dataclass
class MainConfig:
    """Does not include which config group to load experiment from. This is specified from command line via Hydra multirun."""
    model_type: ModelType
    dataset_type: DatasetType

    # Distillation stuff
    is_distill: bool = False
    distill_loss_type: Optional[DistillLossType]
    aug_type: Optional[AugType]
    nonbase_loss_frac: Optional[float]
    """Weighting of this loss term with respect to base soft label matching distillation."""
    dist_temp: Optional[float] = 30
    jac_temp: Optional[float] = 1
    contrast_temp: Optional[float] = 0.1 # 0.1 recommended in paper
    # Initialise these in the main function
    s_layer: Optional[str] = None 
    t_layer: Optional[str] = None
    jacobian_loss_type: LossType = LossType.MSE
    
    # Stuff from other dataclasses
    optimization: OptimizerConfig = field(default_factory=OptimizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)

    # Model-specific init
    resnet_config: Optional[ResNetConfig] = field(default_factory=ResNetConfig)

    # Training 
    distill_iters: int = 8000 # Neither of these values are directly used in the code, but are used to calculate the other values
    teacher_iters: int = 4000
    eval_frequency: int = 100
    """How many iterations between evaluations. If None, assumed to be 1 epoch, if the dataset is not Iterable."""
    num_eval_batches: Optional[int] = 20
    """
    How many batches to evaluate on. If None, evaluate on the entire eval dataLoader.
    Note, this might result in infinite evaluation if the eval dataLoader is not finite.
    """
    teacher_save_path: str = "trained_teachers"
    student_save_path: str = "trained_students"

    # Logging
    log_to_wandb: bool = False # Whether to log to wandb
    save_model_as_artifact: bool = True
    wandb_run_name: str = "run" # Initialise in main function
    is_sweep: bool = False

    def __post_init__(self):  
        match self.dataset_type:
            case DatasetType.DOMINOES:
                self.dataset.output_size = 10
            case DatasetType.SHAPES:
                self.dataset.output_size = 8
            case DatasetType.CIFAR100:
                self.dataset.output_size = 100
            case DatasetType.CIFAR10:
                self.dataset.output_size = 10
            case _:
                raise ValueError(f"Unknown dataset type {self.dataset_type}")
        
        match self.distill_loss_type:
            case DistillLossType.JACOBIAN:
                self.nonbase_loss_frac = 0.5
            case DistillLossType.CONTRASTIVE:
                self.nonbase_loss_frac = 0.01
            case _:
                raise ValueError(f"Unknown loss type {self.distill_loss_type}")
        
        self.wandb_project_name = f"{self.model_type} {self.dataset_type}"
        self.iters_used = self.distill_iters if self.is_distill else self.teacher_iters


@dataclass
class ExperimentConfig:
    """If there is only one mechanism, values of mech 2 frac and random mech 2 do not matter."""
    im_frac: float = 1.0
    m1_frac: float = field(default=0.0, metadata={'tags': ['BOX', 'FLOOR']})
    m2_frac: float = field(default=0.0, metadata={'tags': ['MNIST', 'SCALE']})
    rand_im: bool = False
    rand_m1: bool = False
    rand_m2: bool = False


ExpConfig = namedtuple('ExpConfig', ['name', 'experiment_config'])


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


def config_to_yaml(configs, filename_prefix):
    for i, config in enumerate(configs):
        filename = f"{filename_prefix}_{i}.yaml"
        with open(filename, 'w') as file:
            yaml.dump({'name': config.name, 'experiment_config': vars(config.experiment_config)}, file)


def create_new_configs():
    config_to_yaml(ConfigGroups.targeted_configs, 'lent/configs/targeted_configs/targeted')
    config_to_yaml(ConfigGroups.exhaustive_configs, 'lent/configs/exhaustive_configs/exhaustive')
    config_to_yaml(ConfigGroups.counterfactual_configs, 'lent/configs/counterfactual_configs/counterfactual')
        
        
if __name__ == "__main__":
    pass