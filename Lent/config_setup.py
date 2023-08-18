"""Bunch of dataclasses and enums to group immutable variables into classes."""
from enum import Enum
import yaml
from dataclasses import dataclass, field
from typing import Optional, Any
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


class ConfigType(str, Enum):
    TARGETED = "TARGETED" # Initial targeted experiments set out by Ekdeep
    EXHAUSTIVE = "EXHAUSTIVE"


class BoxPatternType(str, Enum):
    RANDOM = "RANDOM"
    MANDELBROT = "MANDELBROT"
    
    
@dataclass
class WRNArchitectureConfig:
    blocks_per_stage: Optional[int] = None
    width_factor: Optional[int] = None
        
        
@dataclass
class AugmentationConfig:
    """
    If alpha = beta = 1, distribution uniform.
    If alpha >> beta, lam distribution skewed towards 1.
    If alpha << beta, lam distribution skewed towards 0.
    """
    alpha: float = 1.0
    beta: float = 1.0
    mix_prob: float = 0.5
    crop_prob: float = 0.5
    flip_prob: float = 0.5
    rotate_prob: float = 0.5
    
    
@dataclass
class DatasetConfig:
    data_folder: str = "data"
    output_size: Optional[int] = 10
    use_augmentation: bool = False
    box_cue_size: Optional[int] = 4 # Inverse of fraction of image that box cue covers
    box_cue_pattern: Optional[str] = BoxPatternType.MANDELBROT
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class DataLoaderConfig:
    """
    train_fraction: Fraction of dataset to be set aside for training.
    batch_size: For both train and test.
    seed: Random seed for reproducibility, ensuring fn returns same split with same args.
    """
    train_bs: int = 64
    test_bs: int = 64
    num_workers: Optional[int] = 1
    shuffle_train: bool = True
    

@dataclass
class OptimizerConfig:
    optimizer_type: OptimizerType = OptimizerType.SGD
    base_lr = 0.1 # Base LR for SGD
    weight_decay: float = 0.0
    momentum: float = 0.0
    # For cosine LR scheduler
    final_lr: float = 1e-3
    clip_grad: float = float("inf")
    cosine_lr_schedule: bool = True
    optimizer_kwargs: Optional[dict[str, Any]] = field(default_factory=dict)


@dataclass
class MainConfig:
    """Does not include which config group to load experiment from. This is specified from command line via Hydra multirun."""
    model_type: ModelType
    dataset_type: DatasetType
    config_type: str
    aug_type: AugType
    
    # Stuff from other dataclasses
    optimization: OptimizerConfig = field(default_factory=OptimizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)

    # Model-specific init
    wrn_config: Optional[WRNArchitectureConfig] = field(default_factory=WRNArchitectureConfig)

    # Training
    epochs: Optional[int] = None # Instantiate in main function
    num_iters: int = 20000 # Upper bound - see early stopping - do not set lower than iterations for 1 epoch or will not run
    min_iters: int = 8000
    eval_frequency: int = 100
    """How many iterations between evaluations. If None, assumed to be 1 epoch, if the dataset is not Iterable."""
    num_eval_batches: Optional[int] = 50
    """
    How many batches to evaluate on. If None, evaluate on the entire eval dataLoader.
    Note, this might result in infinite evaluation if the eval dataLoader is not finite.
    """
    early_stop_patience: int = 20 # Number of epochs with no accuracy improvement before training stops
    teacher_save_path: Optional[str] = None
    run_description: Optional[str] = ""

    # Logging
    log_to_wandb: bool = False
    is_sweep: bool = False
    sweep_num: Optional[int] = 10
    save_model_as_artifact: bool = True
    wandb_project_name: Optional[str] = None
    wandb_run_name: Optional[str] = None # Initialise in main function


@dataclass
class DistillConfig(MainConfig):
    distill_loss_type: Optional[DistillLossType] = DistillLossType.BASE # Whether to add extra terms in base distillation
    base_distill_loss_type: Optional[LossType] = LossType.KL # Type of base distillation loss
    jacobian_loss_type: LossType = LossType.MSE
    nonbase_loss_frac: Optional[float] = 0
    """Weighting of this loss term with respect to base soft label matching distillation."""
    dist_temp: float = 30
    jac_temp: float = 1 # Currently NOT USED
    contrast_temp: float = 0.1 # 0.1 recommended in paper
    # Initialise these in the main function
    s_layer: Optional[str] = None
    t_layer: Optional[str] = None
    
    use_early_stop = False
    save_model: Optional[bool] = False
    student_save_path: Optional[str] = None
    teacher_accs: Optional[list[float]] = None


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
    
    Numbers relate to M (image mech), S1 (spurious 1: box or floor colour) and S2 (spurious 2: MNIST or object scale) respectively in the ExpConfig names.
    """
    exhaustive = (
        ExpConfig("I", ExperimentConfig(im_frac=1, m1_frac=0, m2_frac=0, rand_im=False, rand_m1=False, rand_m2=False)), 
        ExpConfig("A", ExperimentConfig(im_frac=0, m1_frac=1, m2_frac=0, rand_im=False, rand_m1=False, rand_m2=False)), 
        ExpConfig("B", ExperimentConfig(im_frac=0, m1_frac=0, m2_frac=1, rand_im=False, rand_m1=False, rand_m2=False)), 
        ExpConfig("AB", ExperimentConfig(im_frac=0, m1_frac=1, m2_frac=1, rand_im=False, rand_m1=False, rand_m2=False)), 
        ExpConfig("IB", ExperimentConfig(im_frac=1, m1_frac=0, m2_frac=1, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("IA", ExperimentConfig(im_frac=1, m1_frac=1, m2_frac=0, rand_im=False, rand_m1=False, rand_m2=False)), 
        ExpConfig("IAB", ExperimentConfig(im_frac=1, m1_frac=1, m2_frac=1, rand_im=False, rand_m1=False, rand_m2=False))
    )


    targeted = (
        ExpConfig("No mechanisms (baseline): 100 0 0", ExperimentConfig(im_frac=1, m1_frac=0, m2_frac=0, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("Teacher one spurious: 100 0 60", ExperimentConfig(im_frac=1, m1_frac=0, m2_frac=0.6, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("Teacher both spurious: 100 30 60", ExperimentConfig(im_frac=1, m1_frac=0.3, m2_frac=0.6, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("Student both spurious: 100 60 90", ExperimentConfig(im_frac=1, m1_frac=0.6, m2_frac=0.9, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("Student both spurious: 100 90 60", ExperimentConfig(im_frac=1, m1_frac=0.9, m2_frac=0.6, rand_im=False, rand_m1=False, rand_m2=False))
    )


    targeted_cf = (
        ExpConfig("All mechanisms: 100 100 100", ExperimentConfig(im_frac=1, m1_frac=1, m2_frac=1, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("Only spurious: 0 100 100", ExperimentConfig(im_frac=0, m1_frac=1, m2_frac=1, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("Randomize S1: 100 R 100", ExperimentConfig(im_frac=1, m1_frac=1, m2_frac=1, rand_im=False, rand_m1=True, rand_m2=False)),
        ExpConfig("Randomize S2: 100 100 R", ExperimentConfig(im_frac=1, m1_frac=1, m2_frac=1, rand_im=False, rand_m1=False, rand_m2=True)),
        ExpConfig("Randomize image, both spurious: R 100 100", ExperimentConfig(im_frac=1, m1_frac=1, m2_frac=1, rand_im=True, rand_m1=False, rand_m2=False))
    )
    
    frac = (
        ExpConfig("10 0 0", ExperimentConfig(im_frac=0.1, m1_frac=0, m2_frac=0, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("25 0 0", ExperimentConfig(im_frac=0.25, m1_frac=0, m2_frac=0, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("50 0 0", ExperimentConfig(im_frac=0.5, m1_frac=0, m2_frac=0, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("75 0 0", ExperimentConfig(im_frac=0.75, m1_frac=0, m2_frac=0, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("90 0 0", ExperimentConfig(im_frac=0.9, m1_frac=0, m2_frac=0, rand_im=False, rand_m1=False, rand_m2=False)),
        
        ExpConfig("100 10 0", ExperimentConfig(im_frac=1, m1_frac=0.1, m2_frac=0, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("100 25 0", ExperimentConfig(im_frac=1, m1_frac=0.25, m2_frac=0, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("100 50 0", ExperimentConfig(im_frac=1, m1_frac=0.5, m2_frac=0, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("100 75 0", ExperimentConfig(im_frac=1, m1_frac=0.75, m2_frac=0, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("100 90 0", ExperimentConfig(im_frac=1, m1_frac=0.9, m2_frac=0, rand_im=False, rand_m1=False, rand_m2=False)),
        
        ExpConfig("100 0 10", ExperimentConfig(im_frac=1, m1_frac=0, m2_frac=0.1, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("100 0 25", ExperimentConfig(im_frac=1, m1_frac=0, m2_frac=0.25, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("100 0 50", ExperimentConfig(im_frac=1, m1_frac=0, m2_frac=0.5, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("100 0 75", ExperimentConfig(im_frac=1, m1_frac=0, m2_frac=0.75, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("100 0 90", ExperimentConfig(im_frac=1, m1_frac=0, m2_frac=0.9, rand_im=False, rand_m1=False, rand_m2=False)),
        
        ExpConfig("100 10 100", ExperimentConfig(im_frac=1, m1_frac=0, m2_frac=0.1, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("100 25 100", ExperimentConfig(im_frac=1, m1_frac=0, m2_frac=0.25, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("0 50 100", ExperimentConfig(im_frac=1, m1_frac=0, m2_frac=0.5, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("0 75 100", ExperimentConfig(im_frac=1, m1_frac=0, m2_frac=0.75, rand_im=False, rand_m1=False, rand_m2=False)),
        ExpConfig("0 90 100", ExperimentConfig(im_frac=1, m1_frac=0, m2_frac=0.9, rand_im=False, rand_m1=False, rand_m2=False)),
    )


def config_to_yaml(configs, filename_prefix):
    for i, config in enumerate(configs):
        filename = f"Lent/configs/experiment/{filename_prefix}_{i}.yaml"
        with open(filename, 'w') as file:
            yaml.dump({'config_filename': f"{filename_prefix}_{i}", 'name': config.name, 'experiment_config': vars(config.experiment_config)}, file)
        filename = f"Lent/configs/experiment_s/{filename_prefix}_{i}.yaml"
        with open(filename, 'w') as file:
            yaml.dump({'config_filename': f"{filename_prefix}_{i}", 'name': config.name, 'experiment_config': vars(config.experiment_config)}, file)


def create_new_configs():
    # config_to_yaml(ConfigGroups.targeted, 'targeted')
#     config_to_yaml(ConfigGroups.exhaustive, 'exhaustive')
    # config_to_yaml(ConfigGroups.targeted_cf, 'lent/configs/counterfactual/cf')
    config_to_yaml(ConfigGroups.frac, 'frac')
        
if __name__ == "__main__":
    create_new_configs()