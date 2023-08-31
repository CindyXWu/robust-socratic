# Robust Socratic

## Installation
- Clone repository into new environenment. 
- Navigate to Lent folder and run
```pip install -r requirements.txt```
- Install compatible PyTorch version for your system, with TorchVision (see PyTorch installation page).

## Parameter settings
- All configs are in YAML files managed by Hydra: https://hydra.cc/docs/intro/.
- Base config files provide configs for objects MainConfig and DistillConfig (see Lent/config_setup.py).
- Exact dataset settings for teacher and distillation are then provided separately using `+experiment=teacher_config_filename +experiment_s=student_config_filename` via CLI.
- Bash scripts are also provided for sweeps `sweep.sh, frac.sh` and running experiments with multiple seeds  `multirun.sh`
-   If running sweeps manually (not via `sweep.sh`, use file `distill_sweep.py`) and set the correct config file names at the top of the file.  There is a separate sweep config file, whose name must be specified and parameters checked for correct values before running.
-   `frac.sh` is specifically for experiments iterating over fraction of spurious mechanisms.
-   Sweeps can be done two ways: via Python file or CLI by calling an initialised WandB agent. Due to threading issues with Hydra, the code has not yet been optimised for the former to work without memory leak.
- Project and run names depend on all the model and experiment settings and are set in `run.py, run_distill.py`.

## Example commands to run: teacher training
For all possible combinations of mechanisms in dominoes dataset:
```python Lent/run.py -m +experiment=exhaustive_0,exhaustive_1,exhaustive_2,exhaustive_3,exhaustive_4,exhaustive_5,exhaustive_6```

## Example commands to run: student distillation
For teacher trained on plain CIFAR10 images and student distilled on all possible combinations of mechanisms in dominoes dataset, with Jacobian matching loss:
```python Lent/run_distill.py -m +experiment=exhaustive_0 +experiment_s=exhaustive_0,exhaustive_1,exhaustive_2,exhaustive_3,exhaustive_4,exhaustive_5,exhaustive_6 distill_loss_type=JACOBIAN```


## Recommended hyperparameters from search
Contrastive:
```
nonbase_loss_frac: 0.3
base_lr: 0.05
contrast_temp: 0.1 # Not so important
```
Jacobian:
```
nonbase_loss_frac: 0.15-0.2
```
