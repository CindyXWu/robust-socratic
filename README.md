# Robust Socratic

## Installation
- Clone repository into new environenment. 
- Navigate to Lent folder and run
```pip install -r requirements.txt```
- Install compatible PyTorch version for your system, with TorchVision (see PyTorch installation page).

## YAML scripting and SLURM:
- yaml_s.py, yaml_t.py: create YAML files with user specified configs, given as lists.
  - For a student: set teacher model, student model, loss type, experiment number (mechanism type) and dataset. These are keys (indices) of entries in dictionaries in info_dicts.py. The dataset parameter is necessary because it tells the distillation training which counterfactual dataloaders are needed.
  - For a teacher: set teacher model and experiment number.
  - The format will be a list of dictionaries. The way the config file is read, is by indexing this list of dictionaries with an experiment index. This is passed in a loop in a file which calls sbatch on SLURM files.
- Files of form submit_\*.sh are bash scripting files which pass in YAML config files to call sbatch on the right configurations. All you need to do to run experiments with the correct YAML settings are change the config file name and SLURM script name.
- Files named batch_* and array_s are SLURM files.
- The file submit_array submits an array run in SLURM (better than calling multiple sbatch). You can choose to use this or not.
- I have multiple SLURM scripts because I couldn't be bothered to change the file name being run in the SLURM scripts when calling them. These are batch_s (for distillation), batch_t (for CIFAR teacher training), batch_shapes and batch_dominoes. The ones with manual at the end are ones to be run in command line rather than by calling the bash scripting files.

## Using wandb:
- Project and run names depend on all the model and experiment settings and are set in distill.py.
- If using a wandb sweep, the sweep parameters must be set in the sweep dictionary as given by wandb documentation. Examples are provided in the code.
- Note on sweeps: best sweep types to use are probabably 'grid' (goes through all options) and 'bayes' (uses GP to model distributions of parameter settings).

## Setting parameters:
- The variables edited in the distillation files by the config are:
  - TEACH_NUM, STUDENT_NUM, LOSS_NUM, AUG_NUM, DATASET_NUM which define the exeriment you're running and are keys of dictionaries in info_dicts.py. The entries of these dictionaries are more descriptive strings, e.g. teacher_dict[TEACH_NUM=0] is "LeNet".
- The initial and final learning rate are not edited by the config file. At the current moment I have them set differently depending on the loss function being used, but I may add extra levels of granularity for selection of LR based on e.g. model and dataset. As expected, currently not all models train amazingly.
- All variables not edited by config file:
  - Initial LR, final LR, epochs, temperature (fix at 20-30), tau (temperature for contrastive distillation, fixed at 0.1 for now), alpha (fraction of contrastive/Jacobian loss to use - in future may automatically set by evaluating both once, and setting alpha such that the size of both normal distillation and other losses is comparable), sweep configurations
  - Layer names for contrastive distillation: in image_models.py there is a function that lists a dictionary of all layer names. I use this to extract a per-model name of the last feature map layer, and set this when initialising the models.
  - Batch size: default 64.

