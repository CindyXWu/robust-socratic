# Robust Socratic

## Lent
Image datasets for variety of distillation types. Start with ResNet50 and LeNet5 modified for CIFAR-10. Implement:
- Jacobian matching
- Feature matching
- Contrastive distillation
- Mixup and augmentation techniques

I evaluate distillation on counterfactual datasets for image experiments. The shapes and dominoes datasets have >=2 mechanisms, while CIFAR100/CIFAR10 only has one.

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

## File structure:
### Training:
- All files can be run with 2 options: wandb sweep/run. If argument 'is_sweep' is set to True, a sweep is used (this can also be changed in a variable in the file).
- Choose to input configurations from a config file or run file with parameters hard coded. Both of these are implicit in the arguments passed in. If a YAML configuration file name is passed, then configs are used.
- distill.py: distillation training for all datasets. Does not contain the functions used for training and evaluation, but rather configuration setups. Distillation training takes in 3 optional arguments: 
  - sweep: whether to use a sweep or not. Default False.
  - config_name: name of configuration file to load. Default None, which means no config file is loaded.
  - config_num: index of dictionary in configuarion file to use.
  - Also teacher model, student model, teacher experiment, student experiment, loss type, dataset and whether data is augmented.
  - All configurations are handled as keys to dictionaries in info_dicts.py.
- teacher.py: teacher training files. You need to specify the following either in code or via a config file, via indices into the dictionaries in info_dict.py:
  - Dataset e.g. 0 for CIFAR100, 3 for shapes.
  - Teacher model.
  - Experiment type (specific to dataset, see info_dicts.py).
- info_dicts.py: stores dictionaries relating teacher model / experiment (handles data spurious type and strength) / student model / loss function to the numbers passed around config files. There is a separate experiment dictionary for each dataset, as the names of the counterfactual evaluations are different. You may have to refer to this extensively while setting experiments.
- train_utils.py: utility file for training methods, including evaluation, dataloader creation and teacher/student training.
  - I use a cosine LR scheduler and SGD without momentum.
  - Usual epoch range is 5-30.
  - Counterfactual dataloaders for distillation are loaded in the distillation function using a dictionary in form ['mechanism name to log to wandb plot': dataloader].
  - Data logged every 100 iterations.

### Data:
- shapes_3D.py: class for DeepMind 3D shapes dataset. Object shape and colour is predictive of class label. Colour is a continuous hue variable, which is split into 2 groups. This leads to 8 labels in total. The dataset has by default 6 latent variables. I set 2 spurious mechanisms: the size and floor colour.
- utils_ekdeep.py: Ekdeep's box, colour aug and dominoes datasets for CIFAR-10, CIFAR100 and dominoes dataset. I have editied this so that setting base_dataset to 'dominoes' now automatically means the dominoes dataset with spurious boxes optional on the CIFAR-10 images. This is the best dataset to use for testing as it is fast to train and has 2 mechanisms.
- image_utils.py: dataloader with augmentation for 3D shapes and CIFAR (under construction).

### Losses:
- jacobian_srinivas.py: Jacobian distillation loss. There are two types: a full Jacobian and approximate Jacobian which only calculates the Jacobian on the top-k classes of the output of the model. You can set k in the main Jacobian function (first one in file), but if the model ouptut class number < k it will not use the approximate Jacobian.
- contrastive.py: contrastive distillation loss. This does not use a memory buffer. If the student and teacher are of the same type, then no embedding is used. Normalisation is done on student and teacher feature maps before dot product and exponentiation.
- feature_match.py: feature matching distillation loss (deprecated as of March).

### Models:
- image_models.py: edited models for training. As of April these are: ResNet18, ResNet50, and a variable size wide resnet function. The ResNet18 and ResNet50 are pre-trained from PyTorch's library with adaptive pooling in the last layer for variable output size. The wide resnet also uses adaptive pooling, but is Kaiming initialised and not PyTorch's basic implementation. Its layer numbers also vary as 16, 20, 26...
  - Learning rates for the PyTorch initialised models should be 0.1-0.01 with cosine LR scheduler.
- basic1_models.py: basic FC networks used for 1D testing from Michaelmas.
- The folder Image_Experiments contains all trained teacher models I have thus far. distill.py loads these. They also have a 'test_acc' associated which is automatically printed for comparison.

## Michaelmas
Custom dataset counterfactual training. Dataset is made up of 1D vectors with 3 latent features each - each are slabs from -1 to 1. 1 feature entirely predictive of the dataset and 2 with complex but learnable patterns. A battery of 9 experiments performed to investigate whether distillation preserves mechanistic properties.

- basic1_teacher_train: train, test and save teacher MLP, implementation of Jose Horas’ knowledge distillation Github
- basic1_student_train: train and test student MLP
- utils: contains custom dataset class(es) [1. y binary, x-k-slabs].

