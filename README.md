# Robust Socratic
Repository for all IIB project - robust model distillation.

Model distillation with mechanistic robustness.

## Requirements
- PyTorch
- einops
- h5py
- wandb
- yaml
- argparse
- tdqm, matplotlib, numpy

## Lent
Image datasets for variety of distillation types. Start with ResNet50 and LeNet5 modified for CIFAR-10. Implement:
- Jacobian matching
- Feature matching
- Contrastive distillation
- Mixup and augmentation techniques

I evaluate distillation on counterfactual datasets for image experiments.

## File structure:
### Training:
- cifar_student_train.py, cifar_teacher_train.py: CIFAR10 and CIFAR100 image datasets teacher and distillation training files. These do not contain the functions used for training and evaluation, but rather configuration setups. Distillation training takes in 3 optional arguments: 
  - sweep: whether to use a sweep or not. Default True.
  - config_name: name of configuration file to load. Default None, which means no config file is loaded.
  - config_num: index of dictionary in configuarion file to use.
  - All configurations are handled as keys to a dictionary, entries of form [int key: str descriptive value]. The dictionary values are in info_dictionaries.
- info_dictionaries: stores dictionaries relating teacher model / experiment (handles data spurious type and strength) / student model / loss function to the numbers passed around config files.
- train_utils.py: utility file for training methods, including evaluation and teacher and student training.
- shapes_student_train.py, shapes_teacher_train.py: 3D shapes dataset training. See file on 3D shapes for more information.

### Data:
- shapes_3D.py: class for DeepMind 3D shapes dataset. Object shape and colour is predictive of class label. Colour is a continuous hue variable, which is split into 3 groups. This leads to 12 labels in total. The dataset has by default 6 latent variables.
- utils_ekdeep.py: Ekdeep's box, colour aug and dominoes datasets for CIFAR-10.
- image_utils.py: dataloader with augmentation for 3D shapes and CIFAR.

### Losses:
- jacobian_srinivas.py: Jacobian distillation loss.
- feature_match.py: feature matching distillation loss (deprecated as of March).
- contrastive.py: contrastive distillation loss.

### Models:
- image_models.py: edited models for training. As of Mar, ResNet18, ResNet50, LeNet on CIFAR-10, and ResNet18/50 on 3D shapes.
- basic1_models.py: basic FC networks used for 1D testing from Michaelmas.

### Configs and utils:
- plotting.py: (deprecated for HPC) plotting functions.
- yaml_templating: turns dictionaries into a yaml config file.

## Michaelmas
Custom dataset counterfactual training. Dataset is made up of 1D vectors with 3 latent features each - each are slabs from -1 to 1. 1 feature entirely predictive of the dataset and 2 with complex but learnable patterns. A battery of 9 experiments performed to investigate whether distillation preserves mechanistic properties.

- basic1_teacher_train: train, test and save teacher MLP, implementation of Jose Horas’ knowledge distillation Github
- basic1_student_train: train and test student MLP
- utils: contains custom dataset class(es) [1. y binary, x-k-slabs].

