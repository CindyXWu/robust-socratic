# Robust Socratic
Repository for all IIB project - robust model distillation.

Model distillation with mechanistic robustness.

## Lent
Image datasets for variety of distillation types. Start with ResNet50 and LeNet5 modified for CIFAR-10. Implement:
- Jacobian matching
- Feature matching
- Contrastive distillation
- Mixup and augmentation techniques
I evaluate distillation on counterfactual datasets for image experiments.

## Michaelmas
Custom dataset counterfactual training. Dataset is made up of 1D vectors with 3 latent features each - each are slabs from -1 to 1. 1 feature entirely predictive of the dataset and 2 with complex but learnable patterns. A battery of 9 experiments performed to investigate whether distillation preserves mechanistic properties.

- basic1_teacher_train: train, test and save teacher MLP, implementation of Jose Horas’ knowledge distillation Github
- basic1_student_train: train and test student MLP
- utils: contains custom dataset class(es) [1. y binary, x-k-slabs].

