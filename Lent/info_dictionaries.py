# Look at these as reference to set values for above variables
exp_dict = {0: 'plain', 1: 'box', 2: 'box_random', 3: 'box_half', 4: 'box_random_half'}
student_dict = {0: "LeNet5_CIFAR10"}
teacher_dict = {0: "LeNet5_CIFAR10", 1: "ResNet50_CIFAR10", 2: "ResNet18_CIFAR10", 3: "ResNet18_CIFAR100", 4: "ResNet50_CIFAR100"}
loss_dict  = {0: "Base Distillation", 1: "Jacobian", 2: "Feature Map", 3: "Attention Jacobian"}
aug_dict = {0: "None", 1: "Mixup et al", 2: "Union of datasets"}

# For shapes dataset training
s_teacher_dict = {0: "ResNet18_3DShapes", 1: "ResNet50_3DShapes", 2: "ResNet18_Flexi"}
s_exp_dict = {0: "Plain", 1: "Floor", 2: "Wall", 3: "Scale", 4: "Orientation"}