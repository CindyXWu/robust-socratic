# Look at these as reference to set values for above variables
exp_dict = {0: "plain", 1: "box", 2: "box_random"}
student_dict = {0: "LeNet5_CIFAR10", 1: "ResNet18_CIFAR100", 2: "ResNet18_Flexi"}
teacher_dict = {0: "LeNet5_CIFAR10", 1: "ResNet50_CIFAR10", 2: "ResNet18_CIFAR10", 3: "ResNet18_CIFAR100", 4: "ResNet50_CIFAR100", 5: "ResNet50Flexi_CIFAR100"}
loss_dict  = {0: "Base Distillation", 1: "Jacobian", 2: "Contrastive"}
aug_dict = {0: "None", 1: "Mixup et al", 2: "Union of datasets"}

# For shapes dataset training
s_teacher_dict = {0: "ResNet18_3DShapes", 1: "ResNet50_3DShapes", 2: "ResNet18_Flexi", 3: "ResNet50_Flexi"}
s_exp_dict = {0: "Shape_Colour", 1: "Floor", 2: "Wall", 3: "Scale", 4: "Orientation"}
