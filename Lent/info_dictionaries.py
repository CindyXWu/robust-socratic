# Look at these as reference to set values for above variables
exp_dict = {0: 'plain', 1: 'box', 2: 'box_random', 3: 'box_half', 4: 'box_random_half'}
student_dict = {0: "LeNet5_CIFAR10"}
teacher_dict = {0: "LeNet5_CIFAR10", 1: "ResNet50_CIFAR10", 2: "ResNet18_CIFAR10", 3: "ResNet18_CIFAR100", 4: "ResNet50_CIFAR100", 5: "ResNet18_3DShapes", 6: "ResNet50_3DShapes"}
loss_dict  = {0: "Base Distillation", 1: "Jacobian", 2: "Feature Map", 3: "Attention Jacobian"}