# Look at these as reference to set values for above variables
exp_dict = {0: "plain", 1: "box", 2: "box_random"}
student_dict = {0: "LeNet5_CIFAR10", 2: "ResNet18_AP", 3: "ResNet20_Wide"}
# Wide might be a misnomer but oh well it's stuck now
teacher_dict = {0: "LeNet5_CIFAR10", 1: "ResNet18_AP", 2: "ResNet50_AP", 3: "ResNet20_Wide"}
loss_dict  = {0: "Base Distillation", 1: "Jacobian", 2: "Contrastive"}
aug_dict = {0: "None", 1: "Mixup et al", 2: "Union of datasets"}

s_exp_dict = {0: "Shape_Colour", 1: "Floor", 2: "Wall", 3: "Scale", 4: "Orientation"}
