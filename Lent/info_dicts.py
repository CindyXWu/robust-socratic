# Look at these as reference to set values for above variables
exp_dict = {0: "P", 1: "B", 2: "BR"}
student_dict = {0: "LeNet5_CIFAR10", 1: "ResNet18_AP", 2: "ResNet20_Wide"}
# Wide might be a misnomer but oh well it's stuck now
teacher_dict = {0: "LeNet5_CIFAR10", 1: "ResNet18_AP", 2: "ResNet50_AP", 3: "ResNet20_Wide"}
loss_dict  = {0: "Base Distillation", 1: "Jacobian", 2: "Contrastive"}
aug_dict = {0: "None", 1: "Mixup et al", 2: "Union of datasets"}

shapes_exp_dict = {0: "Shape_Color", 1: "Shape_Color_Floor", 2: "Shape_Color_Scale", 3: "Floor", 4: "Color", 5: "Floor_Color"}
dominoes_exp_dict = {0: "CIFAR10", 1: "MNIST", 2: "Box", 3: "MNIST+Box"}