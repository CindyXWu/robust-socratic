# Look at these as reference to set values for above variables
student_dict = {0: "LeNet5", 1: "ResNet18_AP", 2: "ResNet20_Wide"}
# Wide might be a misnomer but oh well it's stuck now
teacher_dict = {0: "LeNet5", 1: "ResNet18_AP", 2: "ResNet50_AP", 3: "ResNet20_Wide"}
loss_dict  = {0: "Base Distillation", 1: "Jacobian", 2: "Contrastive"}
aug_dict = {0: "None", 1: "Mixup et al", 2: "Union of datasets"}
dataset_dict = {0: "CIFAR100", 1: "Dominoes", 2: "Shapes"}

# Exhaustive experiments
shapes_exp_dict = {0: "Shape_Color", 
                   1: "Floor", 
                   2: "Scale", 
                   3: "Floor_Scale", 
                   4: "Shape_Color_Floor", 
                   5: "Shape_Color_Scale", 
                   6: "Shape_Color_Floor_Scale"}
# List elements: image frac, mech 1 frac (box), mech 2 frac (MNIST), random image, random mech 1, random mech 2
dominoes_exp_dict = {"CIFAR10": [1, 0, 0, False, False, False], 
                     "Box": [1, 1, 0, True, False, False], 
                     "MNIST": [1, 0, 1, True, False, False], 
                     "MNIST_Box": [1, 1, 1, True, False, False], 
                     "CIFAR10_MNIST": [1, 0, 1, False, False, False],
                     "CIFAR10_Box": [1, 1, 0, False, False, False], 
                     "CIFAR10_MNIST_Box": [1, 1, 1, False, False, False]}
# For single mechanism the second element is ignored so all values are just placeholders
cifar_exp_dict = {"P": [1, 0, 0, False, False, False], 
                  "B": [1, 1, 0, True, False, False], 
                  "BR": [1, 1, 0, True, True, False]}

#====================================================================================================================
# List elements: image frac, mech 1 frac, mech 2 frac, random image, random mech 1, random mech 2
# If there is only one mechanism, then skip references to mech 2 frac and random mech 2
exp_dict_all = {"No mechanisms (baseline): M=100% S1=0% S2=0%": [1, 0, 0, False, False, False], 
                  "Teacher one spurious mechanism: M=100% S1=0% S2=60%": [1, 0, 0.6, False, False, False], 
                  "Teacher both spurious mechanisms: M=100% S1=30% S2=60%": [1, 0.3, 0.6, False, False, False],
                  "Student both spurious mechanism: M=100% S1=60% S2=90%": [1, 0.6, 0.9, False, False, False],
                  "Student both spurious mechanisms: M=100% S1=90% S2=60%": [1, 0.9, 0.6, False, False, False]
                  }

# Counterfactual evals
# Experiment number implicit in index of list converted dictionary
# List elements: image frac, mech 1 frac, mech 2 frac, random image, random mech 1, random mech 2
counterfactual_dict_all = {"All mechanisms: M=100% S1=100% S2=100%": 
                           [1, 1, 1, False, False, False], 
                           "Only spurious mechanisms: M=0% S1=100% S2=100%": [0, 1, 1, False, False, False], 
                           "Randomize spurious mechanisms: M=100% S1=randomized S2=100%": [1, 1, 1, False, True, False], 
                           "Randomize spurious mechanisms: M=100% S1=100% S2=randomized": [1, 1, 1, False, False, True], 
                           "Randomize image: M=randomized S1=100% S2=100%": [1, 1, 1, True, False, False]
                           }

# Use shortening for legends in plots
mech_map = {"CIFAR10": "C", 
            "Box": "B", 
            "MNIST": "M", 
            "MNIST_Box": "MB", 
            "CIFAR10_MNIST": "CM", 
            "CIFAR10_Box": "CB", 
            "CIFAR10_MNIST_Box": "CMB"}

new_mech_map = {"M=100% S1=0% S2=0%": "100 0 0", 
                  "M=100% S1=0% S2=60%": "100 0 60", 
                  "M=100% S1=30% S2=60%": "100 30 60",
                  "M=100% S1=60% S2=0%": "100 60 0",
                  "M=100% S1=60% S2=30%": "100 60 30"
                  }

new_mech_list = ["100 0 0", "100 0 60", "100 30 60", "100 60 0", "100 60 30"]

# # Temp to wrestle naming into shape to undo my bug
# temp_dict = {"M=0% S1=100% S2=100%": "M=100% S1=randomized S2=100%", 
#              "M=100% S1=randomized S2=100%": "M=100% S1=100% S2=randomized", 
#              "M=100% S1=100% S2=randomized": "M=randomized S1=100% S2=100% ", 
#              "M=randomized S1=100% S2=100%": "M=100% S1=randomized S2=100%"}