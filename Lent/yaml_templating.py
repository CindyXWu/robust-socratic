from jinja2 import Template
import os
import yaml

# Templating for initial sweeps over loss function and teacher training type for teacher being LeNet5 only
exp_dict = {0: 'plain', 1: 'box', 2: 'box_random', 3: 'box_half', 4: 'box_random_half'}
student_dict = {0: "LeNet5_CIFAR10"}
teacher_dict = {0: "LeNet5_CIFAR10", 1: "ResNet50_CIFAR10", 2: "ResNet18_CIFAR10"}
loss_dict  = {0: "Base Distillation", 1: "Jacobian", 2: "Feature Map", 3: "Attention Jacobian"}

# Create a directory for your configuration files (if it doesn't already exist)
if not os.path.exists("configs"):
    os.makedirs("configs")
# Define the filename for your configuration file
filename = "configs/LeNet_exp_loss_configs.yml"


template = Template("""
experiment_num: {{ experiment_num }}
loss_num: {{ loss_num }}
""")

experiment_nums = [0, 1, 2]
loss_nums = [0, 1]
teacher_num = 0
student_num = 0

experiments = []
for exp_num in experiment_nums:
    for loss_num in loss_nums:
        experiments.append({'experiment_num': exp_num, 'loss_num': loss_num, 'teacher_num': teacher_num, 'student_num': student_num})

with open(filename, "w") as f:
    yaml.dump(experiments, f)