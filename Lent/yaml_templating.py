from jinja2 import Template
import os
import yaml

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Define the filename for your configuration file
filename = "teacher.yaml"
experiment_nums = [0, 1, 2]
teacher_num = 0

# Templating for initial sweeps over loss function and teacher training type for teacher being LeNet5 only
exp_dict = {0: 'plain', 1: 'box', 2: 'box_random', 3: 'box_half', 4: 'box_random_half'}
student_dict = {0: "LeNet5_CIFAR10"}
teacher_dict = {0: "LeNet5_CIFAR10", 1: "ResNet50_CIFAR10", 2: "ResNet18_CIFAR10", 3: "ResNet18_CIFAR100", 4: "ResNet50_CIFAR100"}
loss_dict  = {0: "Base Distillation", 1: "Jacobian", 2: "Feature Map", 3: "Attention Jacobian"}
template = Template("""
experiment_num: {{ experiment_num }}
loss_num: {{ loss_num }}
""")

experiments = []
for exp_num in experiment_nums:
    experiments.append({'experiment_nu100m': exp_num,'teacher_num': teacher_num})

with open(filename, "w") as f:
    yaml.dump(experiments, f)