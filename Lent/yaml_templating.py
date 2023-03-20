from jinja2 import Template
import os
import yaml

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Define the filename for your configuration file
filename = "ResNet_ResNet_CIFAR100_s.yml"
experiment_nums = [0, 1, 2]
teacher_nums = [3, 4]
loss_nums = [0, 1]
student_nums = [1, 2]

template = Template("""
experiment_num: {{ experiment_num }}
teacher_num: {{ teacher_num }}
loss_num: {{loss_num}}
""")

experiments = []
for exp_num in experiment_nums:
    for t_num in teacher_nums:
        for loss_num in loss_nums:
            for s_num in student_nums:
                experiments.append({'experiment_num': exp_num,'teacher_num': t_num, 'student_num': s_num, 'loss_num': loss_num})

with open(filename, "w") as f:
    yaml.dump(experiments, f)
