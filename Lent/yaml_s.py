import os
import yaml

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Define the filename for your configuration file
filename = "CIFAR100_distill.yml"
s_exp_nums = [0, 1, 2]
t_exp_nums = [1, 2]
teacher_nums = [3]
student_nums = [2]
loss_nums = [0]
aug_nums = [0]

# Todo: add augmentation
experiments = []
for s_exp_num in s_exp_nums:
    for t_exp_num in t_exp_nums:
        for t_num in teacher_nums:
            for loss_num in loss_nums:
                for s_num in student_nums:
                    experiments.append({'teacher_num': t_num, 'student_num': s_num, 's_exp_num': s_exp_num, 't_exp_num': t_exp_num, 'loss_num': loss_num})

with open(filename, "w") as f:
    yaml.dump(experiments, f)
