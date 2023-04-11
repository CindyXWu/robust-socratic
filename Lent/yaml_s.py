import os
import yaml

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Define the filename for your configuration file
dataset = "Dominoes"
filename = dataset+"_s.yml"
s_exp_nums = [0, 1, 2, 3, 4, 5, 6]
t_exp_nums = [0, 1, 2, 3, 4, 5, 6]
teacher_nums = [1]
student_nums = [1]
loss_nums = [0, 1, 2]
aug_nums = [0]

# Todo: add augmentation
experiments = []
for s_exp_num in s_exp_nums:
    for t_exp_num in t_exp_nums:
        for t_num in teacher_nums:
            for loss_num in loss_nums:
                for s_num in student_nums:
                    experiments.append({'dataset': dataset, 'teacher_num': t_num, 'student_num': s_num, 's_exp_num': s_exp_num, 't_exp_num': t_exp_num, 'loss_num': loss_num})

with open(filename, "w") as f:
    yaml.dump(experiments, f)
