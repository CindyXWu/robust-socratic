import os
import yaml
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from info_dicts import *

os.chdir(os.path.dirname(os.path.abspath(__file__)))

dataset_num = 2 # Defines filename for yaml file
s_exp_nums = [1, 3, 4]
t_exp_nums = [0, 1, 2]
teacher_nums = [1]
student_nums = [1]
loss_nums = [0, 1]
aug_nums = [0]

dataset = dataset_dict[dataset_num]
filename = dataset+"_s.yml"

# Todo: add augmentation
experiments = []
for t_exp_num in t_exp_nums:
    for s_exp_num in s_exp_nums:
        for t_num in teacher_nums:
            for loss_num in loss_nums:
                for s_num in student_nums:
                    experiments.append({'dataset_num': dataset_num, 
                                        'teacher_num': t_num,
                                        'student_num': s_num,
                                        's_exp_num': s_exp_num, 
                                        't_exp_num': t_exp_num, 
                                        'loss_num': loss_num
                                        })

with open(filename, "w") as f:
    yaml.dump(experiments, f)
