import os
import yaml
from info_dicts import *

os.chdir(os.path.dirname(os.path.abspath(__file__)))

dataset_num = 1 # Defines filename for yaml file
t_exp_nums = [0, 1, 2]
teacher_nums = [3]

dataset = dataset_dict[dataset_num]
filename = dataset+"_t.yml"

experiments = []
for t_exp_num in t_exp_nums:
    for t_num in teacher_nums:
        experiments.append({'teacher_num': t_num, 'exp_num': t_exp_num,})

with open(filename, "w") as f:
    yaml.dump(experiments, f)
