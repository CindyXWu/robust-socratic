import os
import yaml

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Define the filename for your configuration file
filename = "shapes_t.yml"
t_exp_nums = [0]
teacher_nums = [1, 2, 3]

experiments = []
for t_exp_num in t_exp_nums:
    for t_num in teacher_nums:
        experiments.append({'teacher_num': t_num, 't_exp_num': t_exp_num,})

with open(filename, "w") as f:
    yaml.dump(experiments, f)
