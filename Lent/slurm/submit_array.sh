#!/bin/bash

config_file="configs/Dominoes_s.yml"
num_configs=$(python -c "import yaml; print(len(yaml.safe_load(open('$config_file'))))")

sbatch --export=config_file=$config_file --array=0-$((num_configs-1)) array_s
