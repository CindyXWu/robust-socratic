#!/bin/bash

# Set the argument to pass to myscript.sh - this is the file path the training file sees
config_file="configs/Dominoes_t.yml"
# This is the file path this file needs to read the file
rel_config_file="../${config_file}"
configs=$(python -c "import yaml; print(yaml.safe_load(open('$config_file')))")

# Define the number of configurations
num_configs=$(python -c "import yaml; print(len(yaml.safe_load(open('$config_file'))))")

# Loop through each configuration and run the Python script
for ((i=0; i<$num_configs; i++))
do
    sbatch --export=CONFIG_FILE=$config_file,IDX=$i batch_s
done
