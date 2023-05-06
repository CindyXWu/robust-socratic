#!/bin/bash

# Set the argument to pass to myscript.sh
config_file="configs/Dominoes_s.yml"
configs=$(python -c "import yaml; print(yaml.safe_load(open('$config_file')))")

# Define the number of configurations
num_configs=$(python -c "import yaml; print(len(yaml.safe_load(open('$config_file'))))")

# Loop through each configuration and run the Python script
for ((i=0; i<$num_configs; i++))
do
    sbatch --export=CONFIG_FILE=$config_file,IDX=$i batch_s
done
