#!/bin/bash
# Run array job via SLURM passing in individual config files in command line for Hydra
# If not on SLURM I would expect to just pass in all the config files in a list for Hydra multirun
exp_group="targeted"
rel_config_folder="../configs/"
num_configs=$(ls "${rel_config_folder}/${exp_group}"*.yaml 2> /dev/null | wc -l)

sbatch --export=exp_group=$exp_group --array=0-$((num_configs-1)) array_s
