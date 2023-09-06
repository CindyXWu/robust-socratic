#!/bin/bash

# This script manages the main config name via command line passing to Hydra
# It does not manage the sweep file name as of 14/08
SCRIPT_PATH="Lent/run_distill.py"
config_filename="distill_config"
experiment="exhaustive_0" # Frac 0 - i.e. teacher trained on totally clean data
experiment_s="frac_0"
args="config_type=FRAC distill_loss_type=JACOBIAN dataset.box_cue_pattern=RANDOM experiment_s.experiment_config.m1_frac=90,95,96,97,98,99,100"

for i in {1..3}
do
    HYDRA_FULL_ERROR=1 python $SCRIPT_PATH -m +experiment=$experiment +experiment_s=$experiment_s $args --config-name=$config_filename
done