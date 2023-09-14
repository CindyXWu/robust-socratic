#!/bin/bash

# This script manages the main config name via command line passing to Hydra
# It does not manage the sweep file name as of 14/08
SCRIPT_PATH="Lent/run_distill.py"
config_filename="distill_config"
experiment="exhaustive_1" # T mech A
experiment_s="exhaustive_1" # S mech A
args="config_type=EXHAUSTIVE distill_loss_type=BASE dataset.box_cue_pattern=RANDOM"

for i in {1..1}
do
    HYDRA_FULL_ERROR=1 python $SCRIPT_PATH -m +experiment=$experiment +experiment_s=$experiment_s $args --config-name=$config_filename
done