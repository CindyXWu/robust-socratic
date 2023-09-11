#!/bin/bash

# This script manages the main config name via command line passing to Hydra
# It does not manage the sweep file name as of 14/08
SCRIPT_PATH="Lent/run_distill.py"
config_filename="wrn_distill_config"
experiment="exhaustive_0" # T mech I
experiment_s="exhaustive_5" # S mech IA
args="config_type=EXHAUSTIVE distill_loss_type=BASE dataset.box_cue_pattern=RANDOM student_wrn_config.width_factor=1,2,3,4,5"

for i in {1..3}
do
    HYDRA_FULL_ERROR=1 python $SCRIPT_PATH -m +experiment=$experiment +experiment_s=$experiment_s $args --config-name=$config_filename
done