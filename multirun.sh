#!/bin/bash

# This script manages the main config name via command line passing to Hydra
# It does not manage the sweep file name as of 14/08
SCRIPT_PATH="Lent/run_distill.py"
config_filename="distill_config"
experiment="targeted_0"
experiment_s="targeted_0,targeted_3,targeted_4"
args="distill_loss_type=JACOBIAN"

for i in {1..3}
do
    python $SCRIPT_PATH -m +experiment=$experiment +experiment_s=$experiment_s $args --config-name=$config_filename
done