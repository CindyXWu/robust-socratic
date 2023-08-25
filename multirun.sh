#!/bin/bash

# This script manages the main config name via command line passing to Hydra
# It does not manage the sweep file name as of 14/08
SCRIPT_PATH="Lent/run_distill.py"
config_filename="distill_config"
experiment="exhaustive_6"
experiment_s="exhaustive_0,exhaustive_1,exhaustive_2,exhaustive_3,exhaustive_4,exhaustive_5,exhaustive_6"
args="distill_loss_type=JACOBIAN"

for i in {1..3}
do
    python $SCRIPT_PATH -m +experiment=$experiment +experiment_s=$experiment_s $args --config-name=$config_filename
done