#!/bin/bash

# This script manages the main config name via command line passing to Hydra
# It does not manage the sweep file name as of 14/08
SCRIPT_PATH="Lent/run.py"
config_filename="main_config"
args=""

for i in {10..19}
do
    experiment="frac_$i"
    python $SCRIPT_PATH -m +experiment=$experiment $args --config-name=$config_filename
done