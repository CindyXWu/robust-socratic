#!/bin/bash

# This script manages the main config name via command line passing to Hydra
# It does not manage the sweep file name as of 14/08
SCRIPT_PATH="Lent/run_distill.py"
# config_filename="wrn_distill_config"
# experiment="targeted_2"
# experiment_s="targeted_0,targeted_3,targeted_4"
# args="config_type=TARGETED distill_loss_type=BASE dataset.box_cue_pattern=RANDOM log_teacher=False student_wrn_config.width_factor=5,6,7,8"

config_filename="distill_config"
experiment="exhaustive_0"
experiment_s="exhaustive_3,exhaustive_4,exhaustive_5,exhaustive_6"
# experiment_s="exhaustive_1"
args="config_type=EXHAUSTIVE distill_loss_type=JACOBIAN dataset.box_cue_pattern=MANDELBROT log_teacher=False"

# config_filename="distill_config"
# experiment="exhaustive_4"
# experiment_s="exhaustive_3,exhaustive_4,exhaustive_5,exhaustive_6"
# # experiment_s="exhaustive_1"
# args="config_type=EXHAUSTIVE distill_loss_type=JACOBIAN dataset.box_cue_pattern=MANDELBROT log_teacher=False"

for i in {1..2}
do
    HYDRA_FULL_ERROR=1 python $SCRIPT_PATH -m +experiment=$experiment +experiment_s=$experiment_s $args --config-name=$config_filename
done