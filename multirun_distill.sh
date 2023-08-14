#!/bin/bash

# Path to your script
SCRIPT_PATH="Lent/run_distill.py"
experiment="exhaustive_0"
experiment_s="exhaustive_0,exhaustive_1,exhaustive_2,exhaustive_3,exhaustive_4,exhaustive_5,exhaustive_6"
args="distill_loss_type=BASE, dataset.box_cue_type=MANDELBROT"

for i in {1..5}
do
    python $SCRIPT_PATH -m +experiment=$experiment +experiment_s=$experiment_s $args
done