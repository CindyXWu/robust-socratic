#!/bin/bash

# Path to your script
SCRIPT_PATH="Lent/run_distill.py"
experiment="exhaustive_0"
experiment_s="exhaustive_0"
args=""

for i in {1..5}
do
    python $SCRIPT_PATH +experiment=$experiment +experiment_s=$experiment_s
done