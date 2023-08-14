#!/bin/bash

# Path to your script
SCRIPT_PATH="Lent/run_distill.py"
experiment="exhaustive_1"
experiment_s="exhaustive_0,exhaustive_1,exhaustive_2,exhaustive_3,exhaustive_4,exhaustive_5,exhaustive_6"
args=""

for i in {1..5}
do
    python $SCRIPT_PATH -m +experiment=$experiment +experiment_s=$experiment_s
done