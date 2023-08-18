#!/bin/bash

# Change this - filename for sweep setup, not main config
config_filename="distill_sweep_test_config"
sweep_filename="contrastive_sweep"
config_filepath="Lent/configs/$config_filename.yaml"
sweep_filepath="Lent/configs/sweep/$sweep_filename.yaml"

is_sweep=$(python -c "import yaml; config=yaml.safe_load(open('$config_filepath')); print(config.get('is_sweep', False))")

if [ "$is_sweep" == "True" ]; then
    wandb_project_name=$(python -c "import yaml; config=yaml.safe_load(open('$config_filepath')); print(f\"DISTILL-{config['model_type']}-{config['dataset_type']}-{config['config_type']}-{config['dataset']['box_cue_pattern']}{config['wandb_project_name']}\")")
    sweep_num=$(python -c "import yaml; config=yaml.safe_load(open('$config_filepath')); print(config['sweep_num'])")
    # Turns out WandB logs to stderr instead of stdout
    sweep_output=$(wandb sweep --project "$wandb_project_name" "$sweep_filepath" 2>&1)
    sweep_id=$(echo "$sweep_output" | grep -oP 'with ID: \K[^ ]+')
    wandb agent --count $sweep_num "iib-mech-robust/$wandb_project_name/$sweep_id"
fi
