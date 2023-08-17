#!/bin/bash

# Change this - filename for sweep setup, not main config
config_filename='distill_sweep_test_config'


config_filepath="Lent/configs/$config_filename.yaml"
is_sweep=$(python -c "import yaml; config=yaml.safe_load(open($config_filepath)); print(config.get('is_sweep', False))")

if [ "$is_sweep" == "True" ]; then
    wandb_project_name=$(python -c "import yaml; config=yaml.safe_load(open($config_filepath)); print(f\"{config['wandb_project_name']} DISTILL {config['model_type']} {config['dataset_type']} {config['config_type']} {config['dataset']['box_cue_pattern']}\")")

    sweep_num=$(python -c "import yaml; config=yaml.safe_load(open('configs/$config_filename.yaml')); print(config['sweep_num'])")

    sweep_id=$(wandb sweep --project $wandb_project_name "$config_filepath.yaml")

    wandb agent --count $sweep_num $sweep_id
fi