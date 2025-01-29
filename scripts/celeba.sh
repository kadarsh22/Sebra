#!/bin/bash

# Define project paths
PROJECT_HOME_DIR="/user/HS502/ak03476/PycharmProjects/sebra/"
DATASET_DIR="/vol/research/project_storage/data"
CONDA_ENV="sebra"

# Navigate to project directory
cd "$PROJECT_HOME_DIR" || { echo "Error: Failed to change directory to $PROJECT_HOME_DIR"; exit 1; }

# Activate Conda environment (using conda activate instead of source)
if [ -f "/vol/research/project_storage/miniconda3/bin/activate" ]; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
else
    echo "Error: Conda activate script not found at /vol/research/project_storage/miniconda3/bin/activate"
    exit 1
fi

# Ensure working directory is correct
cd "$PROJECT_HOME_DIR" || { echo "Error: Failed to change directory to $PROJECT_HOME_DIR"; exit 1; }
export PYTHONPATH="$PROJECT_HOME_DIR"

# Run Python script with arguments (fixing missing values for --seeds)
python3 celeba_trainers/launcher.py --method sebra --wandb --beta_inverse 0.8 --p_critical 0.7 --lr 0.001 --momentum 0.8 --weight_decay 0.0001 --gap 3 --classifier_weight 1 --temperature 0.05 --save_ckpt --log_models