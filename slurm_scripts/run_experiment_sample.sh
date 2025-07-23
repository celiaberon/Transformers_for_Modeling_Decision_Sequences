#!/bin/bash
#SBATCH --job-name=basic-workflow
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1          
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00  
#SBATCH --mem=100GB
#SBATCH --partition=kempner_requeue

# Source common functions
source "./slurm_scripts/common_functions.sh"

# Load modules
module load python/3.12.5-fasrc01
module load cuda/12.2.0-fasrc01

# Activate the environment using the full path
mamba activate ~/.conda/envs/transformers

BASE_PATH="."  # Current directory
INFERENCE_PATH="${BASE_PATH}/transformer/inference"
RUN_NUMBER=1

# Data generation
python ${BASE_PATH}/synthetic_data_generation/generate_data.py \
        --run=$RUN_NUMBER \
        --domain_id=B \
        --num_steps_train=100000 \
    
# Run evaluation for individual datasets
python ${BASE_PATH}/evaluation/basic_evaluation.py --run $RUN_NUMBER
python ${BASE_PATH}/evaluation/graphs_on_trial_block_transitions.py --run $RUN_NUMBER

# Model training
srun --cpu-bind=none python -m transformer.train \
    --run=$RUN_NUMBER \
    --n_layer=4 \
    --n_head=4 \
    --n_embd=64 

# Transformer evaluation
python ${INFERENCE_PATH}/guess_using_transformer.py --run $RUN_NUMBER
python ${INFERENCE_PATH}/evaluate_transformer_guess.py --run $RUN_NUMBER
python ${INFERENCE_PATH}/graphs_transformer_vs_ground_truth.py --run $RUN_NUMBER
python ${BASE_PATH}/evaluation/inspect_data.py --run $RUN_NUMBER
