#!/bin/bash
#SBATCH --job-name=basic-workflow
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4           
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00  
#SBATCH --mem=80GB
#SBATCH --partition=kempner
#SBATCH --output=slurm_output/%j.out
#SBATCH --error=slurm_output/%j.err

BASE_PATH="."  # Changed BASE_PATH to point to the current directory
INFERENCE_PATH="${BASE_PATH}/transformer/inference"

module load python/3.12.5-fasrc01
module load cuda/12.2.0-fasrc01

# Initialize Conda/Mamba properly
eval "$(conda shell.bash hook)"  # Initialize shell hook

# Activate the environment using the full path
mamba activate ~/.conda/envs/transformers || source ~/.conda/envs/transformers/bin/activate

# Get latest run number
get_next_run() {
    local latest=$(ls -d ${BASE_PATH}/experiments/run_* 2>/dev/null | sort -t_ -k2 -n | tail -n1 | sed 's/.*run_//')
    if [ -z "$latest" ]; then
        echo 1
    else
        echo $((latest + 1))
    fi
}

RUN_NUMBER=$(get_next_run)
# RUN_NUMBER=35

echo "Starting run $RUN_NUMBER"

python ${BASE_PATH}/synthetic_data_generation/generate_data.py --run $RUN_NUMBER --domain_id "A" --num_steps 100000 --no_overwrite
python ${BASE_PATH}/evaluation/basic_evaluation.py --run $RUN_NUMBER
python ${BASE_PATH}/evaluation/graphs_on_trial_block_transitions.py --run $RUN_NUMBER

# Set up distributed training environment variables
export MASTER_PORT=12355 # there may be a smarter way to set this, but this port is almost always open
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE)) # world size is equal to number of nodes and number of tasks per node
echo "WORLD_SIZE=$WORLD_SIZE"
echo "MASTER_PORT=$MASTER_PORT"

# Define a master address for communication between GPUs
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR=$MASTER_ADDR"

srun python ${BASE_PATH}/transformer/train.py --epochs=10000 --run $RUN_NUMBER --checkpoint_interval=1000 --eval_interval=1000 --predict # --compile  # --enforce_data_epochs

# python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER --step_max=100
# python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER --step_max=1000
# python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER --step_min=1000 --step_max=10000
# python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER --step_min=10000 --step_max=100000
# python ${INFERENCE_PATH}/learning.py --run $RUN_NUMBER

# Automatically remove large learning files
# rm "${BASE_PATH}/experiments/run_${RUN_NUMBER}/seqs/learning_model"*"val_preds.txt"

python ${INFERENCE_PATH}/guess_using_transformer.py --run $RUN_NUMBER
python ${INFERENCE_PATH}/evaluate_transformer_guess.py --run $RUN_NUMBER
python ${INFERENCE_PATH}/graphs_transformer_vs_ground_truth.py --run $RUN_NUMBER

# Find checkpoint files and extract base names
for model_file in "${BASE_PATH}/experiments/run_${RUN_NUMBER}/models/model_"*"cp"*".pth"; do
    if [ -f "$model_file" ]; then
        # Extract basename and remove .pth extension
        model_name=$(basename "$model_file" .pth)
        printf '%*s\n' 80 '' | tr ' ' '-'
        echo -e "\nProcessing checkpoint: $model_name"
        python ${INFERENCE_PATH}/guess_using_transformer.py --run $RUN_NUMBER --model_name "$model_name"
        python ${INFERENCE_PATH}/evaluate_transformer_guess.py --run $RUN_NUMBER --model_name "$model_name"
        python ${INFERENCE_PATH}/graphs_transformer_vs_ground_truth.py --run $RUN_NUMBER --model_name "$model_name"
    fi
done

# Must follow checkpoint predictions
printf '%*s\n' 80 '' | tr ' ' '-'
echo -e "plot_checkpoint_comparison.py\n"
python ${INFERENCE_PATH}/plot_checkpoint_comparison.py --run $RUN_NUMBER