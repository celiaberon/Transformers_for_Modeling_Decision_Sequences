#!/bin/bash
#SBATCH --job-name=basic-workflow
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1          
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00  
#SBATCH --mem=60GB
#SBATCH --partition=kempner_requeue

# Source common functions
source "./slurm_scripts/common_functions.sh"

# Setup environment
setup_environment

# Accept parameters from master runner
RUN_NUMBER=${1:-$(get_next_run)}
N_LAYER=${2:-4}
N_HEAD=${3:-4}
EPOCHS=${4:-100}
TRAIN_STEPS=${5:-100000}
CONTEXT_LENGTH=${6:-12}
EMBD_DIM=${7:-64}
BATCH_SIZE=${8:-256}
DOMAIN_CONFIG=${9:-"domains.ini"}
DOMAIN_ID=${10:-"B"}
USE_STANDARD_DATASET=${11:-"false"}
COMPARISON_DIR=${12:-""}
DEBUG_MODE=${13:-"false"}

export DOMAIN_ID=$DOMAIN_ID
export DOMAIN_CONFIG=$DOMAIN_CONFIG
export EXPERIMENT_TYPE="comparison"
export COMPARISON_DIR=$COMPARISON_DIR
export DEBUG_MODE=$DEBUG_MODE

# Setup standard dataset if requested
setup_standard_dataset \
    --use-standard-dataset "$USE_STANDARD_DATASET" \
    --comparison-dir "$COMPARISON_DIR" \
    --domain-config "$DOMAIN_CONFIG" \
    --domain-id "$DOMAIN_ID" \
    --multiple-domains "true" \
    --train-steps "$TRAIN_STEPS" \
    --val-steps "1000000" \
    --run-number "$RUN_NUMBER"

# Export run number
export RUN_NUMBER
echo "Using run number: $RUN_NUMBER"

# Data generation
print_section_header "Data Generation"
if [ "$USE_STANDARD_DATASET" = "true" ]; then
    generate_standard_dataset
else
    python ${BASE_PATH}/synthetic_data_generation/generate_data.py \
        --run $RUN_NUMBER \
        --domain_id $DOMAIN_ID \
        --num_steps_train=$TRAIN_STEPS \
        --num_steps_val=1_000_000 \
        --no_overwrite \
        --config_file "$DOMAIN_CONFIG" \
        --multiple_domains
    
    # Run evaluation for individual datasets
    python ${BASE_PATH}/evaluation/basic_evaluation.py --run $RUN_NUMBER
    python ${BASE_PATH}/evaluation/graphs_on_trial_block_transitions.py --run $RUN_NUMBER
fi

# Setup distributed environment
setup_distributed_environment

print_section_header "Model Training"
srun --cpu-bind=none python -m transformer.train \
    --epochs=$EPOCHS \
    --run $RUN_NUMBER \
    --batch_size=$BATCH_SIZE \
    --n_layer=$N_LAYER \
    --n_head=$N_HEAD \
    --n_embd=$EMBD_DIM \
    --sequence_length=$CONTEXT_LENGTH \
    --checkpoint_interval="1_000_000"

# Setup GPU environment for inference
setup_gpu_environment

# Transformer evaluation
print_section_header "Transformer Evaluation"
python ${INFERENCE_PATH}/guess_using_transformer.py --run $RUN_NUMBER
python ${INFERENCE_PATH}/evaluate_transformer_guess.py --run $RUN_NUMBER
python ${INFERENCE_PATH}/graphs_transformer_vs_ground_truth.py --run $RUN_NUMBER
python ${BASE_PATH}/evaluation/inspect_data.py --run $RUN_NUMBER
