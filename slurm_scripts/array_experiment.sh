#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1          
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00  
#SBATCH --mem=60GB
#SBATCH --partition=kempner_requeue
#SBATCH --partition=kempner
#SBATCH --array=1-12%12
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

source "./slurm_scripts/common_functions.sh"
setup_environment

PARAM_GRID=$1
PARAM_GRID_DIR=$(dirname "$PARAM_GRID")
echo "PARAM_GRID argument: $PARAM_GRID"
LOCK_FILE="${PARAM_GRID}.lock"
N=$(($(wc -l < $PARAM_GRID) - 1))
echo "N: $N"
echo "First few lines of $PARAM_GRID:"
head "$PARAM_GRID"

if ! touch "$LOCK_FILE" 2>/dev/null; then
    echo "ERROR: Cannot write to lock file $LOCK_FILE"
    exit 1
fi
claim_next_run() {
    # Use flock to ensure only one process claims a run at a time
    local claimed_line
    exec 200>"$LOCK_FILE"
    flock 200
    echo "Attempting to claim a run in $PARAM_GRID"
    # Find the first pending or failed run (robust to whitespace/CR)
    claimed_line=$(awk -F, 'NR>1 {gsub(/\r/,"",$NF); gsub(/^[ \t]+|[ \t]+$/, "", $NF); if($NF=="pending" || $NF=="failed") {print NR; exit}}' "$PARAM_GRID")
    if [ -z "$claimed_line" ]; then
        flock -u 200
        return 1
    fi
    # Read the line
    LINE=$(awk -v n="$claimed_line" 'NR==n' "$PARAM_GRID")
    IFS=',' read -r run_number layers heads epochs train_steps context_length embd_dim batch_size domain_config domain_id experiment_type use_standard_dataset debug_mode status <<< "$LINE"
    echo "Claimed line: $LINE"
    echo "Parsed run_number: $run_number, status: $status"

    # Export experiment type and comparison dir for downstream scripts
    export EXPERIMENT_TYPE="$experiment_type"
    export DEBUG_MODE="$debug_mode"
    export USE_STANDARD_DATASET="$use_standard_dataset"
    export DOMAIN_ID="$domain_id"
    export DOMAIN_CONFIG="$domain_config"

    if [ "$experiment_type" = "comparison" ]; then
        export COMPARISON_DIR=$(dirname "$PARAM_GRID")
        echo "Set COMPARISON_DIR to $COMPARISON_DIR"
    else
        export EXPERIMENT_DIR="experiments/$EXPERIMENT_TYPE"
        echo "Set EXPERIMENT_DIR to $EXPERIMENT_DIR"
    fi

    # Mark as launched
    update_run_status "$PARAM_GRID" "$run_number" "launched"
    flock -u 200
    return 0
}

# Try to claim a run; exit if none left
if ! claim_next_run; then
    echo "No pending or failed runs left to process."
    exit 0
fi

# Create logs directory for this run and redirect outputs to separate log files
if [ "$EXPERIMENT_TYPE" = "comparison" ]; then
    RUN_DIR="$COMPARISON_DIR/run_${run_number}"
else
    RUN_DIR="experiments/$EXPERIMENT_TYPE/run_${run_number}"
fi
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "$LOG_DIR"

# Redirect stdout to process.log and stderr to error.log
exec > "${LOG_DIR}/process.log" 2> "${LOG_DIR}/error.log"

echo "======================================================="
echo "Starting experiment: run_${run_number}"
echo "Model config: $layers layers, $heads heads, $embd_dim embedding dim" 
echo "Training config: $epochs epochs, $train_steps train steps, batch size $batch_size"
echo "Context length: $context_length"
echo "Domain config: $domain_config"
echo "Domain ID: $domain_id"
echo "Experiment type: $experiment_type"
echo "======================================================="

# Clean up status
status=$(echo "$status" | tr -d '\r\n ')

# Call the appropriate experiment script
case "$experiment_type" in
    "basic")
        SCRIPT="./slurm_scripts/run_experiment.sh"
        ;;
    "agents_test")
        SCRIPT="./slurm_scripts/run_test_1b_agents.sh"
        ;;
    "environment_test")
        SCRIPT="./slurm_scripts/run_test_1c_environments.sh"
        ;;
    "multi_domain")
        SCRIPT="./slurm_scripts/run_test_1a_multi_domain.sh"
        ;;
    "comparison")
        SCRIPT="./slurm_scripts/model_comparison.sh"
        ;;
    "layer_norm")
        SCRIPT="./slurm_scripts/run_layer_norm.sh"
        ;;
    *)
        echo "Unknown experiment type: $experiment_type"
        update_run_status "$PARAM_GRID" "$run_number" "failed"
        exit 1
        ;;
esac

# Run the experiment
bash $SCRIPT "$run_number" "$layers" "$heads" "$epochs" "$train_steps" "$context_length" "$embd_dim" "$batch_size" "$domain_config" "$domain_id" "$use_standard_dataset" "$debug_mode"

if [ $? -eq 0 ]; then
    update_run_status "$PARAM_GRID" "$run_number" "completed"
else
    update_run_status "$PARAM_GRID" "$run_number" "failed"
fi 