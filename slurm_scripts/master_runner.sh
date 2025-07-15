source "./slurm_scripts/common_functions.sh"
setup_environment

# Parameters for experiment sweeps (or single)
LAYERS_ARRAY=(1)
HEADS_ARRAY=(1)
EPOCHS_ARRAY=(100)
TRAIN_STEPS_ARRAY=(100000)
CONTEXT_LENGTH_ARRAY=(6)
EMBD_DIM_ARRAY=(4)
BATCH_SIZE_ARRAY=(256)
DOMAIN_CONFIG_ARRAY=("three_domains.ini")
DOMAIN_ID_ARRAY=("A") # "B" "C" "B_not_sticky")
EXPERIMENT_TYPE="layer_norm"  # define the experiment you are running
export EXPERIMENT_TYPE  # Export immediately so it's available to all functions
USE_STANDARD_DATASET=true  # Standard dataset flag - when true, uses a shared dataset for all runs
DEBUG_MODE=true  # Debug mode flag - when true, prevents writing to model_summary.csv

# Options are:
#   "basic": run_experiment.sh
#   "multi_domain": run_test_1a_multi_domain.sh
#   "agents_test": run_test_1b_agents.sh
#   "environment_test": run_test_1c_environments.sh
#   "comparison": model_comparison.sh
#   "layer_norm": run_layer_norm.sh

TRACKER_FILE="experiments/${EXPERIMENT_TYPE}/tracker.txt"

# Initialize starting run number - scan existing runs once at the beginning
initialize_run
NEXT_RUN_NUMBER=$RUN_NUMBER

# Initialize comparison directory if needed
if [ "$EXPERIMENT_TYPE" = "comparison" ]; then
    # Ensure comparison base directory exists
    mkdir -p "experiments/comparison"
    
    # Get next comparison number
    COMPARISON_NUMBER=$(ls -d experiments/comparison/comparison_* 2>/dev/null | sort -t_ -k2 -n | tail -n1 | sed 's/.*comparison_//' || echo 0)
    COMPARISON_NUMBER=$((COMPARISON_NUMBER + 1))
    COMPARISON_DIR="experiments/comparison/comparison_${COMPARISON_NUMBER}"
    mkdir -p "$COMPARISON_DIR"
    echo "Created comparison directory: $COMPARISON_DIR"
    
    # Update tracker file for comparison
    TRACKER_FILE="${COMPARISON_DIR}/tracker.txt"
    echo "Comparison ${COMPARISON_NUMBER} started at $(date)" > "$TRACKER_FILE"
fi

# At the top of your script
MAX_CONCURRENT_JOBS=12
# Function to count currently running/pending jobs
count_running_jobs() {
    local username=$(whoami)
    # Count jobs that are in running (R) or pending (PD) state for your user
    local job_count=$(squeue -u $username -h -t RUNNING,PENDING | wc -l)
    echo $job_count
}

# Function to wait until jobs are below threshold
wait_for_job_slot() {
    local max_jobs=$1
    while true; do
        local current_jobs=$(count_running_jobs)
        echo "Currently running/pending jobs: $current_jobs (maximum: $max_jobs)"
        
        if [ "$current_jobs" -lt "$max_jobs" ]; then
            echo "Job slot available, proceeding with submission"
            break
        else
            echo "Too many jobs running. Waiting 60 seconds before checking again..."
            sleep 60
        fi
    done
}

CURRENT_JOB_COUNT=$(count_running_jobs)

# Function to submit a single experiment job
submit_experiment() {
    local experiment_name=$1
    local layers=$2
    local heads=$3
    local epochs=$4
    local train_steps=$5
    local context_length=$6
    local embd_dim=$7
    local batch_size=$8
    local domain_config=$9
    local domain_id=${10}
    local run_number=${11}
    
    # Wait for a job slot to be available
    wait_for_job_slot $MAX_CONCURRENT_JOBS
    
    # Create a temporary script for this specific experiment
    temp_script=$(mktemp)
    
    # Write SLURM directives to the temp script
    cat > $temp_script << EOL
#!/bin/bash
#SBATCH --job-name=${experiment_name}
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --mem=80GB
#SBATCH --partition=kempner
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

source "./slurm_scripts/common_functions.sh"
setup_environment

# Use the pre-assigned run number instead of generating a new one
RUN_NUMBER=${run_number}

# Export environment variables for the training process
export EXPERIMENT_TYPE="$EXPERIMENT_TYPE"
export DEBUG_MODE="$DEBUG_MODE"
if [ "$EXPERIMENT_TYPE" = "comparison" ]; then
    export COMPARISON_DIR="$COMPARISON_DIR"
fi

# Create logs directory for this run and redirect outputs to separate log files
if [ "$EXPERIMENT_TYPE" = "comparison" ]; then
    RUN_DIR="$COMPARISON_DIR/run_\${RUN_NUMBER}"
else
    RUN_DIR="experiments/$EXPERIMENT_TYPE/run_\${RUN_NUMBER}"
fi
LOG_DIR="\${RUN_DIR}/logs"
mkdir -p "\${LOG_DIR}"

# Redirect stdout to process.log and stderr to error.log
exec > "\${LOG_DIR}/process.log" 2> "\${LOG_DIR}/error.log"

echo "======================================================="
echo "Starting experiment: $experiment_name"
echo "Model config: $layers layers, $heads heads, $embd_dim embedding dim"
echo "Training config: $epochs epochs, $train_steps train steps, batch size $batch_size"
echo "Context length: $context_length"
echo "Domain config: $domain_config"
echo "Domain ID: $domain_id"
echo "Run number: \${RUN_NUMBER}"
echo "======================================================="

# Determine which script to run
case "$EXPERIMENT_TYPE" in
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
        echo "Unknown experiment type: $EXPERIMENT_TYPE"
        exit 1
        ;;
esac

# Update tracker file in the requested format BEFORE running the experiment
echo " " >> $TRACKER_FILE
echo "run\${RUN_NUMBER}: $EXPERIMENT_TYPE, $epochs epochs, $train_steps train steps, $layers layers, $heads heads, $context_length context length, $embd_dim embedding dimensions, $domain_config" >> $TRACKER_FILE

# Run the experiment
if [ "$EXPERIMENT_TYPE" = "comparison" ]; then
    bash \$SCRIPT \${RUN_NUMBER} $layers $heads $epochs $train_steps $context_length $embd_dim $batch_size "$domain_config" $domain_id "$USE_STANDARD_DATASET" "$COMPARISON_DIR" "$DEBUG_MODE"
else
    bash \$SCRIPT \${RUN_NUMBER} $layers $heads $epochs $train_steps $context_length $embd_dim $batch_size "$domain_config" $domain_id "$USE_STANDARD_DATASET" "$DEBUG_MODE"
fi

echo "Experiment completed: run\${RUN_NUMBER}"
EOL

    # Submit the job
    echo "Submitting experiment: $experiment_name with run number $run_number"
    sbatch $temp_script
    
    # Clean up the temp script
    rm $temp_script
    
    # Add a small delay to avoid overwhelming the scheduler
    sleep 1
}


# Submit jobs for each combination in the parameter sweep
for layers in "${LAYERS_ARRAY[@]}"; do
    for heads in "${HEADS_ARRAY[@]}"; do
        for epochs in "${EPOCHS_ARRAY[@]}"; do
            for train_steps in "${TRAIN_STEPS_ARRAY[@]}"; do
                for context_length in "${CONTEXT_LENGTH_ARRAY[@]}"; do
                    for embd_dim in "${EMBD_DIM_ARRAY[@]}"; do
                        for batch_size in "${BATCH_SIZE_ARRAY[@]}"; do
                            for domain_config in "${DOMAIN_CONFIG_ARRAY[@]}"; do
                                for domain_id in "${DOMAIN_ID_ARRAY[@]}"; do
                                    experiment_name="l${layers}_h${heads}_e${epochs}_c${context_length}_d${embd_dim}"
                                    if [ $CURRENT_JOB_COUNT -ge $MAX_CONCURRENT_JOBS ]; then
                                        wait_for_job_slot $MAX_CONCURRENT_JOBS
                                        # Reset counter when we've waited
                                        CURRENT_JOB_COUNT=$(count_running_jobs)
                                    fi
                                    submit_experiment "$experiment_name" "$layers" "$heads" "$epochs" "$train_steps" "$context_length" "$embd_dim" "$batch_size" "$domain_config" "$domain_id" "${NEXT_RUN_NUMBER}"
                                    NEXT_RUN_NUMBER=$((NEXT_RUN_NUMBER + 1))
                                    CURRENT_JOB_COUNT=$((CURRENT_JOB_COUNT + 1))
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "All experiment jobs submitted."