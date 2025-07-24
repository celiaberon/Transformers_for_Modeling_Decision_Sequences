source "./slurm_scripts/common_functions.sh"
setup_environment

# Source config file for parameters
CONFIG_FILE="slurm_scripts/experiment_sweep.conf"
source "$CONFIG_FILE"
echo "LAYERS_ARRAY: $LAYERS_ARRAY"

# Convert string variables to arrays
LAYERS_ARRAY=($LAYERS_ARRAY)
HEADS_ARRAY=($HEADS_ARRAY)
EPOCHS_ARRAY=($EPOCHS_ARRAY)
TRAIN_STEPS_ARRAY=($TRAIN_STEPS_ARRAY)
CONTEXT_LENGTH_ARRAY=($CONTEXT_LENGTH_ARRAY)
EMBD_DIM_ARRAY=($EMBD_DIM_ARRAY)
BATCH_SIZE_ARRAY=($BATCH_SIZE_ARRAY)
DOMAIN_CONFIG_ARRAY=($DOMAIN_CONFIG_ARRAY)
DOMAIN_ID_ARRAY=($DOMAIN_ID_ARRAY)

# Options are:
#   "basic": run_experiment.sh
#   "multi_domain": run_test_1a_multi_domain.sh
#   "agents_test": run_test_1b_agents.sh
#   "environment_test": run_test_1c_environments.sh
#   "comparison": model_comparison.sh
#   "layer_norm": run_layer_norm.sh

TRACKER_FILE="experiments/${EXPERIMENT_TYPE}/tracker.txt"

# Initialize starting run number - scan existing runs once at the beginning
# initialize_run
# NEXT_RUN_NUMBER=$RUN_NUMBER

# Parse --resume <instance_dir> option
RESUME_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME_MODE=true
            RESUME_DIR="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if [ "$RESUME_MODE" = true ]; then
    PARAM_GRID_DIR="$RESUME_DIR"
    echo "[master_runner] Resuming previous experiment instance in $PARAM_GRID_DIR"
    # Set COMPARISON_DIR or EXPERIMENT_DIR for downstream logic
    if [[ "$PARAM_GRID_DIR" == *comparison* ]]; then
        COMPARISON_DIR="$PARAM_GRID_DIR"
    else
        EXPERIMENT_DIR="$PARAM_GRID_DIR"
    fi
else
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
        PARAM_GRID_DIR="$COMPARISON_DIR"
    else
        EXPERIMENT_DIR="experiments/$EXPERIMENT_TYPE"
        mkdir -p "$EXPERIMENT_DIR"
        PARAM_GRID_DIR="$EXPERIMENT_DIR"
    fi
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
    local jobs_remaining=$2  # Pass in jobs remaining to submit
    while true; do
        local current_jobs=$(count_running_jobs)
        echo "Currently running/pending jobs: $current_jobs (maximum: $max_jobs)"
        if [ "$jobs_remaining" != "" ]; then
            echo "Jobs remaining to submit: $jobs_remaining"
        fi
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
#SBATCH --time=00:50:00
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

if [ "$RESUME_MODE" = false ]; then
    # Submit the parameter grid generation as a SLURM job and wait for completion
    PARAM_GRID_SCRIPT="slurm_scripts/generate_param_grid.py"
    PARAM_GRID_JOB_SCRIPT=$(mktemp)
    cat > $PARAM_GRID_JOB_SCRIPT << EOL
#!/bin/bash
#SBATCH --job-name=param_grid_gen
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:03:00
#SBATCH --mem=2GB
#SBATCH --partition=kempner_requeue
#SBATCH --output=${PARAM_GRID_DIR}/param_grid_gen.log
#SBATCH --error=${PARAM_GRID_DIR}/param_grid_gen.err

module load python/3.12.5-fasrc01
python $PARAM_GRID_SCRIPT "$CONFIG_FILE" "$PARAM_GRID_DIR"
EOL
    PARAM_GRID_JOBID=$(sbatch --parsable $PARAM_GRID_JOB_SCRIPT)
    echo "Submitted param grid generation job as $PARAM_GRID_JOBID"
    rm $PARAM_GRID_JOB_SCRIPT
    # Wait for the param grid job to finish (proceed as soon as the file is ready)
    echo "Waiting for parameter grid CSV to be created and populated..."
    while [ ! -s "$PARAM_GRID_DIR/param_grid.csv" ]; do
        sleep 2
    done
    # Optionally, wait up to 30 seconds for the SLURM job to disappear from the queue (but do not block if already gone)
    timeout 30s bash -c "while squeue -j $PARAM_GRID_JOBID > /dev/null 2>&1; do sleep 2; done"
    echo "Parameter grid generated. Proceeding with experiment submissions."
fi

PARAM_GRID_CSV="$PARAM_GRID_DIR/param_grid.csv"

# Read the CSV and submit jobs for incomplete runs only
if [ ! -f "$PARAM_GRID_CSV" ]; then
    echo "Parameter grid CSV not found: $PARAM_GRID_CSV"
    exit 1
fi

# Helper function to update run status in the param grid CSV
update_run_status() {
    local csv_file=$1
    local run_number=$2
    local new_status=$3
    python3 - "$csv_file" "$run_number" "$new_status" <<END
import csv
import sys
csv_file = sys.argv[1]
run_number = sys.argv[2]
new_status = sys.argv[3]
rows = []
header = None
try:
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        for line in reader:
            if line and any(field.strip() for field in line):
                header = line
                break
        if not header:
            with open(csv_file, 'w', newline='') as wf:
                pass
            sys.exit(0)
        for row in reader:
            if not row or len(row) != len(header):
                continue
            if row[0] == run_number:
                row[-1] = new_status
            rows.append(row)
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
except Exception as e:
    with open(csv_file) as f:
        print(f.read())
    raise
END
}

NEXT_RUN_NUMBER=1
{
    read -r header  # skip header
    while IFS=, read -r run_number layers heads epochs train_steps context_length embd_dim batch_size domain_config domain_id experiment_type use_standard_dataset debug_mode status; do
        status=$(echo "$status" | tr -d '\r\n ')
        if [ "$status" != "pending" ]; then
            continue
        fi
        if [ "$experiment_type" = "comparison" ]; then
            run_dir="experiments/comparison/comparison_*/run_${run_number}"
        else
            run_dir="experiments/$experiment_type/run_${run_number}"
        fi
        experiment_name="l${layers}_h${heads}_e${epochs}_c${context_length}_d${embd_dim}"
        submit_experiment "$experiment_name" "$layers" "$heads" "$epochs" "$train_steps" "$context_length" "$embd_dim" "$batch_size" "$domain_config" "$domain_id" "$run_number"
        update_run_status "$PARAM_GRID_CSV" "$run_number" "launched"
        NEXT_RUN_NUMBER=$((NEXT_RUN_NUMBER + 1))
    done
} < "$PARAM_GRID_CSV"

echo "All experiment jobs submitted."