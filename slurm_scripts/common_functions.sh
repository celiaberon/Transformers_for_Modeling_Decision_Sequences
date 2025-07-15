#!/bin/bash
# Common functions for SLURM scripts

# === Environment Setup ===
setup_environment() {
    # Set base paths
    BASE_PATH="."  # Current directory
    INFERENCE_PATH="${BASE_PATH}/transformer/inference"
    
    # Set Python path for imports
    export PYTHONPATH="${BASE_PATH}:$PYTHONPATH"
    
    # Load modules
    module load python/3.12.5-fasrc01
    module load cuda/12.2.0-fasrc01
    
    # Initialize Conda/Mamba
    eval "$(conda shell.bash hook)"
    
    # Activate the environment using the full path
    mamba activate ~/.conda/envs/transformers || source ~/.conda/envs/transformers/bin/activate
    
    # Export path variables for other functions to use
    export BASE_PATH INFERENCE_PATH
}

# === Run Number Management ===
get_next_run() {
    # Check if we're in a comparison directory context
    if [ -n "$COMPARISON_DIR" ]; then
        local latest=$(ls -d ${COMPARISON_DIR}/run_* 2>/dev/null | sort -V | tail -n1 | sed 's/.*run_//')
    else
        # Use experiment type to determine directory
        local experiment_type=${EXPERIMENT_TYPE:-basic}
        local search_dir="${BASE_PATH}/experiments/${experiment_type}"
        local latest=$(ls -d ${search_dir}/run_* 2>/dev/null | sort -V | tail -n1 | sed 's/.*run_//')
    fi
    if [ -z "$latest" ]; then
        echo 1
    else
        echo $((latest + 1))
    fi
}

initialize_run() {
    # Get run number or use provided override
    local override_run=$1
    
    if [ -n "$override_run" ]; then
        RUN_NUMBER=$override_run
    else
        RUN_NUMBER=$(get_next_run)
    fi
    
    echo "Starting run $RUN_NUMBER"
    export RUN_NUMBER
}

# === Dataset Configuration Management ===
# Create a dataset configuration structure
create_dataset_config() {
    # Initialize associative array for dataset configuration
    declare -A config
    
    # Set defaults
    config[domain_config]="domains.ini"
    config[domain_id]="default"
    config[multiple_domains]="false"
    config[train_steps]="100000"
    config[val_steps]="1000000"
    config[train_domains]=""
    config[val_domains]=""
    config[use_custom_domains]="false"
    config[run_number]="1"
    config[comparison_dir]=""
    config[use_standard_dataset]="false"
    
    # Parse named arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --domain-config)
                config[domain_config]="$2"
                shift 2
                ;;
            --domain-id)
                config[domain_id]="$2"
                shift 2
                ;;
            --multiple-domains)
                config[multiple_domains]="$2"
                shift 2
                ;;
            --train-steps)
                config[train_steps]="$2"
                shift 2
                ;;
            --val-steps)
                config[val_steps]="$2"
                shift 2
                ;;
            --train-domains)
                config[train_domains]="$2"
                shift 2
                ;;
            --val-domains)
                config[val_domains]="$2"
                shift 2
                ;;
            --use-custom-domains)
                config[use_custom_domains]="$2"
                shift 2
                ;;
            --run-number)
                config[run_number]="$2"
                shift 2
                ;;
            --comparison-dir)
                config[comparison_dir]="$2"
                shift 2
                ;;
            --use-standard-dataset)
                config[use_standard_dataset]="$2"
                shift 2
                ;;
            *)
                echo "Unknown parameter: $1"
                shift
                ;;
        esac
    done
    
    # Export the configuration as a serialized string
    local config_str=""
    for key in "${!config[@]}"; do
        config_str+="${key}=${config[$key]}|"
    done
    echo "${config_str%|}"  # Remove trailing |
}

# Parse dataset configuration from serialized string
parse_dataset_config() {
    local config_str="$1"
    declare -A config
    
    IFS='|' read -ra pairs <<< "$config_str"
    for pair in "${pairs[@]}"; do
        IFS='=' read -r key value <<< "$pair"
        config[$key]="$value"
    done
    
    # Export as global associative array
    for key in "${!config[@]}"; do
        eval "DATASET_CONFIG_${key}='${config[$key]}'"
    done
}

# Generate domain identifier from configuration
generate_domain_identifier() {
    local config_str="$1"
    parse_dataset_config "$config_str"
    
    # For custom domain scripts
    if [ -n "$DATASET_CONFIG_train_domains" ] && [ -n "$DATASET_CONFIG_val_domains" ]; then
        # Sort domains to ensure consistent naming
        local sorted_train=$(echo "$DATASET_CONFIG_train_domains" | tr ' ' '\n' | sort | tr '\n' '_' | sed 's/_$//')
        local sorted_val=$(echo "$DATASET_CONFIG_val_domains" | tr ' ' '\n' | sort | tr '\n' '_' | sed 's/_$//')
        echo "train_${sorted_train}_val_${sorted_val}"
        return
    fi
    
    # For standard multi-domain generation
    if [ "$DATASET_CONFIG_multiple_domains" = "true" ]; then
        # Determine actual domains from config file
        local config_path="${BASE_PATH}/synthetic_data_generation/${DATASET_CONFIG_domain_config}"
        if [ -f "$config_path" ]; then
            # Extract domain IDs from config file
            local domains=$(grep '^\[' "$config_path" | sed 's/\[//g' | sed 's/\]//g' | sort | tr '\n' '_' | sed 's/_$//')
            echo "${domains}"
        else
            echo "multidomain"
        fi
    else
        # Single domain
        echo "${DATASET_CONFIG_domain_id}"
    fi
}

# === Standard Dataset Management ===
setup_standard_dataset() {
    # Create configuration from named arguments
    local config_str=$(create_dataset_config "$@")
    parse_dataset_config "$config_str"
    
    if [ "$DATASET_CONFIG_use_standard_dataset" = "true" ]; then
        # Generate domain-specific identifier
        DATASET_IDENTIFIER=$(generate_domain_identifier "$config_str")
        
        # Use a single shared datasets folder for all standard datasets
        if [ -n "$DATASET_CONFIG_comparison_dir" ]; then
            # For comparison experiments, use shared datasets folder in comparison directory
            STANDARD_DATASET_DIR="${DATASET_CONFIG_comparison_dir}/shared_datasets"
        else
            # For non-comparison experiments, use shared datasets folder within experiment type
            local experiment_type=${EXPERIMENT_TYPE:-basic}
            STANDARD_DATASET_DIR="${BASE_PATH}/experiments/${experiment_type}/shared_datasets"
        fi
        
        mkdir -p "$STANDARD_DATASET_DIR"
        export STANDARD_DATASET_DIR
        export USE_STANDARD_DATASET=true
        export DATASET_IDENTIFIER
        export DATASET_CONFIG_STR="$config_str"
        echo "Using standard dataset in: $STANDARD_DATASET_DIR (identifier: $DATASET_IDENTIFIER)"
    else
        export USE_STANDARD_DATASET=false
        echo "Using individual datasets for each run"
    fi
}

# === Helper Functions for Standard Dataset Management ===

copy_metadata_to_shared() {
    local run_number=$1
    local dataset_dir=$2
    local dataset_identifier=$3
    
    # Use experiment type to determine directory
    local experiment_type=${EXPERIMENT_TYPE:-basic}
    local run_dir="${BASE_PATH}/experiments/${experiment_type}/run_${run_number}"
    
    # Copy metadata with identifier suffix
    if [ -f "${run_dir}/metadata.txt" ]; then
        cp "${run_dir}/metadata.txt" "${dataset_dir}/metadata_${dataset_identifier}.txt"
        echo "Copied metadata: metadata_${dataset_identifier}.txt"
    fi

}

copy_logs_to_shared() {
    local run_number=$1
    local dataset_dir=$2
    local dataset_identifier=$3
    
    # Use experiment type to determine directory
    local experiment_type=${EXPERIMENT_TYPE:-basic}
    local run_dir="${BASE_PATH}/experiments/${experiment_type}/run_${run_number}"

    # Copy data generation log with identifier suffix
    if [ -f "${run_dir}/logs/data_generation.log" ]; then
        mkdir -p "${dataset_dir}/logs"
        cp "${run_dir}/logs/data_generation.log" "${dataset_dir}/logs/data_generation_${dataset_identifier}.log"
        echo "Copied data generation log: data_generation_${dataset_identifier}.log"
    fi
}

# === File Locking for Dataset Generation ===
acquire_dataset_lock() {
    local lock_file="$1"
    local timeout=300  # 5 minutes timeout
    local elapsed=0
    local wait_interval=5
    
    while [ $elapsed -lt $timeout ]; do
        if (set -C; echo $$ > "$lock_file") 2>/dev/null; then
            echo "Acquired dataset generation lock"
            return 0
        fi
        
        echo "Waiting for dataset generation lock... (${elapsed}s/${timeout}s)"
        sleep $wait_interval
        elapsed=$((elapsed + wait_interval))
    done
    
    echo "ERROR: Timeout waiting for dataset generation lock"
    return 1
}

release_dataset_lock() {
    local lock_file="$1"
    if [ -f "$lock_file" ]; then
        rm -f "$lock_file"
        echo "Released dataset generation lock"
    fi
}

# === Safe Dataset Generation ===
generate_standard_dataset() {
    # Use the exported configuration from setup_standard_dataset
    if [ -z "$DATASET_CONFIG_STR" ]; then
        echo "ERROR: No dataset configuration found. Call setup_standard_dataset first."
        return 1
    fi
    
    parse_dataset_config "$DATASET_CONFIG_STR"
    
    local dataset_dir="$STANDARD_DATASET_DIR"
    local lock_file="${dataset_dir}/.generation_lock_${DATASET_IDENTIFIER}"
    local marker_file="${dataset_dir}/seqs/.generation_complete_${DATASET_IDENTIFIER}"
    
    # Check if dataset already exists
    if [ -f "$marker_file" ]; then
        echo "Standard dataset already exists and is complete for identifier: $DATASET_IDENTIFIER"
        return 0
    fi
    
    # Acquire lock for dataset generation
    if acquire_dataset_lock "$lock_file"; then
        # Double-check after acquiring lock
        if [ -f "$marker_file" ]; then
            echo "Standard dataset was generated by another process for identifier: $DATASET_IDENTIFIER"
            release_dataset_lock "$lock_file"
            return 0
        fi
        
        echo "Generating standard dataset..."
        echo "Configuration: domain_config=${DATASET_CONFIG_domain_config}, domain_id=${DATASET_CONFIG_domain_id}"
        echo "               train_steps=${DATASET_CONFIG_train_steps}, val_steps=${DATASET_CONFIG_val_steps}"
        echo "               multiple_domains=${DATASET_CONFIG_multiple_domains}, use_custom_domains=${DATASET_CONFIG_use_custom_domains}"
        if [ -n "$DATASET_CONFIG_train_domains" ]; then
            echo "               train_domains=${DATASET_CONFIG_train_domains}, val_domains=${DATASET_CONFIG_val_domains}"
        fi
        
        mkdir -p "${dataset_dir}/seqs"
        
        # Generate the dataset with appropriate script and arguments
        if [ "$DATASET_CONFIG_use_custom_domains" = "true" ]; then
            # Use custom domain script
            python ${BASE_PATH}/synthetic_data_generation/generate_data_custom_domains.py \
                --run $DATASET_CONFIG_run_number \
                --num_steps_train=$DATASET_CONFIG_train_steps \
                --num_steps_val=$DATASET_CONFIG_val_steps \
                --no_overwrite \
                --config_file "$DATASET_CONFIG_domain_config" \
                --train_domains $DATASET_CONFIG_train_domains \
                --val_domains $DATASET_CONFIG_val_domains
        elif [ "$DATASET_CONFIG_multiple_domains" = "true" ]; then
            # Use standard multi-domain generation
            python ${BASE_PATH}/synthetic_data_generation/generate_data.py \
                --run $DATASET_CONFIG_run_number \
                --num_steps_train=$DATASET_CONFIG_train_steps \
                --num_steps_val=$DATASET_CONFIG_val_steps \
                --no_overwrite \
                --config_file "$DATASET_CONFIG_domain_config" \
                --multiple_domains
        else
            # Use single domain generation
            python ${BASE_PATH}/synthetic_data_generation/generate_data.py \
                --run $DATASET_CONFIG_run_number \
                --domain_id $DATASET_CONFIG_domain_id \
                --num_steps_train=$DATASET_CONFIG_train_steps \
                --num_steps_val=$DATASET_CONFIG_val_steps \
                --no_overwrite \
                --config_file "$DATASET_CONFIG_domain_config"
        fi
        
        # Mark generation as complete
        if [ $? -eq 0 ]; then

            # Copy metadata to shared dataset folder
            copy_metadata_to_shared "$DATASET_CONFIG_run_number" "$dataset_dir" "$DATASET_IDENTIFIER"
            
            # Run evaluation on the generated dataset
            echo "Running evaluation on generated standard dataset..."
            python ${BASE_PATH}/evaluation/basic_evaluation.py --run $DATASET_CONFIG_run_number
            python ${BASE_PATH}/evaluation/graphs_on_trial_block_transitions.py --run $DATASET_CONFIG_run_number
            
            # Copy log files to shared dataset folder
            copy_logs_to_shared "$DATASET_CONFIG_run_number" "$dataset_dir" "$DATASET_IDENTIFIER"
            
            touch "$marker_file"
            echo "Standard dataset generation completed successfully"
        else
            echo "ERROR: Standard dataset generation failed"
            release_dataset_lock "$lock_file"
            return 1
        fi
        
        release_dataset_lock "$lock_file"
        return 0
    else
        echo "ERROR: Could not acquire lock for dataset generation"
        return 1
    fi
}

# === GPU Management ===
setup_gpu_environment() {
    # Detect number of available GPUs
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    if [ -n "$SLURM_GPUS_PER_NODE" ]; then
        NUM_GPUS=$SLURM_GPUS_PER_NODE
    fi
    echo "Detected $NUM_GPUS GPU(s) for execution"
    export NUM_GPUS
}

# Function to assign a GPU ID based on index and available GPUs
assign_gpu() {
    local idx=$1
    echo $((idx % NUM_GPUS))
}

# === Distributed Training Setup ===
setup_distributed_environment() {
    # Set up distributed training environment variables
    export MASTER_PORT=12355 # Default port that's usually available
    export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
    echo "WORLD_SIZE=$WORLD_SIZE"
    echo "MASTER_PORT=$MASTER_PORT"
    
    # Define a master address for communication between GPUs
    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export MASTER_ADDR=$master_addr
    echo "MASTER_ADDR=$MASTER_ADDR"
}

# === Parallel Execution ===
# Run commands in parallel on different GPUs if possible, otherwise sequentially
run_on_gpus() {
    local commands=("$@")
    local num_commands=${#commands[@]}
    
    # Array to track processes
    pids=()
    
    if [ "$NUM_GPUS" -gt 1 ]; then
        echo "Running commands in parallel across GPUs"
        
        # Process each command
        for i in "${!commands[@]}"; do
            local cmd="${commands[$i]}"
            local gpu_id=$(assign_gpu $i)
            
            # Check if we need to wait for a previous job on this GPU
            if [ -n "${pids[$gpu_id]}" ]; then
                if kill -0 ${pids[$gpu_id]} 2>/dev/null; then
                    echo "Waiting for previous job on GPU $gpu_id to finish..."
                    wait ${pids[$gpu_id]}
                fi
            fi
            
            # Run the command on the assigned GPU
            echo "Running on GPU $gpu_id: $cmd"
            CUDA_VISIBLE_DEVICES=$gpu_id $cmd &
            pids[$gpu_id]=$!
        done
        
        # Wait for all processes to complete
        wait
    else
        echo "Running commands sequentially on single GPU"
        for cmd in "${commands[@]}"; do
            CUDA_VISIBLE_DEVICES=0 $cmd
        done
    fi
}


process_checkpoints() {
    # Find checkpoint files and extract base names
    checkpoint_files=()
    # Use experiment type to determine directory
    local experiment_type=${EXPERIMENT_TYPE:-basic}
    for model_file in "${BASE_PATH}/experiments/${experiment_type}/run_${RUN_NUMBER}/models/model_"*"cp"*".pth"; do
        if [ -f "$model_file" ]; then
            checkpoint_files+=("$model_file")
        fi
    done
    
    if [ ${#checkpoint_files[@]} -eq 0 ]; then
        echo "No checkpoints found to process"
        return
    fi
    
    echo "Found ${#checkpoint_files[@]} checkpoints to process"
    
    # Initialize array to keep track of running processes for each GPU
    pids=()
    
    # Process each checkpoint on an available GPU, but ensure commands run in sequence
    for i in "${!checkpoint_files[@]}"; do
        model_file="${checkpoint_files[$i]}"
        model_name=$(basename "$model_file" .pth)
        
        # Assign GPU using modular arithmetic
        # Default to 1 GPU if NUM_GPUS not set
        NUM_GPUS=${NUM_GPUS:-1}
        gpu_id=$((i % NUM_GPUS))
        
        print_section_header "Processing checkpoint: $model_name on GPU $gpu_id"
        
        # If previous job was on this GPU, wait for it to finish
        if [ -n "${pids[$gpu_id]}" ]; then
            if kill -0 ${pids[$gpu_id]} 2>/dev/null; then
                echo "Waiting for previous job on GPU $gpu_id to finish..."
                wait ${pids[$gpu_id]}
            fi
        fi
        
        # Run the processing commands in sequence for this checkpoint on the assigned GPU
        (
            # These must run in sequence for each checkpoint
            CUDA_VISIBLE_DEVICES=$gpu_id python -m transformer.inference.guess_using_transformer --run $RUN_NUMBER --model_name $model_name
            # CUDA_VISIBLE_DEVICES=$gpu_id python ${INFERENCE_PATH}/evaluate_transformer_guess.py --run $RUN_NUMBER --model_name $model_name
            # CUDA_VISIBLE_DEVICES=$gpu_id python ${INFERENCE_PATH}/graphs_transformer_vs_ground_truth.py --run $RUN_NUMBER --model_name $model_name
        ) &
        
        # Store process ID for this GPU
        pids[$gpu_id]=$!
    done
    
    # Wait for all remaining processes to complete
    wait
}

# === Formatting and Logging ===
print_section_header() {
    local title="$1"
    printf '%*s\n' 80 '' | tr ' ' '-'
    echo -e "\n$title\n"
} 