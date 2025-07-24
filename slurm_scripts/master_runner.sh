#!/bin/bash
# Usage: ./launch_sweep.sh [--resume <param_grid.csv>] [--config <config_file>]

source "./slurm_scripts/common_functions.sh"
setup_environment

CONFIG_FILE="slurm_scripts/experiment_sweep.conf"
PARAM_GRID=""
CONCURRENCY=12

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            PARAM_GRID="experiments/$2/param_grid.csv"
            shift 2
            ;;
        --config)
            CONFIG_FILE="slurm_scripts/$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if [ -z "$PARAM_GRID" ]; then
    # Source config to get EXPERIMENT_TYPE
    source "$CONFIG_FILE"
    if [ "$EXPERIMENT_TYPE" = "comparison" ]; then
        mkdir -p experiments/comparison
        # Find next available comparison dir
        COMPARISON_NUMBER=$(ls -d experiments/comparison/comparison_* 2>/dev/null | sort -t_ -k2 -n | tail -n1 | sed 's/.*comparison_//' || echo 0)
        COMPARISON_NUMBER=$((COMPARISON_NUMBER + 1))
        PARAM_GRID_DIR="experiments/comparison/comparison_${COMPARISON_NUMBER}"
        mkdir -p "$PARAM_GRID_DIR"
    else
        PARAM_GRID_DIR="experiments/$EXPERIMENT_TYPE"
        mkdir -p "$PARAM_GRID_DIR"
    fi
    echo "Generating param grid using $CONFIG_FILE in $PARAM_GRID_DIR..."
    python slurm_scripts/generate_param_grid.py "$CONFIG_FILE" "$PARAM_GRID_DIR"
    PARAM_GRID="$PARAM_GRID_DIR/param_grid.csv"
    echo "Generated new param grid: $PARAM_GRID"
else
    echo "Resuming with existing param grid: $PARAM_GRID"
fi

if [ ! -f "$PARAM_GRID" ]; then
    echo "ERROR: Param grid CSV not found: $PARAM_GRID"
    exit 1
fi

# Count number of jobs (excluding header)
N=$(($(wc -l < "$PARAM_GRID") - 1))
if [ "$N" -le 0 ]; then
    echo "No jobs to submit (param grid is empty)."
    exit 0
fi

echo "Submitting $N jobs as a SLURM array (concurrency $CONCURRENCY)..."
sbatch --array=1-$N%$CONCURRENCY slurm_scripts/array_experiment.sh "$PARAM_GRID"