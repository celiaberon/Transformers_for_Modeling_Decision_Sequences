import os
import itertools
import csv
import sys

# Helper to parse config values as arrays
def parse_array(val):
    return val.strip('"').split()

def main():
    CONFIG_PATH = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(CONFIG_PATH):
        print(f"Config file not found: {CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)

    # Read config as key-value pairs
    config = {}
    with open(CONFIG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            if '=' not in line: continue
            k, v = line.split('=', 1)
            config[k.strip()] = v.strip()

    # Required keys
    keys = [
        'LAYERS_ARRAY', 'HEADS_ARRAY', 'EPOCHS_ARRAY', 'TRAIN_STEPS_ARRAY',
        'CONTEXT_LENGTH_ARRAY', 'EMBD_DIM_ARRAY', 'BATCH_SIZE_ARRAY',
        'DOMAIN_CONFIG_ARRAY', 'DOMAIN_ID_ARRAY', 'EXPERIMENT_TYPE',
        'USE_STANDARD_DATASET', 'DEBUG_MODE'
    ]
    for k in keys:
        if k not in config:
            print(f"Missing key in config: {k}", file=sys.stderr)
            sys.exit(1)

    experiment_type = config['EXPERIMENT_TYPE'].strip('"')
    if output_dir is None:
        if experiment_type == 'comparison':
            output_dir = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'comparison')
        else:
            output_dir = os.path.join(os.path.dirname(__file__), '..', 'experiments', experiment_type)
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, 'param_grid.csv')

    # Build parameter grid
    param_grid = {
        'layers': parse_array(config['LAYERS_ARRAY']),
        'heads': parse_array(config['HEADS_ARRAY']),
        'epochs': parse_array(config['EPOCHS_ARRAY']),
        'train_steps': parse_array(config['TRAIN_STEPS_ARRAY']),
        'context_length': parse_array(config['CONTEXT_LENGTH_ARRAY']),
        'embd_dim': parse_array(config['EMBD_DIM_ARRAY']),
        'batch_size': parse_array(config['BATCH_SIZE_ARRAY']),
        'domain_config': parse_array(config['DOMAIN_CONFIG_ARRAY']),
        'domain_id': parse_array(config['DOMAIN_ID_ARRAY']),
    }
    static_params = {
        'experiment_type': experiment_type,
        'use_standard_dataset': config['USE_STANDARD_DATASET'].strip('"'),
        'debug_mode': config['DEBUG_MODE'].strip('"'),
    }

    # Write CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = (
            ['run_number'] +
            list(param_grid.keys()) +
            list(static_params.keys()) +
            ['status']
        )
        writer.writerow(header)
        for i, combo in enumerate(itertools.product(*param_grid.values()), 1):
            row = [i] + list(combo) + [static_params[k] for k in static_params] + ['pending']
            writer.writerow(row)
    print(f"Wrote parameter grid to {output_csv}")

if __name__ == '__main__':
    main() 