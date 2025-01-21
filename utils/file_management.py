import os
import glob

def get_latest_run():
    """Find the highest numbered run directory."""
    base_path = os.path.dirname(os.path.dirname(__file__))
    run_dirs = glob.glob(os.path.join(base_path, "experiments", "run_*"))
    if not run_dirs:
        return -1
    return max([int(d.split('_')[-1]) for d in run_dirs])

def get_run_dir(run=None):
    """Get the directory for a specific run, defaulting to latest."""
    base_path = os.path.dirname(os.path.dirname(__file__))
    if run is None:
        run = get_latest_run()
    return os.path.join(base_path, "experiments", f"run_{run}")

def ensure_run_dir(run):
    """Create run directory if it doesn't exist."""
    run_dir = get_run_dir(run)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def get_file_path(filename, run=None, create_dir=False):
    """Get full path for a file in a run directory."""
    if run is None:
        run = get_latest_run()
    run_dir = get_run_dir(run)
    if create_dir:
        os.makedirs(run_dir, exist_ok=True)
    return os.path.join(run_dir, filename)

def get_experiment_file(filename_template, run=None, suffix='tr'):
    """Get path to an experiment-specific file.
    
    Args:
        filename_template (str): Template like 'behavior_run_{}.txt'
        run (int, optional): Run number. Defaults to latest run.
        suffix (str, optional): Dataset suffix ('tr' or 'v'). Defaults to 'tr'.
    
    Returns:
        str: Full path to the requested file
    """
    if run is None:
        run = get_latest_run()
    
    filename = filename_template.format(f"{run}{suffix}")
    return os.path.join(get_run_dir(run), filename) 

def format_tokens(tokens):
    """Format the number of tokens to nearest thousand (K) or million (M)."""
    if tokens >= 1_000_000:
        return f"{tokens // 1_000_000}M"  # Nearest million
    elif tokens >= 1_000:
        return f"{tokens // 1_000}K"      # Nearest thousand
    else:
        return str(tokens)

def parse_model_info(run=None, model_name=None):
    """Parse model information from metadata.txt"""
    metadata_file = get_experiment_file("metadata.txt", run)
    model_info = {
        'model_name': None,
        'tokens_seen': None,
        'dataloader': {},
        'config': {}
    }
    
    found_model_name = False
    with open(metadata_file, 'r') as f:
        current_section = None
        for line in f:
            line = line.strip()
            if not line: continue
            
            if line.startswith("Model name:"):
                if found_model_name:
                    return model_info  # early return if we don't want to move onto the next model
                model_info['model_name'] = line.split(": ")[1]
                if line.split(": ")[1] == model_name:
                    found_model_name = True
            elif line.startswith("Tokens seen:"):
                model_info['tokens_seen'] = int(line.split(": ")[1].replace(',', ''))
            elif line.startswith("Dataloader parameters:"):
                current_section = 'dataloader'
            elif line.startswith("GPTConfig parameters:"):
                current_section = 'config'
            elif current_section and ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                try:
                    value = int(value)
                except ValueError:
                    pass
                model_info[current_section][key] = value
    
    return model_info

def get_latest_model_name(run=None):
    """Get the name of the latest model based on tokens seen."""
    model_info = parse_model_info(run)
    return model_info['model_name']
    # tokens_m = format_tokens(model_info['tokens_seen'])
    # return f"model_seen{tokens_m}.pth"
