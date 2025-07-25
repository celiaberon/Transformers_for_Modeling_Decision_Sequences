import glob
import logging
import os
import sys


def get_latest_run():
    """
    Find the highest numbered run directory within the current experiment type.
    
    Returns:
        int: Highest run number, or 0 if no runs exist
    """
    base_path = os.path.dirname(os.path.dirname(__file__))
    
    # Check if we're in a comparison context
    comparison_dir = os.environ.get('COMPARISON_DIR')
    if comparison_dir:
        run_dirs = glob.glob(os.path.join(comparison_dir, "run_*"))
    else:
        # Use experiment type to determine directory
        experiment_type = os.environ.get('EXPERIMENT_TYPE', 'basic')
        experiment_dir = os.path.join(base_path, "experiments", experiment_type)
        run_dirs = glob.glob(os.path.join(experiment_dir, "run_*"))
        
    if not run_dirs:
        return 0  # if no runs, return 0 so first run is 1
    return max([int(d.split('_')[-1]) for d in run_dirs])


def get_run_dir(run=None):
    """
    Get the directory for a specific run, defaulting to latest.
    
    Args:
        run (int, optional): Run number. If None, uses the latest run.
        
    Returns:
        str: Full path to the run directory
    """
    base_path = os.path.dirname(os.path.dirname(__file__))
    if run is None:
        run = get_latest_run()
    
    # Check if we're in a comparison context
    comparison_dir = os.environ.get('COMPARISON_DIR')
    if comparison_dir:
        return os.path.join(comparison_dir, f"run_{run}")
    else:
        # Use experiment type to determine directory
        experiment_type = os.environ.get('EXPERIMENT_TYPE', 'basic')
        return os.path.join(base_path, "experiments", experiment_type, f"run_{run}")


def ensure_run_dir(run, overwrite=True, subdir=None):
    """
    Create run directory and subdirectory if they don't exist.
    
    Args:
        run (int): Run number
        overwrite (bool): Whether to overwrite existing directory
        subdir (str, optional): Subdirectory name within run directory
        
    Returns:
        str: Full path to the created directory
    """
    run_dir = get_run_dir(run)
    if subdir:
        run_dir = os.path.join(run_dir, subdir)
    
    os.makedirs(run_dir, exist_ok=overwrite)  # Ensure the directory exists
    return run_dir


def get_file_path(filename, run=None, create_dir=False):
    """
    Get full path for a file in a run directory.
    
    Args:
        filename (str): Name of the file
        run (int, optional): Run number. If None, uses the latest run.
        create_dir (bool): Whether to create the directory if it doesn't exist
        
    Returns:
        str: Full path to the file
    """
    if run is None:
        run = get_latest_run()
    run_dir = get_run_dir(run)
    if create_dir:
        os.makedirs(run_dir, exist_ok=True)
    return os.path.join(run_dir, filename)


def get_experiment_file(filename_template, run=None, suffix='tr', subdir=None, use_standard=False):
    """
    Get path to an experiment-specific file.
    
    Args:
        filename_template (str): Template like 'behavior_run_{}.txt'
        run (int, optional): Run number. Defaults to latest run.
        suffix (str, optional): Dataset suffix ('tr' or 'v'). Defaults to 'tr'.
        subdir (str, optional): Subdirectory within the run directory. Defaults to None.
        use_standard (bool, optional): Whether to use the standard dataset directory. Defaults to False.
    Returns:
        str: Full path to the requested file
    """
    if run is None:
        run = get_latest_run()
    
    # Check if we should use standard dataset (extra logic for metadata)
    if use_standard or (subdir in ['seqs', 'agent_behavior']):
        use_standard = (os.environ.get('USE_STANDARD_DATASET', 'false').lower()
                        == 'true')
    standard_dataset_dir = os.environ.get('STANDARD_DATASET_DIR')
    dataset_identifier = os.environ.get('DATASET_IDENTIFIER', 'default')
    
    if use_standard and standard_dataset_dir and subdir in ['seqs', 'agent_behavior']:
        # Use standard dataset directory for sequence files and agent_behavior
        target_dir = os.path.join(standard_dataset_dir, subdir or '')
        
        # Transform the filename template to use dataset identifier instead of 
        # run number e.g., 'behavior_run_{}.txt' -> 'behavior_{}_tr.txt'
        if 'run_{}' in filename_template:
            # Replace 'run_{}' with '{}' and append suffix
            base_template = filename_template.replace('run_{}', '{}')
            filename = base_template.format(f"{dataset_identifier}_{suffix}")
        else:
            # For agent_behavior files, add dataset identifier suffix
            if subdir == 'agent_behavior':
                # Split filename into base and extension
                parts = filename_template.split('.')
                if len(parts) > 1:
                    base = '.'.join(parts[:-1])
                    ext = parts[-1]
                    filename = f"{base}_{dataset_identifier}.{ext}"
                else:
                    filename = f"{filename_template}_{dataset_identifier}"
            else:
                # Fallback for other templates
                filename = filename_template.format(f"{dataset_identifier}_{suffix}")
    elif use_standard and standard_dataset_dir and filename_template.startswith('metadata'):
        filename = f'metadata_{dataset_identifier}.txt'
        target_dir = standard_dataset_dir
    else:
        # Use regular run directory
        run_dir = get_run_dir(run)
        if subdir:
            run_dir = os.path.join(run_dir, subdir)
        target_dir = run_dir
        filename = filename_template.format(f"{run}{suffix}")
    
    os.makedirs(target_dir, exist_ok=True)  # Ensure the directory exists
    return os.path.join(target_dir, filename)


def format_tokens(tokens):
    """
    Format the number of tokens to a concise string label (K, M, B, etc.)
    
    Args:
        tokens (int): Number of tokens
        
    Returns:
        str: Formatted string (e.g., "500K", "2M")
        
    Raises:
        ValueError: If tokens is negative
    """
    if tokens < 0:
        raise ValueError("Token count cannot be negative.")
    
    suffixes = ['', 'K', 'M', 'B', 'T']  # Add more suffixes as needed
    index = 0
    
    while tokens >= 1000 and index < len(suffixes) - 1:
        tokens /= 1000.0
        index += 1
    
    return f"{int(tokens)}{suffixes[index]}"


def get_domain_params(run=None, domain_id=None, suffix='tr'):
    """
    Get the parameters for a specific domain.
    """
    metadata_file = get_experiment_file("metadata.txt", run, use_standard=True)

    with open(metadata_file, 'r') as f:
        data_section = None
        current_section = None
        domain = None
        for line in f:
            line = line.strip()
            if line.startswith('Dataset'):
                data_section = line.split(' ')[-1]
            if data_section == suffix:
                if current_section == 'domain_id':
                    domain = line
                    current_section = 'post_domain_id'
                if (current_section == 'agent_params') & (domain_id is domain):
                    params = line
                    return domain, eval(params)
                elif (current_section == 'agent_params') & (domain_id is not domain):
                    current_section = 'domain_id'
                if line.startswith("Task parameters:"):
                    current_section = 'domain_id'
                if line.startswith("Agent parameters:"):
                    current_section = 'agent_params'
                
    return None


def read_sequence(file_name):
    """
    Read a sequence from a file.
    
    Args:
        file_name (str): Path to the file to read
        
    Returns:
        str: Sequence as a single string with whitespace removed
    """
    with open(file_name, 'r') as f:
        events = f.read().replace("\n", "").replace(" ", "")
    return events


def write_sequence(file_name, data):
    """
    Write a sequence to a file with line breaks every 100 characters.
    
    Args:
        file_name (str): Path to the output file
        data (list): List of characters to write
    """
    with open(file_name, 'w') as f:
        for i, char in enumerate(data):
            if i % 100 == 0 and i > 0:
                f.write('\n')
            f.write(char)


def get_relative_path(full_path):
    """
    Get the repository name and relative path.
    
    Args:
        full_path (str): Full path to a file
        
    Returns:
        str: Path relative to the repository root
        
    Raises:
        ValueError: If repository name is not found in the path
    """
    repo_name = "Transformers_for_Modeling_Decision_Sequences"
    try:
        # Find the position of the repository name in the path
        repo_index = full_path.index(repo_name)
        # Return everything from the repo name onwards
        return full_path[repo_index:]
    except ValueError:
        raise ValueError(f"Could not find {repo_name} in path: {full_path}")


def convert_to_local_path(original_path):
    """
    Convert a path from the repository to a local path.
    
    Args:
        original_path (str): Original path in the repository
        
    Returns:
        str: Converted local path
    """
    relative_path = get_relative_path(original_path)
    return os.path.join(os.path.expanduser("~"), "GitHub", relative_path)


def check_files_exist(*filepaths, verbose=True):
    """
    Check if all specified files exist.

    Args:
        *filepaths: Variable number of file paths to check
        verbose (bool): Whether to print information about missing files
        
    Returns:
        bool: True if all files exist, False otherwise
    """
    missing_files = [f for f in filepaths if not os.path.exists(f)]

    if missing_files:
        if verbose:
            print("Missing files:")
            for f in missing_files:
                print(f"  {f}")
        return False

    return True


class ConditionalFormatter(logging.Formatter):
    """
    Custom formatter that handles conditional formatting of log messages.
    Allows for bypassing the standard formatting for certain messages.
    """
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
    
    def format(self, record):
        # Add default job_id if not present
        if not hasattr(record, 'job_id'):
            record.job_id = 'unknown'
            
        if getattr(record, 'no_format', False):
            return record.getMessage()
        return super().format(record)


class FormattedLogger(logging.LoggerAdapter):
    """
    Custom logger adapter that adds formatting options.
    
    This adapter extends the standard logger with a raw method for
    outputting messages without the standard formatting.
    """
    def __init__(self, logger, extra):
        super().__init__(logger, extra)
    
    def raw(self, msg, *args):
        """
        Log a message without the standard formatting.
        
        Args:
            msg (str): Message to log
            *args: Arguments for string formatting
        """
        if args:
            msg = msg % args
        self.logger.info(msg, extra={'no_format': True})


def setup_logging(run_number, component_name, module_name=None):
    """
    Set up logging for experiment scripts.
    
    Args:
        run_number (int): The experiment run number
        component_name (str): Name of the component for log file (e.g., 'data_generation', 'training')
        module_name (str, optional): Name of the module for logger. If None, uses component_name
        
    Returns:
        FormattedLogger: Configured logger instance
    """
    run_dir = get_run_dir(run_number)
    log_dir = os.path.join(run_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Get SLURM job ID from environment variable, default to 'local' if not running in SLURM
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    
    # Configure logging to write to both component-specific file and console
    log_file = os.path.join(log_dir, f'{component_name}.log')
    
    # Clear any existing handlers
    logging.getLogger().handlers = []

    formatter = ConditionalFormatter(
        fmt='%(asctime)s - job_%(job_id)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    handlers = [logging.FileHandler(log_file)]
    if job_id == 'local':  # Only add StreamHandler when not running in SLURM
        handlers.append(logging.StreamHandler(sys.stdout))  # Explicitly use stdout
    
    for handler in handlers:
        handler.setFormatter(formatter)
    
    logging.basicConfig(level=logging.INFO, handlers=handlers)
    
    # Create a logger specific to the module
    logger = logging.getLogger(module_name or component_name)
    
    # Add SLURM job ID to logger's context  
    logger = FormattedLogger(logger, {'job_id': job_id})
    return logger


def setup_project_path():
    """
    Add the project root to the Python path if not already present.
    This ensures imports work correctly regardless of where the script is called from.
    """
    # Get the directory containing this file (utils/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (project root)
    project_root = os.path.dirname(current_dir)
    
    # Add to path if not already present
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print(sys.path)
    return project_root


def get_project_root():
    """
    Get the project root directory.
    
    Returns:
        str: Path to the project root directory
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    return os.path.dirname(current_dir) 