"""
Model utility functions for selecting and managing different model types.
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys

import utils.file_management as fm
from transformer.transformer import GPTConfig

def select_model(model_type):
    """
    Select the appropriate model class based on model type.
    
    Args:
        model_type (str): Type of model to select ('GPT', 'GPT_noLN', etc.)
    
    Returns:
        class: The appropriate model class
    
    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type == 'GPT':
        from transformer import GPT
        return GPT
    elif model_type == 'GPT_noLN':
        from alt_transformer import GPT_noLN
        return GPT_noLN
    elif model_type == 'LastTokenGPT':
        from alt_transformer import LastTokenGPT
        return LastTokenGPT
    else:
        raise ValueError(f"Invalid model type: {model_type}. "
                        f"Supported types: {get_available_model_types()}")

def get_available_model_types():
    """
    Get list of available model types.
    
    Returns:
        list: List of available model type strings
    """
    return ['GPT', 'GPT_noLN', 'LastTokenGPT']


def create_model_from_config(model_type, config):
    """
    Create a model instance from a model type and config.
    
    Args:
        model_type (str): Type of model to create
        config (GPTConfig): Model configuration
        
    Returns:
        Model instance
    """
    ModelClass = select_model(model_type)
    return ModelClass(config)

def parse_model_info(run=None, model_name=None):
    """
    Parse model information from metadata.txt
    
    Args:
        run (int, optional): Run number. If None, uses the latest run.
        model_name (str, optional): Filter for specific model name
        
    Returns:
        dict: Dictionary containing parsed model information
    """
    metadata_file = fm.get_experiment_file("metadata.txt", run)
    model_info = {
        'model_name': None,
        'model_type': None,
        'tokens_seen': None,
        'dataloader': {},
        'config': {}
    }
    if model_name is not None:
        model_name = model_name.split('_cp')[0]
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
            elif line.startswith("Model type:"):
                model_info['model_type'] = line.split(": ")[1]
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
    """
    Get the name of the latest model based on tokens seen.
    
    Args:
        run (int, optional): Run number. If None, uses the latest run.
        
    Returns:
        str: Latest model name
    """
    model_info = parse_model_info(run)
    return model_info['model_name']

def load_trained_model(run, model_name, device, **kwargs):
    """
    Load a trained model from disk with proper model type selection.
    
    Args:
        run (int): Run number
        model_name (str): Name of the model to load
        device (str): Device to load the model on
        **kwargs: Additional arguments for torch.load
        
    Returns:
        tuple: (model, model_info, config)
    """
    # Get model info from metadata
    model_info = parse_model_info(run, model_name=model_name)
    if model_name is None:
        model_name = model_info['model_name']
    else:
        assert (model_info['model_name'] == model_name) or (model_info['model_name'] == model_name.split('_cp')[0]), (
            'did not recover correct model')

    # Configure model using metadata
    config = GPTConfig(
        block_size=model_info['config'].get('Block size', 12),
        vocab_size=model_info['config'].get('Vocab size', 4),
        n_layer=model_info['config'].get('Number of layers', 1),
        n_head=model_info['config'].get('Number of heads', 1),
        n_embd=model_info['config'].get('Embedding size', 64)
    )
    
    # Load the trained model using the correct model type
    model_type = model_info.get('model_type', 'GPT')  # Default to GPT for backward compatibility
    model = create_model_from_config(model_type, config)
    model_path = fm.get_experiment_file(f'{model_name}.pth', run, subdir='models')

    try:
        model.load_state_dict(torch.load(model_path, map_location=device, **kwargs))
    except Exception:
        checkpoint = torch.load(model_path, map_location=device, **kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, model_info, config

def save_model(model, model_name, run_number, *, is_checkpoint=False, step=None, compile=False, logger=None, **kwargs):
    """
    Save model weights or checkpoint.
    
    Args:
        model: The model to save
        model_name (str): Name for the saved model
        run_number (int): Run number for file path
        is_checkpoint (bool): Whether this is a checkpoint save
        step (int): Step number for checkpoint naming
        compile (bool): Whether model is compiled
        logger: Optional logger instance
        **kwargs: Additional arguments (optimizer, best_val_loss, etc.)
    """
    suffix = f"_cp{step}" if is_checkpoint else ""
    model_path = fm.get_experiment_file(f'{model_name}{suffix}.pth', run_number, subdir='models')
    
    if logger:
        logger.info("Saving model at: %s", model_path)
    
    # Get state dict based on model type
    if isinstance(model, DDP):
        state_dict = model.module.state_dict()
    elif compile:
        state_dict = model._orig_mod.state_dict()
    else:
        state_dict = model.state_dict()
    
    # Save checkpoint or just weights
    if is_checkpoint:
        checkpoint = {
            'model_state_dict': state_dict,
            'optimizer_state_dict': kwargs.get('optimizer').state_dict(),
            'step': step,
            'best_val_loss': kwargs.get('best_val_loss'),
            'loss_steps': kwargs.get('loss_steps'),
            'val_loss_steps': kwargs.get('val_loss_steps'),
        }
        torch.save(checkpoint, model_path)
    else:
        torch.save(state_dict, model_path) 