import argparse
import getpass
import math
import os
import sys
import time
import glob
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
# import wandb commented out for now because of permission errors
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from transformer import GPT, DataLoaderLite, DataLoader, DDPConfig, GPTConfig, DataLoaderShuffle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.file_management as fm

logger = None

username = getpass.getuser()

def initialize_logger(run_number, is_master_process=False):
    """Initialize the logger with the correct run number."""
    global logger
    if is_master_process:
        logger = fm.setup_logging(run_number, 'training', 'train')
        logger.info(f"Initialized master logger (rank 0)")
    else:
        # Create a null logger for non-master processes
        logger = logging.getLogger('null_logger')
        logger.addHandler(logging.NullHandler())

def parse_args():
    parser = argparse.ArgumentParser(description='Train GPT model with hyperparameter tuning.')
    parser.add_argument('--sequence_length', type=int, default=12, help='Sequence length (T) for training.')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of transformer layers.')
    parser.add_argument('--n_head', type=int, default=1, help='Number of attention heads.')
    parser.add_argument('--n_embd', type=int, default=64, help='Embedding size.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of Epochs looping through the training data.')
    parser.add_argument('--max_lr', type=float, default=6e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--task_id', type=int, default=None, help='SLURM task ID.')
    parser.add_argument('--run_number', type=int, default=None, help='ID of dataset to train/validate on')
    parser.add_argument('--compile', action='store_true', default=False, help='Flag to compile the code for faster training')
    parser.add_argument('--predict', action='store_true', default=False, help='Flag to predict on the validation set')
    parser.add_argument('--eval_interval', type=int, default=None, help='Interval to evaluate the model')
    parser.add_argument('--checkpoint_interval', type=str, default=None, help='Number of epochs between checkpoints or "log" for logarithmic spacing')
    parser.add_argument('--enforce_data_epochs', action='store_true', default=False, help='Flag to force data loader to reset')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--choice_only', action='store_true', default=False,
                        help='Objective to optimize -- choice only excludes reward prediction')

    args = parser.parse_args()

    if args.checkpoint_interval is None:
        args.checkpoint_interval = max(1, int(args.epochs // 10))
    elif args.checkpoint_interval != 'log':
        args.checkpoint_interval = float(args.checkpoint_interval)

    return args

def write_predictions(model_name, predictions, last_step=False):
    # Define the vocabulary and mappings
    vocab = ['R', 'r', 'L', 'l']
    itos = {i: ch for i, ch in enumerate(vocab)}
    
    pred_file = fm.get_experiment_file(f"learning_{model_name}_val_preds.txt", run_number, subdir='seqs')
    
    if predictions['step'][0] == 0:
        # Initialize the validation predictions file.
        with open(pred_file, 'w') as f:
            f.write("Step\tTrue\tPredicted\tIdx\n")

    # Convert tensors to strings/values
    true_tokens = [itos[t.item()] for t in predictions['true_next']]
    pred_tokens = [itos[t.item()] for t in predictions['pred_next']]

    with open(pred_file, 'a') as f:
        for s, true, pred, idx in zip(
            predictions['step'],
            true_tokens,
            pred_tokens,
            predictions['y_indices'].numpy()
        ):
            f.write(f"{s}\t{true}\t{pred}\t{idx}\n")
    if last_step:
        logger.info(f"Sampled validation predictions saved to {pred_file}")

def write_metadata(model, model_name, total_batch_size, max_steps, train_loader, val_loader, config):
    metdata_file = fm.get_experiment_file("metadata.txt", run_number)
    tokens_trained_on = total_batch_size * max_steps

    with open(metdata_file, 'a') as meta_file:
        meta_file.write(f"\nModel name: {model_name}\n")
        meta_file.write(f"  Num Parameters: {sum(p.numel() for p in model.parameters())}\n")
        meta_file.write(f"\nTokens seen: {tokens_trained_on:,}\n")
        meta_file.write(f"\nTotal batch size: {total_batch_size:,}\n")
        meta_file.write(f"\nMax steps: {max_steps:,}\n")
        meta_file.write(f"\nDataloader parameters:\n")
        meta_file.write(f"\nFile trained on: {train_loader.behavior_file}\n")
        meta_file.write(f"\nFile validated on: {val_loader.behavior_file}\n")
        meta_file.write(f"  Batch size (B): {train_loader.B}\n")
        meta_file.write(f"  Sequence length (T): {train_loader.T}\n")
        meta_file.write(f"  Steps per epoch: {train_loader.batches_per_epoch}")
        meta_file.write(f"\nGPTConfig parameters:\n")
        meta_file.write(f"  Block size: {config.block_size}\n")
        meta_file.write(f"  Vocab size: {config.vocab_size}\n")
        meta_file.write(f"  Number of layers: {config.n_layer}\n")
        meta_file.write(f"  Number of heads: {config.n_head}\n")
        meta_file.write(f"  Embedding size: {config.n_embd}\n")
        meta_file.write(f"\n")
    logger.info(f"Metadata saved to {metdata_file}")

def write_experiment_summary(args, model, model_name, val_loss_steps, max_steps):

    import pandas as pd

    def _load_summary(path_to_file: str):
        try:
            summary = pd.read_csv(path_to_file, index_col=None)
            return summary
        except FileNotFoundError:
            return pd.DataFrame()

    def _save_summary(curr_summary: pd.DataFrame):
        path_to_file = os.path.abspath(os.path.join(__file__, '../../', 'model_summary.csv'))
        summary = _load_summary(path_to_file)
        summary = pd.concat((summary, curr_summary)).reset_index(drop=True)
        summary.to_csv(path_to_file, index=False)
        logger.info(f"Experiment summary saved to {path_to_file}")

    losses = {}
    xs = np.concatenate([np.arange(0, max_steps, args.eval_interval), [max_steps]])

    if isinstance(val_loss_steps, dict):
        for key, data in val_loss_steps.items():
            losses[f'best_val_{key}'] = min(data)
            losses[f'best_val_{key}_step'] = xs[data.index(min(data))]
    else:
        losses['best_val_full_loss'] = min(val_loss_steps)
        losses['best_val_full_loss_step'] = xs[val_loss_steps.index(min(val_loss_steps))]
    summary = {
        'model_id': os.environ.get('SLURM_JOB_NAME', 'unknown_job'),
        'experiment_type': os.environ.get('EXPERIMENT_TYPE', None),
        'domain_config': os.environ.get('DOMAIN_CONFIG', None),
        'domain_id': os.environ.get('DOMAIN_ID', None),
        'num_samples': model_name[len("model_seen"):],
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'max_steps': max_steps,
        'run_number': run_number
    }
    args_dict = vars(args)
    summary.update(args_dict)
    summary.update(losses)
    logger.info(f"Experiment summary:\n{summary}")

    df = pd.DataFrame(summary, index=[0])
    _save_summary(df)

def save_model(model, model_name, run_number, *, is_checkpoint=False, step=None, compile=False, **kwargs):
    suffix = f"_cp{step}" if is_checkpoint else ""
    model_path = fm.get_experiment_file(f'{model_name}{suffix}.pth', run_number, subdir='models')
    logger.info("Saving model at: %s", model_path)
    if isinstance(model, DDP):
        state_dict = model.module.state_dict()
    elif compile:
        state_dict = model._orig_mod.state_dict()
    else:
        state_dict = model.state_dict()
    if is_checkpoint:
        checkpoint = {
            'model_state_dict': state_dict,
            'optimizer_state_dict': kwargs.get('optimizer').state_dict(),
            'step': step,  # Save the current step or epoch
            'best_val_loss': kwargs.get('best_val_loss'),
            'loss_steps': kwargs.get('loss_steps'),
            'val_loss_steps': kwargs.get('val_loss_steps'),
        }
        torch.save(checkpoint, model_path)
    else:
        # switch to saving in scratch?
        torch.save(state_dict, model_path)
    # wandb.save(model_path)

def plot_losses(loss_steps, val_loss_steps, max_steps, eval_interval, model_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = np.arange(0, max_steps, eval_interval)
    if not xs[-1] == (max_steps-1):
        xs = np.concatenate([xs, [max_steps-1]])
    ax.plot(xs, loss_steps, label='Training Loss')
    if isinstance(val_loss_steps, dict):
        for key, data in val_loss_steps.items():
            ax.plot(xs, data, label=f'Validation {key}')
    else:
        ax.plot(xs, val_loss_steps, label='Validation Loss')
    ax.set(xlabel='Steps', ylabel='Loss', title='Training and Validation Losses')
    ax.legend()
    fig_path = fm.get_experiment_file(f'losses_{model_name}.png', run_number, subdir='models')
    fig.savefig(fig_path)

def profile_execution(function_to_profile, *args, **kwargs):
    """Profiles the execution of a function and generates a performance plot."""
    import cProfile
    import pstats
    with cProfile.Profile() as pr:
        function_to_profile(*args, **kwargs)
    stats = pstats.Stats(pr)
    stats.sort_stats('cumtime')

    # Extract function statistics for plotting based on cumulative time
    function_names = []
    cumulative_times = []
    for func, stat in stats.stats.items():
        filename, lineno, func_name = func
        cumulative_time = stat[3]  # cumulative time is the 4th element in the tuple
        if cumulative_time > 0.01:  # Threshold for relevance
            function_names.append(f"{lineno}({func_name})")
            cumulative_times.append(cumulative_time)

    # Plot profiling results
    plt.figure(figsize=(10, 6))
    plt.barh(function_names, cumulative_times, color="skyblue")
    plt.xlabel("Cumulative Time (s)")
    plt.ylabel("Function")
    plt.title("Cumulative Time of Key Functions in Profiled Code")
    plt.gca().invert_yaxis()
    plt.show()

def get_lr(step, lr_schedule, max_steps):
    warmup_steps = lr_schedule['warmup_steps']
    max_lr = lr_schedule['max_lr']
    min_lr = lr_schedule['min_lr']
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (1 + math.cos(math.pi * decay_ratio)) * (max_lr - min_lr)

def estimate_loss(model, val_loader, ddp, step, predict=False, policy='argmax'):
    model.eval()
    val_losses = {}
    if predict:
        predictions = {
            'step': [],
            'context': torch.empty((0, val_loader.T), dtype=torch.long),  # [total_samples, T]
            'true_next': torch.empty(0, dtype=torch.long),                # [total_samples]
            'pred_next': torch.empty(0, dtype=torch.long),                # [total_samples]
            'y_indices': torch.empty(0, dtype=torch.long),                # [total_samples]
        }
    for _ in range(val_loader.batches_per_epoch):
        if predict:
            x, y, y_indices = val_loader.next_batch(return_indices=True)
        else:
            x, y = val_loader.next_batch()
        x, y = x.to(ddp.device), y.to(ddp.device)
        with torch.no_grad():
            logits, loss = model(x, y, by_feature=True)
            
            if isinstance(loss, dict):
                for key, value in loss.items():
                    if key not in val_losses:
                        val_losses[key] = []  # Initialize a list for each key if not already present
                    val_losses[key].append(value)  # Append the loss value for this key
            else:
                if _ == 0:
                    val_losses = []
                val_losses.append(loss)  # Handle the single value case

            if predict:
                # Get predicted next tokens
                last_logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
                if policy == 'argmax':
                    pred_tokens = torch.argmax(last_logits, dim=-1)  # Shape: [batch_size]
                elif policy == 'softmax':
                    probs = F.softmax(last_logits, dim=-1)  # drawing from the distribution for each sample
                    pred_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [batch_size]                
                else:
                    raise ValueError(f"Invalid policy: {policy}")

                # Store entire batch at once
                predictions['context'] = torch.cat([predictions['context'], x.cpu()], dim=0)
                predictions['true_next'] = torch.cat([predictions['true_next'], y[:, -1].cpu()])
                predictions['pred_next'] = torch.cat([predictions['pred_next'], pred_tokens.cpu()])
                predictions['y_indices'] = torch.cat([predictions['y_indices'], y_indices[:, -1].cpu()])
                predictions['step'].extend([step] * x.shape[0])

    avg_loss = {}
    # Calculate average loss for each key if losses is a dictionary
    if isinstance(val_losses, dict):
        for key in val_losses.keys():
            avg_loss[key] = torch.stack(val_losses[key]).mean()
    else:
        avg_loss = torch.stack(val_losses).mean()

    # Reduce the loss across all processes if using DDP
    if ddp.ddp:
        if isinstance(avg_loss, dict):
            # dist.all_reduce(avg_loss['full_loss'], op=dist.ReduceOp.SUM)
            for key in avg_loss.keys():
                dist.all_reduce(avg_loss[key], op=dist.ReduceOp.AVG)
                # avg_loss[key] = avg_loss[key] / ddp.world_size
        else:
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            # avg_loss = avg_loss / ddp.world_size
    model.train()  # Switch back to training mode
    if predict:
        return avg_loss, predictions
    return avg_loss

def update_predictions_file(model_name, starting_step):
    pred_file = fm.get_experiment_file(f"learning_{model_name}_val_preds.txt", run_number, subdir='seqs')

    # Check if the predictions file exists
    if not os.path.exists(pred_file):
        logger.info(f"Predictions file {pred_file} does not exist. No updates needed.")
        return None  # Exit the function if the file does not exist

    # Read existing predictions
    with open(pred_file, 'r') as f:
        lines = f.readlines()
    excess_steps = []

    # Write back only the lines that are within the cutoff step
    with open(pred_file, 'w') as f:
        for line in lines:
            if line.startswith("Step"):  # Keep the header
                f.write(line)
            else:
                step = int(line.split('\t')[0])  # Extract the step from the line
                if step <= starting_step:  # Check against the cutoff step
                    f.write(line)
                else:
                    excess_steps.append(step)
    if excess_steps:
        logger.info(f"Excess steps: {excess_steps[0]} to {excess_steps[-1]}")

def trim_loss_steps(losses, starting_step, eval_interval):
    
    if starting_step < eval_interval:
        return losses
    # Get the indices of the losses to keep.
    idcs = np.insert(np.arange(0, starting_step, eval_interval), -1, starting_step)
    num_idcs = len(idcs)

    if isinstance(losses, dict):
        for key in losses.keys():
            losses[key] = losses[key][:num_idcs]
    else:
        losses = losses[:num_idcs]

    return losses

def update_checkpoint_interval(nth_checkpoint=1, max_steps=None):

    min_interval = 1
    max_interval = 3000

    log_factor = 2.5
    
    try:
        checkpoint_steps = max(min_interval, min(max_interval, int(log_factor ** nth_checkpoint)))
    except OverflowError:
        checkpoint_steps = max_interval

    if nth_checkpoint == 1:
        checkpoints = [checkpoint_steps]
        j = 2
        i = checkpoint_steps
        while i < max_steps:
            checkpoints.append(update_checkpoint_interval(j, max_steps))
            j += 1
            i += checkpoints[-1]
        print(len(checkpoints), checkpoints)
        assert(len(checkpoints) < 40), "Excessive number of checkpoints"

    return int(checkpoint_steps)

def steps_per_checkpoint(checkpoint_interval, batches_per_epoch, grad_accum_steps, max_steps=None):

    if checkpoint_interval == 'log':
        checkpoint_steps = update_checkpoint_interval(max_steps=max_steps)
    else:
        steps_per_epoch = batches_per_epoch / grad_accum_steps
        checkpoint_steps = int(checkpoint_interval * steps_per_epoch)
    
    return checkpoint_steps


def main():    
    seed = 200
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_float32_matmul_precision('high')
    args = parse_args()
    ddp = DDPConfig()
    #redeclare variables if they didn't get passed through
    ddp.rank = int(os.environ.get('SLURM_PROCID'))
    ddp.local_rank = int(os.environ.get('SLURM_LOCALID'))
    ddp.master_process = (ddp.rank == 0)
    ddp.world_size = int(os.environ.get('SLURM_NTASKS'))
    # After DDP setup, print again with the configured rank
    print(f"After DDP setup: I am process with rank={ddp.rank}, local_rank={ddp.local_rank}, master_process={ddp.master_process}")
    
    global run_number
    run_number = args.run_number or fm.get_latest_run()
    initialize_logger(run_number, is_master_process=ddp.master_process)
    
    if (args.checkpoint_interval == 'log') or (args.checkpoint_interval < 1):
        logger.info("Checkpoint interval is less than 1. Enforcing data epochs is disabled.")

    if ddp.master_process:
        logger.info("Starting training script with args: %s", args)

    lr_schedule ={
        'max_lr': args.max_lr,
        'min_lr': args.max_lr * 0.1,
        'warmup_steps': 1000,
    }

    # Training setup
    B = args.batch_size  # number of samples per batch
    T = args.sequence_length  # number of trials per sample
    total_batch_size = 2 * B * T * ddp.world_size # number of tokens per batch = inference steps with our dense dataloader
    assert total_batch_size % (B * T * ddp.world_size) == 0, (
        "make sure total batch size is divisible by B * T * ddp.world_size")

    # Number of micro steps to reach total batch size (inner training loop).
    grad_accum_steps = total_batch_size // (B * T * ddp.world_size)

    train_loader = DataLoader(
        B=B,
        T=T,
        process_rank=ddp.rank,
        num_processes=ddp.world_size,
        run_number=run_number,
        suffix='tr'
    )
    val_loader = DataLoader(
        B=2048,
        T=T,
        process_rank=ddp.rank,
        num_processes=ddp.world_size,
        run_number=run_number,
        suffix='v'
    )
    logger.info(f"Train loader class: {train_loader.__class__.__name__}")
    print('valid indices', len(train_loader.process_valid_indices))
    # Number steps required to pass over full dataset x n_epochs.
    max_steps = int(train_loader.batches_per_epoch * args.epochs / grad_accum_steps)
    n_samples = B * train_loader.batches_per_epoch * args.epochs * ddp.world_size
    model_name = f"model_seen{fm.format_tokens(n_samples)}"

    if args.eval_interval is None:
        args.eval_interval = max(1, int(max_steps // 100))
        logger.info(f"Setting eval interval to {args.eval_interval}")

    if ddp.master_process:
        logger.info(f"total desired batch size: {total_batch_size}")
        logger.info(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
        logger.info(f"=> calculated steps: {max_steps}")

    # Create model.
    model = GPT(GPTConfig(
        vocab_size=4,
        block_size=T,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        device=ddp.device
    ))
    model.to(ddp.device)

    # Check whether checkpoint or model already exists.
    if os.path.exists(fm.get_experiment_file(f'{model_name}.pth', run_number, subdir='models')):
        logger.info("Model already exists. Skipping training.")
        return None
    elif any(checkpoints := glob.glob(os.path.join(fm.get_run_dir(run_number), 'models', "*cp*.pth"))):
        if args.checkpoint_interval == 'log':
            raise NotImplementedError('Checkpoint loading not for dynamic checkpointing.')
        # I think optimizer can be configured here?
        optimizer = model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=lr_schedule['max_lr'],
            device=ddp.device,
            master_process=ddp.master_process)

        model_path = sorted(checkpoints)[-1]
        logger.info(f"Checkpoint already exists. Loading checkpoint from {model_path}.")
        checkpoint = torch.load(model_path, map_location=ddp.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        starting_step = checkpoint['step']
        best_val_loss = checkpoint['best_val_loss']  # There is a chance this came after the checkpoint was saved.
        loss_steps = trim_loss_steps(checkpoint['loss_steps'], starting_step, args.eval_interval)
        val_loss_steps = trim_loss_steps(checkpoint['val_loss_steps'], starting_step, args.eval_interval)
        val_loss = val_loss_steps.get('full_loss')[-1] if isinstance(val_loss_steps, dict) else val_loss_steps[-1]
        logger.info(f"Starting from step {starting_step}")
        logger.info('Num loss steps: %d', len(loss_steps))
        logger.info('Adjusted from: %d', len(checkpoint['loss_steps']))
        # Remove any predictions made after the checkpoint was saved.
        update_predictions_file(model_name, starting_step)
        model.to(ddp.device)

    else:
        best_val_loss = float('inf')
        val_loss = None
        loss_steps = []
        val_loss_steps = {}
        starting_step = 0
    
    next_checkpoint_step = steps_per_checkpoint(args.checkpoint_interval, train_loader.batches_per_epoch,
                                                grad_accum_steps, max_steps=max_steps)
    nth_checkpoint = 1

    if ddp.master_process:
        logger.info(f"Number of steps to checkpoint: {next_checkpoint_step}")
        logger.info(f"Number of batches per epoch: {train_loader.batches_per_epoch}")

    if args.compile:
        model = torch.compile(model)
    if ddp.ddp:
        model = DDP(model, device_ids=[ddp.local_rank])
    raw_model = model.module if ddp.ddp else model

    # Configure optimizer.
    optimizer = raw_model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=lr_schedule['max_lr'],
        device=ddp.device,
        master_process=ddp.master_process)

    if ddp.ddp:
        dist.barrier()

    print(f"Rank: {ddp.rank}, Local Rank: {ddp.local_rank}, World Size: {ddp.world_size}")
    print(f"Rank: {ddp.rank},", train_loader.process_valid_indices[:10])
    if ddp.master_process:
        # wandb.init(
        #     project="gpt-training",
        #     config={
        #         "run_number": run_number,
        #         "total_batch_size": total_batch_size,
        #         "max_lr": lr_schedule['max_lr'],
        #         "min_lr": lr_schedule['min_lr'],
        #         "warmup_steps": lr_schedule['warmup_steps'],
        #         "max_steps": max_steps,
        #         "B": B,
        #         "T": T,
        #         "grad_accum_steps": grad_accum_steps,
        #         "vocab_size": 4,
        #         "n_layer": args.n_layer,
        #         "n_head": args.n_head,
        #         "n_embd": args.n_embd,
        #         "task_id": args.task_id
        #     },
        #     name=f"run_task_{args.task_id}",  # Name the run based on the task ID
        #     dir="/tmp",
        # )
        # wandb.watch(model)
        print("DDP WORLD SIZE: ", ddp.world_size)

    # Training loop
    for step in range(starting_step, max_steps):

        if ddp.device_type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0

        # Accumulate gradients over multiple mini-batches (micro_steps)
        for micro_step in range(grad_accum_steps):
            # logger.info(f'micro step {step}: {train_loader.current_position}')
            x, y = train_loader.next_batch()
            x, y = x.to(ddp.device), y.to(ddp.device)

            # Forward pass and loss computation
            with torch.autocast(device_type=ddp.device_type, dtype=torch.bfloat16):
                _, loss = model(x, y, choice_only=args.choice_only)
            loss = loss / grad_accum_steps  # Normalize loss over gradient accumulation steps
            loss_accum += loss.detach()  # Track the total loss
            if ddp.ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            loss.backward()  # Backpropagate gradients

        # Clip gradients to prevent exploding gradients
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step, lr_schedule, max_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()
        if ddp.device_type == 'cuda':
            torch.cuda.synchronize()
        # Time the step and calculate tokens processed per second
        t1 = time.time()
        dt = (t1 - t0) * 1000  # Time in milliseconds
        tokens_per_sec = (train_loader.B * train_loader.T) * grad_accum_steps * ddp.world_size/ (t1 - t0)

        """VALIDATION SAMPLING"""
        # Print logging information every eval_interval steps
        if step % args.eval_interval == 0 or step == max_steps - 1:
            if args.predict:
                val_loss, predictions = estimate_loss(model, val_loader, ddp, step, predict=True, policy='softmax')
                write_predictions(model_name, predictions, step==(max_steps-1))
            else:
                val_loss = estimate_loss(model, val_loader, ddp, step, predict=False)
            
            if ddp.master_process:
                if isinstance(val_loss, dict):
                    for key in val_loss.keys():
                        if key not in val_loss_steps:
                            val_loss_steps[key] = []
                        val_loss_steps[key].append(val_loss[key].item())
                else:
                    if isinstance(val_loss_steps, dict):
                        val_loss_steps = []
                    val_loss_steps.append(val_loss)
        
                val_loss_choice = val_loss.get('choice_loss').item() if isinstance(val_loss, dict) else None
                val_loss_reward = val_loss.get('reward_loss').item() if isinstance(val_loss, dict) else None
                val_loss = val_loss.get('full_loss').item() if isinstance(val_loss, dict) else val_loss.item()
                # wandb.log({
                #     "step": step,
                #     "loss": loss_accum.item(),
                #     "val_loss": val_loss,
                #     "choice_loss": val_loss_choice,
                #     "reward_loss": val_loss_reward,
                #     "lr": lr,
                #     "grad_norm": norm,
                #     "step_time_ms": dt,
                #     "tokens_per_sec": tokens_per_sec,
                # })
                
                if step % (args.eval_interval*10) == 0:
                    logger.info(f"step {step} | loss: {loss_accum.item():.4f} | val_loss: {val_loss:.4f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f} ms | tok/sec: {tokens_per_sec:.2f}")
                
                loss_steps.append(loss_accum.item())

        """CHECKPOINTING"""
        if (step % next_checkpoint_step == 0) and ddp.master_process:
            logger.info(f"Checkpoint at step {step} with dataloader position {train_loader.current_position}")
            if loss_improved := (val_loss < best_val_loss):
                best_val_loss = val_loss
            # Save the model checkpoint
            save_model(model, model_name, run_number, is_checkpoint=True, compile=args.compile,
                        step=step, optimizer=optimizer, best_val_loss=best_val_loss, loss_steps=loss_steps,
                        val_loss_steps=val_loss_steps)
            logger.info(f"New best validation loss: {best_val_loss:.4f}. Model checkpoint saved at step {step}. Validation loss improved: {loss_improved}")

            if args.checkpoint_interval == 'log':
                nth_checkpoint += 1
                next_checkpoint_step = update_checkpoint_interval(nth_checkpoint)
                logger.info(f"Number of steps to checkpoint: {next_checkpoint_step}")

            if args.enforce_data_epochs:
                if ddp.master_process:
                    logger.info(f'prior to data reset (training): {train_loader.current_position}')
                train_loader.current_position = 0
    
    if ddp.ddp:
        dist.barrier()

    if ddp.master_process:
        if ddp.ddp:
            model = model.module
        save_model(model, model_name, run_number, compile=args.compile)
        write_metadata(model, model_name, total_batch_size, max_steps, train_loader, val_loader, model.config)
        # wandb.finish()
        plot_losses(loss_steps, val_loss_steps, max_steps, args.eval_interval, model_name)
        write_experiment_summary(args, model, model_name, val_loss_steps, max_steps)

    if ddp.ddp:
        destroy_process_group()

if __name__ == "__main__":
    print('-' * 80)
    print('train.py\n')
    ENABLE_PROFILING = False
    if ENABLE_PROFILING:
        profile_execution(main)
    else:
        main()