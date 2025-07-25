import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F

from utils.file_management import get_experiment_file, read_sequence


seed = 200
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

class CausalSelfAttention(nn.Module):
    """Causal self-attention layer with masking"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, return_attn_weights=False):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q, k, v = [tensor.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for tensor in (q, k, v)]

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (C ** 0.5)
        attn_weights = attn_weights.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=-1)

        y = torch.matmul(attn_weights, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        if return_attn_weights:
            return y, attn_weights
        return y


class MLP(nn.Module):
    """Multi-layer perceptron for transformer block"""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    """Transformer block with attention and MLP"""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, return_attn_weights=False):
        if return_attn_weights:
            x_residual = x
            x, attn_weights = self.attn(self.ln_1(x), return_attn_weights=True)
            x = x_residual + x
            x = x + self.mlp(self.ln_2(x))
            return x, attn_weights
        else:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x


@dataclass
class GPTConfig:
    """Configuration for the GPT model"""
    block_size: int = 12
    vocab_size: int = 4
    n_layer: int = 1
    n_head: int = 1
    n_embd: int = 64
    device: str = 'cpu'


class GPT(nn.Module):
    """GPT-style transformer model for sequence prediction"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # Weight tying
        self.apply(self._init_weights)
        self.device = config.device

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02 * ((2 * self.config.n_layer) ** -0.5 if hasattr(module, 'NOGPT_SCALE_INIT') else 1)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def combine_logits_by_group(self, logits, targets, group_label=''):
        """Combine logits by grouping tokens (choice or reward)"""
        group_idcs = {
            'choice': [[0, 1], [2, 3]],  # Right and left
            'reward': [[0, 2], [1, 3]]   # Reward and unreward
        }

        grouped_logits = []
        combined_targets = torch.empty_like(targets)

        for i, idcs in enumerate(group_idcs.get(group_label)):
            grouped_logits.append(logits[:, :, idcs].sum(dim=2))
            mask = torch.isin(targets, torch.tensor(idcs).to(self.device))
            combined_targets[mask] = i
    
        combined_logits = torch.stack(grouped_logits, dim=2)
        return combined_logits, combined_targets

    def calculate_loss(self, logits, targets=None, choice_only=False, by_feature=False):
        """Calculate loss with different options (full, choice-only, or by-feature)"""
        if choice_only:
            logits_choice, targets_choice = self.combine_logits_by_group(logits, targets, 'choice')
            loss = F.cross_entropy(logits_choice.view(-1, logits_choice.size(-1)),
                                   targets_choice.view(-1))
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None

        if by_feature:
            logits_choice, targets_choice = self.combine_logits_by_group(logits, targets, 'choice')
            logits_reward, targets_reward = self.combine_logits_by_group(logits, targets, 'reward')
            loss = {'full_loss': loss,
                    'choice_loss': F.cross_entropy(logits_choice.view(-1, logits_choice.size(-1)),
                                                   targets_choice.view(-1)),
                    'reward_loss': F.cross_entropy(logits_reward.view(-1, logits_reward.size(-1)),
                                                    targets_reward.view(-1))
                }
        return loss

    def forward(self, idx, targets=None, return_attn_weights=False, return_residual=False, **kwargs):
        """Forward pass through the model"""
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)

        attn_weights_all_layers = []
        for block in self.transformer.h:
            if return_attn_weights:
                x, attn_weights = block(x, return_attn_weights=True)
                attn_weights_all_layers.append(attn_weights.detach())
            else:
                x = block(x)

        if return_residual:
            residual_snapshot = x.clone().detach()

        # Unembedding: hidden @ wte.weight.T → (vocab_size,)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = self.calculate_loss(logits, targets, **kwargs)
        
        if return_attn_weights:
            # (batch_size, num_heads, seq_len, seq_len)
            return logits, loss, attn_weights_all_layers
        if return_residual:
            return logits, loss, residual_snapshot
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Load pre-trained GPT-2 model weights"""
        from transformers import GPT2LMHeadModel
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args.update(vocab_size=50257, block_size=1024)
        model = GPT(GPTConfig(**config_args))

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd = model.state_dict()

        for k in [key for key in sd_hf.keys() if not key.endswith(('.attn.masked_bias', '.attn.bias'))]:
            if any(k.endswith(w) for w in ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']):
                sd[k].copy_(sd_hf[k].t())
            else:
                sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device, master_process):
        """Configure optimizer with weight decay for training"""
        params = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in params.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in params.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        if master_process:
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            
        fused = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames and device.type == 'cuda'
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=fused)
        return optimizer


class BaseDataLoader:
    """Base data loader for sequence data"""
    def __init__(self, B, T, process_rank, num_processes, run_number=None, suffix='tr'):
        """Initialize data loader for training or validation.
        
        Args:
            B (int): Batch size
            T (int): Sequence length
            process_rank (int): Rank of current process for DDP
            num_processes (int): Total number of processes for DDP
            run_number (int, optional): Run number to load. Defaults to latest run.
            suffix (str, optional): Dataset suffix ('tr' or 'v'). Defaults to 'tr'.
        """

        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        
        # Load data
        behavior_file = get_experiment_file("behavior_run_{}.txt", run_number, suffix, subdir='seqs')
        text = read_sequence(behavior_file)
        
        # Load session transition points
        sessions_file = behavior_file.replace('behavior', 'session_transitions')
        self.session_transitions = []
        with open(sessions_file, 'r') as f:
            for line in f:
                self.session_transitions.append(int(line.strip().split(',')[0]))
                
        self.valid_indices = self.get_valid_indices(T + 1)  # +1 for x (T) and y (1)
        
        # Convert text to tokens
        vocab = ['R', 'r', 'L', 'l']
        stoi = {ch: i for i, ch in enumerate(vocab)}
        tokens = [stoi[ch] for ch in text if ch in stoi]
        if process_rank == 0:
            print(f"read in {len(tokens)} tokens from {behavior_file}")
        
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.original_indices = torch.tensor(range(len(tokens)), dtype=torch.long)
        self.behavior_file = behavior_file

    def get_valid_indices(self, min_length):
        """Get indices where valid sequences of length min_length can start"""
        valid_indices = []
        prev_transition = 0
        
        for transition in self.session_transitions:
            # For each session, find all valid starting points
            for i in range(prev_transition, transition - min_length + 1):
                valid_indices.append(i)
            prev_transition = transition
            
        return torch.tensor(valid_indices, dtype=torch.long)


class DataLoader(BaseDataLoader):
    """Dense dataloader rolling over all possible sequences"""
    def __init__(self, B, T, process_rank, num_processes, run_number=None, suffix='tr'):
        super().__init__(B, T, process_rank, num_processes, run_number, suffix)

        # Assign indices to processes for distributed training
        self.indices_per_process = len(self.valid_indices) // num_processes
        start_idx = process_rank * self.indices_per_process
        end_idx = start_idx + self.indices_per_process if process_rank < num_processes - 1 else len(self.valid_indices)
        self.process_valid_indices = self.valid_indices[start_idx:end_idx]
        
        self.current_position = 0
        self.batches_per_epoch = len(self.process_valid_indices) // B
        self.suffix = suffix

    def handle_batch_overflow(self):
        """Reset position when we run out of batches"""
        if self.suffix == 'tr':
            # For training, add some randomness to diversify batches
            self.current_position = torch.randint(low=0, high=50, size=(1,)).item()
        else:
            self.current_position = 0
    
    def next_batch(self, return_indices=False):
        """Get next batch of data"""
        B, T = self.B, self.T

        if self.current_position + B > len(self.process_valid_indices):
            self.handle_batch_overflow()
        
        # Get starting indices for this batch
        batch_start_indices = self.process_valid_indices[self.current_position:self.current_position+B]
    
        # Create a tensor of shape [B, T] with all required indices
        offsets = torch.arange(T).unsqueeze(0).expand(B, -1)
        indices = batch_start_indices.unsqueeze(1) + offsets
        
        x = self.tokens[indices]
        y = self.tokens[indices + 1]

        self.current_position += B
        
        if return_indices:
            y_indices = self.original_indices[indices + 1]
            return x, y, y_indices
        return x, y

class DataLoaderLite(DataLoader):
    """Lighter data loader that avoids overlapping contexts"""
    def __init__(self, B, T, process_rank, num_processes, run_number=None, suffix='tr'):
        super().__init__(B, T, process_rank, num_processes, run_number, suffix)
        self.valid_indices = self.filter_overlapping_indices(self.valid_indices, T)

        # Reassign indices to processes
        self.indices_per_process = len(self.valid_indices) // num_processes
        start_idx = process_rank * self.indices_per_process
        end_idx = start_idx + self.indices_per_process if process_rank < num_processes - 1 else len(self.valid_indices)
        self.process_valid_indices = self.valid_indices[start_idx:end_idx]
        
        self.current_position = 0
        self.batches_per_epoch = len(self.process_valid_indices) // B
        self.suffix = suffix

    def handle_batch_overflow(self):
        """Reset position when out of batches"""
        self.current_position = 0
    
    def filter_overlapping_indices(self, indices, context_length):
        """Filter start trial indices to avoid overlapping contexts"""
        if len(indices) == 0:
            return indices
            
        # Sort indices first
        sorted_indices = torch.sort(indices)[0]
        
        # Initialize with the first index
        filtered_indices = [sorted_indices[0].item()]
        
        # Iterate through sorted indices
        for idx in sorted_indices[1:]:
            # Check if this index is at least T positions away from the last accepted index
            if idx >= filtered_indices[-1] + context_length:
                filtered_indices.append(idx.item())
        
        return torch.tensor(filtered_indices, dtype=torch.long)
    
class DataLoaderShuffle(DataLoader):
    """Data loader that shuffles indices for each epoch"""
    def __init__(self, B, T, process_rank, num_processes, run_number=None, suffix='tr'):
        super().__init__(B, T, process_rank, num_processes, run_number, suffix)
        
        torch.manual_seed(200 + process_rank)  # Different seed per process

        # Shuffle and assign indices to processes
        start_idx = process_rank * self.indices_per_process
        end_idx = start_idx + self.indices_per_process if process_rank < num_processes - 1 else len(self.valid_indices)
        self.shuffled_indices = torch.randperm(len(self.valid_indices))
        self.process_valid_indices = self.shuffled_indices[start_idx:end_idx]
        
    def handle_batch_overflow(self):
        """Reshuffle for next epoch"""
        self.current_position = 0
        torch.manual_seed(200 + self.process_rank + int(torch.randint(1, 1000, (1,)).item()))
        self.shuffled_indices = torch.randperm(len(self.valid_indices))
        start_idx = self.process_rank * self.indices_per_process
        end_idx = start_idx + self.indices_per_process
        self.process_valid_indices = self.shuffled_indices[start_idx:end_idx]


class DDPConfig:
    """Configuration for Distributed Data Parallel training"""
    def __init__(self):
        # Check if we're in a SLURM environment
        slurm_procid = os.environ.get('SLURM_PROCID')
        slurm_localid = os.environ.get('SLURM_LOCALID')
        
        # Check if we're in a PyTorch DDP environment
        rank = os.environ.get('RANK')
        local_rank = os.environ.get('LOCAL_RANK')
        world_size = os.environ.get('WORLD_SIZE')
        
        print(slurm_procid, rank, local_rank, world_size)
        # Determine if we're in a distributed environment
        self.ddp = (int(os.environ.get('WORLD_SIZE', 1)) > 1) or (int(os.environ.get('SLURM_NTASKS', 1)) > 1)
        
        if self.ddp:
            assert torch.cuda.is_available(), "need CUDA for DDP"
            
            # Set up ranks and process group
            self.local_rank = int(os.environ.get('SLURM_PROCID')) - int(os.environ.get('SLURM_NODEID')) * int(os.environ.get('SLURM_NTASKS_PER_NODE'))
            self.rank = int(os.environ.get('SLURM_PROCID'))
            self.world_size = int(os.environ.get('WORLD_SIZE'))
            self.master_process = (self.rank == 0)
            
            # Initialize the process group
            try:
                print(self.rank, self.local_rank, self.world_size)
                dist.init_process_group(
                    backend="nccl",
                    timeout=timedelta(minutes=30),
                    init_method="env://",
                    rank=self.rank,
                    world_size=self.world_size
                )
                dist.barrier()            
                # Use the local_rank to set the device
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device(f'cuda:{self.local_rank}')

            except Exception as e:
                print(f"Error initializing process group: {e}")
                print(f"Environment variables: RANK={os.environ.get('RANK')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}, "
                      f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}, MASTER_ADDR={os.environ.get('MASTER_ADDR')}, "
                      f"MASTER_PORT={os.environ.get('MASTER_PORT')}")
                # Fall back to single-process mode
                self.ddp = False
                self.rank = 0
                self.local_rank = 0
                self.world_size = 1
                self.master_process = True
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            print(f"Process {self.rank} initialized successfully: "
                  f"local_rank={self.local_rank}, master_process={self.master_process}")
        else:
            # Single process mode
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.master_process = True
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"