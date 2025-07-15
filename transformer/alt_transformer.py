import os
import sys
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.file_management import setup_project_path
# setup_project_path()
from transformer import GPT, MLP, Block, CausalSelfAttention


seed = 200
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

class LastTokenAttentionFinal(nn.Module):
    """Optimized causal self-attention for final layer"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Only need query for last token
        q_last = q[:, -1:, :]  # Shape: (B, 1, n_embd)
        
        # Need full k, v since last token can attend to all positions
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q_last = q_last.view(B, 1, self.n_head, C // self.n_head)
        q_last = q_last.transpose(1, 2)
        
        # Compute attention only for last token
        attn_weights = torch.matmul(q_last, k.transpose(-2, -1)) / (C ** 0.5)
        mask = self.bias[:, :, -1:, :T] == 0
        attn_weights = attn_weights.masked_fill(mask, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # Compute output only for last token
        y = torch.matmul(attn_weights, v)  # (B, n_head, 1, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, 1, C)  # (B, 1, C)
        y = self.c_proj(y)
        
        return y.squeeze(1)  # (B, C)


class LastTokenMLP(MLP):
    """Multi-layer perceptron for transformer block operating only on the last token"""
    def __init__(self, config):
        super().__init__(config)

    def forward(self, x, is_final_layer=False):
        if is_final_layer:
            # Only process last token in final layer - return smaller tensor
            x_last = x[:, -1, :]  # Shape: (B, n_embd)
            return super().forward(x_last)  # Shape: (B, n_embd)
        else:
            # Process all tokens in non-final layers
            return super().forward(x)  # Shape: (B, T, n_embd)


class RegularBlock(nn.Module):
    """Regular transformer block for non-final layers"""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x  # Shape: (B, T, n_embd)


class FinalBlock(nn.Module):
    """Optimized transformer block for final layer"""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn_final = LastTokenAttentionFinal(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = LastTokenMLP(config)

    def forward(self, x):
        # Optimized attention that only computes last token
        x_last = x[:, -1, :]  # Shape: (B, n_embd)
        x_last = x_last + self.attn_final(self.ln_1(x))
        x_last = x_last + self.mlp(self.ln_2(x), is_final_layer=True)
        return x_last  # Shape: (B, n_embd)


class LastTokenGPT(GPT):
    """
    GPT-style transformer model optimized for inference efficiency
    
    Optimizations:
    - Layers 1 to N-1: Full computation (needed for attention dependencies)
    - Layer N: Only computes last token representations
      * Attention: Only computes query/output for last token
      * MLP: Only processes last token  
    - Final projection: Only on last token
    
    Shape flow:
        Input: (B, T)
        Embeddings: (B, T, n_embd)
        Layers 1 to N-1: (B, T, n_embd) → (B, T, n_embd)
        Layer N: 
          - Attention: (B, T, n_embd) → (B, n_embd)  # Only last token
          - MLP: (B, T, n_embd) → (B, n_embd)        # Only last token
          - Output: (B, n_embd)
        LayerNorm: (B, n_embd) → (B, n_embd)
        LM Head: (B, n_embd) → (B, vocab_size)
        Final output: (B, 1, vocab_size)  # Unsqueezed for consistency
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # Create layers: N-1 regular blocks + 1 final block
        regular_blocks = [Block(config) for _ in range(config.n_layer - 1)]
        final_block = [FinalBlock(config)]
        all_blocks = regular_blocks + final_block
        
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'h': nn.ModuleList(all_blocks),
            'ln_f': nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # Weight tying
        self.apply(self._init_weights)
        self.device = config.device

    def load_from_full_model(self, full_model):
        """
        Load weights from a trained full GPT model into this reduced model.
        
        Args:
            full_model: Trained GPT model to copy weights from
        """
        full_state = full_model.state_dict()
        reduced_state = self.state_dict()
        
        def copy_if_exists(src_key, dst_key):
            """Helper to copy weights if both source and destination exist"""
            if src_key in full_state and dst_key in reduced_state:
                reduced_state[dst_key].copy_(full_state[src_key])
        
        # Copy embeddings and final layers
        shared_keys = [
            'transformer.wte.weight', 'transformer.wpe.weight', 
            'transformer.ln_f.weight', 'transformer.ln_f.bias',
            'lm_head.weight'
        ]
        for key in shared_keys:
            copy_if_exists(key, key)
        
        # Copy transformer blocks
        for layer_idx in range(len(full_model.transformer.h)):
            full_prefix = f'transformer.h.{layer_idx}'
            reduced_prefix = f'transformer.h.{layer_idx}'
            
            # Define all weight types for each component
            layer_norm_weights = ['ln_1.weight', 'ln_1.bias', 
                                  'ln_2.weight', 'ln_2.bias']
            attention_weights = ['c_attn.weight', 'c_attn.bias', 
                                 'c_proj.weight', 'c_proj.bias']
            mlp_weights = ['c_fc.weight', 'c_fc.bias', 
                           'c_proj.weight', 'c_proj.bias']
            
            # Copy layer norms
            for weight_name in layer_norm_weights:
                copy_if_exists(f'{full_prefix}.{weight_name}', 
                             f'{reduced_prefix}.{weight_name}')
            
            # Copy attention weights to appropriate attention module
            is_final_layer = (layer_idx == len(full_model.transformer.h) - 1)
            attn_module = 'attn_final' if is_final_layer else 'attn'
            
            for weight_name in attention_weights:
                full_key = f'{full_prefix}.attn.{weight_name}'
                reduced_key = f'{reduced_prefix}.{attn_module}.{weight_name}'
                copy_if_exists(full_key, reduced_key)
            
            # Copy MLP weights
            for weight_name in mlp_weights:
                copy_if_exists(f'{full_prefix}.mlp.{weight_name}', 
                             f'{reduced_prefix}.mlp.{weight_name}')
        
        self.load_state_dict(reduced_state)

    def calculate_loss(self, logits, targets=None, choice_only=False, by_feature=False):
        """Calculate loss with different options (full, choice-only, or by-feature)"""
        if targets is not None:
            # For LastTokenGPT, we only have logits for the last token
            # So we only compute loss for the last token's target
            targets_last = targets[:, -1:]  # Shape: (B, 1)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                 targets_last.view(-1))
        else:
            loss = None
        return loss

    def forward(self, idx, targets=None, **kwargs):
        """Forward pass through the model"""
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)

        # Process through all blocks (each knows its own behavior)
        for block in self.transformer.h:
            x = block(x)

        # After final layer, x has shape (B, n_embd)
        x_final = self.transformer.ln_f(x)
        logits = self.lm_head(x_final)  # Shape: (B, vocab_size)
        
        # Add sequence dimension back for consistency with training
        logits = logits.unsqueeze(1)  # Shape: (B, 1, vocab_size)
        
        loss = self.calculate_loss(logits, targets, **kwargs)

        return logits, loss


class Block_noLN(nn.Module):
    """Transformer block with attention and MLP"""
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class GPT_noLN(GPT):
    """GPT-style transformer model for sequence prediction"""
    def __init__(self, config):
        super().__init__(config)
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'h': nn.ModuleList([Block_noLN(config) for _ in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # Weight tying
        self.apply(self._init_weights)

    def forward(self, idx, targets=None, **kwargs):
        """Forward pass through the model"""
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)

        for block in self.transformer.h:
            x = block(x)

        logits = self.lm_head(x)
        loss = self.calculate_loss(logits, targets, **kwargs)

        return logits, loss
