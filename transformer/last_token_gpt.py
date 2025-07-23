import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from transformer.transformer import GPT, MLP, Block, CausalSelfAttention

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


# class LastTokenMLP(MLP):
#     """Multi-layer perceptron for transformer block operating only on the last token"""
#     def __init__(self, config):
#         super().__init__(config)

#     def forward(self, x, is_final_layer=False):
#         if is_final_layer:
#             # Only process last token in final layer - return smaller tensor
#             x_last = x[:, -1, :]  # Shape: (B, n_embd)
#             return super().forward(x_last)  # Shape: (B, n_embd)
#         else:
#             # Process all tokens in non-final layers
#             return super().forward(x)  # Shape: (B, T, n_embd)


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
        self.attn = LastTokenAttentionFinal(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.is_last_token_gpt = True

    def forward(self, x):
        # Apply attention to get the last token representation
        x_last = x[:, -1, :]  # Shape: (B, n_embd)
        x_last = x_last + self.attn(self.ln_1(x))
        
        # Apply LayerNorm to the current last token state, not full sequence
        x_last_normed = self.ln_2(x_last.unsqueeze(1)).squeeze(1)
        x_last = x_last + self.mlp(x_last_normed)
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
        self.is_last_token_gpt = (
            hasattr(self.transformer.h[-1], 'is_last_token_gpt') and 
            self.transformer.h[-1].is_last_token_gpt
        )

    def load_from_full_model(self, full_model):
        """
        Load weights from a trained full GPT model into this reduced model.
        
        Args:
            full_model: Trained GPT model to copy weights from
        """
        full_state = full_model.state_dict()
        reduced_state = self.state_dict()
        
        # Copy embeddings and final layers (these are identical)
        shared_keys = [
            'transformer.wte.weight', 'transformer.wpe.weight', 
            'transformer.ln_f.weight', 'transformer.ln_f.bias',
            'lm_head.weight'
        ]
        for key in shared_keys:
            if key in full_state and key in reduced_state:
                reduced_state[key].copy_(full_state[key])
        
        # Copy transformer blocks
        for layer_idx in range(len(full_model.transformer.h)):
            prefix = f'transformer.h.{layer_idx}'
            
            # All weight types for each component
            component_weights = {
                'ln_1': ['weight', 'bias'],
                'ln_2': ['weight', 'bias'], 
                'attn.c_attn': ['weight', 'bias'],
                'attn.c_proj': ['weight', 'bias'],
                'mlp.c_fc': ['weight', 'bias'],
                'mlp.c_proj': ['weight', 'bias']
            }
            
            # Copy all weights - same paths for both models now
            for component, weights in component_weights.items():
                for weight in weights:
                    key = f'{prefix}.{component}.{weight}'
                    if key in full_state and key in reduced_state:
                        reduced_state[key].copy_(full_state[key])
        
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


class LastTokenGPTAdapter:
    """
    Adapter that makes LastTokenGPT compatible with analysis tools.
    
    This adapter wraps a LastTokenGPT model and provides the same interface
    as a regular GPT model, but handles the shape differences internally.
    
    IMPORTANT: For LastTokenGPT's final layer activations:
    - Only the last token position contains actual computed values
    - All previous token positions (0 to T-2) are set to ZERO
    - This represents what LastTokenGPT actually computes
    - Analysis tools should interpret zeros as "not computed" for those positions
    """
    def __init__(self, model, verbose: bool = False):
        self.model = model
        self.config = model.config
        self.device = next(model.parameters()).device
        self.verbose = verbose
        self.is_last_token_gpt = getattr(model, 'is_last_token_gpt', False)
        
        # Store reference to the model's methods for delegation
        self._original_forward = model.forward
        
    def forward(self, *args, **kwargs):
        """Forward pass through the underlying model."""
        return self.model(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        """Make the adapter callable like a PyTorch model."""
        return self.model(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying model."""
        # Don't delegate these special attributes
        if name in ['model', 'config', 'device', 'verbose', 'is_last_token_gpt']:
            return super().__getattribute__(name)
        return getattr(self.model, name)
        
    def normalize_activation_shape(self, activation, layer_name, original_input_shape):
        """
        Normalize activation shapes to be compatible with analysis tools.
        
        Args:
            activation: The captured activation tensor (numpy array)
            layer_name: Name of the layer that produced this activation
            original_input_shape: Shape of the original input (B, T)
            
        Returns:
            Normalized activation with shape (B, T, hidden_dim)
            For LastTokenGPT final layers: positions 0 to T-2 are zeros, 
            position T-1 contains the actual computed activation
        """
        if activation is None:
            return activation
            
        B, T = original_input_shape
        
        if self.verbose:
            print(f"Normalizing {layer_name}: shape {activation.shape}, n_layer={self.config.n_layer}")
            print(f"Final layer would be: _{self.config.n_layer-1}")
            print(f"Is final layer? {layer_name.endswith(f'_{self.config.n_layer-1}')}")
        
        # For LastTokenGPT final layer MLP components, expand 2D to 3D
        # Skip attention components as they already have correct shapes
        if (self.is_last_token_gpt and 
                layer_name.endswith(f'_{self.config.n_layer-1}') and
                not any(attn_comp in layer_name for attn_comp in ['qk', 'attn'])):
            # This is the final layer MLP - activations might be 2D (B, hidden_dim)
            if len(activation.shape) == 2:
                if self.verbose:
                    print(f"Converting final layer {layer_name} from 2D {activation.shape} to 3D")
                # Create 3D tensor with zeros for positions 0 to T-2
                # and actual activation at position T-1 (last token)
                B_act, hidden_dim = activation.shape
                expanded = np.zeros((B_act, T, hidden_dim), dtype=activation.dtype)
                # Put the actual computed activation at the last position
                expanded[:, -1, :] = activation
                activation = expanded
                if self.verbose:
                    print(f"Final shape after expansion: {activation.shape}")
        
        return activation

    def create_shape_normalizing_hook_manager(self, base_hook_manager):
        """
        Create a hook manager that normalizes shapes for LastTokenGPT compatibility.
        
        Args:
            base_hook_manager: The original HookManager instance
            
        Returns:
            Modified hook manager with shape normalization
        """
        
        def custom_make_hook(component_name: str, last_token_only: bool = False):
            def hook(module, input, output):
                if base_hook_manager.verbose:
                    print(f"Hook fired for {component_name}: output shape {output.shape}")
                
                # Guard against unexpected hook firings
                if component_name not in base_hook_manager.activations:
                    if base_hook_manager.verbose:
                        print(f"Warning: Hook fired for {component_name}")
                    return
                
                # Convert to numpy
                output_np = output.detach().cpu().numpy()
                
                # Get input shape for normalization (approximate from output)
                B = output_np.shape[0]
                # For shape normalization, we need to know T
                # We can get this from the input or make a reasonable assumption
                if hasattr(module, 'last_input_shape'):
                    T = module.last_input_shape[1]
                else:
                    # Fallback: assume T from context or use a default
                    T = 6  # This should be passed in somehow
                
                # Normalize shape
                normalized_output = self.normalize_activation_shape(
                    output_np, component_name, (B, T)
                )
                
                if last_token_only:
                    if normalized_output.ndim == 3:
                        base_hook_manager.activations[component_name].append(
                            normalized_output[:, -1, :])
                    else:
                        base_hook_manager.activations[component_name].append(
                            normalized_output)
                else:
                    base_hook_manager.activations[component_name].append(
                        normalized_output)
                    
                if base_hook_manager.verbose:
                    print(f"Stored normalized activation for {component_name}")
            return hook
        
        # Replace the hook creation method
        base_hook_manager._make_hook = custom_make_hook
        return base_hook_manager

    def patch_analyzer_for_compatibility(self, analyzer):
        """
        Patch an analyzer to work with LastTokenGPT by modifying its hook manager.
        
        Args:
            analyzer: The analyzer instance to patch
            
        Returns:
            The patched analyzer
        """
        if not self.is_last_token_gpt:
            return analyzer  # No patching needed for regular models
            
        # Store the original _extract_internal_states method
        original_extract = analyzer._extract_internal_states
        
        def patched_extract_internal_states(sequences, components):
            """Patched version that handles LastTokenGPT shape normalization."""
            if analyzer.verbose:
                print("Using LastTokenGPT-compatible activation extraction")
            
            # Get input shape info
            B, T = len(sequences), len(sequences[0]) if sequences else (0, 0)
            
            # Store original hook creation methods
            original_make_hook = analyzer.hook_manager._make_hook
            original_make_attention_hook = analyzer.hook_manager._make_attention_hook
            
            def shape_normalizing_make_hook(component_name: str, last_token_only: bool = False):
                def hook(module, input, output):
                    if analyzer.hook_manager.verbose:
                        print(f"Hook fired for {component_name}: output shape {output.shape}")
                    
                    # Guard against unexpected hook firings
                    if component_name not in analyzer.hook_manager.activations:
                        if analyzer.hook_manager.verbose:
                            print(f"Warning: Hook fired for {component_name}")
                        return
                    
                    # Convert to numpy and normalize shape
                    output_np = output.detach().cpu().numpy()
                    normalized_output = self.normalize_activation_shape(
                        output_np, component_name, (B, T)
                    )
                    
                    if last_token_only:
                        if normalized_output.ndim == 3:
                            analyzer.hook_manager.activations[component_name].append(
                                normalized_output[:, -1, :])
                        else:
                            analyzer.hook_manager.activations[component_name].append(
                                normalized_output)
                    else:
                        analyzer.hook_manager.activations[component_name].append(
                            normalized_output)
                        
                    if analyzer.hook_manager.verbose:
                        print(f"Stored normalized activation for {component_name}")
                return hook
            
            def last_token_attention_hook(component_name: str, last_token_only: bool = False):
                """Custom attention hook for LastTokenGPT that only computes last token attention."""
                def hook(module, input, output):
                    if analyzer.hook_manager.verbose:
                        print(f"LastToken attention hook fired for {component_name}")
                    
                    # Guard against unexpected hook firings
                    if component_name not in analyzer.hook_manager.activations:
                        if analyzer.hook_manager.verbose:
                            print(f"Warning: Attention hook fired for {component_name}")
                        return
                    
                    x = input[0]
                    B, T, C = x.size()
                    
                    # Check if this is the final layer
                    layer_idx = int(component_name.split('_')[-1])
                    is_final_layer = (layer_idx == self.config.n_layer - 1)
                    
                    if is_final_layer:
                        # Use LastTokenAttentionFinal logic
                        qkv = module.c_attn(x)
                        q, k, v = qkv.split(module.n_embd, dim=2)
                        
                        # Only need query for last token
                        q_last = q[:, -1:, :]  # Shape: (B, 1, n_embd)
                        
                        # Need full k, v since last token can attend to all positions
                        k = k.view(B, T, module.n_head, C // module.n_head).transpose(1, 2)
                        v = v.view(B, T, module.n_head, C // module.n_head).transpose(1, 2)
                        q_last = q_last.view(B, 1, module.n_head, C // module.n_head)
                        q_last = q_last.transpose(1, 2)
                        
                        # Compute attention only for last token
                        qk_attn = torch.matmul(q_last, k.transpose(-2, -1)) / (C ** 0.5)
                        mask = module.bias[:, :, -1:, :T] == 0
                        qk_attn = qk_attn.masked_fill(mask, float('-inf'))
                        qk_attn_softmax = torch.softmax(qk_attn, dim=-1)
                        
                        # Create full-size attention matrix with zeros except last row
                        full_qk_attn = torch.zeros(B, module.n_head, T, T, device=x.device, dtype=x.dtype)
                        full_qk_attn[:, :, -1:, :] = qk_attn
                        
                        full_qk_attn_softmax = torch.zeros(B, module.n_head, T, T, device=x.device, dtype=x.dtype)
                        full_qk_attn_softmax[:, :, -1:, :] = qk_attn_softmax
                        
                        # Compute OV output (only last token has meaningful values)
                        ov_output_last = torch.matmul(qk_attn_softmax, v)
                        full_ov_output = torch.zeros(B, module.n_head, T, C // module.n_head, device=x.device, dtype=x.dtype)
                        full_ov_output[:, :, -1:, :] = ov_output_last
                        
                    else:
                        # Use regular attention computation for non-final layers
                        qkv = module.c_attn(x)
                        q, k, v = qkv.split(module.n_embd, dim=2)
                        q = q.view(B, T, module.n_head, C // module.n_head).transpose(1, 2)
                        k = k.view(B, T, module.n_head, C // module.n_head).transpose(1, 2)
                        v = v.view(B, T, module.n_head, C // module.n_head).transpose(1, 2)
                        
                        full_qk_attn = torch.matmul(q, k.transpose(-2, -1)) / (C ** 0.5)
                        full_qk_attn = full_qk_attn.masked_fill(
                            module.bias[:, :, :T, :T] == 0, float('-inf')
                        )
                        full_qk_attn_softmax = torch.softmax(full_qk_attn, dim=-1)
                        full_ov_output = torch.matmul(full_qk_attn_softmax, v)
                    
                    # Store the appropriate activation based on component type
                    if last_token_only:
                        if component_name.startswith('qk_attn_softmax'):
                            activation = full_qk_attn_softmax[:, :, -1, :].detach().cpu().numpy()
                        elif component_name.startswith('qk_'):
                            activation = full_qk_attn[:, :, -1, :].detach().cpu().numpy()
                        elif component_name.startswith('ov_'):
                            activation = full_ov_output[:, :, -1, :].detach().cpu().numpy()
                        else:
                            activation = output.detach().cpu().numpy()
                    else:
                        if component_name.startswith('qk_attn_softmax'):
                            activation = full_qk_attn_softmax.detach().cpu().numpy()
                        elif component_name.startswith('qk_'):
                            activation = full_qk_attn.detach().cpu().numpy()
                        elif component_name.startswith('ov_'):
                            activation = full_ov_output.detach().cpu().numpy()
                        else:
                            activation = output.detach().cpu().numpy()
                    
                    analyzer.hook_manager.activations[component_name].append(activation)
                    
                    if analyzer.hook_manager.verbose:
                        print(f"Stored LastToken attention activation for {component_name}: shape {activation.shape}")
                
                return hook
            
            # Temporarily patch both hook managers
            analyzer.hook_manager._make_hook = shape_normalizing_make_hook
            analyzer.hook_manager._make_attention_hook = last_token_attention_hook
            
            try:
                # Call the original method with patched hooks
                return original_extract(sequences, components)
            finally:
                # Restore original hook creation methods
                analyzer.hook_manager._make_hook = original_make_hook
                analyzer.hook_manager._make_attention_hook = original_make_attention_hook
        
        # Replace the analyzer's method
        analyzer._extract_internal_states = patched_extract_internal_states
        return analyzer
