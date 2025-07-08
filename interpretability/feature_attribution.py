"""Feature attribution methods for analyzing transformer model predictions.

This module provides various methods for attributing model predictions to input features
and intermediate layers. It includes implementations of various perturbation and gradient-based
methods for feature attribution:
- Integrated Gradients
- Embedding Erasure
- Contrastive Attribution
- Layer Perturbation Analysis
- Layer Gradient Attribution

The main class `AttributionAnalyzer` provides a unified interface for all attribution methods.
"""

from contextlib import contextmanager
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from analyzer import BaseAnalyzer
from matplotlib.lines import Line2D

sns.set_theme(
    style='ticks',
    font_scale=1.0,
    rc={'axes.labelsize': 11,
        'axes.titlesize': 11,
        'savefig.transparent': True,
        'legend.title_fontsize': 11,
        'legend.fontsize': 10,
        'legend.borderpad': 0.2,
        'figure.titlesize': 11,
        'figure.subplot.wspace': 0.1,
        })


class AttributionAnalyzer(BaseAnalyzer):
    """Analyzer for feature attribution in transformer models.
    
    This class provides methods to analyze how different parts of the input and
    intermediate layers contribute to model predictions. It supports multiple
    attribution methods including integrated gradients, embedding erasure,
    and contrastive attribution.
    
    Args:
        model: The transformer model to analyze
        model_config: Model configuration object
        layers: Optional dictionary mapping layer names to module paths
        method: The default attribution method to use ('gradients' or 'inputs')
        verbose: Whether to print debug information
        config: Optional configuration object
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        model_config: Any,
        layers: Optional[dict[str, str]] = None,
        method: str = 'gradients',
        verbose: bool = False,
        config: Optional[Any] = None
    ) -> None:
        """Initialize the analyzer with a model and optional layer configuration."""
        # Call parent constructor with proper arguments
        super().__init__(model, verbose=verbose, config=config)
        
        if method == 'gradients':
            self.layers = layers or self._get_layers()
        else:
            self.layers = ['inputs']


    def _get_score(
        self,
        input_tensor: torch.Tensor,
        target_token_idx: int,
        as_prob: bool = False
    ) -> float:
        """Get model's prediction score for target token.
        
        Args:
            input_tensor: Input tensor of shape [1, seq_len]
            target_token_idx: Index of target token in vocabulary
            as_prob: Whether to return probability instead of logit
            
        Returns:
            Model's prediction score for target token
        """
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            base_logits, _ = self.model(input_tensor)
            if as_prob:
                base_probs = F.softmax(base_logits[0, -1], dim=0)
                return base_probs[target_token_idx].item()
            return base_logits[0, -1, target_token_idx].item()

    @contextmanager
    def _register_hooks(
        self,
        hooks: list[tuple[torch.nn.Module, callable]]
    ) -> None:
        """Context manager for registering and cleaning up hooks.
        
        Args:
            hooks: List of (module, hook_function) tuples to register
        """
        handles = [
            module.register_forward_hook(hook)
            for module, hook in hooks
        ]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def _get_transformer_layers(self) -> list[torch.nn.Module]:
        """Get list of transformer layers from the model."""
        return self.model.transformer.h

    def _get_layers(self) -> dict[str, torch.nn.Module]:
        """Get default layer configuration for attribution.
        
        This method sets up hooks to capture activations and gradients at key points
        in the transformer's computation flow:
        
        - token_embedding: Raw token embeddings before position encoding
        - position_embedding: Position encodings added to token embeddings
        - layer0_attn_input: Input projection that creates Q,K,V vectors
            Shape: [batch, seq_len, 3*n_embd] -> split into Q,K,V
        - layer0_attn_output: Output projection after attention computation
            Shape: [batch, seq_len, n_embd] -> projects back to embedding dim
        - layer0_ln1: Layer normalization after attention block
            Normalizes activations before MLP
        - layer0_mlp_input: First linear projection in MLP
            Shape: [batch, seq_len, n_embd] -> [batch, seq_len, 4*n_embd]
        - layer0_mlp_output: Final linear projection in MLP
            Shape: [batch, seq_len, 4*n_embd] -> [batch, seq_len, n_embd]
            
        Returns:
            Dictionary mapping layer names to module instances
        """
        layers = {
            'token_embedding': 'transformer.wte',
            'position_embedding': 'transformer.wpe',
            # 'layer0_attn_input': 'transformer.h.0.attn.c_attn',  # Input projection
            # 'layer0_attn_output': 'transformer.h.0.attn.c_proj',  # Output projection
            # 'layer0_ln1': 'transformer.h.0.ln_1',  # Layer norm after attention
            # 'layer0_mlp_input': 'transformer.h.0.mlp.c_fc',  # Input projection
            # 'layer0_mlp_output': 'transformer.h.0.mlp.c_proj',  # Output projection
        }
        return {
            name: self._get_module_by_path(path)
            for name, path in layers.items()
        }

    def _get_module_by_path(self, path: str) -> torch.nn.Module:
        """Access a module using a string path.
        
        Args:
            path: Dot-separated path to module (e.g. 'transformer.h.0.attn')
            
        Returns:
            The specified module
        """
        modules = path.split('.')
        current = self.model
        for module in modules:
            if module.isdigit():
                current = current[int(module)]
            else:
                current = getattr(current, module)
        return current

    def _get_attribution_function(self, method: str) -> callable:
        """Get the appropriate attribution function for the given method.
        
        Args:
            method: Name of attribution method to use
            
        Returns:
            Function implementing the requested attribution method
            
        Raises:
            ValueError: If method is not recognized
        """
        if method == "embedding_erasure":
            return self.embedding_erasure_attribution
        elif method == "contrastive":
            return self.contrastive_attribution
        elif method == "lime":
            return self.lime_attribution_contrastive
        elif method == "layer_gradient":
            return self.layer_gradient_attribution
        elif method == "integrated_gradients":
            return self.integrated_gradients_attribution
        else:
            raise ValueError(f"Unknown attribution method: {method}")

    def _create_gradient_hook(
        self,
        layer_name: str,
        layer_activations: dict[str, list[torch.Tensor]],
        layer_gradients: dict[str, list[torch.Tensor]]
    ) -> callable:
        """Create a forward hook that captures both activations and gradients.
        
        Args:
            layer_name: Name of the layer being hooked
            layer_activations: Dictionary to store captured activations
            layer_gradients: Dictionary to store captured gradients
            
        Returns:
            Forward hook function that captures activations and registers gradient hook
        """
        def forward_hook(
            module: torch.nn.Module,
            inp: tuple[torch.Tensor, ...],
            out: torch.Tensor
        ) -> torch.Tensor:
            # Store activation immediately
            if isinstance(out, tuple):
                act = out[0]
            else:
                act = out
            layer_activations[layer_name].append(act.detach().clone())
            
            # Ensure output requires gradients
            if isinstance(out, tuple):
                out = (out[0].requires_grad_(True),) + out[1:]
            else:
                out = out.requires_grad_(True)
            
            # Register gradient hook
            def grad_hook(grad: torch.Tensor) -> torch.Tensor:
                layer_gradients[layer_name].append(grad.detach().clone())
                return grad
            
            out.register_hook(grad_hook)
            return out
        
        return forward_hook

    @contextmanager
    def _capture_gradients(
        self,
        layers_dict: dict[str, torch.nn.Module]
    ) -> tuple[dict[str, list[torch.Tensor]], dict[str, list[torch.Tensor]]]:
        """Context manager to capture both activations and gradients for specified layers.
        
        Args:
            layers_dict: Dictionary mapping layer names to module instances
            
        Yields:
            Tuple of (activations, gradients) dictionaries
        """
        activations = {name: [] for name in layers_dict}
        gradients = {name: [] for name in layers_dict}

        hooks = [
            (module, self._create_gradient_hook(name, activations, gradients))
            for name, module in layers_dict.items()
        ]

        with self._register_hooks(hooks):
            yield activations, gradients

    def _compute_layer_attribution(
        self,
        activation: torch.Tensor,
        gradient: torch.Tensor,
    ) -> np.ndarray:
        """Compute attribution from activation and gradient tensors.
        
        Args:
            activation: Layer activation tensor
            gradient: Layer gradient tensor
            
        Returns:
            Attribution scores as numpy array
        """
        # Use standard activation * gradient approach for all layers
        return (activation * gradient).sum(dim=-1).cpu().numpy().squeeze()

    def _process_layer_attributions(
        self,
        layer_activations: dict[str, list[torch.Tensor]],
        layer_gradients: dict[str, list[torch.Tensor]]
    ) -> dict[str, Optional[np.ndarray]]:
        """Process captured activations and gradients into attribution scores.
        
        Args:
            layer_activations: Dictionary of captured activations
            layer_gradients: Dictionary of captured gradients
            
        Returns:
            Dictionary mapping layer names to attribution scores
        """
        attributions = {}

        for layer_name in self.layers:
            if layer_activations[layer_name] and layer_gradients[layer_name]:
                activation = layer_activations[layer_name][0]
                gradient = layer_gradients[layer_name][0]
                
                # Debug prints for signs
                # act_sign = torch.sign(activation).float().mean().item()
                # grad_sign = torch.sign(gradient).float().mean().item()
                # print(f"{layer_name}:")
                # print(f"  Activation sign (mean): {act_sign:.3f}")
                # print(f"  Gradient sign (mean): {grad_sign:.3f}")
                
                attributions[layer_name] = self._compute_layer_attribution(
                    activation,
                    gradient
                )
            else:
                print(f"Warning: No activations or gradients captured for {layer_name}")
                attributions[layer_name] = None

        # Handle special combination logic
        if ('token_embedding' in attributions and 'position_embedding' in attributions and
            attributions['token_embedding'] is not None and attributions['position_embedding'] is not None):
            attributions['combined'] = attributions['token_embedding'] + attributions['position_embedding']

        return attributions

    def embedding_erasure_attribution(
        self,
        sequence: str,
        target_token_idx: int,
        as_prob: bool = False
    ) -> np.ndarray:
        """Measure influence of each input token by zeroing its embeddings.
        
        For each position, measures the effect of zeroing both token and
        positional embeddings on the model's prediction.
        
        Args:
            sequence: Input sequence to analyze
            target_token_idx: Index of target token in vocabulary
            as_prob: Whether to use probability instead of logit
            
        Returns:
            Array of attribution scores for each position
        """
        device = next(self.model.parameters()).device
        input_tensor = self._prepare_input(sequence).to(device)
        base_score = self._get_score(input_tensor, target_token_idx, as_prob)
        attributions = []

        for i in range(len(sequence)):
            def create_erasure_hook(position: int):
                def hook(
                    module: torch.nn.Module,
                    input: tuple[torch.Tensor, ...],
                    output: torch.Tensor
                ) -> torch.Tensor:
                    modified_output = output.clone()
                    if len(modified_output.shape) == 3:  # [batch, seq_len, embedding_dim]
                        modified_output[0, position, :] = 0.0
                    elif len(modified_output.shape) == 2:  # [seq_len, embedding_dim]
                        modified_output[position, :] = 0.0
                    return modified_output
                return hook

            # Create hooks for both token and position embeddings
            token_hook = create_erasure_hook(i)
            position_hook = create_erasure_hook(i)

            hooks = [
                (self.model.transformer.wte, token_hook),
                (self.model.transformer.wpe, position_hook)
            ]
            with self._register_hooks(hooks):
                with torch.no_grad():
                    masked_logits, _ = self.model(input_tensor)
                    if as_prob:
                        masked_probs = F.softmax(masked_logits[0, -1], dim=0)
                        masked_score = masked_probs[target_token_idx].item()
                    else:
                        masked_score = masked_logits[0, -1, target_token_idx].item()
            attributions.append(base_score - masked_score)
        return np.array(attributions)

    def contrastive_attribution(
        self,
        sequence: str,
        target_token_idx: int,
        as_prob: bool = False
    ) -> np.ndarray:
        """Measure each token's importance by comparing against alternatives.
        
        For each position, compares the effect of the actual token against the
        average effect of all other possible tokens in the vocabulary.
        
        Args:
            sequence: Input sequence to analyze
            target_token_idx: Index of target token in vocabulary
            as_prob: Whether to use probability instead of logit
            
        Returns:
            Array of attribution scores for each position
        """
        input_tensor = self._prepare_input(sequence)
        base_score = self._get_score(input_tensor, target_token_idx, as_prob)
        attributions = []
        vocab_size = len(self.stoi)

        for i in range(len(sequence)):
            original_token = input_tensor[0, i]
            alt_scores = []
            for alt_token in range(vocab_size):
                if alt_token == original_token:
                    continue
                alt_seq = input_tensor.clone()
                alt_seq[0, i] = alt_token
                alt_score = self._get_score(alt_seq, target_token_idx, as_prob)
                alt_scores.append(alt_score)
            avg_alt_score = sum(alt_scores) / len(alt_scores)
            attributions.append(base_score - avg_alt_score)
        return np.array(attributions)

    def lime_attribution_contrastive(
        self,
        sequence: str,
        target_token_idx: int,
        n_samples: int = 200
    ) -> np.ndarray:
        """LIME attribution using contrastive perturbation strategy."""
        from sklearn.linear_model import Ridge

        token_ids = self._prepare_input(sequence).squeeze(0)
        seq_len = len(token_ids)
        vocab_size = len(self.stoi)
        
        # Initialize storage
        perturbed_data = np.zeros((n_samples, seq_len))
        predictions = []
        
        for sample_idx in range(n_samples):
            # Generate perturbation mask and alternative tokens in one step
            perturb_mask = np.random.binomial(1, 0.5, size=seq_len)
            perturbed_seq = token_ids.clone()
            
            # For positions to perturb, replace with random alternative
            mask_positions = np.where(perturb_mask == 1)[0]
            for pos in mask_positions:
                # Get all tokens except current one
                alt_tokens = [t for t in range(vocab_size) if t != token_ids[pos].item()]
                perturbed_seq[pos] = np.random.choice(alt_tokens)
            
            # Record perturbations and get prediction
            perturbed_data[sample_idx] = perturb_mask
            with torch.no_grad():
                predictions.append(
                    self._get_score(perturbed_seq.unsqueeze(0), target_token_idx, as_prob=True)
                )

        # Fit interpretable model
        model_lime = Ridge(alpha=1.0)
        model_lime.fit(perturbed_data, predictions)
        
        return -model_lime.coef_

    def attribution_umap(
        self,
        sequences: list[str],
        method: str = 'contrastive',
        n_components: int = 2,
        umap_kwargs: dict = None,
        **attribution_kwargs
    ):
        """
        Run UMAP on attribution matrices where each sequence becomes a single vector.
        For each sequence, concatenate all token-wise attributions into one long vector.
        
        Returns:
            embedding: (n_sequences, n_components) - UMAP embedding of sequences
            matrix: (n_sequences, context_length * vocab_size) - concatenated attributions
            sequences: list of input sequences for reference
        """
        import umap
        
        func = self._get_attribution_function(method)
        sequence_vectors = []
        
        for seq in sequences:
            # Get attribution matrix for this sequence: (vocab_size, context_length)
            attributions = []
            for target_token_idx in range(len(self.vocab)):  # Loop over vocab tokens
                attribution_scores = func(seq, target_token_idx, **attribution_kwargs)
                attributions.append(attribution_scores)  # Each is (seq_len,)
            
            # Convert to numpy array and flatten: (vocab_size * context_length,)
            attribution_matrix = np.array(attributions)  # (vocab_size, seq_len)
            seq_vector = attribution_matrix.flatten()
            sequence_vectors.append(seq_vector)
        
        # Stack all sequence vectors: (n_sequences, context_length * vocab_size)
        big_matrix = np.stack(sequence_vectors, axis=0)
        # Run UMAP on the sequence vectors
        umap_kwargs = umap_kwargs or {}
        # Add default parameters to handle sparse/problematic matrices
        umap_kwargs.setdefault('n_neighbors', min(15, len(sequences) - 1))
        umap_kwargs.setdefault('min_dist', 0.1)
        umap_kwargs.setdefault('metric', 'euclidean')
        
        print(f"Matrix stats - Shape: {big_matrix.shape}, Non-zero: {np.count_nonzero(big_matrix)}, Sparsity: {1 - np.count_nonzero(big_matrix)/big_matrix.size:.3f}")
        
        reducer = umap.UMAP(n_components=n_components, **umap_kwargs)
        embedding = reducer.fit_transform(big_matrix)  # (n_sequences, n_components)
        
        return embedding, big_matrix

    # def layer_perturbation_analysis(
    #     self,
    #     sequence: str,
    #     target_token_idx: int
    # ) -> dict[str, float]:
    #     """Analyze layer importance by perturbing intermediate activations.
        
    #     For each layer, measures the effect of adding noise to its activations
    #     on the model's prediction.
        
    #     Args:
    #         sequence: Input sequence to analyze
    #         target_token_idx: Index of target token in vocabulary
            
    #     Returns:
    #         Dictionary mapping layer names to importance scores
    #     """
    #     input_tensor = self._prepare_input(sequence)
    #     layer_activations = {}
    #     layer_importance = {}

    #     def capture_activations_hook(layer_name: str) -> callable:
    #         def hook(
    #             module: torch.nn.Module,
    #             input: tuple[torch.Tensor, ...],
    #             output: torch.Tensor
    #         ) -> torch.Tensor:
    #             layer_activations[layer_name] = output.detach().clone()
    #             return output
    #         return hook

    #     # Register hooks for each layer to capture activations
    #     hooks = [
    #         (block, capture_activations_hook(f"layer_{i}"))
    #         for i, block in enumerate(self._get_transformer_layers())
    #     ]
    #     with self._register_hooks(hooks):
    #         base_score = self._get_score(input_tensor, target_token_idx)

    #     # Perturb each layer and measure effect
    #     for i, block in enumerate(self._get_transformer_layers()):
    #         layer_name = f"layer_{i}"

    #         def perturb_hook(
    #             module: torch.nn.Module,
    #             input: tuple[torch.Tensor, ...],
    #             output: torch.Tensor
    #         ) -> torch.Tensor:
    #             activation = layer_activations[layer_name]
    #             noise = torch.randn_like(activation) * activation.std() * 0.1
    #             return activation + noise

    #         with self._register_hooks([(block, perturb_hook)]):
    #             perturbed_score = self._get_score(input_tensor, target_token_idx)
    #         layer_importance[layer_name] = abs(base_score - perturbed_score)
    #     return layer_importance

    def layer_gradient_attribution(self, sequence, target_token_idx):
        """Compute gradient-based attributions for specified layers."""

        input_tensor = self._prepare_input(sequence)

        with self._capture_gradients(self.layers) as (activations, gradients):
            # Forward and backward pass
            logits, _ = self.model(input_tensor, targets=None)
            target = torch.zeros_like(logits)
            target[0, -1, target_token_idx] = 1.0
            self.model.zero_grad()
            logits.backward(target, retain_graph=True)
            attributions = self._process_layer_attributions(activations, gradients)
            return attributions

    def integrated_gradients_attribution(
        self,
        sequence: str,
        target_token_idx: int,
        layers_of_interest: Optional[dict[str, torch.nn.Module]] = None,
        steps: int = 20,
        reference_sequence: str = 'RRRRRR'
    ) -> dict[str, np.ndarray]:
        """Compute integrated gradients attribution for specified layers.
        
        Implements the integrated gradients method by interpolating between
        reference and input embeddings and accumulating gradients.
        
        Args:
            sequence: Input sequence to analyze
            target_token_idx: Index of target token in vocabulary
            layers_of_interest: Dict mapping layer names to modules
            steps: Number of interpolation steps
            reference_sequence: Baseline sequence for interpolation
            
        Returns:
            Dictionary mapping layer names to attribution scores
        """
        if layers_of_interest is None:
            layers_of_interest = self.layers

        # Prepare input and baseline tensors
        input_tensor = self._prepare_input(sequence)
        baseline = self._prepare_input(reference_sequence)

        # Store original training mode and set to train for gradient tracking
        original_mode = self.model.training
        self.model.train()

        # Calculate baseline and input embeddings
        with torch.no_grad():
            input_emb = self.model.transformer.wte(input_tensor)
            reference_emb = self.model.transformer.wte(baseline)
            pos_emb = self.model.transformer.wpe(
                torch.arange(
                    len(input_tensor),
                    device=self.model.device
                ).unsqueeze(0)
            )

        # Initialize storage for integrated gradients
        integrated_grads = {name: None for name in layers_of_interest}
        
        # Capture reference activations for all layers
        reference_activations = {}
        with self._capture_gradients(layers_of_interest) as (ref_acts, _):
            # Run forward pass with reference input
            def ref_embedding_hook(
                module: torch.nn.Module,
                input: tuple[torch.Tensor, ...],
                output: torch.Tensor
            ) -> torch.Tensor:
                return reference_emb
            
            ref_hook = self.model.transformer.wte.register_forward_hook(
                ref_embedding_hook
            )
            try:
                self.model(baseline)
                # Store reference activations
                for name in ref_acts:
                    if ref_acts[name]:
                        reference_activations[name] = ref_acts[name][0].detach().clone()
            finally:
                ref_hook.remove()

        # For each interpolation step
        for step in range(steps):
            self.model.zero_grad()

            # Calculate alpha for this step
            alpha = step / (steps - 1)

            # Create interpolated embeddings
            interpolated_emb = reference_emb + alpha * (input_emb - reference_emb)
            interpolated_emb.requires_grad_(True)
            
            # Use existing capture_gradients method for all layers
            with self._capture_gradients(layers_of_interest) as (
                step_activations,
                step_gradients
            ):
                # Forward pass through the model replacing token embeddings with interpolated version
                def embedding_hook(
                    module: torch.nn.Module,
                    input: tuple[torch.Tensor, ...],
                    output: torch.Tensor
                ) -> torch.Tensor:
                    return interpolated_emb
                
                # Register hook to replace token embeddings
                token_emb_hook = self.model.transformer.wte.register_forward_hook(
                    embedding_hook
                )
                
                try:
                    # Run normal forward pass
                    logits, _ = self.model(input_tensor)

                    # Set up target for backpropagation (one-hot for target token)
                    target = torch.zeros_like(logits)
                    target[0, -1, target_token_idx] = 1.0
                    
                    # Backward pass with retain_graph=True
                    logits.backward(target, retain_graph=True)

                    # Store token embedding gradients directly from interpolated_emb
                    if interpolated_emb.grad is not None:
                        step_gradients['token_embedding'] = [interpolated_emb.grad.clone()]
                    
                    # Add step gradients to integrated gradients
                    for name in step_gradients:
                        if step_gradients[name]:
                            grad = step_gradients[name][0]
                            
                            # For intermediate layers, multiply by difference
                            if name != 'token_embedding':
                                if (step_activations[name] and
                                    name in reference_activations):
                                    act = step_activations[name][0]
                                    ref_act = reference_activations[name]
                                    grad = grad * (act - ref_act)
                            
                            if integrated_grads[name] is None:
                                integrated_grads[name] = (grad / steps).clone()
                            else:
                                integrated_grads[name] += (grad / steps).clone()
                        else:
                            print(f"Warning: No gradients captured for layer {name} at step {step}")

                finally:
                    # Clean up the token embedding hook
                    token_emb_hook.remove()

                    # Clear any remaining gradients
                    self.model.zero_grad()
                    if interpolated_emb.grad is not None:
                        interpolated_emb.grad.zero_()

        # Restore model's original training mode
        self.model.train(original_mode)

        # Calculate attributions
        attributions = {}
        for name in integrated_grads:
            if integrated_grads[name] is not None:
                if name == 'token_embedding':
                    # Special handling for token embeddings
                    attribution = integrated_grads[name] * (input_emb - reference_emb)
                else:
                    attribution = integrated_grads[name]
                attributions[name] = attribution.sum(dim=-1).cpu().numpy().squeeze()

        return attributions

    def get_activations(
        self,
        sequences: list[str],
        layer: str | None = None
    ) -> dict[str, np.ndarray] | dict[str, dict[str, np.ndarray]]:
        """Get activations for a list of sequences.
        
        Args:
            sequences: List of sequences to analyze
            layer: Optional layer to get activations from. If None, returns
                activations for all layers.
            
        Returns:
            If layer is specified: Dictionary mapping sequences to their activation
                vectors for that layer
            If layer is None: Dictionary mapping sequences to dictionaries of
                layer activations
        """
        # For attribution analysis, we typically want specific layers
        # This is a simplified implementation - you may want to expand this
        if layer is None:
            layer = 'token_embedding'
        
        # Get activations using the parent class method if available
        # For now, return empty dict as this is primarily for attribution
        return {}

    def plot_sequence_attribution(
        self,
        ax: plt.Axes,
        sequence: str,
        target_token: str,
        attribution_value: np.ndarray
    ) -> None:
        """Plot attribution scores for a sequence.
        
        Args:
            ax: Matplotlib axes to plot on
            sequence: Input sequence
            target_token: Target token being predicted
            attribution_value: Attribution scores for each position
        """
        hm = sns.heatmap(
            attribution_value.reshape(1, -1),
            ax=ax,
            vmin=-1,
            vmax=1,
            cmap="RdBu",
            annot=False,
            cbar=False
        )
        for i, char in enumerate(sequence):
            ax.text(
                i + 0.5, 0.5, char,
                ha='center', va='center',
                fontsize=8, fontweight='bold', color='k'
            )
        ax.yaxis.tick_right()
        ax.set_yticks([0.5], labels=[target_token], rotation=0)
        ax.set_xticks([])

    def get_attribution_all_targets(
        self,
        sequence: str,
        method: str = 'contrastive',
        **kwargs
    ) -> dict[str, dict[str, np.ndarray]]:
        """Get attribution scores for all possible target tokens.
        
        Args:
            sequence: Input sequence to analyze
            method: Attribution method to use
            **kwargs: Additional arguments for attribution method
            
        Returns:
            Dictionary mapping target tokens to attribution scores
        """
        attribution_func = self._get_attribution_function(method)

        attributions = {}
        for t in self.vocab:
            target_idx = self.stoi[t]
            attribution = attribution_func(sequence, target_idx, **kwargs)
            if not isinstance(attribution, dict):
                attribution = {self.layers[0]: attribution}
            attributions[t] = attribution
        return attributions

    def plot_attribution_all_targets(
        self,
        sequences: Union[str, list[str]],
        method: str = 'contrastive',
        ncols: int = 1,
        **kwargs
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Plot attribution scores for all targets and sequences.
        
        Args:
            sequences: Single sequence or list of sequences to analyze
            method: Attribution method to use
            ncols: Number of columns in plot
            **kwargs: Additional arguments for attribution method
            
        Returns:
            Tuple of (figure, axes) for the plot
        """
        if not isinstance(sequences, list):
            sequences = [sequences]

        n_sequences = len(sequences)

        # Get all attributions for all sequences up front
        attributions = {}
        for sequence in sequences:
            attributions[sequence] = self.get_attribution_all_targets(
                sequence,
                method,
                **kwargs
            )

        layers = list(attributions[sequences[0]]['R'].keys())
        n_layers = len(layers)

        fig = plt.figure(
            figsize=(2.1*n_sequences, 0.3*len(self.vocab)*n_layers)
        )
        subfigs = fig.subfigures(nrows=n_layers, hspace=0.04)

        if n_layers == 1:
            subfigs = [subfigs]

        for layer, subfig in zip(layers, subfigs):
            axs = subfig.subplots(
                nrows=len(self.vocab),
                ncols=n_sequences,
                sharex=True,
                gridspec_kw={'wspace': 0.3}
            )
            if n_sequences == 1:
                axs = [axs]
            else:
                axs = axs.T

            for ax0, sequence in zip(axs, sequences):
                for t, ax in zip(self.vocab, ax0):
                    self.plot_sequence_attribution(
                        ax,
                        sequence,
                        t,
                        attributions[sequence][t][layer]
                    )

                if layer == layers[0]:
                    ax0[0].set_title(sequence, y=1.1 + ((n_layers > 1)/2))

            if n_layers > 1:
                subfig.suptitle(layer, y=1.01)

            cbar_width = 0.03 / ncols
            cbar_x = 1.033 - (0.0129 * ncols)
            cbar_ax = fig.add_axes([cbar_x, 0.15, cbar_width, 0.6])
            norm = plt.Normalize(vmin=-1, vmax=1)
            sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
            sm.set_array([])
            fig.colorbar(sm, cax=cbar_ax)

        return fig, axs

    def plot_attribution_contiguous_sequences(
        self,
        sequences: list[str],
        method: str = 'contrastive',
        **kwargs
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Plot attribution scores for sequences with arrows between them.
        
        Args:
            sequences: List of sequences to analyze
            method: Attribution method to use
            **kwargs: Additional arguments for attribution method
            
        Returns:
            Tuple of (figure, axes) for the plot
        """
        n_sequences = len(sequences)
        n_layers = len(self.layers)
        y = 1 + ((n_layers > 1) * 0.03)
        fig, axs = self.plot_attribution_all_targets(
            sequences,
            method,
            ncols=n_sequences,
            **kwargs
        )

        if n_sequences == 1:
            return fig, axs

        for i in range(n_sequences - 1):
            ax1 = axs.T[0][i]
            ax2 = axs.T[0][i+1]
            fig.canvas.draw()
            x1 = ax1.get_position().x1
            x2 = ax2.get_position().x0
            dx = x2 - x1

            # Create a line with an arrow marker
            line = Line2D(
                [x1 + (0.4*dx), x2 - (0.1*dx)],
                [y, y],
                marker='>',
                markeredgecolor='gray',
                markevery=[-1],
                color='gray',
                linewidth=1.5,
                markersize=6,
                figure=fig,
                transform=fig.transFigure
            )
            fig.lines.append(line)

        return fig, axs
