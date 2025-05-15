import inspect
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from interp_helpers import tokenize
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


class PerturbationAnalyzer:
    def __init__(self, model: torch.nn.Module, layers: dict[str, str] = None, method: str = 'gradients'):
        self.model = model
        self.vocab = ['R', 'r', 'L', 'l']
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        if method == 'gradients':
            self.layers = layers or self._get_layers()
        else:
            self.layers = ['inputs']

    def _prepare_input(self, sequence):
        token_ids = tokenize(sequence)
        if isinstance(token_ids, torch.Tensor):
            input_tensor = token_ids.clone().detach().unsqueeze(0)
        else:
            input_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        return input_tensor.to(self.model.device)

    def _get_score(self, input_tensor, target_token_idx, as_prob=False):
        with torch.no_grad():
            base_logits, _ = self.model(input_tensor)
            base_probs = F.softmax(base_logits[0, -1], dim=0)
            if as_prob:
                return base_probs[target_token_idx].item()
            else:
                return base_logits[0, -1, target_token_idx].item()

    @contextmanager
    def _register_hooks(self, hooks):
        handles = [module.register_forward_hook(hook) for module, hook in hooks]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def _get_transformer_layers(self):
        return self.model.transformer.h

    def _get_layers(self):
        layers = {
            'token_embedding': 'transformer.wte',
            'position_embedding': 'transformer.wpe',
            'layer0_attn': 'transformer.h.0.attn',
            'layer0_mlp': 'transformer.h.0.mlp',
        }
        return {name: self._get_module_by_path(path) for name, path in layers.items()}

    def _get_module_by_path(self, path):
        """Access a module using a string path"""
        modules = path.split('.')
        current = self.model
        for module in modules:
            if module.isdigit():
                current = current[int(module)]
            else:
                current = getattr(current, module)
        return current

    def _get_attribution_function(self, method):
        """
        Main entry point: choose attribution method.
        """
        if method == "embedding_erasure":
            return self.embedding_erasure_attribution
        elif method == "contrastive":
            return self.contrastive_attribution
        elif method == "layer_perturbation":
            return self.layer_perturbation_analysis
        elif method == "layer_gradient":
            return self.layer_gradient_attribution
        elif method == "integrated_gradients":
            return self.integrated_gradients_attribution
        else:
            raise ValueError(f"Unknown attribution method: {method}")

    def _create_gradient_hook(self, layer_name, layer_activations, layer_gradients):
        """Create a forward hook that captures both activations and gradients."""
        def forward_hook(module, inp, out):
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
            def grad_hook(grad):
                layer_gradients[layer_name].append(grad.detach().clone())
                return grad
            
            out.register_hook(grad_hook)
            return out
        
        return forward_hook

    @contextmanager
    def _capture_gradients(self, layers_dict):
        """Context manager to capture both activations and gradients for specified layers."""
        activations = {name: [] for name in layers_dict}
        gradients = {name: [] for name in layers_dict}

        hooks = [
            (module, self._create_gradient_hook(name, activations, gradients))
            for name, module in layers_dict.items()
        ]

        with self._register_hooks(hooks):
            yield activations, gradients

    def _compute_layer_attribution(self, activation, gradient):
        """Compute attribution from activation and gradient tensors."""
        return (activation * gradient).sum(dim=-1).cpu().numpy().squeeze()

    def _process_layer_attributions(self, layer_activations, layer_gradients):
        """Process captured activations and gradients into attribution scores."""
        attributions = {}

        for layer_name in self.layers:
            if layer_activations[layer_name] and layer_gradients[layer_name]:
                activation = layer_activations[layer_name][0]
                gradient = layer_gradients[layer_name][0]
                attributions[layer_name] = self._compute_layer_attribution(activation, gradient)
            else:
                print(f"Warning: No activations or gradients captured for {layer_name}")
                attributions[layer_name] = None

        # Handle special combination logic
        if 'token_embedding' in attributions and 'position_embedding' in attributions:
            if attributions['token_embedding'] is not None and attributions['position_embedding'] is not None:
                attributions['combined'] = attributions['token_embedding'] + attributions['position_embedding']

        return attributions

    def embedding_erasure_attribution(self, sequence, target_token_idx, as_prob=False):
        """Measure influence of each input token by zeroing both its token and positional embeddings."""
        input_tensor = self._prepare_input(sequence)
        base_score = self._get_score(input_tensor, target_token_idx, as_prob)
        attributions = []

        for i in range(len(sequence)):
            def token_embedding_hook(module, input, output):
                modified_output = output.clone()
                if len(modified_output.shape) == 3:  # [batch, seq_len, embedding_dim]
                    modified_output[0, i, :] = 0.0
                elif len(modified_output.shape) == 2:  # [seq_len, embedding_dim]
                    modified_output[i, :] = 0.0
                return modified_output

            def position_embedding_hook(module, input, output):
                modified_output = output.clone()
                if len(modified_output.shape) == 3:  # [batch, seq_len, embedding_dim]
                    modified_output[0, i, :] = 0.0
                elif len(modified_output.shape) == 2:  # [seq_len, embedding_dim]
                    modified_output[i, :] = 0.0
                return modified_output

            hooks = [
                (self.model.transformer.wte, token_embedding_hook),
                (self.model.transformer.wpe, position_embedding_hook)
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

    def contrastive_attribution(self, sequence, target_token_idx, as_prob=False):
        """Measure each token's importance by comparing against alternative tokens

        For each position, compares the effect of the actual token against the
        average effect of all other possible tokens in the vocabulary."""

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

    def layer_perturbation_analysis(self, sequence, target_token_idx):
        """Analyze layer importance by perturbing intermediate activations."""
        input_tensor = self._prepare_input(sequence)
        layer_activations = {}
        layer_importance = {}

        def capture_activations_hook(layer_name):
            def hook(module, input, output):
                layer_activations[layer_name] = output.detach().clone()
                return output
            return hook

        # Register hooks for each layer to capture activations
        hooks = [
            (block, capture_activations_hook(f"layer_{i}"))
            for i, block in enumerate(self._get_transformer_layers())
        ]
        with self._register_hooks(hooks):
            base_score = self._get_score(input_tensor, target_token_idx)

        # Perturb each layer and measure effect
        for i, block in enumerate(self._get_transformer_layers()):
            layer_name = f"layer_{i}"

            def perturb_hook(module, input, output):
                activation = layer_activations[layer_name]
                noise = torch.randn_like(activation) * activation.std() * 0.1
                return activation + noise

            with self._register_hooks([(block, perturb_hook)]):
                perturbed_score = self._get_score(input_tensor, target_token_idx)
            layer_importance[layer_name] = abs(base_score - perturbed_score)
        return layer_importance

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
        sequence,
        target_token_idx,
        layers_of_interest=None,
        steps=20,
        reference_sequence='RRRRRR',
    ):
        """
        Compute integrated gradients attribution for specified layers.

        Args:
            sequence: Input token sequence
            target_token_idx: Target token index for attribution
            layers_of_interest: Dict mapping layer names to modules (default: token and position embeddings)
            steps: Number of interpolation steps between baseline and input
            reference_sequence: Baseline sequence for integrated gradients

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
            pos_emb = self.model.transformer.wpe(torch.arange(len(input_tensor), device=self.model.device).unsqueeze(0))

        # Initialize storage for integrated gradients
        integrated_grads = {name: None for name in layers_of_interest}
        # Special handling for token embedding gradients
        token_emb_grads = []

        # Capture reference activations for all layers
        reference_activations = {}
        with self._capture_gradients(layers_of_interest) as (ref_acts, _):
            # Run forward pass with reference input
            def ref_embedding_hook(module, input, output):
                return reference_emb

            ref_hook = self.model.transformer.wte.register_forward_hook(ref_embedding_hook)
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

            # Combined embeddings (token + positional)
            combined_emb = interpolated_emb + pos_emb

            # Use existing capture_gradients method for all layers
            with self._capture_gradients(layers_of_interest) as (step_activations, step_gradients):
                # Forward pass through the model
                # First replace the token embeddings with our interpolated version
                def embedding_hook(module, input, output):
                    # Return the interpolated embeddings directly
                    return interpolated_emb
                # Register hook to replace token embeddings
                token_emb_hook = self.model.transformer.wte.register_forward_hook(embedding_hook)

                try:
                    # Run normal forward pass
                    logits, _ = self.model(input_tensor)

                    # Set up target for backpropagation (one-hot for target token)
                    target = torch.zeros_like(logits)
                    target[0, -1, target_token_idx] = 1.0

                    # Backward pass with retain_graph=True to ensure gradients flow
                    logits.backward(target, retain_graph=True)

                    # Store token embedding gradients directly from interpolated_emb
                    if interpolated_emb.grad is not None:
                        token_emb_grads.append(interpolated_emb.grad.clone())
                        # Also store in step_gradients for consistency
                        step_gradients['token_embedding'] = [interpolated_emb.grad.clone()]

                    # Add step gradients to integrated gradients for all layers
                    for name in step_gradients:
                        if step_gradients[name]:
                            grad = step_gradients[name][0]

                            # For intermediate layers, multiply by the difference from reference
                            if name != 'token_embedding':
                                # Get the activation for this layer
                                if step_activations[name] and name in reference_activations:
                                    act = step_activations[name][0]
                                    ref_act = reference_activations[name]
                                    # Multiply gradient by difference from reference
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

        # Calculate integrated gradients for token embeddings
        if token_emb_grads:
            integrated_grads['token_embedding'] = sum(token_emb_grads) / len(token_emb_grads)

        # Calculate attributions
        attributions = {}

        # For token embeddings, multiply by (input - reference)
        if 'token_embedding' in integrated_grads and integrated_grads['token_embedding'] is not None:
            attribution = integrated_grads['token_embedding'] * (input_emb - reference_emb)
            attributions['token_embedding'] = attribution.sum(dim=-1).cpu().numpy().squeeze()

        # For other layers, use the gradients directly (they're already multiplied by activation differences)
        for name in integrated_grads:
            if name != 'token_embedding' and integrated_grads[name] is not None:
                # Sum over all dimensions except sequence length
                attributions[name] = integrated_grads[name].sum(dim=-1).cpu().numpy().squeeze()

        return attributions

    def plot_sequence_attribution(self, ax, sequence, target_token, attribution_value):

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

    def get_attribution_all_targets(self, sequence, method='contrastive', **kwargs):
        attribution_func = self._get_attribution_function(method)

        attributions = {}
        for t in self.vocab:
            target_idx = self.stoi[t]
            attribution = attribution_func(sequence, target_idx, **kwargs)
            if not isinstance(attribution, dict):
                attribution = {self.layers[0]: attribution}
            attributions[t] = attribution
        return attributions

    def plot_attribution_all_targets(self, sequences, method='contrastive', ncols=1, **kwargs):
        if not isinstance(sequences, list):
            sequences = [sequences]

        n_sequences = len(sequences)

        # Get all attributions for all sequences up front.
        attributions = {}
        for sequence in sequences:
            attributions[sequence] = self.get_attribution_all_targets(sequence, method, **kwargs)

        layers = list(attributions[sequences[0]]['R'].keys())
        n_layers = len(layers)

        fig = plt.figure(figsize=(2.1*n_sequences, 0.3*len(self.vocab)*n_layers))
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
                    self.plot_sequence_attribution(ax, sequence, t, attributions[sequence][t][layer])

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

    def plot_attribution_contiguous_sequences(self, sequences, method='contrastive', **kwargs):

        n_sequences = len(sequences)
        n_layers = len(self.layers)
        y = 1 + ((n_layers > 1) * 0.03)
        fig, axs = self.plot_attribution_all_targets(sequences, method, ncols=n_sequences, **kwargs)

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
            line = Line2D([x1 + (0.4*dx), x2 - (0.1*dx)], [y, y],
                        marker='>', markeredgecolor='gray', markevery=[-1],
                        color='gray', linewidth=1.5,  markersize=6,
                        figure=fig,
                        transform=fig.transFigure)
            fig.lines.append(line)

        return fig, axs
