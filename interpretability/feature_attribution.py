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
        else:
            raise ValueError(f"Unknown attribution method: {method}")

    def _create_gradient_hook(self, layer_name, layer_activations, layer_gradients):
        """Create a forward hook that captures both activations and gradients."""
        def forward_hook(module, inp, out):
            # Ensure output requires gradients
            out.requires_grad_(True)

            # Store activation immediately
            layer_activations[layer_name].append(out.detach().clone())

            # Register gradient hook on this output
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

        n_layers = len(self.layers)
        n_sequences = len(sequences)

        # Get all attributions for all sequences up front.
        attributions = {}
        for sequence in sequences:
            attributions[sequence] = self.get_attribution_all_targets(sequence, method, **kwargs)
            
        # if n_layers > 1:
        # For multi-layer (gradient methods): layers as rows, sequences as columns

        fig = plt.figure(figsize=(2.1*n_sequences, 0.3*len(self.vocab)*n_layers))
        subfigs = fig.subfigures(nrows=n_layers, hspace=0.04)

        if n_layers == 1:
            subfigs = [subfigs]

        for layer, subfig in zip(self.layers, subfigs):
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

                if layer == list(self.layers)[0]:
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
