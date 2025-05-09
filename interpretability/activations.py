"""Module for analyzing model activations across different layers."""

from dataclasses import dataclass
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from interpretability.interp_helpers import (embed_sequence, predict_token,
                                             tokenize)
from transformer.transformer import MLP


@dataclass
class DimensionalityReductionConfig:
    """Configuration for activation analysis.

    Attributes:
        token_pos: Position in sequence to analyze
        sequence_method: Method for sequence embedding
        n_components: Number of components for dimensionality reduction
        method: Embedding method ('pca' or 'tsne')
    """
    token_pos: int = -1
    sequence_method: str = 'token'
    n_components: int = 4
    method: str = 'pca'


class ActivationAnalyzer:
    """Class for analyzing model activations across different layers.

    This class provides methods to capture, analyze and visualize activations
    from different components of the transformer model.

    Attributes:
        model (torch.nn.Module): The transformer model to analyze
        _hooks (list[torch.utils.hooks.RemovableHandle]): Storage for hooks
    """

    def __init__(self, model: torch.nn.Module):
        """Initialize the activation analyzer.

        Args:
            model: The transformer model to analyze
        """
        self.model = model
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

    def _get_layer_composition(self, config: Any) -> dict[str, int]:
        """Get the dimensions of each layer in the model component.

        Args:
            config: Model configuration object

        Returns:
            Dictionary mapping layer names to their dimensions
        """
        pass

    def _setup_hooks(self) -> dict[str, list[np.ndarray]]:
        """Set up hooks to capture activations from model layers.

        Returns:
            Dictionary to store captured activations
        """
        activations = {layer: [] for layer in self.layers}

        def make_hook(layer_name: str):
            def hook(
                module: torch.nn.Module,
                input: tuple[torch.Tensor, ...],
                output: torch.Tensor
            ) -> None:
                if layer_name == 'input':
                    activations[layer_name].append(
                        input[0].detach().cpu().numpy()
                    )
                else:
                    activations[layer_name].append(
                        output.detach().cpu().numpy()
                    )
            return hook

        # Register hooks only for layers we want to track
        for block in self.model.transformer.h:
            if self.model_component == 'mlp':
                if 'input' in self.layers:
                    self._hooks.append(
                        block.mlp.c_fc.register_forward_hook(
                            make_hook('input')
                        )
                    )
                if 'gelu' in self.layers:
                    self._hooks.append(
                        block.mlp.gelu.register_forward_hook(
                            make_hook('gelu')
                        )
                    )
                if 'output' in self.layers:
                    self._hooks.append(
                        block.mlp.c_proj.register_forward_hook(
                            make_hook('output')
                        )
                    )

        return activations

    def get_activations(
        self,
        input_seq: list[str]
    ) -> dict[str, dict[str, np.ndarray]]:
        """Capture activations for a list of input sequences.

        Args:
            input_seq: List of input sequences to analyze

        Returns:
            Dictionary mapping sequences to their layer activations
        """
        tokens = tokenize(input_seq)
        activations = self._setup_hooks()

        # Store original training state
        was_training = self.model.training

        try:
            # Set to eval mode and disable gradients
            self.model.eval()
            with torch.no_grad():
                self.model(tokens)
        finally:
            # Restore original training state
            if was_training:
                self.model.train()

            # Clean up hooks
            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()

        # Reformat activations for easier interpretation
        captured_activations = {seq: {} for seq in input_seq}
        for layer, acts in activations.items():
            for seq, act in zip(input_seq, acts[0]):
                captured_activations[seq][layer] = act

        return captured_activations

    def get_activation_by_position(
        self,
        activations: dict[str, dict[str, np.ndarray]],
        token_pos: int = -1
    ) -> dict[str, dict[str, np.ndarray]]:
        """Extract activations for a specific token position.

        Args:
            activations: Dictionary of captured activations
            token_pos: Position in sequence to analyze (-1 for last token)

        Returns:
            Dictionary of activations at specified position
        """
        return {
            layer_name: {
                seq: act[layer_name][token_pos]
                for seq, act in activations.items()
            }
            for layer_name in self.layers
        }

    def find_maximal_activations(
        self,
        activations: dict[str, dict[str, np.ndarray]],
        layer_name: str,
        sequences: list[str],
        top_n: int = 30
    ) -> dict[int, list[tuple[str, float]]]:
        """Find sequences that maximally activate each neuron.

        Args:
            activations: Dictionary of captured activations
            layer_name: Layer to analyze
            sequences: List of sequences to consider
            top_n: Number of top activating sequences to return

        Returns:
            Dictionary mapping neuron indices to their top activating sequences
        """
        layer_activations = activations[layer_name]
        num_neurons = self.layer_composition[layer_name]

        max_activating_seqs = {}
        for neuron_idx in range(num_neurons):
            # Get activation of this neuron for each sequence
            neuron_acts = {
                seq: layer_activations[seq][neuron_idx]
                for seq in sequences
            }

            # Sort sequences by activation (highest first)
            sorted_seqs = sorted(
                neuron_acts.items(),
                key=lambda x: x[1],
                reverse=True
            )

            max_activating_seqs[neuron_idx] = sorted_seqs[:top_n]

        return max_activating_seqs

    def calculate_selectivity(
        self,
        activations: dict[str, np.ndarray],
        target_token: str | list[str],
        reference_tokens: str | list[str] | None = None,
        token_pos: int = -1,
        method: str = 'diff'
    ) -> np.ndarray:
        """Calculate neuron selectivity for target vs reference tokens.

        Args:
            activations: Dictionary mapping sequences to activation vectors
            target_token: Token(s) to calculate selectivity for
            reference_tokens: Token(s) to compare against. If None, uses all
                tokens from complement set of target tokens.
            token_pos: Position in sequence to analyze. Default is last token.
            method: Method for calculating selectivity:
                - 'diff': Mean difference in activation between target and
                    reference
                - 'raw': Mean activation for target token only
                - 'pchange': Percent change from reference to target activation

        Returns:
            Array of selectivity values for each neuron
        """
        tokens = {'R', 'r', 'L', 'l'}
        sequences = list(activations.keys())
        nneurons = len(activations[sequences[0]])

        if not isinstance(target_token, set):
            target_token = set(target_token)

        if reference_tokens is None:
            reference_tokens = list(tokens - target_token)

        target_acts = [
            activations[seq] for seq in sequences
            if seq[token_pos] in target_token
        ]
        reference_acts = [
            activations[seq] for seq in sequences
            if seq[token_pos] in reference_tokens
        ]

        if not target_acts:
            return np.zeros(nneurons)

        target_mean = np.mean(target_acts, axis=0)
        ref_mean = np.mean(reference_acts, axis=0)

        # Calculate selectivity based on method
        if method == 'diff':
            return target_mean - ref_mean
        elif method == 'raw':
            return target_mean
        elif method == 'pchange':
            return (target_mean - ref_mean) / (ref_mean + target_mean)
        else:
            raise ValueError(f"Unknown selectivity method: {method}")

    def calculate_selectivity_by_layer(
        self,
        all_activations: dict[str, dict[str, np.ndarray]],
        layers: list[str] | None = None,
        token_pos: int = -1,
        method: str = 'diff'
    ) -> dict[str, dict[str, np.ndarray]]:
        """Calculate selectivity metrics for each layer at a specific position.

        Args:
            all_activations: Dictionary mapping sequences to activation values
                for each layer
            layers: Layers to analyze. If None, uses all layers.
            token_pos: Position in sequence to analyze. Default is last token.
            method: Method for calculating selectivity. Default is 'diff'.

        Returns:
            Dictionary mapping layers and token positions to selectivity
            values for each token.
        """
        if layers is None:
            layers = self.layers
        elif not isinstance(layers, list):
            layers = [layers]

        # Determine token pairs based on method
        if method.startswith('choice'):
            tokens = [('R', 'r'), ('L', 'l')]
            method = method.replace('choice', '')
        elif method.startswith('reward'):
            tokens = [('R', 'L'), ('r', 'l')]
            method = method.replace('reward', '')
        else:
            tokens = ['R', 'r', 'L', 'l']

        # Get activations for specified position
        activations = self.get_activation_by_position(
            all_activations,
            token_pos
        )

        # Calculate selectivity for each layer and token
        selectivity_by_layer = {}
        for layer in layers:
            layer_activations = activations[layer]
            selectivity_by_layer[layer] = {}

            for token in tokens:
                sel = self.calculate_selectivity(
                    layer_activations,
                    token,
                    token_pos=token_pos,
                    method=method
                )
                selectivity_by_layer[layer][token] = sel

        return selectivity_by_layer

    def analyze_layer_specialization(
        self,
        all_activations: dict[str, dict[str, np.ndarray]],
        layers: list[str] | None = None,
        token_pos: int | list[int] = -1,
        plot_bars: bool = False,
        **kwargs
    ) -> dict[str, dict[str, np.ndarray]]:
        """Analyze and visualize neuron specialization across layers.

        Args:
            all_activations: Dictionary of captured activations
            layers: Layers to analyze
            token_pos: Position(s) in sequence to analyze
            plot_bars: Whether to plot activation bars
            **kwargs: Additional arguments for selectivity calculation

        Returns:
            Dictionary of selectivity values by layer and token
        """
        if layers is None:
            layers = self.layers
        elif not isinstance(layers, list):
            layers = [layers]

        if not isinstance(token_pos, (list, np.ndarray)):
            token_pos = [token_pos]

        fig = plt.figure(figsize=(3*len(layers)+1, 1*len(token_pos)+1.5))
        subfigs = fig.subfigures(nrows=len(token_pos), wspace=0.01)
        subfigs = [subfigs] if len(token_pos) == 1 else subfigs

        # Analyze each position
        for i, (subfig, pos) in enumerate(zip(subfigs, token_pos)):

            selectivity = self.calculate_selectivity_by_layer(
                all_activations,
                layers,
                token_pos=pos,
                **kwargs
            )

            axs = subfig.subplots(ncols=len(layers))
            for j, (ax, layer) in enumerate(zip(axs, layers)):
                if plot_bars:
                    self.plot_activation_bars(selectivity[layer], layer)

                self.plot_activation_heatmap(
                    selectivity[layer],
                    layer,
                    ax=ax
                )

                # Adjust subplot labels
                if len(token_pos) > 1:
                    if i == len(token_pos) - 1:
                        ax.set(title='', xlabel='Neuron Index')
                    if i == 0:
                        ax.set(
                            title=f'{layer.capitalize()} Layer',
                            xticks=[],
                            xlabel=''
                        )
                    else:
                        ax.set(xlabel='', xticks=[], title='')
                    if j == 0:
                        ax.set(ylabel='Token')
                    else:
                        ax.set(yticks=[], ylabel='')

            subfig.suptitle(f'pos: {pos}', y=0.95, x=0.05, ha='left')

        fig.suptitle(
            f'Neuron selectivity by token position\n{kwargs.get("method")}',
            y=0.95
        )
        fig.tight_layout()

        return selectivity

    def plot_activation_bars(
        self,
        activations: dict[str, np.ndarray],
        layer: str
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot activation bars for each token.

        Args:
            activations: Dictionary mapping tokens to activation vectors
            layer: Layer name for plot title

        Returns:
            Figure and axes objects
        """
        fig, axs = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(15, 10),
            sharex=True,
            sharey=True
        )

        for (token, diff), ax in zip(activations.items(), axs.flatten()):
            bars = ax.bar(range(len(diff)), diff)

            for bar in bars:
                if bar.get_height() > 0:
                    bar.set_color('green')
                else:
                    bar.set_color('red')

            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set(
                title=f'{token} Tokens',
                xlabel='Neuron Index',
                ylabel='Activation Difference'
            )

        fig.suptitle(f'{layer.capitalize()} Layer', fontsize=14)
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        return fig, axs

    def plot_activation_heatmap(
        self,
        activations: dict[str, np.ndarray],
        layer: str,
        ax: plt.Axes | None = None,
        **kwargs
    ) -> plt.Axes:
        """Plot activation heatmap.

        Args:
            activations: Dictionary mapping tokens to activation vectors
            layer: Layer name for plot title
            ax: Optional axes to plot on
            **kwargs: Additional plotting arguments

        Returns:
            Axes object with heatmap
        """
        cmap_by_layer = {
            'input': (-3, 3),
            'gelu': (-1, 1),
            'output': (-1, 1)
        }

        diff_data = np.vstack(list(activations.values()))

        ax = sns.heatmap(
            diff_data,
            cmap="RdBu_r",
            center=0,
            annot=False,
            fmt=".2f",
            xticklabels=range(diff_data.shape[1]),
            yticklabels=activations.keys(),
            ax=ax,
            # vmin=cmap_by_layer[layer][0],
            # vmax=cmap_by_layer[layer][1],
            **kwargs
        )

        ax.set(
            title=f'{layer.capitalize()} Layer',
            xlabel='Neuron Index',
            ylabel='Token'
        )

        return ax

    def prepare_pca_embeddings(
        self,
        activations: dict[str, dict[str, np.ndarray]],
        sequences: list[str],
        layer: str,
        config: DimensionalityReductionConfig
    ) -> tuple[Any, np.ndarray]:
        """Prepare PCA embeddings for visualization.
        
        Args:
            activations: Dictionary of captured activations
            sequences: List of sequences to analyze
            layer: Layer to analyze
            config: Analysis configuration
            
        Returns:
            Tuple of (model, embeddings)
        """
        if config.sequence_method == 'token':
            pos_acts = self.get_activation_by_position(
                activations,
                token_pos=config.token_pos
            )
            X = np.array([pos_acts[layer][seq] for seq in sequences])
        elif config.sequence_method == 'concat':
            X = np.concatenate([
                activations[seq][layer].reshape(1, -1)
                for seq in sequences
            ], axis=0)

        if config.method == 'tsne':
            model = TSNE(n_components=2)
            X_embedding = model.fit_transform(X)
        elif config.method == 'pca':
            model = PCA(n_components=config.n_components)
            X_embedding = model.fit_transform(X)

        return model, X_embedding

    def plot_pca_by_layer(
        self,
        activations: dict[str, dict[str, np.ndarray]],
        sequences: list[str],
        config: DimensionalityReductionConfig,
        counts: Optional[list[int]] = None,
        layers: Optional[list[str]] = None,
        variance_explained: bool = True
    ) -> tuple[plt.Figure, tuple[np.ndarray, Optional[np.ndarray]]]:
        """Plot PCA visualization of activations by layer.

        Args:
            activations: Dictionary of captured activations
            sequences: List of sequences to analyze
            config: Configuration for dimensionality reduction
            counts: Optional list of counts for each sequence
            layers: Layers to analyze. If None, uses all layers.
            variance_explained: Whether to plot variance explained

        Returns:
            Figure and axes objects
        """
        if layers is None:
            layers = self.layers
        elif not isinstance(layers, list):
            layers = [layers]

        # Create figure with appropriate layout
        n_layers = len(layers)
        if variance_explained:
            fig = plt.figure(figsize=(n_layers * 3, 4))
            gs = fig.add_gridspec(2, n_layers, height_ratios=[2, 1])
            axs1 = [fig.add_subplot(gs[0, i]) for i in range(n_layers)]
            axs2 = [fig.add_subplot(gs[1, i]) for i in range(n_layers)]
        else:
            fig, axs1 = plt.subplots(
                ncols=n_layers,
                figsize=(n_layers * 3, 2.5),
                layout='constrained'
            )
            axs2 = None

        if n_layers == 1:
            axs1 = [axs1]
            axs2 = [axs2] if axs2 is not None else None

        # Normalize counts if provided
        if counts is not None:
            norm_counts = np.log1p(counts) / np.log1p(np.max(counts))

        for j, (ax, layer) in enumerate(zip(axs1, layers)):
            model, X_embedding = self.prepare_pca_embeddings(
                activations,
                sequences,
                layer,
                config
            )

            if counts is not None:
                for i, (seq, c) in enumerate(zip(sequences, norm_counts)):
                    ax.scatter(
                        X_embedding[i, 0],
                        X_embedding[i, 1],
                        color='red' if seq[config.token_pos] in ('R', 'r')
                        else 'blue',
                        marker='o' if seq[config.token_pos] in ('R', 'L')
                        else 'x',
                        alpha=c,
                        s=10 * c
                    )
                    if len(sequences) < 20:
                        ax.annotate(
                            seq[config.token_pos],
                            (X_embedding[i, 0], X_embedding[i, 1]),
                            fontsize=12
                        )
            else:
                for i, seq in enumerate(sequences):
                    ax.scatter(
                        X_embedding[i, 0],
                        X_embedding[i, 1],
                        color='red' if seq[config.token_pos] in ('R', 'r')
                        else 'blue',
                        marker='o' if seq[config.token_pos] in ('R', 'L')
                        else 'x',
                        s=10
                    )

            ax.set(
                title=layer.capitalize(),
                xlabel='Dim 1' if j == 0 else '',
                ylabel='Dim 2' if j == 0 else ''
            )

            # Plot variance explained if requested
            if (config.method == 'pca' and variance_explained
                    and axs2 is not None):
                axs2[j].plot(
                    range(1, config.n_components + 1),
                    np.cumsum(model.explained_variance_ratio_),
                    marker='o',
                    color='red'
                )
                axs2[j].set(
                    xlabel='Dimension' if j == 0 else '',
                    xticks=range(1, config.n_components + 1),
                    ylabel='Explained Variance' if j == 0 else '',
                    ylim=(0, 1.05)
                )

        title = (
            f'{self.model_component.upper()} Activations with '
            f'{config.sequence_method} method'
        )
        fig.suptitle(title)
        plt.subplots_adjust(top=0.9)
        sns.despine()

        return fig, (axs1, axs2)

    def plot_pca_across_trials(
        self,
        activations: dict[str, dict[str, np.ndarray]],
        sequences: list[str],
        block_sequences: list[list[str]],
        layers: list[str] | str,
        config: DimensionalityReductionConfig
    ) -> tuple[plt.Figure, tuple[np.ndarray, Optional[np.ndarray]]]:
        """Plot PCA visualization of activations across trials.

        Args:
            activations: Dictionary of captured activations
            sequences: List of sequences to analyze
            block_sequences: List of sequences for each block
            layers: Layer(s) to analyze
            config: Configuration for dimensionality reduction

        Returns:
            Figure and axes objects
        """
        if isinstance(layers, str):
            layers = [layers]

        # Just for aesthetics in the PCA plot
        fake_counts = [100000]
        fake_counts.extend([1] * (len(sequences) - 1))

        # Create base plot with all layers
        fig, axs = self.plot_pca_by_layer(
            activations,
            sequences,
            config,
            counts=fake_counts,
            layers=layers,
            variance_explained=False
        )

        # Get predicted tokens for block sequences
        predicted_token = predict_token(self.model, block_sequences)
        palette = sns.color_palette('magma', n_colors=len(block_sequences))

        # Add transition points to each layer's plot
        for i, layer in enumerate(layers):
            # Compute PCA for this layer
            pca, _ = self.prepare_pca_embeddings(
                activations,
                sequences,
                layer,
                config
            )

            # Get transition activations
            transition_acts = self.get_activations(block_sequences.values)

            # Prepare transition embeddings
            if config.sequence_method == 'token':
                pos_acts = self.get_activation_by_position(
                    transition_acts,
                    token_pos=config.token_pos
                )
                X = np.array([
                    pos_acts[layer][seq]
                    for seq in block_sequences
                ])
            elif config.sequence_method == 'concat':
                X = np.concatenate([
                    transition_acts[seq][layer].reshape(1, -1)
                    for seq in block_sequences
                ], axis=0)
            transition_embeddings = pca.transform(X)

            # Plot transition points
            for j, (seq, t) in enumerate(zip(block_sequences, predicted_token)):
                axs[0][i].scatter(
                    transition_embeddings[j, 0],
                    transition_embeddings[j, 1],
                    color=palette[j],
                    s=30,
                    label=f'{seq}->({t})' if i == 0 else None
                )

        fig.legend(
            bbox_to_anchor=(1.05, 0.1),
            loc='lower left',
            borderaxespad=0.,
            fontsize=8,
            frameon=False
        )
        return fig, axs


class MLPAnalyzer(ActivationAnalyzer):
    def __init__(
        self,
        model: torch.nn.Module,
        config: Any,
        layers: Optional[list[str]] = None
    ):
        """Initialize MLP analyzer.

        Args:
            model: The transformer model to analyze
            config: Model configuration object
            layers: Optional list of layers to analyze
        """
        super().__init__(model)
        self.model_component = 'mlp'
        self.layer_composition = self._get_layer_composition(config)
        self.layers = layers or list(self.layer_composition.keys())

    def _get_layer_composition(self, config: Any) -> dict[str, int]:
        """Get the dimensions of each layer in the model component.

        Args:
            config: Model configuration object

        Returns:
            Dictionary mapping layer names to their dimensions
        """
        dummy_mlp = MLP(config)
        return {
            'input': dummy_mlp.c_fc.in_features,
            'gelu': dummy_mlp.c_fc.out_features,
            'output': dummy_mlp.c_proj.out_features
        }


class EmbeddingAnalyzer(ActivationAnalyzer):
    def __init__(
        self,
        model: torch.nn.Module,
        config: Any,
        layers: Optional[list[str]] = None
    ):
        """Initialize embedding analyzer.

        Args:
            model: The transformer model to analyze
            config: Model configuration object
            layers: Optional list of layers to analyze
        """
        super().__init__(model)
        self.model_component = 'embedding'
        self.layer_composition = self._get_layer_composition(config)
        self.layers = layers or list(self.layer_composition.keys())

    def _get_layer_composition(self, config: Any) -> dict[str, int]:
        """Get the dimensions of each layer in the model component.

        Args:
            config: Model configuration object

        Returns:
            Dictionary mapping layer names to their dimensions
        """
        return {'embed': config.n_embd}

    def get_activations(
        self,
        input_seq: list[str]
    ) -> dict[str, dict[str, np.ndarray]]:
        """Capture activations for a list of input sequences.

        Args:
            input_seq: List of input sequences to analyze

        Returns:
            Dictionary mapping sequences to their layer activations
        """
        if not isinstance(input_seq, (list, np.ndarray)):
            input_seq = [input_seq]

        captured_activations = {seq: {} for seq in input_seq}
        for seq in input_seq:
            embed = embed_sequence(self.model, seq, flatten=False)
            captured_activations[seq]['embed'] = embed.squeeze(1)

        return captured_activations
