"""Module for analyzing model activations across different layers."""

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from analyzer import BaseAnalyzer, BaseVisualizer

from interpretability.interp_helpers import embed_sequence
from transformer.transformer import MLP


class ActivationVisualizer(BaseVisualizer):
    """
    Visualizer for activation and MLP analysis.
    Provides plotting utilities for neuron activations, selectivity,
    and MLP structure.
    """
    def plot_activation_bars(
        self, activations: dict[str, np.ndarray], layer: str
    ) -> tuple[plt.Figure, np.ndarray]:
        """
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
        """
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

        if self.num_tokens > 1:
            if self.token_idx == self.num_tokens - 1:
                ax.set(title='', xlabel='Neuron Index')
            if self.token_idx == 0:
                ax.set(
                    title=f'{layer.capitalize()} Layer',
                    xticks=[],
                    xlabel=''
                )
            else:
                ax.set(xlabel='', xticks=[], title='')
            if self.layer_idx == 0:
                ax.set(ylabel='Token')
            else:
                ax.set(yticks=[], ylabel='')
        else:
            ax.set(
                title=f'{layer.capitalize()} Layer',
                xlabel='Neuron Index',
                ylabel='Token'
            )

        return ax

    def create_mlp_visualization(
        self,
        fig_title: str = None
    ) -> tuple[plt.Figure, dict[str, list[plt.Axes]]]:
        
        """Create a matplotlib figure with subfigures for neural network visualization.

        Args:
            fig_title: Overall figure title

        Returns:
            Tuple of (figure object, dict mapping layer names to axes lists)
        """
        n_layers = len(self.analyzer.layer_composition)
        figsize = (max(self.analyzer.layer_composition.values()) * 1, n_layers * 1.8)
        fig = plt.figure(figsize=figsize, constrained_layout=True)

        if fig_title:
            fig.suptitle(fig_title, fontsize=12)

        subfigs = fig.subfigures(n_layers, 1)  # subfigure for each layer
    
        if n_layers == 1:
            subfigs = [subfigs]

        axes_dict = {}
        for subfig, (layer_name, n_neurons) in zip(
            subfigs, self.analyzer.layer_composition.items()
        ):
            title = f"{layer_name.capitalize()} Layer ({n_neurons} neurons)"
            subfig.suptitle(title, fontsize=12)
            axes = subfig.subplots(nrows=1, ncols=n_neurons)  # ax for each neuron
            if n_neurons == 1:
                axes = [axes]
            axes_dict[layer_name] = axes

        return fig, axes_dict 


class ActivationAnalyzer(BaseAnalyzer):
    """Class for analyzing model activations across different layers.

    This class provides methods to capture, analyze and visualize activations
    from different components of the transformer model.

    Attributes:
        model (torch.nn.Module): The transformer model to analyze
        _hooks (list[torch.utils.hooks.RemovableHandle]): Storage for hooks
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_config: Any,
        layers: Optional[list[str]] = None
    ):
        """Initialize the activation analyzer.

        Args:
            model: The transformer model to analyze
            model_config: Model configuration object
            layers: Optional list of layers to analyze
        """
        super().__init__(model, visualizer_class=ActivationVisualizer)
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self.model_component = 'Activation'  # Will be overridden by subclasses
        self.layer_composition = self._get_layer_composition(model_config)
        self.model_config = model_config

    def _get_layer_composition(self, config: Any) -> dict[str, int]:
        """Get the dimensions of each layer in the model component.

        Args:
            config: Model configuration object

        Returns:
            Dictionary mapping layer names to their dimensions
        """
        raise NotImplementedError(
            "Subclasses must implement _get_layer_composition"
        )

    def get_activations(
        self,
        sequences: list[str]
    ) -> dict[str, dict[str, np.ndarray]]:
        """Get activations for all layers and sequences.
        
        Args:
            sequences: List of sequences to analyze
            
        Returns:
            Dictionary mapping sequences to dictionaries of layer activations.
            Structure: {seq: {layer: activation_array}}
        """

        # Use unified hook system to capture activations
        raw_activations = self._extract_internal_states(sequences, self.layers)

        # Convert raw activations to the expected format
        # raw_activations: {layer: tensor[batch_size, seq_len, hidden_dim]}
        # -> {seq: {layer: activation_array}}
        activations = {
            seq: {
                layer_name: acts[i]  # [i] gets i-th sequence
                for layer_name, acts in raw_activations.items()
            }
            for i, seq in enumerate(sequences)
        }

        return activations


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

    def get_activations_by_block(
        self,
        activations: dict[str, dict[str, np.ndarray]],
        block_idx: int
    ) -> dict[str, dict[str, np.ndarray]]:
        """Extract activations for a specific block."""
        block_activations = {}
        for layer_block, acts in activations.items():
            layer, block = layer_block.split('_')
            if int(block) == block_idx:
                block_activations[layer] = acts
        return block_activations

    def find_maximal_activations(
        self,
        activations: dict[str, dict[str, np.ndarray]],
        layer_name: str,
        sequences: list[str],
        top_n: int = 30
    ) -> dict[int, list[tuple[str, float]]]:
        """Find sequences that maximally activate each neuron.

        Args:
            activations: Dictionary mapping sequences to their layer activations
            layer_name: Layer to analyze
            sequences: List of sequences to consider
            top_n: Number of top activating sequences to return

        Returns:
            Dictionary mapping neuron indices to their top activating sequences
        """
        # Get activations for this layer
        layer_acts = activations[layer_name]

        layer_name = layer_name.split('_')[0]
        num_neurons = self.layer_composition[layer_name]

        max_activating_seqs = {}
        for neuron_idx in range(num_neurons):
            # Get activation of this neuron for each sequence
            neuron_acts = {
                seq: layer_acts[seq][neuron_idx]
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

    def get_average_sequence(
        self,
        token_counts_df,
        single_threshold=0.6,
        joint_threshold=0.4
    ) -> list[str]:
        """Extract tokens based on probability rules:
        1. If highest prob token > single_threshold, use that token
        2. If top two tokens together > joint_threshold, use "(token1/token2)"
        3. Otherwise use "-"

        Args:
            token_counts_df: DataFrame where rows are sequence positions, columns
                are tokens, and values are probabilities
            single_threshold: Threshold for using a single token
            joint_threshold: Threshold for using a pair of tokens

        Returns:
            A list of tokens, token pairs, or dashes for each position
        """
        token_sequence = []

        for idx, row in token_counts_df.iterrows():
            # Sort the tokens by probability in descending order
            sorted_tokens = row.sort_values(ascending=False)

            top_token = sorted_tokens.index[0]
            top_prob = sorted_tokens.iloc[0]

            # If the top token is confident enough, use it
            if top_prob > single_threshold:
                token_sequence.append(top_token)
            else:
                # Check if top two tokens indicate specific uncertainty
                second_token = sorted_tokens.index[1]
                second_prob = sorted_tokens.iloc[1]

                if (top_prob > joint_threshold and
                        second_prob > joint_threshold):
                    # Model is specifically uncertain between two tokens
                    token_sequence.append(f"({top_token}/{second_token})")
                else:
                    # Model is uncertain across many tokens
                    token_sequence.append('-')

        return ''.join(token_sequence)

    def describe_pattern_verbose(
        self,
        token_counts: pd.DataFrame,
        layer: str,
        neuron_idx: int,
        seqs: list[str]
    ) -> None:
        """Analyze if this neuron might be detecting specific patterns"""
        print(f"\nPatterns detected by {layer} layer, Neuron {neuron_idx}:")

        # Check for position-specific token preferences
        for row, freq in token_counts.iterrows():
            if any(freq > 0.7):
                max_token = freq[freq > 0.7].index[0]
                print(
                    f"Position {row}: Strong preference for '{max_token}'"
                )

        # Check if neuron responds to recent history
        last_tokens = [seq[-1] for seq in seqs]
        if (all(token in ['R', 'r'] for token in last_tokens) or 
            all(token in ['L', 'l'] for token in last_tokens)):
            print("→ This neuron strongly responds to the type of the most "
                  "recent decision (R/r vs L/l)")

        # Check for alternating patterns
        alternating_count = sum(1 for seq in seqs if 'RL' in seq or 'LR' in seq)
        if alternating_count >= len(seqs) * 0.7:
            print("→ This neuron may be detecting alternating patterns "
                  "between R and L")

        # Check for repeated patterns
        repeat_count = sum(1 for seq in seqs if 'RR' in seq or 'LL' in seq)
        if repeat_count >= len(seqs) * 0.7:
            print("→ This neuron may be detecting repeated tokens of "
                  "the same type")

    def analyze_neuron_patterns(
        self,
        max_activations: dict[str, dict[str, list[tuple[str, float]]]],
        layer_name: str,
        neuron_idx: int,
        ax: plt.Axes | None = None,
        verbose: bool = True,
        **kwargs
    ) -> tuple[plt.Axes, pd.DataFrame]:
        """Analyze patterns in sequences that maximally activate a specific neuron.

        Args:
            max_activations: Dictionary mapping layers and neurons to sequences
                that maximally activate them
            layer_name: Name of layer to analyze
            neuron_idx: Index of neuron to analyze
            ax: Optional matplotlib axes to plot on
            verbose: Whether to print detailed pattern analysis
            **kwargs: Additional arguments passed to seaborn heatmap

        Returns:
            Tuple of (matplotlib axes, token counts DataFrame)
        """
        # Get sequences that maximally activate this neuron
        max_seqs = [seq for seq, _ in max_activations[layer_name][neuron_idx]]

        token_counts = self.count_tokens(max_seqs)

        # Normalize counts
        token_counts = token_counts.div(token_counts.sum(axis=1), axis=0)

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.set(
                title=(f'{layer_name.capitalize()} Layer, '
                       f'Neuron {neuron_idx} - Token Patterns')
            )
            if verbose:
                self.describe_pattern_verbose(
                    token_counts,
                    layer_name,
                    neuron_idx,
                    max_seqs
                )

        sns.heatmap(
            token_counts.T,
            cmap='viridis',
            annot=False,
            fmt='.2f',
            ax=ax,
            vmin=0,
            vmax=1,
            **kwargs
        )

        ax.set(xlabel='Position in Sequence', ylabel='Token')

        return ax, token_counts

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

        # Get activations for target and reference tokens
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

        self.visualizer.num_tokens = len(token_pos)
        self.visualizer.num_layers = len(layers)

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
                self.visualizer.token_idx = i
                self.visualizer.layer_idx = j
                if plot_bars:
                    self.visualizer.plot_activation_bars(selectivity[layer], layer)

                self.visualizer.plot_activation_heatmap(
                    selectivity[layer],
                    layer,
                    ax=ax
                )

            subfig.suptitle(f'pos: {pos}', y=0.95, x=0.05, ha='left')

        fig.suptitle(
            f'Neuron selectivity by token position\n{kwargs.get("method")}',
            y=0.95
        )
        fig.tight_layout()

        self.reset_state()

        return selectivity, fig

    def reset_state(self):
        self.visualizer.token_idx = None
        self.visualizer.layer_idx = None
        self.visualizer.num_tokens = None
        self.visualizer.num_layers = None


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
        super().__init__(model, config, layers)
        self.model_component = 'MLP'
        self.mlp_components = layers or list(self.layer_composition.keys())
        self.layers = [f'{c}_{i}' for c in self.mlp_components for i in range(config.n_layer)]

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
        super().__init__(model, config, layers)
        self.model_component = 'Embedding'
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
            captured_activations[seq]['embed'] = embed#.squeeze(1)

        return captured_activations
