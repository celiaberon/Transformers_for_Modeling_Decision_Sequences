"""Helper functions for analyzing transformer model components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from interp_helpers import embed_sequence, pca_embeddings, predict_token
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class DimensionalityReductionConfig:
    """Configuration for activation analysis.

    Attributes:
        token_pos: Position in sequence to analyze
        sequence_method: Method for sequence embedding
        n_components: Number of components for dimensionality reduction
        method: Embedding method ('pca' or 'tsne')
    """

    def __init__(
        self,
        token_pos: int = -1,
        sequence_method: str = 'token',
        n_components: int = 4,
        method: str = 'pca'
    ):
        self.token_pos = token_pos
        self.sequence_method = sequence_method
        self.n_components = n_components
        self.method = method


class BaseVisualizer:
    """Base class for visualizing model components.
    
    This class provides common visualization functionality for different
    parts of transformer models.
    
    Args:
        analyzer: The analyzer instance to visualize
    """
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.plot_context = None  # Can be 'single', 'multi_head', etc.
        self.seq_idx = None
        
    def plot_token_probs(
        self,
        probs: np.ndarray,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Axes:
        """Plot probability distribution over tokens.
        
        Args:
            probs: Probability distribution
            ax: Optional axes to plot on
            **kwargs: Additional plotting arguments
            
        Returns:
            Axes object with heatmap
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 1))
            
        hm = sns.heatmap(
            probs.reshape(1, -1),
            ax=ax,
            vmin=-1,
            vmax=1,
            cmap="RdBu",
            annot=False,
            cbar=False,
            **kwargs
        )
        
        for i, char in enumerate(self.analyzer.vocab):
            ax.text(
                i + 0.5, 0.5, char,
                ha='center', va='center',
                fontsize=8, fontweight='bold', color='k'
            )
        ax.set_yticks([])
        ax.set_xticks([])
        
        return ax

    def plot_pca_by_layer(
        self,
        sequences: list[str],
        config: DimensionalityReductionConfig,
        counts: Optional[list[int]] = None,
        layers: Optional[list[str]] = None,
        variance_explained: bool = True
    ) -> tuple[plt.Figure, tuple[np.ndarray, Optional[np.ndarray]]]:
        """Plot PCA visualization of activations by layer.

        Args:
            sequences: List of sequences to analyze
            config: Configuration for dimensionality reduction
            counts: Optional list of counts for each sequence
            layers: Layers to analyze. If None, uses all layers.
            variance_explained: Whether to plot variance explained

        Returns:
            Figure and axes objects
        """
        if layers is None:
            layers = self.analyzer.layers
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
                figsize=(n_layers * 3, 2.5)
            )
            axs2 = None

        if n_layers == 1:
            axs1 = [axs1]
            axs2 = [axs2] if axs2 is not None else None

        # Normalize counts if provided
        if counts is not None:
            norm_counts = np.log1p(counts) / np.log1p(np.max(counts))

        for j, (ax, layer) in enumerate(zip(axs1, layers)):
            model, X_embedding = self.analyzer.prepare_pca_embeddings(
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
                ylabel='Dim 2'
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
                    xlabel='Dimension',
                    xticks=range(1, config.n_components + 1),
                    ylabel='Explained Variance' if j == 0 else '',
                    ylim=(0, 1.05)
                )

        title = (
            f'{self.analyzer.model_component} Activations\n with '
            f'{config.sequence_method} method'
        )
        fig.suptitle(title, y=0.9)
        # plt.subplots_adjust(top=0.98)
        sns.despine()
        plt.tight_layout()
        return fig, (axs1, axs2)

    def plot_pca_across_trials(
        self,
        sequences: list[str],
        block_sequences: list[list[str]],
        layers: list[str] | str,
        config: DimensionalityReductionConfig
    ) -> tuple[plt.Figure, tuple[np.ndarray, Optional[np.ndarray]]]:
        """Plot PCA visualization of activations across trials.

        Args:
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
        fake_counts = [10000]
        fake_counts.extend([1] * (len(sequences) - 1))

        # Create base plot with all layers
        fig, axs = self.plot_pca_by_layer(
            sequences,
            config,
            counts=fake_counts,
            layers=layers,
            variance_explained=False
        )

        # Get predicted tokens for block sequences
        predicted_token = predict_token(self.analyzer.model, block_sequences)
        palette = sns.color_palette('magma', n_colors=len(block_sequences))

        # Add transition points to each layer's plot
        for i, layer in enumerate(layers):
            # Compute PCA for this layer using the base sequences
            pca, _ = self.analyzer.prepare_pca_embeddings(
                sequences,
                layer,
                config
            )

            # Get transition activations for the block sequences
            transition_acts = self.analyzer.get_layer_activations(
                block_sequences,
                layer
            )
            if self.analyzer.verbose:
                print(f"Transition acts keys: {list(transition_acts.keys())}")

            # Prepare transition embeddings
            if config.sequence_method == 'token':
                X = np.array([
                    act[config.token_pos]
                    for act in transition_acts.values()
                ])
            elif config.sequence_method == 'concat':
                X = np.concatenate([
                    act.reshape(1, -1)
                    for act in transition_acts.values()
                ], axis=0)

            # Transform block sequences using the PCA from base sequences
            transition_embeddings = pca.transform(X)

            # Create mapping from sequence to its embedding
            seq_to_embedding = {
                seq: emb for seq, emb in zip(transition_acts.keys(), transition_embeddings)
            }

            # Plot transition points
            for j, (seq, t) in enumerate(zip(block_sequences, predicted_token)):
                embedding = seq_to_embedding[seq]
                axs[0][i].scatter(
                    embedding[0],
                    embedding[1],
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


class BaseAnalyzer(ABC):
    """Base class for analyzing transformer model components.
    
    This class provides common functionality for analyzing different parts
    of transformer models (attention, activations, embeddings, etc.).
    
    Args:
        model: The transformer model to analyze
        vocab: List of vocabulary tokens
        stoi: Dictionary mapping tokens to indices
        visualizer_class: Optional visualizer class to use
    """
    
    def __init__(
        self,
        model,
        verbose=False,
        visualizer_class=BaseVisualizer
    ):
        self.model = model
        self.verbose = verbose
        self.vocab = ['R', 'r', 'L', 'l']
        self.stoi = {token: idx for idx, token in enumerate(self.vocab)}
        self.n_layers = model.config.n_layer
        self.n_heads = model.config.n_head
        self.n_embd = model.config.n_embd
        self.visualizer = visualizer_class(self)
    
    def tokenize(self, sequences):
        if isinstance(sequences, str):
            return torch.tensor([self.stoi[char] for char in sequences])
        else:
            return torch.tensor([[self.stoi[char] for char in sequence]
                                for sequence in sequences])
        
    def _prepare_input(self, sequence: str) -> torch.Tensor:
        """Convert input sequence to tensor format."""
        token_ids = self.tokenize(sequence)
        if isinstance(token_ids, torch.Tensor):
            input_tensor = token_ids.clone().detach().unsqueeze(0)
        else:
            input_tensor = torch.tensor(
                token_ids,
                dtype=torch.long
            ).unsqueeze(0)
        return input_tensor.to(self.model.device)
    
    def get_embeddings(self, sequence: str) -> np.ndarray:
        """Get embeddings for a sequence."""
        return embed_sequence(self.model, sequence)
    
    def pca_embeddings(
        self,
        n_components: int = 2,
        **kwargs
    ) -> Tuple[object, np.ndarray]:
        """Get PCA of embeddings."""
        return pca_embeddings(
            self.model,
            n_components=n_components,
            token_mapping=self.stoi,
            **kwargs
        )
    
    def predict_next_token(self, sequence: str) -> np.ndarray:
        """Get model prediction for the next token."""
        input_ids = self._prepare_input(sequence)
        with torch.no_grad():
            logits, _ = self.model(input_ids)
        next_token_logits = logits[:, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        return probs.detach().cpu().numpy()[0]
    
    def get_token_probs(self, sequence: str) -> Dict[str, float]:
        """Get probability distribution over next tokens."""
        probs = self.predict_next_token(sequence)
        return dict(zip(self.vocab, probs))

    def count_tokens(self, seqs: list[str]) -> pd.DataFrame:
        """Count token frequencies at each position across a list of sequences.

        Args:
            seqs: List of sequences to analyze

        Returns:
            DataFrame with token counts at each position
        """

        token_counts = []
        for pos in range(len(seqs[0])):  # Assuming all sequences have same length
            pos_counts = {'R': 0, 'r': 0, 'L': 0, 'l': 0}
            for seq in seqs:
                pos_counts[seq[pos]] += 1
            token_counts.append(pos_counts)

        return pd.DataFrame(token_counts)

    @abstractmethod
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
        pass

    def prepare_pca_embeddings(
        self,
        sequences: list[str],
        layer: str,
        config: DimensionalityReductionConfig
    ) -> tuple[Any, np.ndarray]:
        """Prepare PCA embeddings for visualization.
        
        Args:
            sequences: List of sequences to analyze
            layer: Layer to analyze
            config: Analysis configuration
            
        Returns:
            Tuple of (model, embeddings)
        """
        # Get activations in the format specific to this analyzer
        activations = self.get_layer_activations(sequences, layer)
        
        if config.sequence_method == 'token':
            # Get activations at specific position
            X = np.array([
                act[config.token_pos]
                for act in activations.values()
            ])
        elif config.sequence_method == 'concat':
            X = np.concatenate([act.reshape(1, -1) for act in activations.values()], axis=0)

        if config.method == 'tsne':
            model = TSNE(n_components=2)
            X_embedding = model.fit_transform(X)
        elif config.method == 'pca':
            model = PCA(n_components=config.n_components)
            X_embedding = model.fit_transform(X)

        return model, X_embedding

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
