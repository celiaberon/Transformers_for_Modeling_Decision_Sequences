"""Helper functions for analyzing transformer model components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from interp_helpers import embed_sequence, pca_embeddings
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

    def plot_multi_trial_token_probs(
        self,
        sequences: list[str],
        max_sequences: int = 5,
        **kwargs
    ) -> None:
        """Analyze attention patterns for multiple sequences.
        
        Args:
            sequences: List of sequences to analyze
            max_sequences: Maximum number of sequences to analyze
            **kwargs: Additional arguments passed to plotting functions
        """
        sequences_to_analyze = sequences[:max_sequences]
            
        n_cols = len(sequences_to_analyze)
        
        # Create figure with minimal margins
        fig, axs = plt.subplots(ncols=n_cols, figsize=(3*n_cols, 0.5))        
            
        for ax, seq in zip(axs, sequences_to_analyze):
            # Use analyzer's methods to get probabilities and next token
            probs = self.analyzer.predict_next_token_probs(seq)
            next_token = self.analyzer.predict_next_token(seq, probs)
            
            # Plot probabilities using the base method
            self.plot_token_probs(probs, ax=ax)
                
            ax.set(title=f'{seq} → ({next_token})')

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        return fig

    def plot_embedding_points(
        self,
        ax: plt.Axes,
        embeddings: np.ndarray,
        sequences: list[str],
        config: DimensionalityReductionConfig,
        counts: Optional[list[int]] = None,
        color_map: Optional[dict[str, str]] = None,
        marker_map: Optional[dict[str, str]] = None,
        labels: Optional[list[str]] = None,
        **kwargs
    ) -> plt.Axes:
        """Plot points in embedding space.
        
        Args:
            ax: Axes to plot on
            embeddings: Array of embeddings to plot
            sequences: List of sequences corresponding to embeddings
            config: Configuration for dimensionality reduction
            counts: Optional list of counts for each sequence
            color_map: Optional mapping from sequence to color
            marker_map: Optional mapping from sequence to marker
            labels: Optional list of labels for each point
            **kwargs: Additional plotting arguments
            
        Returns:
            Axes with plotted points
        """
        if counts is not None:
            norm_counts = np.log1p(counts) / np.log1p(np.max(counts))
            
        for i, (seq, emb) in enumerate(zip(sequences, embeddings)):
            # Determine color
            if color_map is not None:
                color = color_map[seq]
            else:
                color = 'red' if seq[config.token_pos] in ('R', 'r') else 'blue'
                
            # Determine marker
            if marker_map is not None:
                marker = marker_map[seq]
            else:
                marker = 'o' if seq[config.token_pos] in ('R', 'L') else 'x'
                
            # Determine size and alpha
            if counts is not None:
                alpha = norm_counts[i]
                size = 10 * norm_counts[i]
            else:
                alpha = 1.0
                size = 10
                
            # Plot point
            ax.scatter(
                emb[0],
                emb[1],
                color=color,
                marker=marker,
                alpha=alpha,
                s=size,
                label=labels[i] if labels is not None else None,
                **kwargs
            )
            
            # Add annotation for small datasets
            if len(sequences) < 20:
                ax.annotate(
                    seq[config.token_pos],
                    (emb[0], emb[1]),
                    fontsize=12
                )
                
        return ax

    def transform_sequences_to_pca_space(
        self,
        sequences: list[str],
        layer: str,
        config: DimensionalityReductionConfig,
        pca_model: Any,
        seq_to_embedding: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Transform sequences into an existing PCA space.
        
        Args:
            sequences: List of sequences to transform
            layer: Layer to get activations from
            config: Configuration for dimensionality reduction
            pca_model: Existing PCA model to use for transformation
            seq_to_embedding: Mapping from sequences to their embeddings
            
        Returns:
            Array of embeddings in the PCA space, preserving original sequence order
        """
        # Split sequences into existing and new
        new_seqs = [seq for seq in sequences if seq not in seq_to_embedding]
        
        # Initialize result array
        embeddings = np.zeros((len(sequences), config.n_components))  # Assuming 2D PCA space
        
        # Fill in existing sequences
        for i, seq in enumerate(sequences):
            if seq in seq_to_embedding:
                embeddings[i] = seq_to_embedding[seq]
        
        # Transform new sequences if any
        if new_seqs:
            # Get activations for new sequences
            new_acts = self.analyzer.get_layer_activations(new_seqs, layer)
            
            # Prepare data for transformation
            if config.sequence_method == 'token':
                X_new = np.array([
                    act[config.token_pos] for act in new_acts.values()
                ])
            else:  # concat
                X_new = np.concatenate([
                    act.reshape(1, -1) for act in new_acts.values()
                ], axis=0)
                
            # Transform using existing PCA model
            new_embeddings = pca_model.transform(X_new)
            
            # Fill in new sequences
            for i, seq in enumerate(sequences):
                if seq in new_seqs:
                    new_idx = new_seqs.index(seq)
                    embeddings[i] = new_embeddings[new_idx]
        
        return embeddings

    def plot_pca_by_layer(
        self,
        sequences: list[str],
        config: DimensionalityReductionConfig,
        counts: Optional[list[int]] = None,
        layers: Optional[list[str]] = None,
        variance_explained: bool = True,
        additional_points: Optional[list[tuple[list[str], dict]]] = None
    ) -> tuple[plt.Figure, tuple[np.ndarray, Optional[np.ndarray]]]:
        """Plot PCA visualization of activations by layer.

        Args:
            sequences: List of sequences to analyze
            config: Configuration for dimensionality reduction
            counts: Optional list of counts for each sequence
            layers: Layers to analyze. If None, uses all layers.
            variance_explained: Whether to plot variance explained
            additional_points: Optional list of (sequences, plot_kwargs) tuples
                for additional points to plot in the same embedding space.
                Sequences can be either in the base set or new sequences.

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

        for j, (ax, layer) in enumerate(zip(axs1, layers)):
            # Compute PCA embeddings
            model, embeddings, seq_to_embedding = self.analyzer.compute_pca_embeddings(
                sequences,
                layer,
                config
            )
            
            # Plot base sequences
            self.plot_embedding_points(
                ax,
                embeddings,
                sequences,
                config,
                counts=counts
            )
            
            if additional_points is not None:
                for add_seqs, plot_kwargs in additional_points:
                    # Transform sequences to PCA space
                    add_embeddings = self.transform_sequences_to_pca_space(
                        add_seqs,
                        layer,
                        config,
                        model,
                        seq_to_embedding
                    )

                    ax.scatter(
                        add_embeddings[:, 0],
                        add_embeddings[:, 1],
                        label=plot_kwargs.pop('label', None),
                        **plot_kwargs
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
        sns.despine()

        fig.legend(
            bbox_to_anchor=(1.05, 0.1),
            loc='lower left',
            borderaxespad=0.,
            fontsize=8,
            frameon=False
        )
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

        # Get predicted tokens for block sequences
        predicted_token = [self.analyzer.predict_next_token(seq) for seq in block_sequences]
        palette = sns.color_palette('magma', n_colors=len(block_sequences))

        # Create additional points configuration
        additional_points = []
        for i, (seq, t) in enumerate(zip(block_sequences, predicted_token)):
            additional_points.append((
                [seq],  # Single sequence
                {
                    'color': palette[i],  # Use 'color' instead of 'c'
                    'label': f'{seq} → ({t})',
                    's': 30
                }
            ))

        # Create plot with base sequences and additional points
        return self.plot_pca_by_layer(
            sequences,
            config,
            counts=fake_counts,
            layers=layers,
            variance_explained=False,
            additional_points=additional_points
        )


class BaseAnalyzer(ABC):
    """Base class for analyzing transformer model components.
    
    This class provides common functionality for analyzing different parts
    of transformer models (attention, activations, embeddings, etc.).
    
    Args:
        model: The transformer model to analyze
        verbose: Whether to print debug information
        visualizer_class: Class to use for visualization
    """
    
    def __init__(
        self,
        model,
        verbose=False,
        visualizer_class=BaseVisualizer
    ):
        """Initialize the analyzer with a model."""
        self.model = model
        self.device = next(model.parameters()).device
        self.verbose = verbose
        self.vocab = ['R', 'r', 'L', 'l']
        self.stoi = {token: idx for idx, token in enumerate(self.vocab)}
        self.n_layers = model.config.n_layer
        self.n_heads = model.config.n_head
        self.n_embd = model.config.n_embd
        self.visualizer = visualizer_class(self)

    def tokenize(
        self,
        sequences: str | list[str],
        batch: bool = False
    ) -> torch.Tensor:
        """Convert input sequence(s) to tensor format.
        
        Args:
            sequences: Input sequence or list of sequences
            batch: Whether to return batched tensor, even for single sequence
            
        Returns:
            Tensor of token IDs, shape (batch_size, seq_len) if batch=True or
            multiple sequences, otherwise (seq_len,)
        """
        if isinstance(sequences, str):
            token_ids = [self.stoi[char] for char in sequences]
            if batch:
                return torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            return torch.tensor(token_ids, dtype=torch.long, device=self.device)
        else:
            token_ids = [[self.stoi[char] for char in sequence] for sequence in sequences]
            return torch.tensor(token_ids, dtype=torch.long, device=self.device)

    def _prepare_input(
        self,
        sequences: str | list[str],
        batch: bool = False
    ) -> torch.Tensor:
        """Prepare input tensor for model.
        
        Args:
            sequences: Input sequence or list of sequences
            batch: Whether to return batched tensor
            
        Returns:
            Tensor ready for model input
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        input_tensor = self.tokenize(sequences, batch=True)
        return input_tensor.to(self.device)

    def get_embeddings(self, sequence: str, **kwargs) -> np.ndarray:
        """Get embeddings for a sequence."""
        return embed_sequence(self.model, sequence, **kwargs)
    
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
    
    def predict_next_token_probs(self, sequence: str) -> np.ndarray:
        """Get model probabilities for the next token."""
        input_ids = self._prepare_input(sequence, batch=True)
        with torch.no_grad():
            logits, _ = self.model(input_ids)
        next_token_logits = logits[:, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        return probs.detach().cpu().numpy()[0]
    
    def predict_next_token(self, sequence: str, probs=None) -> str:
        """Get model prediction for the next token."""
        if probs is None:
            probs = self.predict_next_token_probs(sequence)
        return self.vocab[np.argmax(probs)]
    
    def get_token_probs(self, sequence: str) -> Dict[str, float]:
        """Get probability distribution over next tokens."""
        probs = self.predict_next_token_probs(sequence)
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

    def compute_pca_embeddings(
        self,
        sequences: list[str],
        layer: str,
        config: DimensionalityReductionConfig
    ) -> tuple[Any, np.ndarray]:
        """Compute PCA embeddings for a set of sequences.
        
        Args:
            sequences: List of sequences to analyze
            layer: Layer to analyze
            config: Configuration for dimensionality reduction
            
        Returns:
            Tuple of (pca model, embeddings array, sequence to embedding mapping)
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

        # Create mapping from sequence to embedding
        seq_to_embedding = {
            seq: emb for seq, emb in zip(activations.keys(), X_embedding)
        }

        return model, X_embedding, seq_to_embedding

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
