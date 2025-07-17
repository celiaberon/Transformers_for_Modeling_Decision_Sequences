"""Core base classes for interpretability: BaseAnalyzer, BaseVisualizer, HookManager."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from interpretability.core.config import (DimensionalityReductionConfig,
                                          InterpretabilityConfig)
from transformer.models import GPT


class HookManager:
    """Manages hooks for capturing model activations.
    
    This class handles the registration, execution, and cleanup of hooks
    for capturing activations from different model components.
    """
    
    def __init__(self, model: GPT, verbose: bool = False):
        """Initialize the hook manager.
        
        Args:
            model: The model to attach hooks to
            verbose: Whether to print debug information
        """
        self.model = model
        self.verbose = verbose
        self.hooks = []
        self.activations = {}
        
    def setup_hooks(self, components: List[str], last_token_only: bool = False) -> None:
        """Set up hooks for specified components.
        Args:
            components: List of component names to capture
            last_token_only: If True, only capture activations for the last token
        """
        if self.verbose:
            print(f"Setting up hooks for components: {components}")
            
        for component in components:
            self.activations[component] = []
            hook = self._register_component_hook(component, last_token_only=last_token_only)
            self.hooks.append(hook)
        if self.verbose:
            print(f"Registered {len(self.hooks)} hooks")

    def _register_component_hook(self, component: str, last_token_only: bool = False) -> torch.utils.hooks.RemovableHandle:
        """Register a hook for a specific component.
        Args:
            component: Component name to hook
            
        Returns:
            Hook handle
        """
        if component == 'embed':
            return self.model.transformer.wte.register_forward_hook(
                self._make_hook('embed', last_token_only=last_token_only)
            )
        elif component == 'final':
            return self.model.transformer.ln_f.register_forward_hook(
                self._make_hook('final', last_token_only=last_token_only)
            )
        else:
            layer_idx = int(component.split('_')[-1])
            if layer_idx >= len(self.model.transformer.h):
                raise ValueError(f"Layer index {layer_idx} out of range")
            return self._register_layer_component_hook(component, layer_idx, last_token_only=last_token_only)

    def _register_layer_component_hook(self, component: str, layer_idx: int, last_token_only: bool = False) -> torch.utils.hooks.RemovableHandle:
        layer = self.model.transformer.h[layer_idx]
        if component.startswith(('layer_', 'attn_')):
            return layer.register_forward_hook(self._make_hook(component, last_token_only=last_token_only))
        elif component.startswith(('qk_', 'ov_')):
            return layer.attn.register_forward_hook(self._make_attention_hook(component, last_token_only=last_token_only))
        elif component.startswith('input'):
            return layer.mlp.c_fc.register_forward_hook(self._make_hook(component, last_token_only=last_token_only))
        elif component.startswith('gelu'):
            return layer.mlp.gelu.register_forward_hook(self._make_hook(component, last_token_only=last_token_only))
        elif component.startswith('output'):
            return layer.mlp.c_proj.register_forward_hook(self._make_hook(component, last_token_only=last_token_only))
        else:
            raise ValueError(f"Unknown component: {component}")

    def _make_hook(self, component_name: str, last_token_only: bool = False):
        def hook(module, input, output):
            if last_token_only:
                if output.dim() == 3:
                    self.activations[component_name].append(output[:, -1, :].detach().cpu().numpy())
                else:
                    self.activations[component_name].append(output.detach().cpu().numpy())
            else:
                self.activations[component_name].append(output.detach().cpu().numpy())
        return hook

    def _make_attention_hook(self, component_name: str, last_token_only: bool = False):
        def hook(module, input, output):
            x = input[0]
            B, T, C = x.size()
            qkv = module.c_attn(x)
            q, k, v = qkv.split(module.n_embd, dim=2)
            q = q.view(B, T, module.n_head, C // module.n_head).transpose(1, 2)
            k = k.view(B, T, module.n_head, C // module.n_head).transpose(1, 2)
            v = v.view(B, T, module.n_head, C // module.n_head).transpose(1, 2)
            qk_attn = torch.matmul(q, k.transpose(-2, -1)) / (C ** 0.5)
            qk_attn = qk_attn.masked_fill(
                module.bias[:, :, :T, :T] == 0, float('-inf')
            )
            qk_attn_softmax = F.softmax(qk_attn, dim=-1)
            
            ov_output = torch.matmul(qk_attn_softmax, v)
            if last_token_only:
                if component_name.startswith('qk_attn_softmax'):
                    self.activations[component_name].append(qk_attn_softmax[:, :, -1, :].detach().cpu().numpy())
                elif component_name.startswith('qk_'):
                    self.activations[component_name].append(qk_attn[:, :, -1, :].detach().cpu().numpy())
                elif component_name.startswith('ov_'):
                    self.activations[component_name].append(ov_output[:, :, -1, :].detach().cpu().numpy())
                else:
                    self.activations[component_name].append(output.detach().cpu().numpy())
            else:
                if component_name.startswith('qk_attn_softmax'):
                    self.activations[component_name].append(
                        qk_attn_softmax.detach().cpu().numpy()
                    )
                elif component_name.startswith('qk_'):
                    self.activations[component_name].append(
                        qk_attn.detach().cpu().numpy()
                    )
                elif component_name.startswith('ov_'):
                    self.activations[component_name].append(
                        ov_output.detach().cpu().numpy()
                    )
                else:
                    self.activations[component_name].append(
                        output.detach().cpu().numpy()
                    )
        return hook
    
    def validate_outputs(self, expected_count: int, layer_name: Optional[str] = None) -> None:
        """Validate that hooks captured the expected activations.
        
        Args:
            expected_count: Expected number of activation sets
            layer_name: Optional specific layer to validate
            
        Raises:
            ValueError: If activations were not captured correctly
        """
        layers_to_check = [layer_name] if layer_name else list(self.activations.keys())
        
        for layer in layers_to_check:
            if not self.activations[layer]:
                raise ValueError(f"No activations captured for layer {layer}")
            if len(self.activations[layer]) != 1:
                raise ValueError(
                    f"Expected 1 forward pass for layer {layer}, "
                    f"got {len(self.activations[layer])}"
                )
            
            # Validate shapes
            act = self.activations[layer][0]
            if layer.startswith(('qk_', 'ov_')):
                if len(act.shape) != 4:
                    raise ValueError(
                        f"Expected 4D attention tensor for {layer}, "
                        f"got shape {act.shape}"
                    )
            else:
                if len(act.shape) != 3:
                    raise ValueError(
                        f"Expected 3D activation tensor for {layer}, "
                        f"got shape {act.shape}"
                    )
            
            if act.shape[0] != expected_count:
                raise ValueError(
                    f"Expected batch size {expected_count} for {layer}, "
                    f"got {act.shape[0]}"
                )
    
    def get_activations(self) -> Dict[str, np.ndarray]:
        """Get captured activations.
        
        Returns:
            Dictionary mapping component names to their activations
        """
        return {
            component: acts_list[0] if acts_list else None
            for component, acts_list in self.activations.items()
        }
    
    def clear_activations(self) -> None:
        """Clear stored activations."""
        for component in self.activations:
            self.activations[component].clear()
    
    def cleanup(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.clear_activations()


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
        self.vocab = analyzer.vocab
        self._set_theme()

    def _set_theme(self):
        sns.set_theme(style='ticks',
                      font_scale=1.0,
                      rc={'axes.labelsize': 10,
                          'axes.titlesize': 10,
                          'savefig.transparent': True,
                          'legend.title_fontsize': 10,
                          'legend.fontsize': 10,
                          'legend.borderpad': 0.2,
                          'figure.titlesize': 10,
                          'figure.subplot.wspace': 0.1,
                          })
        
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
        
        for i, char in enumerate(self.vocab):
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
        vertical=False,
        **kwargs
    ) -> None:
        """Analyze attention patterns for multiple sequences.
        
        Args:
            sequences: List of sequences to analyze
            max_sequences: Maximum number of sequences to analyze
            **kwargs: Additional arguments passed to plotting functions
        """
        sequences_to_analyze = sequences[:max_sequences]
            
        n_rows = 1
        n_cols = len(sequences_to_analyze)
        
        if vertical:
            n_rows, n_cols = n_cols, n_rows
        
        # Create figure with minimal margins
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols,
                                figsize=(3*n_cols-vertical, 0.5*n_rows+vertical))        
            
        for ax, seq in zip(axs.ravel(), sequences_to_analyze):
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
            labels: Optional list of labels for each point
            **kwargs: Additional plotting arguments
            
        Returns:
            Axes with plotted points
        """
            
        for i, (seq, emb) in enumerate(zip(sequences, embeddings)):
            # Determine color
            if color_map is not None:
                color = color_map[seq]
            else:
                color = 'red' if seq[config.token_pos] in ('R', 'r') else 'blue'
                
            # Determine marker
            marker = 'o' if seq[config.token_pos] in ('R', 'L') else 'x'
                
            # Determine alpha
            if 'alpha' in kwargs:
                pass
            elif counts is not None:
                norm_counts = np.log1p(counts) / np.log1p(np.max(counts))
                kwargs['alpha'] = norm_counts[i]
            else:
                kwargs['alpha'] = 1.0
                
            # Plot point
            ax.scatter(
                emb[0],
                emb[1],
                color=color,
                marker=marker,
                s=10,
                label=labels[i] if labels is not None else None,
                **kwargs
            )
            
            # Add annotation for small datasets
            if len(sequences) < 20:
                ax.annotate(
                    seq,
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
        additional_points: Optional[list[tuple[list[str], dict]]] = None,
        **kwargs
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

        if n_layers == 1 and not variance_explained:
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
                counts=counts,
                **kwargs
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
                    color='k'
                )
                axs2[j].set(
                    xlabel='Dimension',
                    xticks=range(1, config.n_components + 1),
                    ylabel='Explained Variance\n(Cumulative)' if j == 0 else '',
                    ylim=(0, 1.05)
                )

        title = (
            f'{self.analyzer.model_component} Activations\n '
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
            layers=layers,
            variance_explained=False,
            additional_points=additional_points,
            alpha=0.05
        )

    def plot_2d_surface(
        self,
        embeddings: np.ndarray,
        ax: plt.Axes = None,
        method: str = 'kde',
        bins: int = 100,
        scale: str = 'linear',
        normalize: bool = False,
        additional_points: Optional[list[tuple[list[str], dict]]] = None,
        **kwargs
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot a 2D surface over PC1 vs PC2 using density estimation (KDE) or smoothed histogram.
        Args:
            embeddings: (N, 2) array
            ax: matplotlib Axes (optional, will create if None)
            method: 'kde' (kernel density) or 'hist' (smoothed histogram)
            bins: number of bins for histogram
            scale: 'linear', 'log', 'sqrt', or 'power' (power=0.5)
            normalize: whether to normalize to [0,1]
            additional_points: Optional list of (sequences, plot_kwargs) tuples
                for additional points to plot on the surface
            **kwargs: passed to pcolormesh/contourf
        Returns:
            tuple of (figure, axes)
        """
        # Initialize figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,6))
        else:
            fig = ax.figure
            
        x, y = embeddings[:, 0], embeddings[:, 1]
        xi = np.linspace(x.min(), x.max(), bins)
        yi = np.linspace(y.min(), y.max(), bins)
        xi, yi = np.meshgrid(xi, yi)
        if method == 'kde':
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(np.vstack([x, y]))
            zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
        elif method == 'hist':
            from scipy.ndimage import gaussian_filter
            H, xedges, yedges = np.histogram2d(x, y, bins=bins)
            zi = gaussian_filter(H.T, sigma=2)
            xi, yi = np.meshgrid(
                (xedges[:-1] + xedges[1:]) / 2,
                (yedges[:-1] + yedges[1:]) / 2
            )
        else:
            raise ValueError("method must be 'kde' or 'hist'")
        
        # Apply scaling
        zi_scaled = self._apply_scaling(zi, scale, normalize)
        
        surf = ax.pcolormesh(xi, yi, zi_scaled, **kwargs)
        plt.colorbar(surf, ax=ax, label=f'Density ({scale})')
        ax.set(xlabel='PC1', ylabel='PC2')
        
        # Add additional points if provided
        if additional_points is not None:
            for add_seqs, plot_kwargs in additional_points:
                # For additional points, we need to get their embeddings
                # This assumes the sequences are in the same embedding space
                # You might need to adjust this based on your specific use case
                ax.scatter(
                    add_seqs[:, 0] if hasattr(add_seqs, 'shape') else add_seqs[0],
                    add_seqs[:, 1] if hasattr(add_seqs, 'shape') else add_seqs[1],
                    label=plot_kwargs.pop('label', None),
                    **plot_kwargs
                )
        
        return fig, ax

    def plot_3d_surface(
        self,
        embeddings: np.ndarray,
        ax = None,
        method: str = 'kde',
        bins: int = 50,
        scale: str = 'linear',
        normalize: bool = False,
        additional_points: Optional[list[tuple[list[str], dict]]] = None,
        **kwargs
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot a 3D surface using PC1, PC2, PC3 as a density surface.
        Args:
            embeddings: (N, 3) array
            ax: 3D matplotlib Axes (optional, will create if None)
            method: 'kde' or 'hist'
            bins: number of bins for histogram
            cmap: colormap
            scale: 'linear', 'log', 'sqrt', or 'power' (power=0.5)
            normalize: whether to normalize to [0,1]
            additional_points: Optional list of (sequences, plot_kwargs) tuples
                for additional points to plot on the surface
            **kwargs: passed to plot_surface
        Returns:
            tuple of (figure, axes)
        """
        # Initialize figure and axes if not provided
        if ax is None:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure
            
        x, y, z = embeddings[:, 0], embeddings[:, 1], embeddings[:, 2]
        xi = np.linspace(x.min(), x.max(), bins)
        yi = np.linspace(y.min(), y.max(), bins)
        xi, yi = np.meshgrid(xi, yi)
        if method == 'kde':
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(np.vstack([x, y]))
            zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
        elif method == 'hist':
            from scipy.ndimage import gaussian_filter
            H, xedges, yedges = np.histogram2d(x, y, bins=bins)
            zi = gaussian_filter(H.T, sigma=2)
            xi, yi = np.meshgrid(
                (xedges[:-1] + xedges[1:]) / 2,
                (yedges[:-1] + yedges[1:]) / 2
            )
        else:
            raise ValueError("method must be 'kde' or 'hist'")
        
        # Apply scaling
        zi_scaled = self._apply_scaling(zi, scale, normalize)
        
        surf = ax.plot_surface(xi, yi, zi_scaled, edgecolor='none', **kwargs)
        ax.set(xlabel='PC1', ylabel='PC2', zlabel=f'Density ({scale})')
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Add additional points if provided
        if additional_points is not None:
            for add_seqs, plot_kwargs in additional_points:
                # For additional points, we need to get their embeddings
                # This assumes the sequences are in the same embedding space
                ax.scatter(
                    add_seqs[:, 0] if hasattr(add_seqs, 'shape') else add_seqs[0],
                    add_seqs[:, 1] if hasattr(add_seqs, 'shape') else add_seqs[1],
                    add_seqs[:, 2] if hasattr(add_seqs, 'shape') else add_seqs[2],
                    label=plot_kwargs.pop('label', None),
                    **plot_kwargs
                )
        
        return fig, ax

    def _apply_scaling(self, zi: np.ndarray, scale: str, normalize: bool) -> np.ndarray:
        """
        Apply scaling and normalization to density values.
        Args:
            zi: density values
            scale: 'linear', 'log', 'sqrt', or 'power'
            normalize: whether to normalize to [0,1]
        Returns:
            scaled density values
        """
        zi_scaled = zi.copy()
        
        # Apply scaling
        if scale == 'log':
            # Add small constant to avoid log(0)
            zi_scaled = np.log(zi_scaled + 1e-10)
        elif scale == 'sqrt':
            zi_scaled = np.sqrt(zi_scaled)
        elif scale == 'power':
            zi_scaled = np.power(zi_scaled, 0.5)
        elif scale == 'linear':
            pass  # no scaling
        else:
            raise ValueError("scale must be 'linear', 'log', 'sqrt', or 'power'")
        
        # Apply normalization
        if normalize:
            zi_min, zi_max = zi_scaled.min(), zi_scaled.max()
            if zi_max > zi_min:
                zi_scaled = (zi_scaled - zi_min) / (zi_max - zi_min)
        
        return zi_scaled

    def plot_pca_surface_by_layer(
        self,
        sequences: list[str],
        config: DimensionalityReductionConfig,
        layers: Optional[list[str]] = None,
        surface_method: str = 'density',
        **kwargs
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Plot PCA embeddings as surfaces by layer.

        Args:
            sequences: List of sequences to analyze
            config: Configuration for dimensionality reduction
            layers: Layers to analyze. If None, uses all layers.
            surface_method: Surface method ('density', 'contour', 'heatmap')
            **kwargs: Additional arguments for surface plotting

        Returns:
            Figure and axes objects
        """
        if layers is None:
            layers = self.layers
        elif not isinstance(layers, list):
            layers = [layers]

        n_layers = len(layers)
        fig, axs = plt.subplots(
            ncols=n_layers,
            figsize=(n_layers * 4, 3)
        )

        if n_layers == 1:
            axs = [axs]

        for j, (ax, layer) in enumerate(zip(axs, layers)):
            # Compute PCA embeddings
            model, embeddings, seq_to_embedding = self.compute_pca_embeddings(
                sequences,
                layer,
                config
            )
            
            # Plot surface
            self.visualizer.plot_2d_surface(
                ax,
                embeddings,
                sequences,
                config,
                method=surface_method,
                **kwargs
            )

            ax.set(
                title=f'{layer.capitalize()} - {surface_method.title()}',
                xlabel='Dim 1' if j == 0 else '',
                ylabel='Dim 2'
            )

        title = (
            f'{self.model_component} Activations Surface\n '
            f'{config.sequence_method} method - {surface_method}'
        )
        fig.suptitle(title, y=0.95)
        sns.despine()
        plt.tight_layout()
        
        return fig, axs


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
        model: GPT,
        verbose: bool = False,
        visualizer_class=BaseVisualizer,
        config: Optional[InterpretabilityConfig] = None
    ):
        """Initialize the analyzer with a model."""
        self.model = model
        self.device = next(model.parameters()).device
        self.verbose = verbose
        self.vocab = ['R', 'r', 'L', 'l']
        self.stoi = {token: idx for idx, token in enumerate(self.vocab)}
        self.itos = {idx: token for idx, token in enumerate(self.vocab)}
        self.n_layers = model.config.n_layer
        self.n_heads = model.config.n_head
        self.n_embd = model.config.n_embd
        self.visualizer = visualizer_class(self)
        self.config = config or InterpretabilityConfig()
        self.hook_manager = HookManager(model, verbose)

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
                return torch.tensor(
                    token_ids, dtype=torch.long, device=self.device
                ).unsqueeze(0)
            return torch.tensor(token_ids, dtype=torch.long, device=self.device)
        else:
            token_ids = [
                [self.stoi[char] for char in sequence]
                for sequence in sequences
            ]
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
        probs = self.logits_to_probabilities(next_token_logits)
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

    def logits_to_probabilities(
        self, logits: torch.Tensor, dim: int = -1
    ) -> torch.Tensor:
        """Convert logits to probabilities using softmax.
        
        Args:
            logits: Input logits tensor
            dim: Dimension to apply softmax over (default: -1)
            
        Returns:
            Probability tensor
        """
        return F.softmax(logits, dim=dim)
    
    def extract_target_logits(
        self, 
        logits_by_layer: Dict[str, torch.Tensor], 
        target_position: int = -1
    ) -> Dict[str, torch.Tensor]:
        """Extract logits for a specific target position from layer logits.
        
        Args:
            logits_by_layer: Dictionary mapping layer names to logits with shape [batch, seq_len, vocab_size]
            target_position: Position of target token to extract
            
        Returns:
            Dictionary mapping layer names to target position logits with shape [batch, vocab_size]
        """
        return {
            layer_name: logits[:, target_position, :]
            for layer_name, logits in logits_by_layer.items()
        }

    def get_sequence_targets(
        self, sequences: list[str], target_position: int = -1
    ) -> list[int]:
        """Create target labels from sequences.
        
        Args:
            sequences: List of input sequences
            target_position: Position of target token
            
        Returns:
            List of target labels
        """
        targets = []
        for seq in sequences:
            if len(seq) > abs(target_position):
                target_token = seq[target_position]
                targets.append(self.stoi[target_token])
            else:
                raise ValueError(
                    f"Sequence {seq} is too short to have a token at position {target_position}"
                )
        return targets

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

    def get_layer_activations(
        self,
        sequences: list[str],
        layer: str
    ) -> dict[str, np.ndarray]:
        """Get activations for a specific layer for a list of sequences.
        
        Args:
            sequences: List of sequences to analyze
            layer: Layer to get activations from
            
        Returns:
            Dictionary mapping sequences to their activation vectors for the
            specified layer. Structure: {seq: activation_array}
        """
        # Get all layer activations
        all_activations = self.get_activations(sequences)
        
        # Extract just the requested layer
        return {
            seq: acts[layer]
            for seq, acts in all_activations.items()
        }

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
            X = np.concatenate([
                act.reshape(1, -1) for act in activations.values()
            ], axis=0)

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

    def _extract_internal_states(
        self,
        sequences: List[str],
        components: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Extract internal states from any model components.
        
        Args:
            sequences: List of input sequences
            components: List of component names to capture
            
        Returns:
            Dictionary mapping component names to their states as {layer: tensor[batch_size, seq_len, hidden_dim]}
        """
        # Tokenize sequences
        tokens = self._prepare_input(sequences, batch=True)
        
        # Store original training state
        was_training = self.model.training
        
        # Set up hooks
        self.hook_manager.setup_hooks(components)
        
        try:
            # Set to eval mode and disable gradients
            self.model.eval()
            with torch.no_grad():
                self.model(tokens)
                # Validate hook outputs
                self.hook_manager.validate_outputs(len(sequences))
        finally:
            # Restore original training state
            if was_training:
                self.model.train()
            
            # Get activations and cleanup
            activations = self.hook_manager.get_activations()
            self.hook_manager.cleanup()
        
        return activations

    def _extract_last_token_states(self, sequences: list[str], layers: list[str]):
        """Extract states from LastTokenGPT architecture."""
        # Import here to avoid circular imports
        from interpretability.last_token_hooks import \
            LastTokenActivationAnalyzer

        # Use specialized analyzer for LastTokenGPT
        last_token_analyzer = LastTokenActivationAnalyzer(
            self.model, self.model_config, layers
        )
        
        # Get activations and convert to expected format
        raw_activations = last_token_analyzer.get_activations(sequences)
        
        # Convert format: {seq: {layer: tensor}} -> {layer: tensor[batch, seq, hidden]}
        converted_activations = {}
        for layer in layers:
            if layer in raw_activations[sequences[0]]:
                # Stack activations from all sequences
                layer_activations = [
                    raw_activations[seq][layer] for seq in sequences
                ]
                converted_activations[layer] = torch.stack(layer_activations)
        
        return converted_activations
