"""Helper functions for analyzing transformer model components."""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from activations import EmbeddingAnalyzer
from analyzer import BaseAnalyzer, BaseVisualizer

import interpretability.interp_helpers as interp


class AttentionVisualizer(BaseVisualizer):
    """Helper class for visualizing attention patterns."""
    
    def _get_plot_func(
        self,
        layer_idx: Optional[int],
        head_idx: Optional[int]
    ) -> callable:
        """Get the appropriate plotting function based on indices.
        
        Args:
            layer_idx: Optional layer index
            head_idx: Optional head index
            
        Returns:
            Callable plotting function
        """
        if layer_idx is not None and head_idx is not None:
            return self.plot_attention_head
        elif layer_idx is not None:
            return self.plot_layer_heads
        else:
            return self.plot_all_layers
    
    def plot_attention_head(
        self,
        sequence: str,
        att_map: list[np.ndarray],
        head_idx: int,
        layer_idx: Optional[int] = 0,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Axes:
        """Plot attention pattern for a single head.
        
        Args:
            sequence: Input sequence string
            att_map: Attention map tensor of shape 
                [batch, heads, seq_len, seq_len]
            head_idx: Index of the attention head to plot
            layer_idx: Layer index to plot
            ax: Matplotlib axes to plot on
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 3))
        
        att = att_map[layer_idx][0, head_idx]
        if np.any(att < 0):  # For diffing attention maps
            cmap = 'RdBu'
            vmin = -1
        else:  # For raw attention map
            cmap = 'viridis'
            vmin = 0
            
        sns.heatmap(
            att,
            annot=False,
            fmt='.2f',
            cmap=cmap,
            xticklabels=list(sequence),
            yticklabels=list(sequence),
            vmin=vmin,
            vmax=1,
            ax=ax,
            cbar=False,
            square=True,
            annot_kws={"size": 8},
            mask=np.triu(np.ones_like(att), k=1),  # Hide upper diagonal
        )
        
        # Only show labels based on context
        if self.plot_context == 'multi_sequence':
            # In multi-sequence context, only show labels on first sequence and last layer
            if self.seq_idx == 0 and head_idx == 0:  # First sequence
                ax.set(ylabel='Query (attending)')
            if layer_idx == len(att_map) - 1:  # Last layer
                ax.set(xlabel='Key (attended to)')
            ax.set(title=f'Layer {layer_idx}, Head {head_idx}')
        else:
            # In single sequence context, show all labels
            ax.set(
                title=f'Layer {layer_idx}, Head {head_idx}',
                xlabel='Key (attended to)',
                ylabel='Query (attending)'
            )
            
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        return ax
    
    def plot_layer_heads(
        self,
        sequence: str,
        att_map: list[np.ndarray],
        head_idx: list[int] = None,
        max_heads: Optional[int] = None,
        fig: Optional[plt.Figure] = None,
        **kwargs
    ) -> plt.Figure:
        """Plot all attention heads in a single layer as columns.
        
        Args:
            att_map: Attention map tensor of shape 
                [batch, heads, seq_len, seq_len]
            sequence: Input sequence string
            layer_idx: Index of the layer
            max_heads: Optional maximum number of heads to plot
            **kwargs: Additional arguments passed to plot_attention_head
            
        Returns:
            Matplotlib figure object
        """
        n_heads = self.analyzer.model.config.n_head
        if max_heads is not None:
            n_heads = min(n_heads, max_heads)
        
        if fig is None:
            fig = plt.figure(figsize=(3*n_heads, 3))

        if head_idx is None:
            head_idx = range(n_heads)
        else:
            head_idx = [head_idx]

        axes = fig.subplots(1, n_heads)
        axes = [axes] if n_heads == 1 else axes
        for ax, h in zip(axes, head_idx):
            ax = self.plot_attention_head(
                sequence, att_map, h, ax=ax, **kwargs
            )
        if self.plot_context != 'multi_sequence':
            plt.tight_layout(rect=[0, 0, 1, 0.98])
        return fig
    
    def plot_all_layers(
        self,
        sequence: str,
        att_map: list[np.ndarray],
        layer_idx: list[int] = None,
        fig: Optional[plt.Figure] = None,
        **kwargs
    ) -> List[plt.Figure]:
        """Plot attention patterns for all layers.
        
        Each layer is plotted in a separate figure with its heads as columns.
        
        Args:
            attention_maps: List of attention maps, one per layer
            sequence: Input sequence string
            max_heads: Optional maximum number of heads to plot
            **kwargs: Additional arguments passed to plot_layer_heads
            
        Returns:
            List of matplotlib figures, one per layer
        """
        n_heads = self.analyzer.model.config.n_head
        n_layers = len(att_map)

        if fig is None:
            fig = plt.figure(figsize=(3*n_layers, n_heads*3))
        subfigs = fig.subfigures(nrows=n_layers, wspace=0.0, hspace=0.7)
        subfigs = [subfigs] if n_layers == 1 else subfigs

        if layer_idx is None:
            layer_idx = range(n_layers)

        for subfig, l in zip(subfigs, layer_idx):
            subfig = self.plot_layer_heads(
                sequence,
                att_map,
                layer_idx=l,
                fig=subfig,
                **kwargs)
            
        if self.plot_context != 'multi_sequence':
            # Adjust spacing for all subfigures at once
            fig.subplots_adjust(wspace=-0.7)
        else:
            fig.subplots_adjust(wspace=0.1)
        return fig


class AttentionAnalyzer(BaseAnalyzer):
    """Analyzer for attention patterns in transformer models."""
    
    def __init__(self, model, model_config):
        super().__init__(model, visualizer_class=AttentionVisualizer)
        self.model_config = model_config
        self.n_layers = model_config.n_layer
        self.layers = [f'qk_{i}' for i in range(self.n_layers)]

    def get_attention_maps(self, sequence: str) -> List[np.ndarray]:
        """Get attention weights for each layer and head."""
        
        T = len(sequence)
        assert T <= self.model.config.block_size, (
            f"Sequence length {T} exceeds block size "
            f"{self.model.config.block_size}"
        )
        # Use unified hook system to get QK attention weights
        qk_components = self.layers
        qk_states = self._extract_internal_states([sequence], qk_components)
        
        # Convert to the expected format: List[np.ndarray] where each array is [1, n_heads, seq_len, seq_len]
        attention_maps = []
        for i in range(self.n_layers):
            if f'qk_{i}' in qk_states:
                attention_maps.append(qk_states[f'qk_{i}'])
            else:
                print(f'No attention map found for layer {i}')

        return attention_maps

    def get_activations(self, sequences: list[str], layer: str) -> np.ndarray:
        raise NotImplementedError("Need to implement multi-sequence activation grabbing")
        return self.get_attention_maps(sequences, layer)
        
    def diff_attention_from_reference(
        self,
        att_map: np.ndarray,
        ref_seq: str = None,
        **kwargs
    ):
        if ref_seq is None:
            raise ValueError("Reference sequence must be provided")
        ref_maps = self.get_attention_maps(ref_seq)
        diff_maps = []
        for layer_idx in range(self.n_layers):
            # Create a new array with same shape as att_map[layer_idx]
            layer_diff = np.zeros_like(att_map[layer_idx])
            for head_idx in range(self.n_heads):
                layer_diff[0, head_idx] = (
                    att_map[layer_idx][0, head_idx] - 
                    ref_maps[layer_idx][0, head_idx]
                )
            diff_maps.append(layer_diff)

        return diff_maps
    
    def attention_factorization(
        self,
        sequence: str,
        layer_idx: int = 0,
        head_idx: int = 0,
        nth_feature: int = 0,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Factorize attention patterns using SVD.
        
        Args:
            sequence: Input sequence
            layer_idx: Layer index
            head_idx: Head index
            k: Index of singular value to use
            
        Returns:
            Tuple of (left features, right features) in PCA space
        """
        # Calculate A = QKT
        A = self.get_attention_maps(sequence)[layer_idx][0, head_idx]
        X = self.get_embeddings(sequence, flatten=False)
        
        # SVD(A) = USVT
        U, S, V = np.linalg.svd(A)
        features_left = X.T @ U[:, nth_feature]
        features_right = X.T @ V[:, nth_feature] 
        
        return features_left, S, features_right

    def project_attention_features(self, block_sequence, pca=None, sequences=None, config=None, **kwargs):
        if pca is None:
            pca = self._get_embedding_pca(sequences, config)

        V_pca = pca.components_

        # Project each singular vector
        f_proj_left = []
        f_proj_right = []
        for i in range(len(block_sequence)):
            features_left, S, features_right = self.attention_factorization(
                block_sequence,
                nth_feature=i,
                **kwargs
            )
            f_proj_left.append(V_pca @ features_left)
            f_proj_right.append(V_pca @ features_right)

        return f_proj_left, S, f_proj_right

    def _get_embedding_pca(self, sequences, config):
        embed_analyzer = EmbeddingAnalyzer(self.model, self.model_config)
        pca, _, _ = embed_analyzer.compute_pca_embeddings(sequences, 'embed', config)
        return pca

    def plot_attention_features(self, sequences, block_sequences, config, **kwargs):
        pca = self._get_embedding_pca(sequences, config)
        projections = {'left (U)': [], 'right (V)': []}
        eigenvalues = []
        block_sequences = interp.trim_leading_duplicates(block_sequences)
        
        for seq in block_sequences:
            f_proj_left, S, f_proj_right = self.project_attention_features(seq, pca, **kwargs)
            projections['left (U)'].append(f_proj_left)
            projections['right (V)'].append(f_proj_right)
            eigenvalues.append(S)
            
        # Create figure with subplots for each singular vector
        n_singular_vectors = len(eigenvalues[0])
        fig, axs = plt.subplots(nrows=n_singular_vectors, ncols=2, 
                               figsize=(8, 3*n_singular_vectors),
                               sharex='col', sharey='col')
        
        # Plot eigenvalues in first subplot
        eigenvalues_df = pd.DataFrame(eigenvalues, 
                                    columns=[f'SV{i+1}' for i in range(n_singular_vectors)])
        eigenvalues_df['seq'] = block_sequences
        eigenvalues_melted = eigenvalues_df.melt(id_vars=['seq'], 
                                               var_name='SV', 
                                               value_name='Value')
        
        # Plot each singular vector's projections
        for sv_idx in range(n_singular_vectors):
            # Plot left singular vector projections
            left_projs = []
            for seq_idx, seq_projs in enumerate(projections['left (U)']):
                left_projs.append({
                    'seq': block_sequences[seq_idx],
                    'PC1': seq_projs[sv_idx][0],
                    'PC2': seq_projs[sv_idx][1]
                })
            left_df = pd.DataFrame(left_projs)
            
            sns.scatterplot(data=left_df, x='PC1', y='PC2',
                          hue='seq', palette='magma',
                          ax=axs[sv_idx, 0])
            axs[sv_idx, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axs[sv_idx, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
            axs[sv_idx, 0].set(title=f'Left (U) - SV{sv_idx+1}',
                              xlabel='PC1' if sv_idx == n_singular_vectors-1 else '',
                              ylabel='PC2')
            
            # Plot right singular vector projections
            right_projs = []
            for seq_idx, seq_projs in enumerate(projections['right (V)']):
                right_projs.append({
                    'seq': block_sequences[seq_idx],
                    'PC1': seq_projs[sv_idx][0],
                    'PC2': seq_projs[sv_idx][1]
                })
            right_df = pd.DataFrame(right_projs)
            
            sns.scatterplot(data=right_df, x='PC1', y='PC2',
                          hue='seq', palette='magma',
                          ax=axs[sv_idx, 1])
            axs[sv_idx, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axs[sv_idx, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
            axs[sv_idx, 1].set(title=f'Right (V) - SV{sv_idx+1}',
                              xlabel='PC1' if sv_idx == n_singular_vectors-1 else '',
                              ylabel='PC2')
            
            # Add singular value to title
            sv_value = eigenvalues[0][sv_idx]
            axs[sv_idx, 0].set_title(f'Left (U) - SV{sv_idx+1} (σ={sv_value:.2f})')
            axs[sv_idx, 1].set_title(f'Right (V) - SV{sv_idx+1} (σ={sv_value:.2f})')
            
            # Only show legend for last row
            if sv_idx < n_singular_vectors-1:
                axs[sv_idx, 0].legend().remove()
                axs[sv_idx, 1].legend().remove()
        
        fig.suptitle("Projection of Attention Features into Embedding Layer PCA Space")
        plt.tight_layout()
        return fig

    def plot_attention(
        self,
        sequence: str,
        layer_idx: Optional[int] = None,
        head_idx: Optional[int] = None,
        as_diff: bool = False,
        **kwargs
    ) -> Union[plt.Figure, List[plt.Figure]]:
        """Plot attention patterns with flexible granularity.
        
        Args:
            sequence: Input sequence string
            layer_idx: Optional layer index to analyze
            head_idx: Optional head index to analyze
            **kwargs: Additional arguments passed to plotting functions
            
        Returns:
            Matplotlib figure(s) showing attention patterns
        """
        
        attention_maps = self.get_attention_maps(sequence)
        probs = self.predict_next_token_probs(sequence)
        
        next_token = self.predict_next_token(sequence, probs)

        if as_diff:
            attention_maps = self.diff_attention_from_reference(
                attention_maps,
                **kwargs
            )

        plot_func = self.visualizer._get_plot_func(layer_idx, head_idx)
        fig = plot_func(
            sequence,
            attention_maps,
            layer_idx=layer_idx,
            head_idx=head_idx,
            **kwargs
        )
        return next_token, probs, fig

    def plot_attention_multiple_sequences(
        self,
        sequences: List[str],
        max_sequences: int = 5,
        layer_idx: int = None,
        head_idx: int = None,
        **kwargs
    ) -> None:
        """Analyze attention patterns for multiple sequences.
        
        Args:
            sequences: List of sequences to analyze
            max_sequences: Maximum number of sequences to analyze
            layer_idx: Optional layer index to analyze
            head_idx: Optional head index to analyze
            **kwargs: Additional arguments passed to plotting functions
        """
        sequences_to_analyze = sequences[:max_sequences]
        
        # Set context for multi-sequence plotting
        self.visualizer.plot_context = 'multi_sequence'
        
        n_rows = self.n_layers
        n_cols = len(sequences_to_analyze)
        
        # Create figure with minimal margins
        fig = plt.figure(
            figsize=(3*n_cols, 3*n_rows + 0.3)
        )  # Add extra height for prob row
        fig.subplots_adjust(left=0, right=1, top=0.98, bottom=0)
        
        # Create subfigures with increased spacing between columns
        subfigs = fig.subfigures(
            ncols=n_cols,
            wspace=0.2,  # Increase spacing between sequence columns
            hspace=0.0
        )
        
        # Reduce margin factor to prevent overlap
        margin_factor = np.clip(0.02 * len(sequences_to_analyze), 0.0, 0.2)
        for i, (subfig, seq) in enumerate(
            zip(subfigs, sequences_to_analyze)
        ):
            self.visualizer.seq_idx = i
            # Create subplots within each subfigure with minimal spacing
            
            subfig.subplots_adjust(
                left=0 + margin_factor,  # Slightly reduce left margin
                right=1 - margin_factor,  # Slightly reduce right margin
                top=0.98,
                bottom=0
            )
            
            # Create two subfigures within each subfigure
            top_subfig, bottom_subfig = subfig.subfigures(
                2, 1,
                height_ratios=[
                    1, 
                    (n_rows*len(seq)+1)
                ],  # Make prob row same height as one attention row
                hspace=0.1
            )
            
            # Plot attention maps in bottom subfigure
            next_token, probs, bottom_subfig = self.plot_attention(
                seq, 
                layer_idx=layer_idx, 
                head_idx=head_idx,
                fig=bottom_subfig,
                **kwargs
            )
            
            # Plot probability heatmap in top subfigure
            gs = top_subfig.add_gridspec(1, 4)  # Divide width into 4 parts
            ax_prob = top_subfig.add_subplot(gs[0, 1:3])
            self.visualizer.plot_token_probs(probs, ax=ax_prob)
            
            subfig.suptitle(
                f'{seq} → ({next_token})', 
                y=1.1, 
                fontsize=12
            )
        
        # Reset context
        self.visualizer.plot_context = None
        self.visualizer.seq_idx = None
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        return fig


