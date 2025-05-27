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
        if (self.plot_context != 'multi_sequence') or (self.seq_idx == 0):
            ax.set(
                title=f'Layer {layer_idx}, Head {head_idx}',
                xlabel='Key (attended to)',
                ylabel='Query (attending)'
            )
        else:
            ax.set(
                title=f'Layer {layer_idx}, Head {head_idx}',
                xlabel='Key (attended to)',
                ylabel=''
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
        subfigs = fig.subfigures(nrows=n_layers, wspace=0.0, hspace=0.0)
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
        return fig


class AttentionAnalyzer(BaseAnalyzer):
    """Analyzer for attention patterns in transformer models."""
    
    def __init__(self, model, model_config):
        super().__init__(model, visualizer_class=AttentionVisualizer)
        self.model_config = model_config

    def _attention_hook(self, module, input, output) -> np.ndarray:
        """Hook function to extract attention weights."""
        x = input[0]
        B, T, C = x.size()
        
        # Compute Q, K, V
        qkv = module.c_attn(x)
        q, k, v = qkv.split(module.n_embd, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, module.n_head, C // module.n_head).transpose(1, 2)
        k = k.view(B, T, module.n_head, C // module.n_head).transpose(1, 2)
        
        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / (C ** 0.5))
        att = att.masked_fill(
            module.bias[:, :, :T, :T] == 0, 
            float('-inf')
        )
        att = F.softmax(att, dim=-1)
        
        return att.detach().cpu().numpy()
    
    def get_attention_maps(self, sequence: str) -> List[np.ndarray]:
        """Get attention weights for each layer and head."""
        input_ids = self._prepare_input(sequence, batch=True)
        
        B, T = input_ids.size()
        assert T <= self.model.config.block_size, (
            f"Sequence length {T} exceeds block size "
            f"{self.model.config.block_size}"
        )
        
        attention_maps = []
        hooks = []
        
        # Register hooks for each attention layer
        for block in self.model.transformer.h:
            hook = block.attn.register_forward_hook(
                lambda m, i, o: attention_maps.append(
                    self._attention_hook(m, i, o)
                )
            )
            hooks.append(hook)
        
        try:
            with torch.no_grad():
                self.model(input_ids)
        finally:
            for hook in hooks:
                hook.remove()
        
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
        features_left = X.T @ U[:, 0]
        features_right = X.T @ V[:, 0] 
        
        return features_left, features_right

    def project_attention_features(self, block_sequence, pca=None, sequences=None, config=None, **kwargs):
        
        if pca is None:
            pca = self._get_embedding_pca(sequences, config)

        V_pca = pca.components_

        features_left, features_right = self.attention_factorization(block_sequence, **kwargs)

        f0_proj_left = V_pca @ features_left
        f0_proj_right = V_pca @ features_right

        return f0_proj_left, f0_proj_right

    def _get_embedding_pca(self, sequences, config):
        embed_analyzer = EmbeddingAnalyzer(self.model, self.model_config)
        pca, _, _ = embed_analyzer.compute_pca_embeddings(sequences, 'embed', config)
        return pca

    def plot_attention_features(self, sequences, block_sequences, config, **kwargs):
        
        pca = self._get_embedding_pca(sequences, config)
        projections = {'left': [], 'right': []}
        block_sequences = interp.trim_leading_duplicates(block_sequences)
        for seq in block_sequences:
            f0_proj_left, f0_proj_right = self.project_attention_features(seq, pca, **kwargs)
            projections['left'].append(f0_proj_left)
            projections['right'].append(f0_proj_right)

        fig, axs = plt.subplots(ncols=2, figsize=(5, 3), sharex=True, sharey=True)
        for ax, (singular_vector, values) in zip(axs, projections.items()):
            tmp = pd.DataFrame(values, columns=[f'PC{i}' for i in range(1, config.n_components+1)])
            tmp['seq'] = interp.trim_leading_duplicates(block_sequences)

            # Melt the dataframe to get PCs on x-axis
            tmp_melted = tmp.melt(id_vars=['seq'], var_name='PC', value_name='Projection')

            sns.barplot(data=tmp_melted, x='PC', y='Projection', hue='seq', palette='magma', ax=ax, legend=singular_vector=='right')
            ax.axhline(y=0, color='k')
            ax.set(title=f'{singular_vector.capitalize()} features', xlabel='', ylabel='Projection Magnitude')

        axs[1].legend(bbox_to_anchor=(1, 0), loc='lower left')
        axs[1].set(ylim=(-2.0, 2.0))
        fig.suptitle("Projection of Attention Feature Direction (f0)\ninto Embedding Layer PCA Space")

        plt.tight_layout()

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
        if self.model.config.n_head > 1:
            for seq in sequences_to_analyze:
                self.plot_attention(
                    seq, 
                    layer_idx=layer_idx, 
                    head_idx=head_idx, 
                    **kwargs
                )
        else:
            # Set context for multi-sequence plotting
            self.visualizer.plot_context = 'multi_sequence'
            
            n_rows = self.n_layers
            n_cols = len(sequences_to_analyze)
            
            # Create figure with minimal margins
            fig = plt.figure(
                figsize=(3*n_cols, 3*n_rows + 0.3)
            )  # Add extra height for prob row
            fig.subplots_adjust(left=0, right=1, top=0.98, bottom=0)
            
            # Create subfigures with minimal spacing
            subfigs = fig.subfigures(
                ncols=n_cols,
                wspace=0.0,
                hspace=0.0
            )
            
            margin_factor = np.clip(0.05 * len(sequences_to_analyze), 0.0, 0.45)
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


