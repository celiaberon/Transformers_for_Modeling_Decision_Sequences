"""Helper functions for analyzing transformer model components."""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from activations import EmbeddingAnalyzer
from analyzer import BaseAnalyzer, BaseVisualizer
from sklearn.decomposition import PCA

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
        if np.any(att > 1) or np.any(att < 0):
            cmap = 'viridis'
            cbar = True
            vmin = att.min()
            vmax = att.max()
        elif np.any(att < 0):  # For diffing attention maps
            cmap = 'RdBu'
            vmin = -1
            vmax = 1
            cbar = False
        else:  # For raw attention map
            cmap = 'viridis'
            vmin = 0
            vmax = 1
            cbar = False
            
        sns.heatmap(
            att,
            annot=False,
            fmt='.2f',
            cmap=cmap,
            xticklabels=list(sequence),
            yticklabels=list(sequence),
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            cbar=cbar,
            square=True,
            annot_kws={"size": 8},
            mask=np.triu(np.ones_like(att), k=1),  # Hide upper diagonal
        )
        
        # Only show labels based on context
        if self.plot_context == 'multi_sequence':
            # In multi-sequence context, only show labels on first sequence
            # and last layer
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

    def plot_attention_features(
        self, sequences, block_sequences, config, **kwargs
    ):
        pca = self.analyzer._get_embedding_pca(sequences, config)
        projections = {'left (U)': [], 'right (V)': []}
        eigenvalues = []
        block_sequences = interp.trim_leading_duplicates(block_sequences)
        for seq in block_sequences:
            f_proj_left, S, f_proj_right = (
                self.analyzer.project_attention_features(
                    seq, pca, **kwargs
                )
            )
            projections['left (U)'].append(f_proj_left)
            projections['right (V)'].append(f_proj_right)
            eigenvalues.append(S)
        # Create figure with subplots for each singular vector
        n_singular_vectors = len(eigenvalues[0])
        fig, axs = plt.subplots(
            nrows=n_singular_vectors, ncols=2,
            figsize=(8, 3 * n_singular_vectors),
            sharex='col', sharey='col'
        )
        # Plot eigenvalues in first subplot
        eigenvalues_df = pd.DataFrame(
            eigenvalues,
            columns=[f'SV{i+1}' for i in range(n_singular_vectors)]
        )
        eigenvalues_df['seq'] = block_sequences
        eigenvalues_melted = eigenvalues_df.melt(
            id_vars=['seq'], var_name='SV', value_name='Value'
        )
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
            sns.scatterplot(
                data=left_df, x='PC1', y='PC2',
                hue='seq', palette='magma', ax=axs[sv_idx, 0]
            )
            axs[sv_idx, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axs[sv_idx, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
            axs[sv_idx, 0].set(
                title=f'Left (U) - SV{sv_idx+1}',
                xlabel='PC1' if sv_idx == n_singular_vectors-1 else '',
                ylabel='PC2'
            )
            # Plot right singular vector projections
            right_projs = []
            for seq_idx, seq_projs in enumerate(projections['right (V)']):
                right_projs.append({
                    'seq': block_sequences[seq_idx],
                    'PC1': seq_projs[sv_idx][0],
                    'PC2': seq_projs[sv_idx][1]
                })
            right_df = pd.DataFrame(right_projs)
            sns.scatterplot(
                data=right_df, x='PC1', y='PC2',
                hue='seq', palette='magma', ax=axs[sv_idx, 1]
            )
            axs[sv_idx, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axs[sv_idx, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
            axs[sv_idx, 1].set(
                title=f'Right (V) - SV{sv_idx+1}',
                xlabel='PC1' if sv_idx == n_singular_vectors-1 else '',
                ylabel='PC2'
            )
            # Add singular value to title
            sv_value = eigenvalues[0][sv_idx]
            axs[sv_idx, 0].set_title(
                f'Left (U) - SV{sv_idx+1} (σ={sv_value:.2f})'
            )
            axs[sv_idx, 1].set_title(
                f'Right (V) - SV{sv_idx+1} (σ={sv_value:.2f})'
            )
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
        component: str = 'qk_attn_softmax',
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
        attention_maps = self.analyzer.get_attention_maps(sequence, component)
        probs = self.analyzer.predict_next_token_probs(sequence)
        next_token = self.analyzer.predict_next_token(sequence, probs)

        if as_diff:
            attention_maps = self.analyzer.diff_attention_from_reference(
                attention_maps,
                **kwargs
            )

        plot_func = self._get_plot_func(layer_idx, head_idx)
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
        component: str = 'qk_attn_softmax',
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
        self.plot_context = 'multi_sequence'
        
        n_rows = self.analyzer.n_layers
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
            self.seq_idx = i
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
                component=component,
                **kwargs
            )
            
            # Plot probability heatmap in top subfigure
            gs = top_subfig.add_gridspec(1, 4)  # Divide width into 4 parts
            ax_prob = top_subfig.add_subplot(gs[0, 1:3])
            self.plot_token_probs(probs, ax=ax_prob)
            
            subfig.suptitle(
                f'{seq} → ({next_token})', 
                y=1.1, 
                fontsize=12
            )
        
        # Reset context
        self.plot_context = None
        self.seq_idx = None
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        return fig

    def _fit_pca_on_states(
        self,
        pre_states: List[np.ndarray],
        post_states: Optional[List[np.ndarray]] = None,
        n_components: int = 2,
        last_token_only: bool = False,
        fit_on_both: bool = False,
        **kwargs
    ) -> np.ndarray:
        """Fit PCA on flattened pre-attention states.
        
        Args:
            pre_states: List of pre-attention states
            post_states: List of post-attention states (optional)
            n_components: Number of PCA components
            last_token_only: Whether to focus on last token only
            fit_on_both: Whether to fit PCA on both pre and post states collectively
            
        Returns:
            Tuple of (PCA components matrix, flattened data for centering)
        """
        if last_token_only:
            pre_states = [state[-1, :] for state in pre_states]
            if fit_on_both and post_states is not None:
                post_states = [state[-1, :] for state in post_states]

        # Flatten all pre-attention states for PCA fitting
        flattened_pre = np.array([state.flatten() for state in pre_states])
        
        if fit_on_both and post_states is not None:
            # Also include post-states in PCA fitting
            flattened_post = np.array([state.flatten() for state in post_states])
            # Combine pre and post states for PCA fitting
            flattened_combined = np.vstack([flattened_pre, flattened_post])
            
            pca = PCA(n_components=n_components)
            pca.fit(flattened_combined)
            
            # Return pre-states mean for consistency with projection
            return pca.components_, flattened_combined
        else:
            # Original behavior: fit only on pre-states
            pca = PCA(n_components=n_components)
            pca.fit(flattened_pre)
            return pca.components_, flattened_pre

    def _compute_attention_motion(
        self,
        sequences: List[str],
        layer_idx: int,
        pre_component: Optional[str] = None,
        post_component: Optional[str] = None,
    ) -> List[np.ndarray]:
        """Compute attention-based motion for each sequence.
        
        This method extracts the residual stream states directly before and 
        after attention to get the exact motion caused by the attention mechanism.
        
        Args:
            sequences: List of input sequences
            layer_idx: Layer index to analyze
            pre_component: Component to use for pre-state (if None, uses default)
            post_component: Component to use for post-state (if None, uses default)
            
        Returns:
            List of (pre_attention_state, post_attention_state) tuples
        """
        motion_vectors = []
        
        # Default component selection
        if pre_component is None:
            pre_component = f'layer_{layer_idx-1}' if layer_idx > 0 else 'embed'
        if post_component is None:
            post_component = f'attn_{layer_idx}'
        
        for seq in sequences:
            # Extract residual stream states before and after attention
            states = self.analyzer._extract_internal_states([seq], [pre_component, post_component])
            
            # Get pre and post attention residual stream states
            R_pre = states[pre_component][0]  # (seq_len, d_model)
            R_post = states[post_component][0]  # (seq_len, d_model)
            
            motion_vectors.append((R_pre, R_post))
        
        return motion_vectors

    def _project_motion_to_pca(
        self,
        motion_vectors: List[tuple],
        pca_components: np.ndarray,
        pca_mean: np.ndarray,
        last_token_only: bool = False
    ) -> np.ndarray:
        """Project motion vectors to PCA space.
        
        Args:
            motion_vectors: List of (pre_state, post_state) tuples
            pca_components: PCA components matrix
            pca_mean: Mean of PCA training data
            last_token_only: Whether to focus on last token only
            
        Returns:
            Array of shape (N, 4) with [pre_x, pre_y, post_x, post_y] for each sequence
        """
        arrows = []
        
        for R_pre, R_post in motion_vectors:
            if last_token_only:
                # Focus only on the last token
                pre_flat = R_pre[-1, :]  # (d_model,)
                post_flat = R_post[-1, :]  # (d_model,)
            else:
                # Use entire sequence (flattened)
                pre_flat = R_pre.flatten()  # (seq_len * d_model,)
                post_flat = R_post.flatten()  # (seq_len * d_model,)
            
            # Center and project
            pre_xy = (pre_flat - pca_mean) @ pca_components.T
            post_xy = (post_flat - pca_mean) @ pca_components.T
            
            arrows.append([*pre_xy, *post_xy])
        
        return np.array(arrows)

    def _plot_pca_motion(
        self,
        arrows: np.ndarray,
        sequences: List[str],
        pre_component: str,
        post_component: str,
        ax: plt.Axes,
        sub_sample: Optional[int] = None,
        last_token_only: bool = False,
        **kwargs
    ) -> None:
        """Plot the motion visualization in PCA space.
        
        Args:
            arrows: Array of shape (N, 4) with motion vectors
            sequences: List of input sequences
            pre_component: Name of pre-state component
            post_component: Name of post-state component
            ax: Matplotlib axes to plot on
            sub_sample: Optional subsampling factor
            last_token_only: Whether to focus on last token only
        """
        # Plot starting points
        ax.scatter(
            arrows[:, 0], arrows[:, 1], 
            alpha=0.6,
            label="Pre-state",
            s=20
        )

        # Sub-sample arrows and labels (but keep them matched)
        if sub_sample is not None:
            idcs = np.random.choice(len(sequences), sub_sample, replace=False)
            arrows_sub = arrows[idcs, :]
            sequences_sub = [sequences[i] for i in idcs]
        else:
            arrows_sub = arrows
            sequences_sub = sequences

        # Plot motion arrows (subsampled)
        ax.quiver(
            arrows_sub[:, 0], arrows_sub[:, 1],
            arrows_sub[:, 2] - arrows_sub[:, 0], arrows_sub[:, 3] - arrows_sub[:, 1],
            angles='xy', scale_units='xy', scale=1, 
            alpha=0.7, width=0.003
        )
        
        # Add sequence labels (subsampled, matching arrows)
        for i, seq in enumerate(sequences_sub):
            if last_token_only:
                x = arrows_sub[i, 2]
                y = arrows_sub[i, 3]
            else:
                x = arrows_sub[i, 0]
                y = arrows_sub[i, 1]
            ax.annotate(
                seq,
                (x, y),
                xytext=(5, 5), textcoords='offset points',
                fontsize=10, alpha=0.8
            )
        
        # Set axis limits to encompass all arrow start and end points
        all_x = np.concatenate([arrows[:, 0], arrows_sub[:, 2]])
        all_y = np.concatenate([arrows[:, 1], arrows_sub[:, 3]])
        
        # Add some padding (5% of the range)
        x_range = all_x.max() - all_x.min()
        y_range = all_y.max() - all_y.min()
        padding_x = x_range * 0.05
        padding_y = y_range * 0.05

        ax.set(
            xlabel="PC1",
            ylabel="PC2",
            title=f"Motion of sequences in PCA plane\n"
                  f"({pre_component} → {post_component})",
            xlim=(all_x.min() - padding_x, all_x.max() + padding_x),
            ylim=(all_y.min() - padding_y, all_y.max() + padding_y)
        )
        ax.legend()
        sns.despine()
        return ax

    def plot_attention_motion_in_pca(
        self,
        sequences: List[str],
        layer_idx: int = 0,
        pre_component: Optional[str] = None,
        post_component: Optional[str] = None,
        last_token_only: bool = False,
        fit_on_both: bool = False,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        """Plot how attention updates move sequences through PCA embedding space.
        
        This visualization shows the motion of sequences in the PCA plane when
        attention updates the residual stream. Each arrow shows the movement from
        pre-attention to post-attention state using exact model computations.
        
        Args:
            sequences: List of sequences to analyze
            layer_idx: Layer index to analyze (used for default component selection)
            pre_component: Component for pre-state (e.g., 'embed', 'layer_0_pre_attn')
            post_component: Component for post-state (e.g., 'attn_0', 'layer_0_post_attn')
            last_token_only: Whether to focus PCA on last token only
            fit_on_both: Whether to fit PCA on both pre and post states collectively
            ax: Optional axes to plot on
            **kwargs: Additional plotting arguments
            
        Returns:
            Figure with the motion visualization
            
        Examples:
            # Default: layer-to-layer motion
            plot_attention_motion_in_pca(sequences, layer_idx=1)
            
            # Embed to post-attention
            plot_attention_motion_in_pca(sequences, pre_component='embed', post_component='attn_0')
            
            # Within-attention motion
            plot_attention_motion_in_pca(sequences, pre_component='layer_0_pre_attn', post_component='layer_0_post_attn')
            
            # Focus on last token only
            plot_attention_motion_in_pca(sequences, last_token_only=True)
            
            # Fit PCA on both pre and post states for better motion visualization
            plot_attention_motion_in_pca(sequences, fit_on_both=True)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = ax.figure
        
        # Step 1: Extract pre and post states directly from model
        motion_vectors = self._compute_attention_motion(
            sequences, layer_idx, pre_component, post_component
        )
        
        # Get actual component names for plotting
        actual_pre = pre_component or (f'layer_{layer_idx-1}' if layer_idx > 0 else 'embed')
        actual_post = post_component or f'attn_{layer_idx}'
        
        # Step 2: Fit PCA on pre-attention states (and optionally post-states)
        pre_states = [R_pre for R_pre, _ in motion_vectors]
        post_states = [R_post for _, R_post in motion_vectors] if fit_on_both else None
        
        pca_components, flattened_data = self._fit_pca_on_states(
            pre_states, 
            post_states=post_states,
            last_token_only=last_token_only, 
            fit_on_both=fit_on_both,
            **kwargs
        )
        
        # Step 3: Project motion to PCA space
        arrows = self._project_motion_to_pca(
            motion_vectors, pca_components, flattened_data.mean(0),
            last_token_only
        )
        
        # Step 4: Plot the results
        self._plot_pca_motion(
            arrows, sequences, actual_pre, actual_post, ax,
            last_token_only=last_token_only, **kwargs
        )
        
        plt.tight_layout()
        return fig


class AttentionAnalyzer(BaseAnalyzer):
    """Analyzer for attention patterns in transformer models."""
    
    def __init__(self, model, model_config):
        super().__init__(model, visualizer_class=AttentionVisualizer)
        self.model_config = model_config
        self.n_layers = model_config.n_layer
        self.layers = [f'qk_{i}' for i in range(self.n_layers)]

    def get_attention_maps(self, sequence: str, component='qk_attn_softmax') -> List[np.ndarray]:
        """Get attention weights for each layer and head."""
        
        T = len(sequence)
        assert T <= self.model.config.block_size, (
            f"Sequence length {T} exceeds block size "
            f"{self.model.config.block_size}"
        )
        # Use unified hook system to get QK attention weights
        components = [f'{component}_{i}' for i in range(self.n_layers)]
        states = self._extract_internal_states([sequence], components)
        
        # Convert to the expected format: List[np.ndarray] where each array is [1, n_heads, seq_len, seq_len]
        attention_maps = []
        for i in range(self.n_layers):
            if f'{component}_{i}' in states:
                attention_maps.append(states[f'{component}_{i}'])
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
        component: str = 'qk_attn_softmax',
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
        A = self.get_attention_maps(sequence, component)[layer_idx][0, head_idx]
        X = self.get_embeddings(sequence, flatten=False)

        # Set upper diagonal (future tokens) to 0 (causal mask)
        seq_len = A.shape[0]
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        A = A.copy()  # Don't modify original
        A[mask == 1] = 0.0
        
        # Center and scale each row
        row_max = A.max(axis=-1, keepdims=True)
        A_center = A - row_max  # like softmax does
        row_std = A_center.std(axis=-1, keepdims=True)
        row_std = np.maximum(row_std, 1e-4)  # clamp_min equivalent
        A_norm = A_center / row_std
        
        # SVD(A) = USVT
        U, S, V = np.linalg.svd(A_norm)
        features_left = X.T @ U[:, nth_feature]
        features_right = X.T @ V[nth_feature, :] 
        
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

    