"""Helper functions for analyzing transformer model components."""

from typing import List, Tuple

import numpy as np

from interpretability.analyzers.activations import EmbeddingAnalyzer
from interpretability.core.base import BaseAnalyzer
from interpretability.visualizers.attention_viz import AttentionVisualizer


class AttentionAnalyzer(BaseAnalyzer):
    """Analyzer for attention patterns in transformer models."""
    
    def __init__(self, model, model_config, verbose=False):
        super().__init__(model, verbose=verbose, visualizer_class=AttentionVisualizer)
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
        
        if self.verbose:
            print(f"Getting attention maps for sequence: {sequence}")
            print(f"Component: {component}")
            print(f"Number of layers: {self.n_layers}")
        
        # Use unified hook system to get QK attention weights
        components = [f'{component}_{i}' for i in range(self.n_layers)]
        
        if self.verbose:
            print(f"Setting up hooks for components: {components}")
        
        # Extract internal states returns the activations directly
        activations = self._extract_internal_states([sequence], components)
        
        if self.verbose:
            print(f"Captured activations keys: {list(activations.keys())}")
            print(f"Activations: {activations}")
        
        # Convert to the expected format: List[np.ndarray] where each array is [1, n_heads, seq_len, seq_len]
        attention_maps = []
        for i in range(self.n_layers):
            key = f'{component}_{i}'
            if key in activations:
                attention_maps.append(activations[key])
                if self.verbose:
                    print(f"Found attention map for layer {i}: shape {activations[key].shape}")
            else:
                print(f'No attention map found for layer {i}')

        if self.verbose:
            print(f"Returning {len(attention_maps)} attention maps")

        return attention_maps

    def get_activations(self, sequences: list[str], layer: str) -> np.ndarray:
        raise NotImplementedError("Need to implement multi-sequence activation grabbing")
        # Example for future: self._extract_internal_states(sequences, [layer]); return self.hook_manager.get_activations()
        # return self.get_attention_maps(sequences, layer)
        
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

    