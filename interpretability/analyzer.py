"""Helper functions for analyzing transformer model components."""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from interp_helpers import embed_sequence, pca_embeddings, tokenize


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


class BaseAnalyzer:
    """Base class for analyzing transformer model components.
    
    This class provides common functionality for analyzing different parts
    of transformer models (attention, activations, embeddings, etc.).
    
    Args:
        model: The transformer model to analyze
        vocab: List of vocabulary tokens
        stoi: Dictionary mapping tokens to indices
        visualizer_class: Optional visualizer class to use
    """
    
    def __init__(self, model, visualizer_class=BaseVisualizer):
        self.model = model
        self.vocab = ['R', 'r', 'L', 'l']
        self.stoi = {token: idx for idx, token in enumerate(self.vocab)}
        self.n_layers = model.config.n_layer
        self.n_heads = model.config.n_head
        self.n_embd = model.config.n_embd
        self.visualizer = visualizer_class(self)
        
    def _prepare_input(self, sequence: str) -> torch.Tensor:
        """Convert input sequence to tensor format."""
        token_ids = tokenize(sequence)
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
