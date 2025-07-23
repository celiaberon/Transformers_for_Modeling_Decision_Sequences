"""Configuration classes for interpretability analysis."""

from dataclasses import dataclass


@dataclass
class InterpretabilityConfig:
    """Base configuration for interpretability analysis."""
    def __init__(
        self,
        device: str = 'cpu',
        batch_size: int = 32,
        max_sequences: int = 1000,
        random_state: int = 42,
        verbose: bool = False
    ):
        self.device = device
        self.batch_size = batch_size
        self.max_sequences = max_sequences
        self.random_state = random_state
        self.verbose = verbose


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


@dataclass
class LensConfig:
    """Configuration for lens analysis."""
    
    device: str = 'cpu'
    
    # Logit lens settings
    normalize_logits: bool = False
    
    # Tuned lens settings
    probe_type: str = 'linear'  # 'linear' or 'mlp'
    probe_hidden_dim: int = 128
    random_state: int = 42
    
    # Training settings
    probe_lr: float = 1e-3
    probe_epochs: int = 100
    probe_weight_decay: float = 1e-4
