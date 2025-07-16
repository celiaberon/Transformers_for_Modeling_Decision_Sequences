
import torch
import torch.nn as nn

from transformer import GPT, MLP, CausalSelfAttention

seed = 200
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


class Block_noLN(nn.Module):
    """Transformer block with attention and MLP"""
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class GPT_noLN(GPT):
    """GPT-style transformer model for sequence prediction"""
    def __init__(self, config):
        super().__init__(config)
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'h': nn.ModuleList([Block_noLN(config) for _ in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # Weight tying
        self.apply(self._init_weights)

    def forward(self, idx, targets=None, **kwargs):
        """Forward pass through the model"""
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)

        for block in self.transformer.h:
            x = block(x)

        logits = self.lm_head(x)
        loss = self.calculate_loss(logits, targets, **kwargs)

        return logits, loss
