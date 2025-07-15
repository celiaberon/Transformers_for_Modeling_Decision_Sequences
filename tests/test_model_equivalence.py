import os
import sys

import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer.alt_transformer import LastTokenGPT
from transformer.transformer import GPT, GPTConfig


def test_model_equivalence():
    """Test that full and reduced models produce identical outputs"""
    
    # Create test configuration
    config = GPTConfig(
        block_size=8,
        vocab_size=4,
        n_layer=2,
        n_head=2,
        n_embd=16,
        device='cpu'
    )
    
    # Create models
    full_model = GPT(config)
    reduced_model = LastTokenGPT(config)
    
    # Load weights using the convenient method
    reduced_model.load_from_full_model(full_model)
    
    # Set both models to eval mode
    full_model.eval()
    reduced_model.eval()
    
    # Create test input
    batch_size = 200
    seq_len = 6
    test_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\nTesting with input shape: {test_input.shape}")
    print(f"Test input:\n{test_input}")
    
    # Run inference
    with torch.no_grad():
        # Full model inference
        full_logits, _ = full_model(test_input)
        full_last_token_logits = full_logits[:, -1, :]  # Extract last token logits
        
        # Reduced model inference
        reduced_logits, _ = reduced_model(test_input)
        reduced_last_token_logits = reduced_logits.squeeze(1)  # Remove seq dimension
        
        print(f"\nFull model logits shape: {full_logits.shape}")
        print(f"Full model last token logits shape: {full_last_token_logits.shape}")
        print(f"Reduced model logits shape: {reduced_logits.shape}")
        print(f"Reduced model last token logits shape: {reduced_last_token_logits.shape}")
        
        # Compare outputs
        max_diff = torch.max(torch.abs(full_last_token_logits - reduced_last_token_logits))
        avg_diff = torch.mean(torch.abs(full_last_token_logits - reduced_last_token_logits))
        
        print(f"\nComparison Results:")
        print(f"Max absolute difference: {max_diff.item():.2e}")
        print(f"Average absolute difference: {avg_diff.item():.2e}")
        print(f"Logits are equivalent: {torch.allclose(full_last_token_logits, reduced_last_token_logits, atol=1e-6)}")
        
        # Test predictions
        full_predictions = torch.argmax(full_last_token_logits, dim=-1)
        reduced_predictions = torch.argmax(reduced_last_token_logits, dim=-1)
        
        print(f"\nPredictions:")
        print(f"Full model predictions: {full_predictions}")
        print(f"Reduced model predictions: {reduced_predictions}")
        print(f"Predictions match: {torch.equal(full_predictions, reduced_predictions)}")
        
        return max_diff.item() < 1e-6

def test_with_different_sequence_lengths():
    """Test equivalence with different sequence lengths"""
    
    config = GPTConfig(
        block_size=12,
        vocab_size=4,
        n_layer=3,
        n_head=2,
        n_embd=8,
        device='cpu'
    )
    
    full_model = GPT(config)
    reduced_model = LastTokenGPT(config)
    reduced_model.load_from_full_model(full_model)
    
    full_model.eval()
    reduced_model.eval()
    
    sequence_lengths = [2, 4, 6, 8, 10]
    batch_size = 3
    
    print(f"\nTesting with different sequence lengths:")
    
    all_equivalent = True
    
    for seq_len in sequence_lengths:
        test_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            full_logits, _ = full_model(test_input)
            full_last_token_logits = full_logits[:, -1, :]
            
            reduced_logits, _ = reduced_model(test_input)
            reduced_last_token_logits = reduced_logits.squeeze(1)
            
            is_equivalent = torch.allclose(full_last_token_logits, reduced_last_token_logits, atol=1e-6)
            max_diff = torch.max(torch.abs(full_last_token_logits - reduced_last_token_logits))
            
            print(f"  Seq len {seq_len}: {'✓' if is_equivalent else '✗'} (max diff: {max_diff.item():.2e})")
            
            if not is_equivalent:
                all_equivalent = False
    
    return all_equivalent

if __name__ == "__main__":
    test_model_equivalence()
    test_with_different_sequence_lengths()