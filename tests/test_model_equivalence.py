import numpy as np
import torch

from interpretability.analyzers.activations import MLPAnalyzer
from interpretability.analyzers.attention import AttentionAnalyzer
from interpretability.core.utils import generate_random_sequences
from transformer.models import (GPT, GPTConfig, LastTokenGPT,
                                LastTokenGPTAdapter)


def test_model_equivalence():
    """Test that full and reduced models produce identical outputs"""
    
    # Create test configuration
    config = GPTConfig(
        block_size=6,
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


def test_activation_differences():
    """Test activation differences between models."""
    
    # Create models
    config = GPTConfig(
        block_size=6,
        vocab_size=4,
        n_layer=2,
        n_head=2,
        n_embd=16,
        device='cpu'
    )
    full_model = GPT(config)
    
    # Create reduced model and load weights
    reduced_model = LastTokenGPT(config)
    reduced_model.load_from_full_model(full_model)
    
    # Test sequences
    sequences = generate_random_sequences(200, 6, ['R', 'r', 'L', 'l'])
    print("=== TESTING ACTIVATION DIFFERENCES ===")
    
    # Full model analysis
    full_analyzer = MLPAnalyzer(full_model, config)
    full_activations = full_analyzer.get_activations(sequences)
    
    # Reduced model analysis
    adapter = LastTokenGPTAdapter(reduced_model)
    reduced_analyzer = MLPAnalyzer(adapter, config)
    adapter.patch_analyzer_for_compatibility(reduced_analyzer)
    reduced_activations = reduced_analyzer.get_activations(sequences)
    
    layer_differences = {layer: [] for layer in full_analyzer.layers}

    for seq in sequences:
        for layer in full_analyzer.layers:
            full_last = full_activations[seq][layer][-1]
            reduced_last = reduced_activations[seq][layer][-1]
            
            max_diff = np.abs(full_last - reduced_last).max()
            layer_differences[layer].append(max_diff)

    print("\nLayer Differences:")
    for layer, diffs in layer_differences.items():
        print(f"Layer {layer} (max diff): {np.max(diffs):.2e}")
        print(f"Layer {layer} (mean diff): {np.mean(diffs):.2e}")

def test_last_token_attention_patching():
    """Test if the new attention patching works correctly."""
    
    # Create a single-layer model
    config = GPTConfig(n_layer=1, n_head=1, n_embd=64, block_size=6)
    model = GPT(config)
    analyzer = AttentionAnalyzer(model, config, verbose=False)

    reduced_model = LastTokenGPT(config)
    reduced_model.load_from_full_model(model)
    
    print("=== TESTING LAST TOKEN ATTENTION PATCHING ===")
    
    # Test with adapter and patching
    adapter = LastTokenGPTAdapter(reduced_model, verbose=False)
    reduced_analyzer = AttentionAnalyzer(adapter, config, verbose=False)
    adapter.patch_analyzer_for_compatibility(reduced_analyzer)
    
    sequences = generate_random_sequences(200, 6, ['R', 'r', 'L', 'l'])

    attention_ativation_differences = []
    for sequence in sequences:
        reduced_attention_maps = reduced_analyzer.get_attention_maps(sequence, component='qk_attn_softmax')
        full_attention_maps = analyzer.get_attention_maps(sequence, component='qk_attn_softmax')
        
        for i, att_map in enumerate(reduced_attention_maps):
            
            # Check if only the last row has non-zero values (for final layer)
            if i == config.n_layer - 1:  # Final layer
                # Shape should be (1, n_head, seq_len, seq_len)
                last_row_reduced = att_map[0, 0, -1, :]  # Last row of attention
                last_row_full = full_attention_maps[i][0, 0, -1, :]  # Last row of attention

                attention_ativation_differences.append(np.max(np.abs(last_row_reduced - last_row_full)))
            else:
                pass

    print(f"Max attention activation difference: {np.max(attention_ativation_differences)}")
    print(f"Mean attention activation difference: {np.mean(attention_ativation_differences)}")


if __name__ == "__main__":
    test_model_equivalence()
    test_with_different_sequence_lengths()
    test_activation_differences()   
    test_last_token_attention_patching()
