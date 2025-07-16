import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

import interp_helpers as interp
from feature_attribution import AttributionAnalyzer

import evaluation.inspect_data as inspect
import utils.file_management as fm
import utils.parse_data as parse
import utils.model_utils as model_utils
from interpretability.activations import EmbeddingAnalyzer, MLPAnalyzer
from interpretability.analyzer import DimensionalityReductionConfig
from interpretability.attention_helpers import AttentionAnalyzer

# Set seaborn theme
sns.set_theme(
    style='ticks',
    font_scale=1.0,
    rc={
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'savefig.transparent': True,
        'legend.title_fontsize': 11,
        'legend.fontsize': 10,
        'legend.borderpad': 0.2,
        'figure.titlesize': 11,
        'figure.subplot.wspace': 0.1,
    }
)

logger = None

vocab = ['R', 'r', 'L', 'l']
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}


def initialize_logger(run: int) -> None:
    """Initialize logger for the interpretability analysis.
    
    Args:
        run: Run number to initialize logger for
    """
    global logger
    logger = fm.setup_logging(run, 'interpretability', 'interp')


def setup_device() -> torch.device:
    """Set up the device for model computation.
    
    Returns:
        torch.device: Device to use for computation
    """
    device = torch.device(
        'mps' if torch.backends.mps.is_available()
        else 'cuda' if torch.cuda.is_available()
        else 'cpu'
    )
    print(f"Using {device} device")
    return device


def load_model_and_config(
    run: Optional[int],
    model_name: Optional[str],
    device: torch.device
) -> Tuple[torch.nn.Module, Dict, Dict, str]:
    """Load model and configuration.
    
    Args:
        run: Run number to load model from
        model_name: Name of model to load
        device: Device to load model onto
        
    Returns:
        Tuple containing:
            - Loaded model
            - Model info dictionary
            - Model configuration
            - Model name
    """
    run = run or fm.get_latest_run()
    initialize_logger(run)
    model, model_info, config = model_utils.load_trained_model(
        run,
        model_name=model_name,
        device=device,
        weights_only=False
    )
    model.to(device)
    
    if model_name is None:
        model_name = model_info['model_name']
    else:
        assert (
            model_info['model_name'] == model_name or
            model_info['model_name'] == model_name.split('_cp')[0]
        ), 'did not recover correct model'
        
    return model, model_info, config, model_name


def analyze_sequence_similarity(
    model: torch.nn.Module,
    sequences: List[str],
    run: int,
    sample_size: int = 20
) -> None:
    """Analyze similarity between sequences.
    
    Args:
        model: Model to analyze
        sequences: List of sequences to analyze
        run: Run number
        sample_size: Number of sequences to sample
    """
    sample_sequences = np.random.choice(sequences, size=sample_size)
    embeddings, similarities = interp.sequence_embedding_similarity(
        model, sample_sequences, stoi
    )
    
    ordered_sequences, ordered_sim_matrix, _ = interp.cluster_sequences_hierarchical(
        similarities, sample_sequences, replot=False, is_similarity=True
    )
    
    fig, ax = interp.plot_similarity(ordered_sim_matrix, ordered_sequences)
    ax.set(title=f'run {run}')
    fig_path = fm.get_experiment_file(
        'sample_sequence_similarity.png', run, subdir='interp'
    )
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close()


def compare_train_val_embeddings(model, model_info, run):

    model_name = model_info['model_name']
    train_events = parse.parse_simulated_data(*parse.get_data_filenames(run, suffix='tr'))
    aligned_data = parse.load_predictions(run, model_name, suffix='v')

    maxT = model_info['dataloader']['Sequence length (T)']
    minT = maxT

    aligned_data = inspect.compare_train_val(run, train_events, aligned_data,
                                    maxT=maxT, minT=minT, show_plot=False)

    val_unique_sequences = aligned_data.query('seq6_overlap == False')['seq6_RL'].value_counts()[:20].index

    embeddings, similarities = interp.sequence_embedding_similarity(model, val_unique_sequences, stoi)

    ordered_sequences, ordered_sim_matrix, Z_ordered = interp.cluster_sequences_hierarchical(similarities, val_unique_sequences, replot=False, is_similarity=True)

    matched_sequences = inspect.match_by_base_sequence(aligned_data, ordered_sequences, base_T=5, min_count=20)

    fig, axs = plt.subplots(ncols=3, figsize=(12, 3.5), layout='constrained')

    fig, axs[0] = interp.plot_similarity(ordered_sim_matrix, ordered_sequences, fig=fig, ax=axs[0])
    axs[0].set(title='Validation Only Sequences')

    _, similarities = interp.sequence_embedding_similarity(model, matched_sequences, stoi)
    fig, axs[1] = interp.plot_similarity(similarities, matched_sequences, fig=fig, ax=axs[1])
    axs[1].set(title='Matched Sequences')

    # Get upper triangular values (excluding diagonal) from both matrices
    triu_idx = np.triu_indices_from(ordered_sim_matrix, k=1)
    val_similarities = ordered_sim_matrix[triu_idx]

    matched_sim_values = similarities[triu_idx]

    # Plot the similarity values against each other
    axs[2].scatter(val_similarities, matched_sim_values, alpha=0.5)
    axs[2].set(xlabel='Validation Sequence Similarities', 
        ylabel='Matched Sequence Similarities',
        title='Comparing Similarity Distributions')

    # Add diagonal line for reference
    lims = [
        np.min([axs[2].get_xlim(), axs[2].get_ylim()]),
        np.max([axs[2].get_xlim(), axs[2].get_ylim()])
    ]
    axs[2].plot(lims, lims, 'k--', alpha=0.5, zorder=0)
    axs[2].text(lims[0], lims[1], f'r={np.corrcoef(val_similarities, matched_sim_values)[0, 1]:.2f}', ha='left', va='top', fontsize=12)
    sns.despine()

    fig_path = fm.get_experiment_file(
        'train_val_embeddings.png', run, subdir='interp'
    )
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close()


def analyze_uncertainty(
    model: torch.nn.Module,
    sequences: List[str],
    run: int,
    include_reward: bool = False
) -> None:
    """Analyze model uncertainty on sequences.
    
    Args:
        model: Model to analyze
        sequences: List of sequences to analyze
        run: Run number
        include_reward: Whether to include reward in uncertainty analysis
    """
    # Get uncertainty sequences
    if include_reward:
        high_uncertainty_sequences, _ = interp.get_uncertain_sequences(
            sequences, model, is_uncertain=True, threshold=0.5
        )
        low_uncertainty_sequences, _ = interp.get_uncertain_sequences(
            sequences, model, is_uncertain=False, threshold=0.7
        )
        title = 'Overall Uncertainty (including reward)'
        filename = 'similarity_uncertainty_overall.png'
    else:
        # Analyze choice uncertainty
        selected_sequences, uncertainty = interp.get_uncertain_sequences(
            sequences, model, feature='choice', is_uncertain=True, threshold=0
        )

        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot(uncertainty)
        ax.set(
            title='Uncertainty of Choice',
            xlabel='Sequence',
            ylabel='Uncertainty'
        )
        sns.despine()
        fig_path = fm.get_experiment_file(
            'uncertainty_choice.png', run, subdir='interp'
        )
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close()

        high_uncertainty_sequences, _ = interp.get_uncertain_sequences(
            sequences, model, feature='choice', is_uncertain=True, threshold=0.7
        )
        low_uncertainty_sequences, _ = interp.get_uncertain_sequences(
            sequences, model, feature='choice', is_uncertain=False, threshold=0.7
        )
        title = 'Choice Uncertainty'
        filename = 'similarity_uncertainty_choice.png'

    # Plot high/low uncertainty sequences
    fig, axs = plt.subplots(ncols=2, figsize=(7, 3.5), layout='constrained')

    # Plot high uncertainty sequences
    _, similarities = interp.sequence_embedding_similarity(
        model, high_uncertainty_sequences[:20], stoi
    )
    ordered_sequences, ordered_sim_matrix, _ = interp.cluster_sequences_hierarchical(
        similarities, high_uncertainty_sequences[:20], replot=False, is_similarity=True
    )
    fig, axs[0] = interp.plot_similarity(
        ordered_sim_matrix, ordered_sequences, fig=fig, ax=axs[0]
    )
    axs[0].set(title='High Uncertainty Sequences')

    # Plot low uncertainty sequences
    _, similarities = interp.sequence_embedding_similarity(
        model, low_uncertainty_sequences[:20], stoi
    )
    ordered_sequences, ordered_sim_matrix, _ = interp.cluster_sequences_hierarchical(
        similarities, low_uncertainty_sequences[:20], replot=False, is_similarity=True
    )
    fig, axs[1] = interp.plot_similarity(
        ordered_sim_matrix, ordered_sequences, fig=fig, ax=axs[1]
    )
    axs[1].set(title='Low Uncertainty Sequences')
    
    fig.suptitle(title)
    fig_path = fm.get_experiment_file(filename, run, subdir='interp')
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close()


def analyze_block_transitions(
    model: torch.nn.Module,
    events: List[str],
    T: int,
    run: int
) -> Dict[str, List[str]]:
    """Analyze block transition sequences.
    
    Args:
        model: Model to analyze
        events: List of events
        T: Sequence length
        run: Run number
        
    Returns:
        Dictionary of test sequences
    """
    block_sequences = interp.get_block_transition_sequences(events, T, high_port=1)
    block_sequences = [list(b.values) for b in block_sequences]

    test_sequences = {i: s for i, s in enumerate(block_sequences[10:14], start=10)}
    test_sequences = test_sequences | {
        'fixed_0': ['LLLLLL', 'LLLLLL', 'LLLLLL', 'LLLLLL', 'LLLLLL', 'LLLLLl',
                   'LLLLll', 'LLLllr', 'LLllrR', 'LllrRR', 'llrRRR', 'lrRRRR'],
        'fixed_1': ['LLLLLL', 'LLLLLl', 'LLLLlL', 'LLLlLL', 'LLlLLl', 'LlLLll',
                   'lLLlll', 'LLllll', 'Llllll', 'llllll', 'llllll', 'llllll'],
        'fixed_2': ['LLLLLr', 'LLLLrl', 'LLLrlL', 'LLrlLL', 'LrlLLl', 'rlLLlL',
                   'lLLlLl', 'LLlLll', 'LlLlll', 'lLlllR', 'LlllRl', 'lllRlR'],
        'fixed_3': ['LlLLLL', 'lLLLLl', 'LLLLlL', 'LLLlLL', 'LLlLLl', 'LlLLll',
                   'lLLlll', 'LLllll', 'LllllR', 'llllRR', 'lllRRR', 'llRRRR']
    }
    
    return test_sequences


def analyze_mlp_layer_specialization(
    model: torch.nn.Module,
    sequences: List[str],
    run: int
) -> None:
    """Analyze MLP layer specialization.
    
    Args:
        model: Model to analyze
        sequences: List of sequences to analyze
        run: Run number
    """
    mlp_analyzer = MLPAnalyzer(model, model.config)
    activations = mlp_analyzer.get_activations(sequences)
    
    for method in ['raw', 'choicediff', 'rewarddiff', 'choicepchange', 'rewardpchange']:
        _, fig = mlp_analyzer.analyze_layer_specialization(
            activations, token_pos=-1, method=method
        )
        fig_path = fm.get_experiment_file(
            f'mlp_selectivity_last_token_{method}.png', run, subdir='interp'
        )
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close()


def analyze_mlp_dimensionality(
    model: torch.nn.Module,
    sequences: List[str],
    counts: List[int],
    run: int,
    test_sequences: Dict[str, List[str]]
) -> None:
    """Analyze MLP dimensionality using PCA.
    
    Args:
        model: Model to analyze
        sequences: List of sequences to analyze
        counts: List of counts for each sequence
        run: Run number
        test_sequences: Dictionary of test sequences to analyze
    """
    mlp_analyzer = MLPAnalyzer(model, model.config)
    
    # Use the first layer's components for analysis (layer 0)
    mlp_components = ['input_0', 'gelu_0', 'output_0']
    
    for i, seq in test_sequences.items():
        dr_config = DimensionalityReductionConfig(
            token_pos=-1,
            sequence_method='token',
            n_components=4,
            method='pca'
        )
        fig, axs = mlp_analyzer.visualizer.plot_pca_across_trials(
            sequences, seq, mlp_components, dr_config
        )
        fig_path = fm.get_experiment_file(
            f'mlp_pca_last_token.png', run, subdir=f'interp/bt_{i}'
        )
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close()

        dr_config = DimensionalityReductionConfig(
            token_pos=-1,
            sequence_method='concat',
            n_components=2,
            method='pca'
        )
        fig, axs = mlp_analyzer.visualizer.plot_pca_across_trials(
            sequences, seq, mlp_components, dr_config
        )
        fig_path = fm.get_experiment_file(
            f'mlp_pca_concat.png', run, subdir=f'interp/bt_{i}'
        )
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close()


def analyze_test_sequences(
    model: torch.nn.Module,
    sequences: List[str],
    test_sequences: Dict[str, List[str]],
    run: int
) -> None:
    """Analyze test sequences using various methods.
    
    Args:
        model: Model to analyze
        sequences: List of sequences to analyze
        test_sequences: Dictionary of test sequences to analyze
        run: Run number
    """
    embedding_analyzer = EmbeddingAnalyzer(model, model.config)
    attention_analyzer = AttentionAnalyzer(model, model.config)
    attribution_analyzer = AttributionAnalyzer(model, model.config, method='inputs')
    
    for i, seq in test_sequences.items():
        # Embedding analysis
        dr_config = DimensionalityReductionConfig(
            token_pos=-1,
            sequence_method='concat',
            n_components=2,
            method='pca'
        )
        fig, axs = embedding_analyzer.visualizer.plot_pca_across_trials(
            sequences, seq, 'embed', dr_config
        )
        fig_path = fm.get_experiment_file(
            f'embed_pca_concat.png', run, subdir=f'interp/bt_{i}'
        )
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close()
        
        # Attribution analysis
        seq_ = interp.trim_leading_duplicates(seq)
        fig, axs = attribution_analyzer.plot_attribution_contiguous_sequences(
            seq_, method='embedding_erasure', as_prob=True
        )
        fig_path = fm.get_experiment_file(
            f'embedding_erasure.png', run, subdir=f'interp/bt_{i}'
        )
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close()
        
        fig, axs = attribution_analyzer.plot_attribution_contiguous_sequences(
            seq_, method='contrastive', as_prob=True
        )
        fig_path = fm.get_experiment_file(
            f'contrastive.png', run, subdir=f'interp/bt_{i}'
        )
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close()

        fig, axs = attribution_analyzer.plot_attribution_contiguous_sequences(
            seq_, method='lime'
        )
        fig_path = fm.get_experiment_file(
            f'lime.png', run, subdir=f'interp/bt_{i}'
        )
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close()

        # Attention analysis
        fig = attention_analyzer.visualizer.plot_attention_multiple_sequences(
            seq_, max_sequences=10
        )
        fig_path = fm.get_experiment_file(
            f'attention_raw.png', run, subdir=f'interp/bt_{i}'
        )
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close()
        
        fig = attention_analyzer.visualizer.plot_attention_multiple_sequences(
            seq_[1:],
            max_sequences=10,
            as_diff=True,
            ref_seq=seq_[0]
        )
        fig_path = fm.get_experiment_file(
            f'attention_diff.png', run, subdir=f'interp/bt_{i}'
        )
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close()

        dr_config = DimensionalityReductionConfig(
            token_pos=-1,
            sequence_method='token',
            n_components=2
        )
        fig = attention_analyzer.visualizer.plot_attention_features(
            sequences, seq, dr_config
        )
        fig_path = fm.get_experiment_file(
            f'qk_features.png', run, subdir=f'interp/bt_{i}'
        )
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close()


def analyze_neuron_correlations(
    model: torch.nn.Module,
    sequences: List[str],
    run: int
) -> None:
    """Analyze correlations between neurons.
    
    Args:
        model: Model to analyze
        sequences: List of sequences to analyze
        run: Run number
    """
    mlp_analyzer = MLPAnalyzer(model, model.config)
    mlp_activations = mlp_analyzer.get_activations(sequences)
    mlp_last_pos_by_layer = mlp_analyzer.get_activation_by_position(
        mlp_activations, token_pos=-1
    )
    
    fig, axs = plt.subplots(ncols=3, figsize=(8, 2.5), layout='constrained')
    
    for ax, layer in zip(axs, mlp_analyzer.layers):
        # Calculate correlation between neurons
        neuron_corr = pd.DataFrame(mlp_last_pos_by_layer[layer]).T.corr()
        ordered_sequences, ordered_sim_matrix, _ = interp.cluster_sequences_hierarchical(
            neuron_corr.to_numpy(), np.arange(len(neuron_corr)), replot=False,
            metric='euclidean'
        )
        plt.close()
        interp.plot_similarity(ordered_sim_matrix, ordered_sequences, ax=ax)
        ax.set_title(layer.capitalize())
    
    fig.suptitle("Within-Layer Neuron Correlations")
    fig_path = fm.get_experiment_file('mlp_interneuron_corr.png', run, subdir='interp')
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close()


def analyze_maximal_activations(
    model: torch.nn.Module,
    sequences: List[str],
    run: int
) -> None:
    """Analyze maximal activations for each layer.
    
    Args:
        model: Model to analyze
        sequences: List of sequences to analyze
        run: Run number
    """
    # MLP analysis
    mlp_analyzer = MLPAnalyzer(model, model.config)
    mlp_activations = mlp_analyzer.get_activations(sequences)
    mlp_last_pos_by_layer = mlp_analyzer.get_activation_by_position(
        mlp_activations, token_pos=-1
    )
    
    mlp_max_activations = {}
    fig, axes_dict = mlp_analyzer.visualizer.create_mlp_visualization()
    
    for layer_name, axes in axes_dict.items():
        mlp_max_activations[layer_name] = mlp_analyzer.find_maximal_activations(
            mlp_last_pos_by_layer, layer_name, sequences
        )
        
        for neuron_idx, ax in enumerate(axes):
            ax, token_counts = mlp_analyzer.analyze_neuron_patterns(
                mlp_max_activations, layer_name, neuron_idx,
                ax=ax, cbar=neuron_idx == (len(axes)-1)
            )
            avg_sequence = mlp_analyzer.get_average_sequence(
                token_counts, single_threshold=0, joint_threshold=0
            )
            avg_sequence_thresholded = mlp_analyzer.get_average_sequence(
                token_counts, single_threshold=0.6, joint_threshold=0.4
            )
            ax.set(
                title=f"Neuron {neuron_idx+1}\n{avg_sequence}\n{avg_sequence_thresholded}",
                xticks=[]
            )
            ax.set_aspect(1)
            if neuron_idx > 0:
                ax.set(ylabel='', yticks=[])
            ax.set(xlabel='')
    
    fig_path = fm.get_experiment_file('mlp_max_activations.png', run, subdir='interp')
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close()
    
    # Embedding analysis
    embedding_analyzer = EmbeddingAnalyzer(model, model.config)
    embedding_activations = embedding_analyzer.get_activations(sequences)
    embedding_last_pos_by_layer = embedding_analyzer.get_activation_by_position(
        embedding_activations, token_pos=-1
    )
    
    embedding_max_activations = {}
    for layer_name in embedding_analyzer.layers:
        embedding_max_activations[layer_name] = embedding_analyzer.find_maximal_activations(
            embedding_last_pos_by_layer, layer_name, sequences
        )
    
    fig, axes_dict = embedding_analyzer.visualizer.create_mlp_visualization()
    
    for neuron_idx, ax in enumerate(axes_dict['embed']):
        ax, token_counts = embedding_analyzer.analyze_neuron_patterns(
            embedding_max_activations, layer_name, neuron_idx, verbose=False,
            ax=ax, cbar=neuron_idx == (len(axes_dict['embed'])-1)
        )
        avg_sequence = embedding_analyzer.get_average_sequence(
            token_counts, single_threshold=0, joint_threshold=0
        )
        avg_sequence_thresholded = embedding_analyzer.get_average_sequence(
            token_counts, single_threshold=0.6, joint_threshold=0.4
        )
        ax.set(
            title=f"Neuron {neuron_idx+1}\n{avg_sequence}\n{avg_sequence_thresholded}",
            xticks=[]
        )
        ax.set_aspect(1)
        if neuron_idx > 0:
            ax.set(ylabel='', yticks=[])
        ax.set(xlabel='')
    
    fig_path = fm.get_experiment_file('embed_max_act.png', run, subdir='interp')
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close()


def main(run: Optional[int] = None, model_name: Optional[str] = None) -> None:
    """Main function for running interpretability analysis.
    
    Args:
        run: Run number to analyze
        model_name: Name of model to analyze
    """
    device = setup_device()
    model, model_info, config, model_name = load_model_and_config(
        run, model_name, device
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of parameters: {total_params}")
    
    # Print model configuration details
    logger.info(f"Number of layers: {config.n_layer}")
    logger.info(f"Number of attention heads: {config.n_head}")
    logger.info(f"Embedding dimension: {config.n_embd}")
    logger.info(f"Vocabulary size: {config.vocab_size}")
    logger.info(f"Block size (context length): {config.block_size}")
    
    # Analyze embedding evolution
    embedding_evolution = interp.analyze_embedding_evolution(run, save_results=True)
    
    T = model_info['dataloader']['Sequence length (T)']
    num_sequences = 300
    events, sequences, counts = interp.get_common_sequences(
        T, run=run, k=num_sequences
    )
    
    analyze_sequence_similarity(model, sequences, run)
    analyze_uncertainty(model, sequences, run)  # Choice uncertainty
    analyze_uncertainty(model, sequences, run, include_reward=True)  # Overall uncertainty
    compare_train_val_embeddings(model, model_info, run)
    
    test_sequences = analyze_block_transitions(model, events, T, run)
    analyze_mlp_layer_specialization(model, sequences, run)
    analyze_mlp_dimensionality(model, sequences, counts, run, test_sequences)
    analyze_test_sequences(model, sequences, test_sequences, run)
    analyze_neuron_correlations(model, sequences, run)
    analyze_maximal_activations(model, sequences, run)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    args = parser.parse_args()
    
    main(run=args.run, model_name=args.model_name)