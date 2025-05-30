import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath('../'))
import interp_helpers as interp
import pandas as pd
from feature_attribution import AttributionAnalyzer

import utils.file_management as fm
from interpretability.activations import MLPAnalyzer
from interpretability.analyzer import DimensionalityReductionConfig
from utils.parse_data import load_trained_model

sns.set_theme(
    style='ticks',
    font_scale=1.0,
    rc={'axes.labelsize': 11,
        'axes.titlesize': 11,
        'savefig.transparent': True,
        'legend.title_fontsize': 11,
        'legend.fontsize': 10,
        'legend.borderpad': 0.2,
        'figure.titlesize': 11,
        'figure.subplot.wspace': 0.1,
        })


def main(run: int | None = None):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    run = run or fm.get_latest_run()
    model, model_info, config = load_trained_model(run, model_name=None, device=device, weights_only=False)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")

    # Print model configuration details
    print(f"Number of layers: {config.n_layer}")
    print(f"Number of attention heads: {config.n_head}")
    print(f"Embedding dimension: {config.n_embd}")
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Block size (context length): {config.block_size}")

    T = model_info['dataloader']['Sequence length (T)']

    num_sequences = 300
    events, sequences, counts = interp.get_common_sequences(T, run=run, k=num_sequences)

    block_sequences  = interp.get_block_transition_sequences(events, T, high_port=1)
    block_sequences = [list(b.values) for b in block_sequences]

    attribution_analyzer = AttributionAnalyzer(model, method='inputs')
    
    mlp_analyzer = MLPAnalyzer(model, config)
    activations = mlp_analyzer.get_activations(sequences)
    for method in ['raw', 'choicediff', 'rewarddiff', 'choicepchange', 'rewardpchange']:
        _, fig = mlp_analyzer.analyze_layer_specialization(activations, token_pos=-1, method=method)
        fig_path = fm.get_experiment_file(f'mlp_selectivity_last_token_{method}.png', run, subdir=f'interp')
        fig.savefig(fig_path, bbox_inches='tight')

    dr_config = DimensionalityReductionConfig(
        token_pos=-1,
        sequence_method='token',
        n_components=4
    )
    for sm in ['token', 'concat']:
        dr_config.sequence_method = sm
        fig, axs = mlp_analyzer.visualizer.plot_pca_by_layer(sequences, dr_config, counts=counts)
        fig_path = fm.get_experiment_file(f'mlp_pca_{sm}.png', run, subdir=f'interp')
        fig.savefig(fig_path, bbox_inches='tight')


    dr_config.method = 'tsne'
    for sm in ['token', 'concat']:
        dr_config.sequence_method = sm
        fig, axs = mlp_analyzer.visualizer.plot_pca_by_layer(sequences, dr_config, counts=counts, variance_explained=False)
        fig_path = fm.get_experiment_file(f'mlp_tsne_{sm}.png', run, subdir='interp')
        fig.savefig(fig_path, bbox_inches='tight')

    for i, seq in enumerate(block_sequences[10:14], start=10):

        dr_config = DimensionalityReductionConfig(
            token_pos=-1,
            sequence_method='token',
            n_components=4
        )
        fig, axs = mlp_analyzer.visualizer.plot_pca_across_trials(sequences, seq, ['input', 'gelu', 'output'], dr_config)
        fig_path = fm.get_experiment_file(f'mlp_pca_last_token.png', run, subdir=f'interp/bt_{i}')
        fig.savefig(fig_path, bbox_inches='tight')

        dr_config = DimensionalityReductionConfig(
            token_pos=-1,
            sequence_method='concat',
            n_components=2
        )
        fig, axs = mlp_analyzer.visualizer.plot_pca_across_trials(sequences, seq, ['input', 'gelu', 'output'], dr_config)
        fig_path = fm.get_experiment_file(f'mlp_pca_concat.png', run, subdir=f'interp/bt_{i}')
        fig.savefig(fig_path, bbox_inches='tight')
    
        seq_ = interp.trim_leading_duplicates(seq)
        fig, axs = attribution_analyzer.plot_attribution_contiguous_sequences(seq_, method='embedding_erasure', as_prob=True)
        fig_path = fm.get_experiment_file(f'embedding_erasure.png', run, subdir=f'interp/bt_{i}')
        fig.savefig(fig_path, bbox_inches='tight')

        fig, axs = attribution_analyzer.plot_attribution_contiguous_sequences(seq_, method='contrastive', as_prob=True)
        fig_path = fm.get_experiment_file(f'contrastive.png', run, subdir=f'interp/bt_{i}')
        fig.savefig(fig_path, bbox_inches='tight')

        fig, axs = attribution_analyzer.plot_attribution_contiguous_sequences(seq_, method='lime')
        fig_path = fm.get_experiment_file(f'lime.png', run, subdir=f'interp/bt_{i}')
        fig.savefig(fig_path, bbox_inches='tight')

    events, sequences, counts = interp.get_common_sequences(T, events=events, k=1000)
    activations = mlp_analyzer.get_activations(sequences)
    last_pos_by_layer = mlp_analyzer.get_activation_by_position(activations, token_pos=-1)
    fig, axs = plt.subplots(ncols=3,figsize=(8, 2.5), layout='constrained')
    for ax, layer in zip(axs, mlp_analyzer.layers):

        # Calculate correlation between neurons (input layer)
        neuron_corr = pd.DataFrame(last_pos_by_layer[layer]).T.corr()

        ordered_sequences, ordered_sim_matrix, Z_ordered = interp.cluster_sequences_hierarchical(neuron_corr.to_numpy(), np.arange(len(neuron_corr)), replot=False)
        plt.close()
        interp.plot_similarity(ordered_sim_matrix, ordered_sequences, ax=ax)
        ax.set_title(layer.capitalize())

    fig.suptitle("Within-Layer Neuron Correlations")

    fig_path = fm.get_experiment_file('mlp_interneuron_corr.png', run, subdir='interp')
    fig.savefig(fig_path, bbox_inches='tight')