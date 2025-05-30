import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from typing import Dict, List, Optional, Union
sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

import utils.file_management as fm
from evaluate_transformer_guess import compute_confusion_matrix
from evaluation.graph_helper import calc_bpos_behavior
from utils.checkpoint_processing import (add_checkpoint_colorbar,
                                         generate_checkpoint_colormap,
                                         get_checkpoint_files)
from utils.parse_data import (align_predictions_with_gt, get_data_filenames,
                              parse_simulated_data)

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../../behavior-helpers/')))
from bh.visualization import plot_trials as pts

def plot_confusion_matrix_with_bars(
    confusion_matrices: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
    labels: List[str],
    model_names: Optional[List[str]] = None,
    cmap: str = 'Blues',
    normalize: str = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot a confusion matrix where each cell contains a barplot showing the distribution
    of predictions across multiple models/checkpoints.
    
    Args:
        confusion_matrices: Either a single confusion matrix or a list/dict of matrices
            for multiple models. Each matrix should be of shape (n_classes, n_classes).
        labels: List of class labels
        model_names: Optional list of model names (required if confusion_matrices is a list/dict)
        
    Returns:
        Tuple of (figure, axes array)
    """
    if isinstance(confusion_matrices, np.ndarray):
        confusion_matrices = [confusion_matrices]
        if model_names is None:
            model_names = ["Model"]
    elif isinstance(confusion_matrices, dict):
        model_names = list(confusion_matrices.keys())
        confusion_matrices = list(confusion_matrices.values())
    
    n_classes = len(labels)
    n_models = len(confusion_matrices)
    fig_width = (n_classes*0.3)*(n_models*0.2 +1)
    figsize=(fig_width, fig_width)
    fig, axes = plt.subplots(n_classes, n_classes, figsize=figsize, sharey=True, sharex=True)
    axes = np.array(axes).reshape(n_classes, n_classes)
    
    if isinstance(cmap, str):
        cmap = sns.color_palette(cmap, n_models)
    for i in range(n_classes):
        for j in range(n_classes):
            ax = axes[i, j]
            
            # Get values for this cell across all models
            cell_values = [cm[i, j] for cm in confusion_matrices]

            bars = ax.barh(
                np.arange(n_models),
                cell_values,
                height=0.8,
                color=[cmap.get(i) for i in model_names],
                label=model_names
            )
            
            if normalize in ['row', 'col']:
                ax.set(xlim=(0, 1), xticks=[], yticks=[])
            else:
                ax.set(xticks=[], yticks=[])

            # Add labels for first row and column
            if i == 0:
                ax.set_title(labels[j], fontsize=10)
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=10)

        if (i != n_classes-1) and (j != n_classes-1):
            plt.legend().remove()
    plt.legend(bbox_to_anchor=(1, 0.9), loc='upper left', borderaxespad=0.,
               bbox_transform=plt.gcf().transFigure)
    plt.tight_layout()
    
    # Add overarching labels
    fig.text(0.5, 0.02, 'Predicted', ha='center', fontsize=12)
    fig.text(-0.05, 0.5, 'Actual', va='center', rotation='vertical', fontsize=12)
    
    return fig, axes

def main(run=None, suffix: str = 'v'):

    """Plot behavior comparisons across checkpoints and domains."""
    sns.set_theme(style='ticks', font_scale=1.0, rc={'axes.labelsize': 12,
                  'axes.titlesize': 12, 'savefig.transparent': False})

    run = run or fm.get_latest_run()

    checkpoint_files = get_checkpoint_files(run, include_final=False, subdir='seqs', pattern='pred_model', ext='.txt')
    indices_files = get_checkpoint_files(run, include_final=False, subdir='seqs', pattern='pred_indices', ext='.txt')
    model_files = get_checkpoint_files(run, include_final=True)

    if not checkpoint_files:
        print(f"No checkpoint models found in run {run}")
        return None
    elif len(checkpoint_files) == 1:
        print(f"Only one checkpoint model found in run {run}")
        return None
    model_name = model_files[0].split('/')[-1].split('_cp')[0]

    # Load ground truth data once
    files = get_data_filenames(run, suffix=suffix)
    gt_events = parse_simulated_data(*files)
    domains = sorted(gt_events['domain'].unique())

    gt_policies = pts.calc_conditional_probs(gt_events, htrials=2, sortby='pevent', pred_col='switch', add_grps='domain')
    gt_policies['model'] = 'ground truth'

    fig, axes = plt.subplots(3, len(domains), figsize=(4.5*len(domains), 6),
                             sharex=False, layout='constrained')
      
    # If we have only one domain, reshape to maintain 2D structure
    if len(domains) == 1:
        axes = axes.reshape(3, 1)

    # When generating the colormap, use numbers without 'cp' prefix as keys
    cmap = generate_checkpoint_colormap(checkpoint_labels=checkpoint_files)
    cmap['colors'] = {label.replace('cp', ''): color 
                      for label, color in cmap['colors'].items()}
    cmap['colors']['ground truth'] = 'k'

    confusion_matrices = {d:[] for d in domains}
    confusion_matrices_normalized = {d:[] for d in domains}
    checkpoint_models = []
    for pred_file, indices_file in zip(checkpoint_files, indices_files):

        # Extract checkpoint numbers
        assert pred_file.split('_cp')[-1].replace('.txt', '') == indices_file.split('_cp')[-1].replace('.txt', ''), (
            f"Checkpoint numbers don't match between prediction file ({pred_file}) and indices file ({indices_file})"
        )

        # Load predictions for this checkpoint
        predictions = fm.read_sequence(pred_file)
        with open(indices_file, 'r') as f:
            indices = [int(line.strip()) for line in f]
        events = align_predictions_with_gt(gt_events, predictions, indices)
        bpos_ = calc_bpos_behavior(events,
                                   add_cond_cols=['domain', 'session'],
                                   add_agg_cols=['pred_switch', 'pred_selected_high'])

        label = pred_file.split("_cp")[-1].replace(".txt", "")
        color = cmap['colors'][label]
        for ax_, (domain, bpos_domain) in zip(axes.T, bpos_.groupby('domain')):
            sns.lineplot(bpos_domain.query('iInBlock.between(-11, 21)'),
                         x='iInBlock', y='pred_selected_high', ax=ax_[0], color=color, legend=False)
            sns.lineplot(bpos_domain.query('iInBlock.between(-11, 21)'),
                         x='iInBlock', y='pred_switch', ax=ax_[1], color=color, label=label)
            
            _, conf_matrix, confusion_labels = compute_confusion_matrix(events, normalize=None)
            confusion_matrices[domain].append(conf_matrix)
            

            _, conf_matrix, _ = compute_confusion_matrix(events, normalize='col')
            confusion_matrices_normalized[domain].append(conf_matrix)
        checkpoint_models.append(label)
        
        pred_policies = pts.calc_conditional_probs(events, add_grps='domain', htrials=2, sortby='pevent', pred_col='pred_switch')
        pred_policies['model'] = label
        gt_policies = pd.concat([gt_policies, pred_policies])


    # Ground truth data -- mimic as a checkpoint
    for ax_, (domain, bpos_domain) in zip(axes.T, bpos_.groupby('domain')):
        sns.lineplot(bpos_domain.query('iInBlock.between(-11, 21)'),
                     x='iInBlock', y='selHigh', ax=ax_[0], color='k', legend=False, linewidth=3, errorbar=None)
        sns.lineplot(bpos_domain.query('iInBlock.between(-11, 21)'),
                     x='iInBlock', y='Switch', ax=ax_[1], color='k', label='ground truth', linewidth=3, errorbar=None)
        ax_[0].vlines(x=0, ymin=-1, ymax=1.5, ls='--', color='k', zorder=0)
        ax_[1].vlines(x=0, ymin=-1, ymax=1.5, ls='--', color='k', zorder=0)
        ax_[0].set(title=domain, xlim=(-10, 20), ylabel='P(high)', ylim=(0, 1.1))
        ax_[1].set(xlabel='block position', xlim=(-10, 20),
                   ylabel='P(switch)', ylim=(0, 0.45))

        _, ax_[2] = pts.plot_sequences(gt_policies.query('model == "ground truth" & domain == @domain'), ax=ax_[2])

        fig, ax_[2] = pts.plot_sequence_points(gt_policies.query('model != "ground truth" & domain == @domain'), grp='model',
                                               palette=cmap['colors'], yval='pevent', size=3, ax=ax_[2], fig=fig, legend=False)

    add_checkpoint_colorbar(fig, axes, cmap)
    sns.despine()

    fig_path = fm.get_experiment_file(f'bpos_checkpoints_{model_name}.png', run, subdir='predictions')
    fig.savefig(fig_path, bbox_inches='tight')
    print(f'Saved checkpoint comparison plot to {fig_path}')

    for domain in domains:
        fig, axes = plot_confusion_matrix_with_bars(
            confusion_matrices[domain],
            labels=confusion_labels,
            model_names=checkpoint_models,
            normalize=None,
            cmap=cmap['colors'],
    )
        fig_path = fm.get_experiment_file(f'cm_checkpoints_{model_name}_{domain}.png', run, subdir='predictions')
        fig.savefig(fig_path, bbox_inches='tight')

        fig, axes = plot_confusion_matrix_with_bars(
            confusion_matrices_normalized,
            labels=confusion_labels,
            model_names=checkpoint_models,
            normalize='col',
            cmap=cmap['colors'],
        )
        fig_path = fm.get_experiment_file(f'cm_checkpoints_colnorm_{model_name}_{domain}.png', run, subdir='predictions')
        fig.savefig(fig_path, bbox_inches='tight')
        print(f'Saved confusion matrix plot to {fig_path}')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, default=None)
    args = parser.parse_args()
    main(run=args.run)
