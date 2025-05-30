import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.file_management as fm
import utils.parse_data as parse
from utils.checkpoint_processing import (add_checkpoint_colorbar,
                                         generate_checkpoint_colormap,
                                         process_checkpoints)

vocab = ['R', 'r', 'L', 'l']
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}


def tokenize(sequences):
    if isinstance(sequences, str):
        return torch.tensor([stoi[char] for char in sequences])
    else:
        return torch.tensor([[stoi[char] for char in sequence]
                             for sequence in sequences])


def get_common_sequences(T, run=None, events=None, min_count=50, k=10):
    if events is None:
        events = parse.parse_simulated_data(*parse.get_data_filenames(run, suffix='v'))

    if f'seq{T}_RL' not in events.columns or all(events[f'seq{T}_RL'].isna()):
        events = parse.add_sequence_columns(events, T)

    vc = events[f'seq{T}_RL'].value_counts()
    sequences = vc[vc > min_count].index.tolist()[:k]
    sequences_values = vc[vc > min_count].values.tolist()[:k]
    return events, sequences, sequences_values


def get_block_transition_sequences(events, T, trial_range=(-3, 9), high_port=1):
    """Get block transitions from (e.g.) left->right blocks (high_port=1)
    where agent had ~stable selection of Left at transition
    """
    if high_port == 1:
        seq = 'LL'
    elif high_port == 0:
        seq = 'RR'
    block_starts_1 = events.query(f'block_position == 0\
         & block_id > 0 & high_port == {high_port} & seq2_RL == "{seq}"\
         & block_length > 10 & prev_block_length > 10').trial_number.values
    sequences = [events.loc[np.arange(trial_range[0], trial_range[1]) + b, 'seq6_RL']
              for b in block_starts_1]
    return sequences


def trim_leading_duplicates(seq_list):
    if not seq_list:
        return []
    first = seq_list[0]
    # Find the index where the first non-duplicate occurs
    i = 1
    while i < len(seq_list) and seq_list[i] == first:
        i += 1
    return [first] + seq_list[i:]


# def predict_token(model, sequences):
#     tokenized_sequences = tokenize(sequences)
#     device = next(model.parameters()).device
#     tokenized_sequences = tokenized_sequences.to(device)
#     logits, _ = model(tokenized_sequences)
#     probs = F.softmax(logits[:, -1, :], dim=-1)
#     max_probs, predicted_token = probs.max(dim=1)
#     predicted_token = [itos[t] for t in predicted_token.detach().cpu().numpy()]
#     return predicted_token


def get_uncertain_sequences(sequences, model, feature='all',
                            is_uncertain=True, threshold=0.7):

    def combine_logits_by_group(logits, group_label=''):
        """Combine logits by grouping tokens (choice or reward)"""
        group_idcs = {
            'choice': [[0, 1], [2, 3]],  # Right and left
            'reward': [[0, 2], [1, 3]]   # Reward and unreward
        }

        grouped_logits = []

        for i, idcs in enumerate(group_idcs.get(group_label)):
            grouped_logits.append(logits[:, :, idcs].sum(dim=2))

        combined_logits = torch.stack(grouped_logits, dim=2)
        return combined_logits

    tokenized_sequences = tokenize(sequences)
    device = next(model.parameters()).device
    tokenized_sequences = tokenized_sequences.to(device)

    logits, _ = model(tokenized_sequences)

    if feature == 'choice':
        combined_logits = combine_logits_by_group(logits, group_label='choice')
        probs = F.softmax(combined_logits[:, -1, :], dim=1)
        uncertainty = 1-abs(0.5 - probs.detach().cpu().numpy()[:, 0])
    else:
        probs = F.softmax(logits[:, -1, :], dim=1)
        uncertainty, predicted_token = probs.max(dim=1)
        uncertainty = 1-uncertainty.detach().cpu().numpy()

    # Sort sequences by uncertainty/certainty
    sorted_indices = np.argsort(uncertainty)
    sorted_sequences = [sequences[i] for i in sorted_indices]
    sorted_uncertainty = uncertainty[sorted_indices]

    if is_uncertain:
        # Get sequences above threshold
        mask = sorted_uncertainty > threshold
        select_sequences = [seq for i, seq in enumerate(sorted_sequences) if mask[i]]
        select_uncertainty = sorted_uncertainty[mask]
    else:
        mask = sorted_uncertainty < threshold
        select_sequences = [seq for i, seq in enumerate(sorted_sequences) if mask[i]]
        select_uncertainty = sorted_uncertainty[mask]

    return select_sequences, select_uncertainty


def embed_sequence(model, sequence, flatten=True):
    device = next(model.parameters()).device
    input_tensor = tokenize(sequence).unsqueeze(0).to(device)

    pos = torch.arange(0, len(input_tensor), dtype=torch.long,
                       device=device)

    # Get token embeddings and positional embeddings (exactly as in model.forward)
    token_embeddings = model.transformer.wte(input_tensor)
    pos_embeddings = model.transformer.wpe(pos)

    # Combine token and positional embeddings (exactly as the model does)
    combined_embeddings = token_embeddings + pos_embeddings

    if flatten:
        # Flatten to get a single vector representation of the entire sequence
        combined_embeddings = combined_embeddings.squeeze(0).flatten()
    else:
        combined_embeddings = combined_embeddings.squeeze(0)

    return combined_embeddings.detach().cpu().numpy()


def token_embedding_similarity(model, vocab_mappings, ax=None):
    """
    Analyze the token embeddings from the input embedding layer
    (model.transformer.wte). Weights are equivalent to embeddings.

    Parameters:
    -----------
    model : transformer model
        The trained transformer model
    vocab : list or dict
        Vocabulary list or dictionary mapping tokens to indices

    Returns:
    --------
    embeddings : numpy.ndarray
        The token embedding matrix
    similarity_matrix : numpy.ndarray
        Matrix of cosine similarities between token embeddings
    """
    # Extract the token embedding matrix
    token_embeddings = model.transformer.wte.weight.detach().cpu().numpy()

    # Compute cosine similarity between token embeddings
    similarity_matrix = cosine_similarity(token_embeddings)
    ax = plot_similarity(similarity_matrix, vocab_mappings.keys(), ax=ax, annot=True)

    return token_embeddings, similarity_matrix


def sequence_embedding_similarity(model, sequences, stoi):
    """
    Analyze sequences as they would actually be encoded for the attention layers,
    including positional embeddings and proper concatenation.

    Parameters:
    -----------
    model : transformer model
        The trained transformer model
    sequences : list of str
        List of sequences to analyze
    stoi : dict
        Mapping from tokens to indices

    Returns:
    --------
    sequence_embeddings : numpy.ndarray
        Array of sequence embeddings with positional information
    similarity_matrix : numpy.ndarray
        Matrix of cosine similarities between sequence embeddings
    """
    # Get the full embeddings as they're processed by the model
    sequence_embeddings = []

    for sequence in sequences:
        sequence_embeddings.append(embed_sequence(model, sequence, stoi))

    sequence_embeddings = np.array(sequence_embeddings)

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(sequence_embeddings)

    return sequence_embeddings, similarity_matrix


def plot_similarity(similarity_matrix, sequences, ax=None, annot=False):
    """
    Plot the similarity matrix (e.g. token embedding similarity or sequence embedding similarity).
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3), layout='constrained')

    sns.heatmap(
        similarity_matrix,
        annot=annot,
        fmt='.2f',
        cmap='RdBu',
        vmin=-1,
        vmax=1,
        xticklabels=sequences,
        yticklabels=sequences,
        ax=ax
    )
    ax.set(title='Token Embedding Similarity')
    return ax


def pca_embeddings(model, n_components=2, token_mapping=None):
    """
    Perform PCA on model's token embeddings and visualize them.

    Parameters:
    -----------
    model : transformer model
        The model containing token embeddings
    n_components : int, default=2
        Number of PCA components to extract
    token_labels : list, optional
        Labels for each token in the vocabulary

    Returns:
    --------
    pca : sklearn.decomposition.PCA
        Fitted PCA model
    embedded : numpy.ndarray
        Embeddings transformed to the PCA space
    """
    # Extract token embeddings
    embeddings = model.transformer.wte.weight.detach().cpu().numpy()

    pca = PCA(n_components=n_components)
    embedded = pca.fit_transform(embeddings)

    # Create default token labels if not provided
    if token_mapping is None:
        token_mapping = {i: f"Token {i}" for i in range(embeddings.shape[0])}

    n_plots = n_components + 1
    fig, axs = plt.subplots(ncols=n_plots, figsize=(n_plots*3, 3),
                            layout='constrained')

    if n_plots == 1:
        axs = [axs]

    df = pd.DataFrame({
        f'PC{i+1}': embedded[:, i] for i in range(n_components)
    } | {'Token': token_mapping.keys()})

    for i, ax in enumerate(axs[:-2], start=1):
        sns.scatterplot(x=f'PC{i}', y=f'PC{i+1}', data=df, ax=ax, s=100)
        for j, txt in enumerate(token_mapping.keys()):
            ax.annotate(txt, (df[f'PC{i}'][j], df[f'PC{i+1}'][j]), fontsize=12)
        ax.set(title=f"PC{i} vs PC{i+1}")

    # Analyze explained variance
    axs[-2].plot(range(1, n_components + 1),
                 np.cumsum(pca.explained_variance_ratio_),
                 marker='o', color='red')
    axs[-2].set(xlabel='PC', xticks=range(1, n_components + 1),
                ylabel='Explained Variance Ratio', ylim=(0, 1.05) )

    embeddings, _ = token_embedding_similarity(model, token_mapping, ax=axs[-1])

    sns.despine()

    return pca, embedded


def cluster_sequences_hierarchical(similarity_matrix, sequences, replot=True,
                                   method='ward', metric='euclidean'):
    """
    Perform hierarchical clustering with optimal leaf ordering.

    Parameters:
    -----------
    similarity_matrix : numpy.ndarray
        Matrix of similarities between sequences. Can be either:
        - Square matrix (n x n) of similarities/distances
        - Condensed distance matrix (1D array of upper triangular elements)
    sequences : list of str
        List of sequences 
    method : str, default='ward'
        Linkage method: 'ward', 'complete', 'average', or 'single'
    metric : str, default='euclidean'
        Distance metric (use 'precomputed' if similarity_matrix is a distance matrix)
    """
    from scipy.cluster.hierarchy import (dendrogram, linkage,
                                         optimal_leaf_ordering)
    from scipy.spatial.distance import squareform

    # Convert similarity to distance if needed (distance = 1 - similarity)
    if metric != 'precomputed':
        distance_matrix = 1 - similarity_matrix
    else:
        distance_matrix = similarity_matrix

    # Check if matrix is already in square form
    is_square = (len(distance_matrix.shape) == 2 and 
                 distance_matrix.shape[0] == distance_matrix.shape[1])

    if not is_square:
        # Convert condensed distance matrix to square form
        distance_matrix = squareform(distance_matrix)

    # Compute linkage matrix
    Z = linkage(distance_matrix, method=method)

    # Apply optimal leaf ordering to get a better visualization
    Z_ordered = optimal_leaf_ordering(Z, distance_matrix)

    # Plot dendrogram
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    dendrogram(
        Z_ordered,
        labels=sequences,
        leaf_rotation=90,
        leaf_font_size=10,
        color_threshold=0.7*np.max(Z_ordered[:,2]),  # Threshold for coloring branches
        ax=ax
    )
    ax.set(title=f'Hierarchical Clustering ({method} linkage)')

    # Get the optimal ordering of sequences
    ordered_indices = dendrogram(Z_ordered, no_plot=True)['leaves']
    ordered_sequences = [sequences[i] for i in ordered_indices]

    # Reorder similarity matrix according to this ordering
    ordered_sim_matrix = similarity_matrix[ordered_indices][:, ordered_indices]

    if replot:
        fig, ax = plt.subplots(figsize=(0.18 * len(sequences), 0.18 * len(sequences)),
                               layout='constrained')
        ax = plot_similarity(ordered_sim_matrix, ordered_sequences, ax=ax)
        ax.set(title='Similarity Matrix (Ordered by Hierarchical Clustering)')

    return ordered_sequences, ordered_sim_matrix, Z_ordered


def extract_token_embeddings(model, **kwargs):
    """Extract token embedding weights from a model."""
    return model.transformer.wte.weight.detach().cpu().numpy()


def embedding_processor(model, **kwargs):
    """Process a model to extract its token embeddings."""
    embeddings = extract_token_embeddings(model, **kwargs)
    return embeddings


def analyze_embedding_evolution(run, reference_type='final', save_results=False):
    """
    Analyze the evolution of token embeddings across checkpoints.

    Parameters:
    -----------
    run : int
        Run number
    reference_type : str, default='final'
        Type of reference model to use ('final' or 'first')
    save_results : bool, default=False
        Whether to save visualizations and data to disk
    """
    # Process all checkpoints to extract embeddings
    checkpoint_data = process_checkpoints(
        run=run,
        processor_fn=embedding_processor,
        reference_type=reference_type,
        save_results=save_results
    )

    if not checkpoint_data:
        return

    all_embeddings = checkpoint_data['results']
    checkpoint_numbers = checkpoint_data['checkpoint_numbers']
    model_labels = checkpoint_data['model_labels']
    reference_idx = checkpoint_data['reference_idx']

    # Get reference embeddings
    reference_embeddings = all_embeddings[reference_idx]

    # Fit PCA on reference embeddings
    pca = PCA(n_components=2)
    pca.fit(reference_embeddings)

    # Project all embeddings using the same PCA
    projected_embeddings = [pca.transform(emb) for emb in all_embeddings]

    # Create visualization
    plot_embedding_evolution(projected_embeddings, model_labels,
                             run, save_results)

    # Calculate distances from reference
    calculate_embedding_distances(all_embeddings, checkpoint_numbers, model_labels, 
                               reference_idx, run, save_results)

    return projected_embeddings


def plot_embedding_evolution(projected_embeddings, model_labels,
                             run, save_results=False):
    """Visualize the evolution of embeddings in 2D space."""

    n_tokens = projected_embeddings[0].shape[0]

    fig, ax = plt.subplots(figsize=(4, 3), layout='constrained')
    cmap = generate_checkpoint_colormap(checkpoint_labels=model_labels, )

    cmap['colors']['final'] = 'red'

    for token_idx in range(n_tokens):
        # Get embeddings for this token across all checkpoints
        token_positions = np.array([emb[token_idx] for emb in projected_embeddings])
        
        token_positions = pd.DataFrame(token_positions, columns=['PC1', 'PC2'])
        token_positions['model'] = [m.replace('cp', '') for m in model_labels]
        sns.scatterplot(x='PC1', y='PC2', data=token_positions, palette=cmap['colors'], 
                      s=60, marker='o', ax=ax, hue='model',
                      edgecolor=None)
        sns.lineplot(x='PC1', y='PC2', data=token_positions, color='gray', 
                     ax=ax, linewidth=1, alpha=0.5, sort=False)

        ax.text(token_positions.query('model=="final"')['PC1'].item(),
                token_positions.query('model=="final"')['PC2'].item(), 
                str(itos[token_idx]), 
                fontsize=8, ha='center', va='center', zorder=len(model_labels)+1)
        
        ax.text(token_positions.query('model=="0"')['PC1'].item(),
                token_positions.query('model=="0"')['PC2'].item(), 
                '*', color='red', 
                fontsize=14, ha='center', va='center', zorder=len(model_labels)+1)
    
    ax.set(xlabel='PC 1',
           ylabel='PC 2',
           title=f'Token Embeddings Across Training')

    add_checkpoint_colorbar(fig, ax, cmap, ground_truth=False,
                            colorbar_kwargs = {'shrink': 0.8, 'pad': 0.02, 'location': 'right'})
    
    sns.despine()
    
    if save_results:
        out_path = fm.get_experiment_file(f'token_embedding_evolution.png', run, subdir='predictions')
        plt.savefig(out_path, bbox_inches='tight', dpi=300)


def calculate_embedding_distances(all_embeddings, checkpoint_numbers, model_labels, 
                               reference_idx, run, save_results=False):
    """Calculate distances between embeddings across checkpoints."""
    reference_embeddings = all_embeddings[reference_idx]

    cos_sims = []
    eucl_dists = []

    for embeddings in all_embeddings:
        # Average cosine similarity across all tokens
        cos_sim = np.mean([
            np.dot(embeddings[i], reference_embeddings[i]) / 
            (np.linalg.norm(embeddings[i]) * np.linalg.norm(reference_embeddings[i]))
            for i in range(len(embeddings))
        ])
        cos_sims.append(cos_sim)

        # Average Euclidean distance across all tokens
        eucl_dist = np.mean([
            np.linalg.norm(embeddings[i] - reference_embeddings[i])
            for i in range(len(embeddings))
        ])
        eucl_dists.append(eucl_dist)

    df = pd.DataFrame({
        'Checkpoint': checkpoint_numbers,
        'Model': model_labels,
        'Cosine Similarity': cos_sims,
        'Euclidean Distance': eucl_dists
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 3.5), sharex=True, layout='constrained')

    plot_df = df[df.index != reference_idx]

    sns.lineplot(data=plot_df, x='Checkpoint', y='Cosine Similarity', ax=ax1, marker='o', color='k')
    ax1.set(ylabel='Cosine\nSimilarity',
            title=f'Embedding Similarity\nto Reference Model')

    sns.lineplot(data=plot_df, x='Checkpoint', y='Euclidean Distance', ax=ax2, marker='o', color='k')
    ax2.set(ylabel='Euclidean\nDistance',
            xlabel='Checkpoint')
    sns.despine()

    if save_results:
        out_path = fm.get_experiment_file(f'token_embedding_distances.png', run, subdir='predictions')
        plt.savefig(out_path, bbox_inches='tight', dpi=300)


def plot_attribution_all_targets(attribution_func, model, sequences, vocab, stoi, ncols=1, **kwargs):

    for sequence in sequences:
        fig, axs = plt.subplots(ncols=ncols, nrows=4, figsize=(3*ncols, 1.5), layout='constrained', sharex=True)
        if ncols == 1:
            axs = [np.array(ax) for ax in axs]
        for t, ax in zip(vocab, axs):
            target_idx = stoi[t]
            attribution = attribution_func(model, sequence, stoi, target_idx, **kwargs)

            if not isinstance(attribution, dict):
                attribution = {attribution_func.__name__: attribution}

            for ax_, attribution_type in zip(ax.flatten(), attribution.keys()):
                data = attribution[attribution_type].reshape(1, -1)  # Reshape to a 1 × sequence_length matrix  
                hm = sns.heatmap(data, ax=ax_, vmin=-1, vmax=1, cmap="RdBu", annot=False, cbar=False)

                for i, char in enumerate(sequence):
                    ax_.text(i + 0.5, 0.5, char,
                            ha='center',
                            va='center',
                            fontsize=10,
                            fontweight='bold',
                            color='k')

                ax_.yaxis.tick_right()
                ax_.set_yticks([0.5], labels=[t], rotation=0)
                if t == vocab[0]:
                    ax_.set_title(attribution_type)
            fig.suptitle(f"Attribution for sequence: {sequence}", y=1.1)
            ax_.set_xticks([]);

        cbar_ax = fig.add_axes([1.01, 0.15, 0.01, 0.6])  # [x, y, width, height]
        norm = plt.Normalize(vmin=-1, vmax=1)
        sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax)
