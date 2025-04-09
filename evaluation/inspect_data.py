import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(__file__, '../../')))
import utils.file_management as fm
from utils.parse_data import (add_sequence_columns, get_data_filenames,
                              load_predictions, parse_simulated_data)

logger = None


def initialize_logger(run_number):
    global logger
    logger = fm.setup_logging(run_number, 'data_generation', 'inspect_data')


def compare_train_val(run, train_events, val_events, maxT=12, minT=2,
                      fig=None, axs=None, show_plot=True):

    T_overlap_normalized = {}
    T_overlap_raw = {}
    for T in range(minT, maxT+1):
        train_events = add_sequence_columns(train_events, T)
        val_events = add_sequence_columns(val_events, T)

        unique_train_sequences = set(train_events[f'seq{T}_RL'])
        unique_val_sequences = set(val_events[f'seq{T}_RL'])

        overlap = unique_train_sequences.intersection(unique_val_sequences)
        overlap_percentage = len(overlap) / len(unique_val_sequences) * 100
        T_overlap_normalized[T] = overlap_percentage

        raw_overlap = val_events.query(f'seq{T}_RL in @overlap')
        T_overlap_raw[T] = len(raw_overlap) / len(val_events) * 100

        val_events[f'seq{T}_overlap'] = None
        val_events.loc[val_events[f'seq{T}_RL'].notna(), f'seq{T}_overlap'] = val_events.loc[val_events[f'seq{T}_RL'].notna(), f'seq{T}_RL'].isin(overlap)

    # Convert dictionaries to DataFrames for proper plotting
    overlap_norm_df = pd.DataFrame({'Sequence Length': list(T_overlap_normalized.keys()),
                                    'Overlap (%)': list(T_overlap_normalized.values())})

    overlap_raw_df = pd.DataFrame({'Sequence Length': list(T_overlap_raw.keys()),
                                   'Overlap (%)': list(T_overlap_raw.values())})

    if show_plot and fig is None:
        fig, axs = plt.subplots(ncols=2, figsize=(8, 3), layout='constrained')

    if show_plot:
        sns.barplot(x='Sequence Length', y='Overlap (%)', data=overlap_norm_df, ax=axs[0])
        axs[0].set(xlabel='sequence length', ylabel=r'($\%$) overlap',
                title='Overlap of training and validation sequences\nAs percentage of validation SEQUENCES')

        sns.barplot(x='Sequence Length', y='Overlap (%)', data=overlap_raw_df, ax=axs[1])
        axs[1].set(xlabel='sequence length', ylabel=r'($\%$) overlap',
                title='Overlap of training and validation sequences\nAs percentage of validation DATASET');
        sns.despine()
        fig.savefig(fm.get_experiment_file("dataset_overlap.png", run, subdir='agent_behavior'))

    if __name__ == "__main__":
        logger.info(f"Total T={maxT} sequences in training data: {train_events[f'seq{maxT}_RL'].nunique()}")
        logger.info(f"Total T={maxT} sequences in validation data: {val_events[f'seq{maxT}_RL'].nunique()}")

    if show_plot:
        return fig, axs, val_events
    else:
        return val_events


def match_by_base_sequence(events, sequences, base_T=3, min_count=20):
    """
    Match events to sequences based on the last base_T elements of each sequence.
    Return one matching event for each sequence in the original order.
    Only consider sequences that appear at least min_count times in the dataset.

    Args:
        events: DataFrame containing event data with sequence columns
        sequences: List of sequences to match
        base_T: Length of subsequence to use for matching

    Returns:
        List of matched events in the same order as input sequences
    """
    T = len(sequences[0])

    if f'seq{T}_RL' not in events.columns or all(events[f'seq{T}_RL'].isna()):
        events = add_sequence_columns(events, T)

    if f'seq{base_T}_RL' not in events.columns or all(events[f'seq{base_T}_RL'].isna()):
        events = add_sequence_columns(events, base_T)

    base_seqs = [seq[-base_T:] for seq in sequences]
    events_tth_seqs = events.query(f'seq{base_T}_RL in @base_seqs').copy()
    vc = events_tth_seqs[f'seq{T}_RL'].value_counts()
    frequent_seqs = vc[vc > min_count].index

    # Filter events to only include frequent sequences
    frequent_events = events_tth_seqs.query(f'seq{T}_RL in @frequent_seqs')

    # For each base_seq in the original order, find one matching event
    matched_events = []
    for full_seq, base_seq in zip(sequences, base_seqs):
        # Find events that match base_seq but are different from the full sequence
        matching = frequent_events.query(f'seq{base_T}_RL == @base_seq & seq{T}_RL != @full_seq')
        if len(matching) > 0:
            matched_events.append(matching[f'seq{T}_RL'].sample(1).item())
        elif len(events.query(f'seq{base_T}_RL == @base_seq & seq{T}_RL != @full_seq')) > 0:
            print(f'Warning: {base_seq} not found in frequent events')
            matched_events.append(events.query(f'seq{base_T}_RL == @base_seq & seq{T}_RL != @full_seq')[f'seq{T}_RL'].sample(1).item())
        else:
            matched_events.append(None)

    return matched_events


def compare_model_performance(run, train_events, model_name=None, minT=2, show_plot=True):

    model_info = fm.parse_model_info(run, model_name=model_name)

    if model_name is None:
        model_name = model_info['model_name']

    aligned_data = load_predictions(run, model_name, suffix='v')

    fig, axs = plt.subplots(ncols=3, figsize=(12, 3), layout='constrained')

    maxT = model_info['dataloader']['Sequence length (T)']
    if minT <= 0:
        minT = maxT + minT

    fig, axs, aligned_data = compare_train_val(run, train_events, aligned_data,
                                               maxT=maxT, minT=minT, fig=fig, axs=axs, show_plot=show_plot)

    if __name__ == "__main__":
        logger.info(f"Overall prediction accuracy: {aligned_data['pred_correct_k0'].mean() * 100:.2f}%")

    if show_plot:
        T_accuracy_unique = {}
        for T in range(minT, maxT + 1):
            T_accuracy_unique[T] = aligned_data.query(f'seq{T}_overlap == False')['pred_correct_k0'].mean() * 100

        accuracy_df = pd.DataFrame({
            'Sequence Length': list(T_accuracy_unique.keys()),
            'Accuracy (%)': list(T_accuracy_unique.values())})

        sns.barplot(x='Sequence Length', y='Accuracy (%)', data=accuracy_df, ax=axs[-1])
        axs[-1].axhline(y=aligned_data['pred_correct_k0'].mean() * 100, color='k', linestyle='--')
        axs[-1].set(xlabel='sequence length', ylabel='prediction accuracy (%)',
                    title='Model prediction accuracy\nby unique sequences')
        sns.despine()
        fig.savefig(fm.get_experiment_file(f"model_performance_{model_name}.png", run, subdir='models'))


def inspect_batches():
    pass


def main(run=None, model_name=None):

    initialize_logger(run)

    train_events = parse_simulated_data(*get_data_filenames(run, suffix='tr'))
    try:
        compare_model_performance(run, train_events, model_name)
    except AssertionError:
        val_events = parse_simulated_data(*get_data_filenames(run, suffix='v'))
        _ = compare_train_val(run, train_events, val_events)


if __name__ == "__main__":

    print('-' * 80)
    print('inspect_data.py\n')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, default=None)
    args = parser.parse_args()
    main(run=args.run)
