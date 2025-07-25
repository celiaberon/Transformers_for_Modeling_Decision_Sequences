import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import utils.file_management as fm
from utils.parse_data import (align_predictions_with_gt, get_data_filenames,
                              load_predictions, parse_simulated_data)
from utils.model_utils import parse_model_info

def initialize_logger(run):
    global logger
    logger = fm.setup_logging(run, 'inference', 'evaluate_transformer_guess')

def print_accuracy(aligned_data):

    # Compute accuracy of all 4 possible characters
    correct = np.sum(aligned_data['k0'] == aligned_data['pred_k0'])
    total = len(aligned_data)
    accuracy = correct / total if total > 0 else 0
    logger.raw(f"\n{' ':>37}Accuracy: {accuracy:.2%} ({correct}/{total} correct predictions)")

    # Compute and print the adjusted accuracy (choice only scoring)
    choice_accuracy = np.mean(aligned_data['pred_choice'] == aligned_data['choice'])
    logger.raw(f"{' ':>10}Choice only Accuracy (Right/Left same): {choice_accuracy:.2%}")

    # Compute and print the accuracy on reward predictions
    reward_accuracy = np.mean(aligned_data['pred_reward'] == aligned_data['reward'])
    logger.raw(f"{' ':>15}Reward Accuracy (Upper/Lower same): {reward_accuracy:.2%}")


def compute_confusion_matrix(aligned_data, normalize=None):
    aligned_data = aligned_data.copy().dropna(subset=['k0', 'pred_k0'])
    ground_truth_tokens = list(aligned_data['k0'].values)
    prediction_tokens = list(aligned_data['pred_k0'].values)
    confusion = Counter((gt, pred) for gt, pred in zip(ground_truth_tokens, prediction_tokens))
    labels = sorted(set(ground_truth_tokens + prediction_tokens))
    label_map = {label: i for i, label in enumerate(labels)}

    conf_matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for (gt_char, pred_char), count in confusion.items():
        i, j = label_map[gt_char], label_map[pred_char]
        conf_matrix[i, j] = count

    if normalize == 'all':
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum()
    elif normalize == 'row':
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    elif normalize == 'col':
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=0)[np.newaxis, :]

    return confusion, conf_matrix, labels


def plot_confusion_matrix(aligned_data, run, model_name, domain=''):

    confusion, conf_matrix, labels = compute_confusion_matrix(aligned_data)
    logger.raw("\nConfusion Matrix:")
    logger.raw("Ground Truth -> Prediction: Count")
    for (gt_char, pred_char), count in sorted(confusion.items(), key=lambda x: x[1], reverse=True):
        logger.raw(f"{gt_char} -> {pred_char}: {count}")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set(xlabel="Predicted Label", ylabel="Ground Truth Label", title="Confusion Matrix")
    cm_file = fm.get_experiment_file("cm_pred_run_{}.png", run, f"_{model_name}_{domain}", subdir='predictions')
    logger.info(f'saved confusion matrix to {cm_file}')
    fig.savefig(cm_file)


def print_switches(aligned_data):
    # # Compute and print switch percentages
    gt_switch_percentage = round(aligned_data.switch.mean() * 100, 2)
    pred_switch_percentage = round(aligned_data.pred_switch.mean() * 100, 2)
    logger.raw(f"\nSwitch Percentage (model): {pred_switch_percentage:.2f}%")
    logger.raw(f"Switch Percentage (ground truth): {gt_switch_percentage:.2f}%")


def calculate_accuracy_ignore_case(ground_truth, predictions):
    """Calculate accuracy ignoring 'R'-'r' and 'L'-'l' differences"""
    correct = sum(
        gt.upper() == pred.upper()
        for gt, pred in zip(ground_truth, predictions)
    )
    return correct / len(ground_truth) if len(ground_truth) > 0 else 0


def main(run=None, model_name=None):

    if run is None:
        run = fm.get_latest_run()

    initialize_logger(run)

    if model_name is None:
        # Get model info from metadata
        model_info = parse_model_info(run, model_name=model_name)
        model_name = model_info['model_name']
    aligned_data = load_predictions(run, model_name, suffix='v')
    aligned_data = aligned_data.dropna()

    for domain, data in aligned_data.groupby('domain'):
        logger.raw(f'\nAnalysis for Domain {domain}')
        print_accuracy(data)
        print_switches(data)
        if model_name.split('_')[-1].startswith('cp'):
            plot_confusion_matrix(data, run, model_name, domain)

if __name__ == "__main__":
    print('-' * 80)
    print('evaluate_transformer_guess.py\n')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    args = parser.parse_args()
    main(run=args.run, model_name=args.model_name)
