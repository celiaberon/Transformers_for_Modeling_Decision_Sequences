"""Functions for plotting confusion matrices with barplots in each cell."""

from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_confusion_matrix_with_bars(
    confusion_matrices: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
    labels: List[str],
    model_names: Optional[List[str]] = None,
    figsize: tuple = (12, 10),
    title: str = "Confusion Matrix with Prediction Distributions",
    cmap: str = "Blues",
    normalize: bool = True,
    show_values: bool = True,
    value_format: str = "{:.2f}",
    fontsize: int = 10,
    bar_width: float = 0.8,
    bar_spacing: float = 0.1
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot a confusion matrix where each cell contains a barplot showing the distribution
    of predictions across multiple models/checkpoints.
    
    Args:
        confusion_matrices: Either a single confusion matrix or a list/dict of matrices
            for multiple models. Each matrix should be of shape (n_classes, n_classes).
        labels: List of class labels
        model_names: Optional list of model names (required if confusion_matrices is a list/dict)
        figsize: Figure size as (width, height)
        title: Plot title
        cmap: Colormap for the background heatmap
        normalize: Whether to normalize the confusion matrices
        show_values: Whether to show numerical values in cells
        value_format: Format string for cell values
        fontsize: Font size for labels and values
        bar_width: Width of bars in the barplots
        bar_spacing: Spacing between bars in the barplots
        
    Returns:
        Tuple of (figure, axes array)
    """
    # Convert input to list of matrices
    if isinstance(confusion_matrices, np.ndarray):
        confusion_matrices = [confusion_matrices]
        if model_names is None:
            model_names = ["Model"]
    elif isinstance(confusion_matrices, dict):
        model_names = list(confusion_matrices.keys())
        confusion_matrices = list(confusion_matrices.values())
    
    n_classes = len(labels)
    n_models = len(confusion_matrices)
    
    # Create figure and axes
    fig, axes = plt.subplots(n_classes, n_classes, figsize=figsize)
    axes = np.array(axes).reshape(n_classes, n_classes)
    
    # Normalize matrices if requested
    if normalize:
        confusion_matrices = [
            cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            for cm in confusion_matrices
        ]
    
    # Plot each cell
    for i in range(n_classes):
        for j in range(n_classes):
            ax = axes[i, j]
            
            # Get values for this cell across all models
            cell_values = [cm[i, j] for cm in confusion_matrices]

            bars = ax.bar(
                np.arange(n_models),
                cell_values,
                width=bar_width,
                color=sns.color_palette(cmap, n_models)
            )
            
            # Customize cell appearance
            ax.set_ylim(0, 1 if normalize else max(max(cm.flatten()) for cm in confusion_matrices))
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add labels for first row and column
            if i == 0:
                ax.set_title(labels[j], fontsize=fontsize)
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=fontsize)
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3)
    
    # Add model names as x-axis labels at the bottom
    if model_names:
        fig.text(
            0.5,
            0.02,
            "Models",
            ha='center',
            fontsize=fontsize
        )
        for i, name in enumerate(model_names):
            fig.text(
                0.5 + (i - (n_models-1)/2) * 0.1,
                0.02,
                name,
                ha='center',
                fontsize=fontsize-2,
                rotation=45
            )
    
    plt.suptitle(title, fontsize=fontsize+2, y=0.95)
    plt.tight_layout()
    
    return fig, axes

def example_usage():
    """Example usage of the confusion matrix plotting function."""
    # Create example data
    n_classes = 4
    n_models = 3
    
    # Generate random confusion matrices
    confusion_matrices = [
        np.random.randint(0, 100, size=(n_classes, n_classes))
        for _ in range(n_models)
    ]
    
    # Define labels
    labels = ['R', 'r', 'L', 'l']
    model_names = ['Model A', 'Model B', 'Model C']
    
    # Plot confusion matrix
    fig, axes = plot_confusion_matrix_with_bars(
        confusion_matrices,
        labels=labels,
        model_names=model_names,
        normalize=True,
        title="Example Confusion Matrix with Multiple Models"
    )
    
    plt.show()

if __name__ == "__main__":
    example_usage() 