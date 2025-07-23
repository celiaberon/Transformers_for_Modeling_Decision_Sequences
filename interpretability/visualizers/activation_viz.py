"""ActivationVisualizer: plotting utilities for neuron activations, selectivity, and MLP structure."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from interpretability.core.base import BaseVisualizer

class ActivationVisualizer(BaseVisualizer):
    """
    Visualizer for activation and MLP analysis.
    Provides plotting utilities for neuron activations, selectivity,
    and MLP structure.
    """
    def plot_activation_bars(
        self, activations: dict[str, np.ndarray], layer: str
    ) -> tuple[plt.Figure, np.ndarray]:
        fig, axs = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(15, 10),
            sharex=True,
            sharey=True
        )
        for (token, diff), ax in zip(activations.items(), axs.flatten()):
            bars = ax.bar(range(len(diff)), diff)
            for bar in bars:
                if bar.get_height() > 0:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set(
                title=f'{token} Tokens',
                xlabel='Neuron Index',
                ylabel='Activation Difference'
            )
        fig.suptitle(f'{layer.capitalize()} Layer', fontsize=14)
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        return fig, axs

    def plot_activation_heatmap(
        self,
        activations: dict[str, np.ndarray],
        layer: str,
        ax: plt.Axes | None = None,
        **kwargs
    ) -> plt.Axes:
        cmap_by_layer = {
            'input': (-3, 3),
            'gelu': (-1, 1),
            'output': (-1, 1)
        }
        diff_data = np.vstack(list(activations.values()))
        ax = sns.heatmap(
            diff_data,
            cmap="RdBu_r",
            center=0,
            annot=False,
            fmt=".2f",
            xticklabels=range(diff_data.shape[1]),
            yticklabels=activations.keys(),
            ax=ax,
            # vmin=cmap_by_layer[layer][0],
            # vmax=cmap_by_layer[layer][1],
            **kwargs
        )

        if self.num_tokens > 1:
            if self.token_idx == self.num_tokens - 1:
                ax.set(title='', xlabel='Neuron Index')
            if self.token_idx == 0:
                ax.set(
                    title=f'{layer.capitalize()} Layer',
                    xticks=[],
                    xlabel=''
                )
            else:
                ax.set(xlabel='', xticks=[], title='')
            if self.layer_idx == 0:
                ax.set(ylabel='Token')
            else:
                ax.set(yticks=[], ylabel='')
        else:
            ax.set(
                title=f'{layer.capitalize()} Layer',
                xlabel='Neuron Index',
                ylabel='Token'
            )

        return ax

    def create_mlp_visualization(
        self,
        fig_title: str = None
    ) -> tuple[plt.Figure, dict[str, list[plt.Axes]]]:
        n_layers = len(self.analyzer.layer_composition)
        figsize = (max(self.analyzer.layer_composition.values()) * 1, n_layers * 1.8)
        fig = plt.figure(figsize=figsize, constrained_layout=True)

        if fig_title:
            fig.suptitle(fig_title, fontsize=12)

        subfigs = fig.subfigures(n_layers, 1)  # subfigure for each layer
    
        if n_layers == 1:
            subfigs = [subfigs]

        axes_dict = {}
        for subfig, (layer_name, n_neurons) in zip(
            subfigs, self.analyzer.layer_composition.items()
        ):
            title = f"{layer_name.capitalize()} Layer ({n_neurons} neurons)"
            subfig.suptitle(title, fontsize=12)
            axes = subfig.subplots(nrows=1, ncols=n_neurons)  # ax for each neuron
            if n_neurons == 1:
                axes = [axes]
            axes_dict[layer_name] = axes

        return fig, axes_dict 