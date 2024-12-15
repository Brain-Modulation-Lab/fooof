"""Plots for visualizing bic opt."""

import numpy as np

from fooof.core.modutils import safe_import, check_dependency
from fooof.plts.spectra import plot_spectra
from fooof.plts.settings import PLT_FIGSIZES
from fooof.plts.style import style_spectrum_plot, style_plot
from fooof.plts.utils import check_ax, savefig

plt = safe_import('.pyplot', 'matplotlib')

###################################################################################################
###################################################################################################

@savefig
@style_plot
@check_dependency(plt, 'matplotlib')
def plot_bic_values(models):
    """
    Plots a barplot of BIC values for all models, with bar colors based on -log10(BF).
    """
    if not models:
        raise ValueError("No models available to plot.")
    
    # Extract model indices, BIC values, and BF values
    model_indices = [f'Model {i+1}' for i in range(len(models))]
    bic_values = [model['BIC'] for model in models]
    bf_values = [model['BF'] for model in models]  # Assuming BF is already calculated

    # Compute -log10(BF) for coloring the bars
    log_bf = [-np.log10(bf) if bf > 0 else 0 for bf in bf_values]  # Avoid log(0)

    # Normalize the log_bf values for coloring
    norm = plt.Normalize(vmin=min(log_bf), vmax=max(log_bf))
    colors = plt.cm.viridis(norm(log_bf))  # Using 'viridis' color map

    # Create a Seaborn barplot with custom colors
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_indices, bic_values, color=colors, alpha=0.7)

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='-log10(BF) [Bayes evidence]')

    # Adding labels and title
    plt.xlabel('Model')
    plt.ylabel('BIC')
    plt.title('BIC Values for All Models Colored by -log10(BF)')

    # Show the plot
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()