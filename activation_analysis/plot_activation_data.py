#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from matplotlib.ticker import LogLocator, NullFormatter
import configs
from utils.activation_plot_utils import categorize_layer, get_layer_color
from typing import List, Dict
import cli.logger_formatter as logger_formatter
from cli.parsers import ActivationAnalysisPlotParser

USE_BOLD_FONT = False
YR_EXPANSION = 100 
N_TICKS = 7

MARKERSIZE = 11
N_BINS = 200

BOLD_BOUNDARY_TICKS = False  # <--- Toggle bolding of left/0/right x-tick labels
USE_BOLD_FONT = False 

SHORT_NAMES = {
    configs.GPT2: "GPT2",
    f"{configs.FACEBOOK_BART.replace('/','_')}": "BART",
    configs.VIT_BASE_PATCH16_224: "ViT",
    configs.SWIN_BASE_PATCH4_WINDOW7_224: "Swin",
}

MODEL_ALIASES = {
    configs.FACEBOOK_BART: configs.FACEBOOK_BART.replace("/", "_"),
}

# Layer types to plot (in order)
LAYER_TYPES = [
    # "embedding",
    "block",
    "attention", 
    "mlp"
]


PALETTE = {
    "block": ("Block", '#4169E1'),
    "attention": ("MHA", '#FF8C00'),
    "mlp": ("MLP", '#00b050'),
    None: (str(None), "lightgray")
}

def plot_per_op_histogram(models, model_aliases: Dict[str, str], data_path: str, layer_types: str, output_path: str, logger=None, n_bins: int = N_BINS, short_names: Dict[str, str] = SHORT_NAMES):

    # First, collect all the data organized by layer type
    plot_data = {layer_type: [] for layer_type in LAYER_TYPES}

    for model_config, dataset_config in models.items():
        model_name = model_config
        dataset_name = dataset_config
        if model_name in model_aliases:
            model_name = model_aliases[model_name]
        combined_hists = np.load(
            os.path.join(data_path, f"{model_name}-{dataset_name}-fp32-0-layer_histograms_per_layer_type.npz"), 
            allow_pickle=True
        )
        for layer_type in LAYER_TYPES:
            if layer_type not in combined_hists.files:
                # Add placeholder if layer type doesn't exist for this model
                plot_data[layer_type].append(None)
                continue
            layer_dict = combined_hists[layer_type].item()
            hist = layer_dict["hist"]
            vmin = layer_dict["min"]
            vmax = layer_dict["max"]

            logger.debug(f"Processing {layer_type} for {model_name}: vmin={vmin:.0f}, vmax={vmax:.0f}")

            # xs = np.linspace(vmin, vmax, 200)

            # Compute symmetric range
            vabs = max(abs(vmin), abs(vmax))
            xs = np.linspace(-vabs, vabs, n_bins)

            # Original x positions for the histogram
            orig_xs = np.linspace(vmin, vmax, len(hist))

            # Create a new zero-padded histogram over the symmetric range
            new_hist = np.zeros_like(xs)

            # Find the indices in xs that correspond to the original range
            mask = (xs >= vmin) & (xs <= vmax)

            # Interpolate or map the original histogram values to those indices
            new_hist[mask] = np.interp(xs[mask], orig_xs, hist)

            # Replace the histogram and x-values
            hist = new_hist
            vmin = -vabs
            vmax = vabs

            plot_data[layer_type].append({
                'model_name': model_name,
                'dataset_name': dataset_name,
                'hist': hist,
                'xs': xs,
                'vmin': vmin,
                'vmax': vmax
            })

    # Get number of models
    n_models = len(models)

    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("notebook")

    # Calculate GLOBAL y-axis limits for each layer type (row)
    y_limits = {}
    for layer_type in LAYER_TYPES:
        global_min = float('inf')
        global_max = 0
        
        for data in plot_data[layer_type]:
            if data is not None:
                # Get non-zero histogram values
                nonzero_vals = data['hist'][data['hist'] > 0]
                if len(nonzero_vals) > 0:
                    global_min = min(global_min, np.min(nonzero_vals))
                    global_max = max(global_max, np.max(nonzero_vals))
        
        if global_max > 0:
            y_limits[layer_type] = (global_min * 0.5, global_max * 2)
        else:
            y_limits[layer_type] = (1e-1, 1e5)  # Default

    # Create the subplot grid: 3 rows (layer types) x n_models columns
    n_rows = len(LAYER_TYPES)
    fig, axes = plt.subplots(n_rows, n_models, figsize=(5*n_models, 9), sharey="row")

    # Handle case where there's only one model
    if n_models == 1:
        axes = axes.reshape(-1, 1)

    for row_idx, layer_type in enumerate(LAYER_TYPES):
        layer_data = plot_data[layer_type]
        
        for col_idx, data in enumerate(layer_data):
            ax = axes[row_idx, col_idx]
            
            if data is None:
                # No data for this layer type in this model
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            ax.set_yscale('log', nonpositive='clip')
            ax.bar(data['xs'], data['hist'], 
                width=(data['vmax']-data['vmin'])/200, 
                color=sns.color_palette("deep")[row_idx], 
                edgecolor="none", alpha=0.8)
            
            # CRITICAL: Set the same y-limits for all plots in this row
            ax.set_ylim(y_limits[layer_type])
            
            # Set exactly 5 x-ticks: left, mid-left, 0, mid-right, right
            vmin, vmax = data['vmin'], data['vmax']
            x_ticks = [vmin, vmin/2, 0, vmax/2, vmax]
            ax.set_xticks(x_ticks)
            
            # Format x-tick labels and optionally make boundary ticks + 0 bold
            x_labels = []
            for i, tick in enumerate(x_ticks):
                label = f"{tick:.0f}"
                if BOLD_BOUNDARY_TICKS and (i == 0 or i == 2 or i == 4):
                    x_labels.append(r"$\mathbf{" + label + "}$")
                else:
                    x_labels.append(label)
            ax.set_xticklabels(x_labels)
            
            # Title only on top row
            if row_idx == 0:
                ax.set_title(f"{short_names.get(data['model_name'], data['model_name'])}", 
                            fontsize=26, fontweight='bold' if USE_BOLD_FONT else 'normal')
            
            # X-label only on bottom row
            if row_idx == 2:
                ax.set_xlabel("Activation value", fontsize=24, fontweight='bold' if USE_BOLD_FONT else 'normal')
            
            fig.supylabel(
                "Frequency (log-scale)",
                fontsize=24,
                fontweight='bold' if USE_BOLD_FONT else 'normal',
                x=0.003
            )
            # Y-label and layer type on first column
            # if col_idx == 0:
            #     if layer_type == "attention":
            #         layer_type_label = "MHA"
            #     else:
            #         layer_type_label = layer_type
            #     ax.set_ylabel(f"{layer_type_label.upper()}\nFrequency", fontsize=22, fontweight='bold' if USE_BOLD_FONT else 'normal')
            
            if col_idx == 0:
                if layer_type == "attention":
                    layer_type_label = "MHA"
                else:
                    layer_type_label = layer_type.upper()

                ax.set_ylabel(
                    layer_type_label,
                    fontsize=24,
                    fontweight='bold' if USE_BOLD_FONT else 'normal'
                )

            ax.tick_params(labelsize=20)
            
            # Add grid for better readability
            ax.grid(True, alpha=0.9, which='both', ls='--')

    # plt.suptitle("Layer Histograms by Type and Model", fontsize=18, y=0.995, fontweight='bold' if USE_BOLD_FONT else 'normal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"combined_layer_type_histograms_logscale.pdf"), dpi=300)
    plt.show()


def plot_min_max_bounds(models: Dict[str, str], input_path: str, output_path: str, logger, model_aliases: Dict[str, str] = MODEL_ALIASES, short_names: Dict[str, str] = SHORT_NAMES, markersize: int = MARKERSIZE, palette: Dict[str, str] = PALETTE):
    # First pass: collect all data and find global min/max
    all_data = []
    global_min = float('inf')
    global_max = float('-inf')

    for model_config, dataset_config in models.items():
        model_name = model_config
        dataset_name = dataset_config
        
        if model_name in model_aliases:
            model_name = model_aliases[model_name]

        layer_bounds = np.load(os.path.join(input_path, f"{model_name}-{dataset_name}-fp32-0-layer_bounds.npz"), allow_pickle=True)
        layers = list(layer_bounds.files)
        
        mins, maxs = [], []
        for layer in layers:
            bounds = layer_bounds[layer]
            if isinstance(bounds, np.ndarray):
                bounds = bounds.item()
            mins.append(bounds["min"])
            maxs.append(bounds["max"])
        
        all_data.append({
            'model_name': model_name,
            'dataset_name': dataset_name,
            'layers': layers,
            'mins': mins,
            'maxs': maxs
        })
        
        global_min = min(global_min, min(mins))
        global_max = max(global_max, max(maxs))

    logger.debug(f"Global min: {global_min}, Global max: {global_max}")

    sns.set_theme(style="whitegrid")
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(10*n_models, 6), sharey=True)
    if n_models == 1:
        axes = [axes]

    for col_idx, data in enumerate(all_data):
        ax = axes[col_idx]

        layers = data['layers']
        mins   = data['mins']
        maxs   = data['maxs']
        model_name = data['model_name']

        # --- SYMLOG SCALE ---
        lin = 1e1   # must match linthresh
        ax.set_yscale("symlog", linthresh=lin)

        yr = max(abs(global_min), abs(global_max)) * YR_EXPANSION

        # --- ADD COMPLETE LOG GRID ---
        # Major log ticks: ±10^n outside linear zone
        major_pos = np.logspace(1, np.ceil(np.log10(global_max)), num=20)
        major_neg = -major_pos

        # Minor log ticks
        minor_pos = np.concatenate([10**(k + np.log10(np.arange(2,10)*0.1)) 
                                    for k in range(1, int(np.ceil(np.log10(global_max))))])
        minor_neg = -minor_pos

        # Keep only values inside plotting range
        major_ticks = np.concatenate([major_neg, major_pos])
        major_ticks = major_ticks[(np.abs(major_ticks) > lin) &
                                (major_ticks < global_max*2)]

        minor_ticks = np.concatenate([minor_neg, minor_pos])
        minor_ticks = minor_ticks[abs(minor_ticks) < yr*1.5]

        # Draw horizontal log-grid lines
        # for y in major_ticks:
        #     ax.axhline(y, color="gray", lw=1.2, ls="--", alpha=0.8)

        for y in minor_ticks:
            ax.axhline(y, color="gray", lw=0.6, ls=":", alpha=0.5)

        layer_idx_list = []
        layer_count = 0
        for i, (layer_name, mn, mx) in enumerate(zip(layers, mins, maxs)):
            layer_type = categorize_layer(layer_name)
            if layer_type is None:
                continue
            _, color = palette[layer_type]

            layer_idx_list.append(i)

            # Main vertical interval
            ax.plot([layer_count, layer_count], [mn, mx], color=color, linewidth=3)
            ax.scatter(layer_count, mn, s=markersize**2, color=color)
            ax.scatter(layer_count, mx, s=markersize**2, color=color)
            layer_count += 1

        # Uniform limits
        # ax.set_ylim(global_min * 1.5, global_max * 1.5)
        yr = max(abs(global_min), abs(global_max))
        ax.set_ylim(-yr*1.5, yr*1.5)

        ax.set_title(short_names.get(model_name, model_name),
                    fontsize=54, fontweight='bold' if USE_BOLD_FONT else 'normal')

        ax.set_xlabel("Layer index", fontsize=48, fontweight='bold' if USE_BOLD_FONT else 'normal')
        if col_idx == 0:
            ax.set_ylabel("Activation range\n(log-scale)", fontsize=48, fontweight='bold' if USE_BOLD_FONT else 'normal')
        ax.tick_params(labelsize=30, width=2, length=6)

        for spine in ax.spines.values():
            spine.set_linewidth(2.2)
            spine.set_color('black')

        ax.axhline(0, color="black", linewidth=2)

        # step = max(1, len(layers) // 20)
        # ax.set_xticks(range(0, layer_count, step))
        # ax.set_xticklabels([str(i) for i in layer_idx_list[0:layer_count:step]])
        layer_count = len(layer_idx_list)  # how many layers we actually plotted

        if layer_count == 0:
            ax.set_xticks([])
        else:
            if layer_count <= N_TICKS:
                # one tick per plotted layer
                tick_positions = np.arange(layer_count)
            else:
                # spread N_TICKS positions over [0, layer_count-1]
                tick_positions = np.linspace(0, layer_count - 1, N_TICKS)
                tick_positions = np.round(tick_positions).astype(int)
                tick_positions = np.unique(tick_positions)  # avoid duplicates for small layer_count

            # Map plotted positions -> real layer indices
            tick_labels = [str(layer_idx_list[pos]) for pos in tick_positions]

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)

    legend_elements = []

    for layer_type, (label, color) in palette.items():
        if layer_type is None:
            continue
        legend_elements.append(
            plt.Line2D([0], [0], color=color, lw=4, label=label)
        )

    fig.legend(
        handles=legend_elements,
        loc='upper center',
        ncol=len(legend_elements),
        fontsize=48,
        frameon=False,
        bbox_to_anchor=(0.5, 1.20)  # move up/down as needed
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "combined_min_max.pdf"),
                dpi=300, bbox_inches="tight")
    plt.show()

def plot_min_max_bounds_2x2(models: Dict[str, str], input_path: str, output_path: str, logger, model_aliases: Dict[str, str] = MODEL_ALIASES, short_names: Dict[str, str] = SHORT_NAMES, markersize: int = MARKERSIZE, palette: Dict[str, str] = PALETTE):
    # First pass: collect all data and find global min/max
    all_data = []
    global_min = float('inf')
    global_max = float('-inf')

    for model_config, dataset_config in models.items():
        model_name = model_config
        dataset_name = dataset_config
        
        if model_name in model_aliases:
            model_name = model_aliases[model_name]
        
        layer_bounds = np.load(os.path.join(input_path, f"{model_name}-{dataset_name}-fp32-0-layer_bounds.npz"), allow_pickle=True)
        layers = list(layer_bounds.files)
        
        mins, maxs = [], []
        for layer in layers:
            bounds = layer_bounds[layer]
            if isinstance(bounds, np.ndarray):
                bounds = bounds.item()
            mins.append(bounds["min"])
            maxs.append(bounds["max"])
        
        all_data.append({
            'model_name': model_name,
            'dataset_name': dataset_name,
            'layers': layers,
            'mins': mins,
            'maxs': maxs
        })
        
        global_min = min(global_min, min(mins))
        global_max = max(global_max, max(maxs))

    logger.debug(f"Global min: {global_min}, Global max: {global_max}")

    sns.set_theme(style="whitegrid")

    # --- 2×2 GRID ---
    fig, axes = plt.subplots(
        2, 2,
        figsize=(40, 17),   # adjust size to taste
        sharey=True
    )
    axes = axes.flatten()
    n_models = len(models)
    if n_models == 1:
        axes = [axes]

    for col_idx, data in enumerate(all_data):
        ax = axes[col_idx]

        layers = data['layers']
        mins   = data['mins']
        maxs   = data['maxs']
        model_name = data['model_name']

        # --- SYMLOG SCALE ---
        lin = 1e1   # must match linthresh
        ax.set_yscale("symlog", linthresh=lin)

        yr = max(abs(global_min), abs(global_max)) * YR_EXPANSION

        # --- ADD COMPLETE LOG GRID ---
        # Major log ticks: ±10^n outside linear zone
        major_pos = np.logspace(1, np.ceil(np.log10(global_max)), num=20)
        major_neg = -major_pos

        # Minor log ticks
        minor_pos = np.concatenate([10**(k + np.log10(np.arange(2,10)*0.1)) 
                                    for k in range(1, int(np.ceil(np.log10(global_max))))])
        minor_neg = -minor_pos

        # Keep only values inside plotting range
        major_ticks = np.concatenate([major_neg, major_pos])
        major_ticks = major_ticks[(np.abs(major_ticks) > lin) &
                                (major_ticks < global_max*2)]

        minor_ticks = np.concatenate([minor_neg, minor_pos])
        minor_ticks = minor_ticks[abs(minor_ticks) < yr*1.5]

        # Draw horizontal log-grid lines
        # for y in major_ticks:
        #     ax.axhline(y, color="gray", lw=1.2, ls="--", alpha=0.8)

        for y in minor_ticks:
            ax.axhline(y, color="gray", lw=0.6, ls=":", alpha=0.5)

        layer_idx_list = []
        layer_count = 0
        for i, (layer_name, mn, mx) in enumerate(zip(layers, mins, maxs)):
            layer_type = categorize_layer(layer_name)
            if layer_type is None:
                continue
            _, color = palette[layer_type]

            layer_idx_list.append(i)

            # Main vertical interval
            ax.plot([layer_count, layer_count], [mn, mx], color=color, linewidth=3)
            ax.scatter(layer_count, mn, s=markersize**2, color=color)
            ax.scatter(layer_count, mx, s=markersize**2, color=color)
            layer_count += 1

        # Uniform limits
        # ax.set_ylim(global_min * 1.5, global_max * 1.5)
        yr = max(abs(global_min), abs(global_max))
        ax.set_ylim(-yr*1.5, yr*1.5)

        ax.set_title(short_names.get(model_name, model_name),
                    fontsize=54, fontweight='bold' if USE_BOLD_FONT else 'normal')

        if col_idx > 1:
            ax.set_xlabel("Layer index", fontsize=48, fontweight='bold' if USE_BOLD_FONT else 'normal')
        # if col_idx == 0:
        # ax.set_ylabel("Activation bounds\n(log-scale)", fontsize=48, fontweight='bold' if USE_BOLD_FONT else 'normal')
        ax.tick_params(labelsize=40, width=2, length=6)

        for spine in ax.spines.values():
            spine.set_linewidth(2.2)
            spine.set_color('black')

        ax.axhline(0, color="black", linewidth=2)

        # step = max(1, len(layers) // 20)
        # ax.set_xticks(range(0, layer_count, step))
        # ax.set_xticklabels([str(i) for i in layer_idx_list[0:layer_count:step]])
        layer_count = len(layer_idx_list)  # how many layers we actually plotted

        if layer_count == 0:
            ax.set_xticks([])
        else:
            if layer_count <= N_TICKS:
                # one tick per plotted layer
                tick_positions = np.arange(layer_count)
            else:
                # spread N_TICKS positions over [0, layer_count-1]
                tick_positions = np.linspace(0, layer_count - 1, N_TICKS)
                tick_positions = np.round(tick_positions).astype(int)
                tick_positions = np.unique(tick_positions)  # avoid duplicates for small layer_count

            # Map plotted positions -> real layer indices
            tick_labels = [str(layer_idx_list[pos]) for pos in tick_positions]

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)

    legend_elements = []

    for layer_type, (label, color) in palette.items():
        if layer_type is None:
            continue
        legend_elements.append(
            plt.Line2D([0], [0], color=color, lw=4, label=label)
        )

    fig.supylabel(
        "Activation range (log-scale)",
        fontsize=48,
        fontweight='bold' if USE_BOLD_FONT else 'normal',
        x=0.003
    )

    fig.legend(
        handles=legend_elements,
        loc='upper center',
        ncol=len(legend_elements),
        fontsize=48,
        frameon=False,
        bbox_to_anchor=(0.5, 1.05)  # move up/down as needed
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "combined_min_max2x2.pdf"),
                dpi=300, bbox_inches="tight")
    plt.show()

def main():
    models = {
        configs.GPT2: configs.GLUE_MNLI,
        configs.FACEBOOK_BART: configs.GLUE_MNLI,
        configs.VIT_BASE_PATCH16_224: configs.IMAGENET,
        configs.SWIN_BASE_PATCH4_WINDOW7_224: configs.IMAGENET,
    }

    parser = ActivationAnalysisPlotParser()
    args = parser.parse_args()
    data_path = args.input_dir
    output_path = args.output_dir
    verbose = args.verbose

    logger = logger_formatter.logging_setup(__name__, None, False, verbose)

    logger.info("Plotting per-operation histograms...")
    plot_per_op_histogram(
        models=models,
        model_aliases=MODEL_ALIASES,
        data_path=data_path,
        layer_types=LAYER_TYPES,
        output_path=output_path,
        logger=logger,
        n_bins=N_BINS
    )

    logger.info("Plotting min/max bounds...")
    plot_min_max_bounds_2x2(
        models=models,
        input_path=data_path,
        output_path=output_path,
        logger=logger,
        model_aliases=MODEL_ALIASES,
        short_names=SHORT_NAMES,
        markersize=MARKERSIZE,
        palette=PALETTE
    )
    plot_min_max_bounds(
        models=models,
        input_path=data_path,
        output_path=output_path,
        logger=logger,
        model_aliases=MODEL_ALIASES,
        short_names=SHORT_NAMES,
        markersize=MARKERSIZE,
        palette=PALETTE
    )

if __name__ == "__main__":
    main()

