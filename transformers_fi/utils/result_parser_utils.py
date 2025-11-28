import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from itertools import product
from matplotlib.backends.backend_pdf import PdfPages

def parse_filename(filename):
    """
    Parse CSV filenames to extract components.
    
    Args:
        filename (str): CSV filename with format 
                        "<model>-<dataset>-<precision>-<microop>-<float_threshold>-<seed>-layer_<layer>-it_<injection_type>-rrmode_<rrmode>.csv"
    
    Returns:
        dict: Dictionary containing the extracted components
    """
    # Remove the .csv extension
    filename = filename.replace('.csv', '')

    # Split the filename by hyphens
    parts = filename.split('-')[:]

    # as float_threshold may contain hyphens, we need to reconstruct it
    rr_mode = parts[-1].split('_')[1:]
    rr_mode = "_".join(rr_mode) if len(rr_mode) > 1 else rr_mode[0]
    injection_type = parts[-2].split('_')[1:]
    injection_type = "_".join(injection_type) if len(injection_type) > 1 else injection_type[0]
    layer = parts[-3].split('_')[1]
    seed = int(parts[-4])
    float_threshold_parts = parts[-6:-4]
    float_threshold = float("-".join(float_threshold_parts))
    
    microop = parts[-7]
    precision = parts[-8]
    dataset = parts[-9]
    model_parts = parts[:-9]
    model = "_".join(model_parts)

    return {
        "model": model,
        "dataset": dataset,
        "precision": precision,
        "microop": microop,
        "float_threshold": float_threshold,
        "seed": seed,
        "layer": layer,
        "injection_type": injection_type,
        "rr_mode": rr_mode
    }


def get_result_files(file_path):
    """
    Get the list of result files in the given path.
    
    Args:
        file_path (str): Path to the directory containing the result files
    
    Returns:
        list: List of CSV files in the directory
    """
    # Get the list of files in the directory
    files = os.listdir(file_path)
    
    # Filter out only the CSV files
    result_files = [f for f in files if f.endswith('.csv')]
    
    return result_files


def get_result_for_config(path, file):
    """
    Get the results for a given configuration.
    
    Args:
        file (str): Path to the CSV file containing the results
    
    Returns:
        pd.DataFrame: DataFrame containing the results
    """
    # Read the CSV file
    df = pd.read_csv(os.path.join(path, file))
    total = len(df)
    
    df = df[df["prediction_without_fault"] != df["prediction_with_fault"]]
    critical = len(df)

    return df, critical, total

def get_result_for_config2(file):
    """
    Get the results for a given configuration.
    
    Args:
        file (str): Path to the CSV file containing the results
    
    Returns:
        pd.DataFrame: DataFrame containing the results
    """
    # Read the CSV file
    df = pd.read_csv(file)
    total = len(df)
    
    df = df[df["prediction_without_fault"] != df["prediction_with_fault"]]
    critical = len(df)

    return df, critical, total

def get_criticality(file, threshold=1e-5):
    """
    Get the criticality for a given configuration.
    
    Args:
        file (str): Path to the CSV file containing the results
    
    Returns:
        pd.DataFrame: DataFrame containing the criticality
    """
    # Read the CSV file
    df = pd.read_csv(file)
    
    total = len(df)
    
    df_crit = df[df["prediction_without_fault"] != df["prediction_with_fault"]]
    critical = len(df_crit)

    # Calculate the criticality
    df["Crit SDC"] = df["prediction_without_fault"] != df["prediction_with_fault"]
    df["SDC"] = ((df["top1_wo_fault"] - df["top1_w_fault"]) > threshold) | (df["prediction_without_fault"] != df["prediction_with_fault"])

    sdc = df[df["SDC"] == True]
    sdc = len(sdc)
    
    return df, sdc, critical, total


def get_global_df(folder_list, filename_parser=parse_filename):
    """
    Aggregate results from multiple folders into a single DataFrame.
    
    Args:
        folder_list (list): List of folder paths containing the result files
    """
    result_files_fi = [(folder, get_result_files(folder)) for folder in folder_list]
    results_fi = []

    for folder, files in result_files_fi:
        results_fi.extend([(filename_parser(file), *get_criticality(os.path.join(folder, file), threshold=1e-5)) for file in files])

    res_fi = []
    for r in results_fi:
        res_fi.append({
            **r[0],
            "df": r[1],
            "sdc": r[2],
            "critical": r[3],
            "total images": r[4],
        })
    df_fi = pd.DataFrame(res_fi)
    df_fi["SDC%"] = df_fi["sdc"] / df_fi["total images"] * 100
    df_fi["Critical SDC%"] = df_fi["critical"] / df_fi["total images"] * 100
    df_fi["layer"] = df_fi["layer"].astype(int)

    return df_fi, res_fi


def get_global_df_per_model(res_list, model_list, threshold=1e-5):
    df_per_model = {}

    for model in model_list:
        for res in res_list:
            if res["model"] == model:
                if model not in df_per_model:
                    df_per_model[model] = {}
                if res["injection_type"] not in df_per_model[model]:
                    df_per_model[model][res["injection_type"]] = []
                # check if key exists
                if "n_faults" not in df_per_model[model]:
                    df_per_model[model]["n_faults"] = 0
                
                df_per_model[model]["n_faults"] += 1
                df = res["df"]
                df["confidence"] = df["top1_wo_fault"] - df["top2_wo_fault"]
                df["SDC"] = ((df["top1_wo_fault"] - df["top1_w_fault"]) > threshold) | (df["prediction_without_fault"] != df["prediction_with_fault"])
                df["Crit SDC"] = df["prediction_without_fault"] != df["prediction_with_fault"]
                df = df.sort_values(by=["confidence"], ascending=True)
                df_per_model[model][res["injection_type"]].append(df)

    return df_per_model

def bar_plot_confidence_levels_facet(models_data, bin_ranges, bold_labels=False):
    """
    Create a facet grid with bar plots for multiple models.
    Args:
        models_data: Dictionary with model names as keys and tuples (df_dict, n_faults) as values
        bin_ranges: List of tuples with (bin_name, lower_bound, upper_bound) for x-axis labels
    """

    # font sizes
    title_fontsize = 30
    general_label_fontsize = 26
    label_fontsize = 24
    tick_fontsize = 22
    bar_percentage_fontsize = 21

    n_models = len(models_data)
    # Determine grid layout
    n_cols = n_models  # All models in one row
    n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    # Flatten axes array for easy iteration
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_models > 1 else [axes]
    
    # Generate colors
    n_bars = len(bin_ranges)
    if n_bars == 3:
        colors = ['#fd6363', '#ffa500', '#00b050']
    elif n_bars == 4:
        colors = ['#fd6363', '#ff8c42', '#ffd700', '#00b050']
    else:
        import matplotlib.cm as cm
        cmap = cm.get_cmap('RdYlGn')
        colors = [cmap(i / (n_bars - 1)) for i in range(n_bars)]
    
    # Plot each model
    for idx, (model_name, (df_dict, n_faults)) in enumerate(models_data.items()):
        ax = axes[idx]
        
        confidence_levels = []
        crit_sdc_rates = []
        sample_counts = []
        
        # Calculate critical SDC rate for each bin
        for bin_name in df_dict.keys():
            df = df_dict[bin_name]
            rate = (df['Crit SDC'].sum() / (len(df) * n_faults)) * 100 if len(df) > 0 else 0
            confidence_levels.append(bin_name)
            crit_sdc_rates.append(rate)
            sample_counts.append(len(df))
        
        # Calculate percentages
        total_samples = sum(sample_counts)
        sample_percentages = [(count / total_samples * 100) if total_samples > 0 else 0 
                              for count in sample_counts]
        
        # Create bar plot with black edges
        bars = ax.bar(range(len(confidence_levels)), crit_sdc_rates, 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
        
        # Add value labels on top of bars
        for bar, rate in zip(bars, crit_sdc_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2. + 0.15, height,
                    f'{rate:.3f}%',
                    ha='center', va='bottom', fontsize=bar_percentage_fontsize, fontweight='bold' if bold_labels else 'normal')
            # print(f'{rate:.3f}%')
        
        # Set x-axis labels with increased font size
        ax.set_xticks(range(len(confidence_levels)))
        # ax.set_xticklabels([f"{label}\n{ranges[0]:.2f}-{ranges[1]:.2f}\n({perc:.1f}%)" 
        #                     for label, ranges, perc in zip(confidence_levels, bin_ranges, sample_percentages)], 
        #                    fontsize=label_fontsize, fontweight='bold')

        # ax.set_xticklabels([f"{label}\n{count}" 
        #                     for label, count in zip(confidence_levels, sample_counts)], 
        #                    fontsize=tick_fontsize, fontweight='bold' if bold_labels else 'normal')

        ax.set_xticklabels([f"{label}" 
                            for label in confidence_levels], 
                           fontsize=tick_fontsize, fontweight='bold' if bold_labels else 'normal')
        
        # Labels and title for each subplot with increased font sizes
        ax.set_ylabel('Critical SDC PVF%', fontsize=general_label_fontsize, fontweight='bold' if bold_labels else 'normal')
        ax.set_title(f'{model_name}', fontsize=title_fontsize, fontweight='bold' if bold_labels else 'normal', pad=15)
        
        # Make tick labels bold with increased font size
        ax.tick_params(axis='both', which='both', labelsize=tick_fontsize, colors='black', width=2, length=6)
        for label in ax.get_yticklabels():
            label.set_fontweight('bold' if bold_labels else 'normal')
            label.set_color('black')
            
        
        # Styling
        ax.grid(False)
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['left'].set_edgecolor('black')
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['bottom'].set_edgecolor('black')
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    # Add common x-label with increased font size
    fig.text(0.5, 0.01, 'Confidence Level', ha='center', fontsize=general_label_fontsize, fontweight='bold' if bold_labels else 'normal')
    
    fig.patch.set_facecolor('white')
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    
    return fig