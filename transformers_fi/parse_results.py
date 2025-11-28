#!/usr/bin/env python3

from cli.parsers import ResultsParser
import cli.logger_formatter as logger_formatter
from utils import result_parser_utils
from typing import Tuple, List
import pandas as pd
import os
import shutil
from matplotlib.backends.backend_pdf import PdfPages

# Define confidence bins
CONFIDENCE_BINS = [
    ("Very Low", 0.0, 0.25),
    ("Low", 0.25, 0.5),
    ("High", 0.5, 0.75),
    ("Very High", 0.75, 1.0),
]
INJECTION_TYPES = [
    "MULTIPLE_RANDOM",
    "FIXED",
    "ROW",
    "COL",
]

# Define the desired order
DESIRED_ORDER = [
    "gpt2",
    "facebook_bart-large-mnli",
    "vit_base_patch16_224",
    "swin_base_patch4_window7_224",
]

PRETTY_NAMES = {
    "vit_base_patch16": "ViT",
    "vit_base_patch16_224": "ViT",
    "swin_base_patch4_window7_224": "Swin",
    "gpt2": "GPT2",
    "facebook_bart-large-mnli": "BART",
    "facebook_bart_large_mnli": "BART",
}



def generate_dfs(results_dir) -> Tuple[pd.DataFrame, List, pd.DataFrame, pd.DataFrame]:
    fi_folders = [
        os.path.join(results_dir, folder) for folder in os.listdir(results_dir)
    ]

    df_fi, res_fi = result_parser_utils.get_global_df(fi_folders)

    per_microop = df_fi.groupby(['model', 'dataset', 'microop']).agg({
        'sdc': 'sum',
        'critical': 'sum',
        'total images': 'sum',
        'seed': 'count',
    })

    # rename 'seed' to 'num_faults'
    per_microop = per_microop.rename(columns={'seed': 'num_faults'})
    per_microop['SDC%'] = per_microop['sdc'] / per_microop['total images'] * 100

    per_microop['Critical SDC%'] = per_microop['critical'] / per_microop['total images'] * 100

    # per model
    per_model = df_fi.groupby(['model', 'dataset']).agg({
        'sdc': 'sum',
        'critical': 'sum',
        'total images': 'sum',
    })
    per_model['SDC%'] = per_model['sdc'] / per_model['total images'] * 100
    per_model['Critical SDC%'] = per_model['critical'] / per_model['total images'] * 100

    return df_fi, res_fi, per_microop, per_model

def plot_crit_sdc_per_conf_bin(df_fi, output_dir, logger, confidence_bins=CONFIDENCE_BINS, it=INJECTION_TYPES, desired_order=DESIRED_ORDER, pretty_names=True):

    pdf_path = os.path.join(output_dir, "crit_sdc_per_conf_bin.pdf")

    # Collect data for all models
    models_data = {}
    for model in df_fi:
        df_list = []
        for injection_type in it:
            if injection_type not in df_fi[model]:
                continue
            df_list += (df_fi[model][injection_type])
        n_faults = df_fi[model]["n_faults"]
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df = combined_df.groupby(["ground_truth", "prediction_without_fault", "confidence"]).agg({
            "SDC": "sum",
            "Crit SDC": "sum",
        }).sort_values(by=["confidence"], ascending=True).reset_index()
        
        # Create bins dynamically based on confidence_bins
        grouped_conf_dfs = {}
        bin_ranges = []
        for bin_name, lower, upper in confidence_bins:
            if lower == 0.0:
                grouped_conf_dfs[bin_name] = combined_df[combined_df['confidence'] <= upper]
            else:
                grouped_conf_dfs[bin_name] = combined_df[
                    (combined_df['confidence'] > lower) & (combined_df['confidence'] <= upper)
                ]
            bin_ranges.append((lower, upper))
        
        model_name = PRETTY_NAMES[model] if pretty_names and model in PRETTY_NAMES else model
        models_data[model_name] = (grouped_conf_dfs, n_faults)

    # Sort models_data according to desired_order
    ordered_models_data = {}
    for model in desired_order:
        model_name = PRETTY_NAMES[model] if pretty_names and model in PRETTY_NAMES else model
        if model_name in models_data:
            ordered_models_data[model_name] = models_data[model_name]

    # Add any remaining models not in desired_order (optional)
    for model_name in models_data:
        if model_name not in ordered_models_data:
            ordered_models_data[model_name] = models_data[model_name]

    # Create facet grid plot
    logger.info("Generating facet grid plot for all models")
    fig = result_parser_utils.bar_plot_confidence_levels_facet(ordered_models_data, bin_ranges)
    # Save to PDF
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)

    logger.info(f"Saved Critical SDC per confidence bin plot to {pdf_path}")

def main():
    parser = ResultsParser()
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir
    erase_output_dir = args.erase_output_dir
    verbose = args.verbose

    logger = logger_formatter.logging_setup(__name__, None, False, verbose)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        if erase_output_dir:
            logger.info(f"Erasing existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        else:
            logger.warning(f"Output directory {output_dir} already exists. Use --erase-output-dir to erase it.")
            # Exit to avoid overwriting existing results
            return
    
    logger.info(f"Parsing results from {results_dir} and saving to {output_dir}")

    df_fi, res_fi, per_microop, per_model = generate_dfs(results_dir)

    per_microop_path = os.path.join(output_dir, "per_microop_results.xlsx")
    per_model_path = os.path.join(output_dir, "per_model_results.xlsx")

    per_microop.to_excel(per_microop_path)
    per_model.to_excel(per_model_path)


    models = df_fi["model"].unique()

    df_per_model = result_parser_utils.get_global_df_per_model(res_fi, models)

    plot_crit_sdc_per_conf_bin(df_per_model, output_dir, logger)



if __name__ == "__main__":
    main()