#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
import pandas as pd
from pathlib import Path


MINIIMAGENET_CLASS_NAMES = {48: "Guitar", 57: "Hourglass", 68: "Printer", 85: "Piano", 95: "Orange"}

GALAXYMNIST_CLASS_NAMES = {
    0: "Smooth Round",
    1: "Smooth Cigar-shaped",
    2: "Edge-on Disk",
    3: "Unbarred Spiral",
}


def create_latex_table(df, caption="", label=""):
    """Create a LaTeX table from the DataFrame with selected columns."""
    # Select and rename columns for the table
    columns = {
        "anomaly_class": "Anomaly Class",
        "final_auroc": "AUROC",
        "final_auprc": "AUPRC",
        "top_0.1pct_anomalies_found": "Anomalies Found at 0.1\\% Data [\\%]",
        "top_0.1pct_precision": "Precision at 0.1\\% Data [\\%]",
        "top_1.0pct_anomalies_found": "Anomalies Found at 1\\% Data [\\%]",
        "top_1.0pct_precision": "Precision at 1\\% Data [\\%]",
    }

    df_selected = df[columns.keys()].copy()

    # Map class numbers to names
    if df["dataset"].iloc[0] == "miniimagenet":
        df_selected["anomaly_class"] = (
            df_selected["anomaly_class"].astype(int).map(MINIIMAGENET_CLASS_NAMES)
        )
    elif df["dataset"].iloc[0] == "galaxymnist":
        df_selected["anomaly_class"] = (
            df_selected["anomaly_class"].astype(int).map(GALAXYMNIST_CLASS_NAMES)
        )

    # Calculate mean and std
    numeric_cols = list(set(columns.keys()) - {"anomaly_class"})

    # Format numeric values with appropriate precision
    format_rules = {
        "final_auroc": "{:.2f}",
        "final_auprc": "{:.2f}",
        "top_0.1pct_anomalies_found": "{:.2f}",
        "top_0.1pct_precision": "{:.2f}",
        "top_1.0pct_anomalies_found": "{:.2f}",
        "top_1.0pct_precision": "{:.2f}",
    }

    for col in numeric_cols:
        df_selected[col] = df_selected[col].map(format_rules[col].format)

    # Calculate means and stds in the correct order
    means = {
        "final_auroc": df_selected["final_auroc"].astype(float).mean(),
        "final_auprc": df_selected["final_auprc"].astype(float).mean(),
        "top_0.1pct_anomalies_found": df_selected["top_0.1pct_anomalies_found"]
        .astype(float)
        .mean(),
        "top_0.1pct_precision": df_selected["top_0.1pct_precision"].astype(float).mean(),
        "top_1.0pct_anomalies_found": df_selected["top_1.0pct_anomalies_found"]
        .astype(float)
        .mean(),
        "top_1.0pct_precision": df_selected["top_1.0pct_precision"].astype(float).mean(),
    }

    stds = {
        "final_auroc": df_selected["final_auroc"].astype(float).std(),
        "final_auprc": df_selected["final_auprc"].astype(float).std(),
        "top_0.1pct_anomalies_found": df_selected["top_0.1pct_anomalies_found"].astype(float).std(),
        "top_0.1pct_precision": df_selected["top_0.1pct_precision"].astype(float).std(),
        "top_1.0pct_anomalies_found": df_selected["top_1.0pct_anomalies_found"].astype(float).std(),
        "top_1.0pct_precision": df_selected["top_1.0pct_precision"].astype(float).std(),
    }

    # Create the LaTeX table
    latex_str = "\\begin{table*}[t]\\scriptsize\n"
    latex_str += "\\caption{" + caption + "\\label{" + label + "}}\n"
    latex_str += "\\begin{center}\n"
    latex_str += "{\\tabulinesep=1.2mm\n"
    latex_str += "\\setlength\\tabcolsep{2pt}\n"

    # Create column specification with X columns for better spacing
    col_widths = ["0.4cm", "0.5cm", "0.5cm", "0.7cm", "0.7cm", "0.7cm", "0.7cm"]
    col_spec = " | ".join([f"X[{w}]" for w in col_widths])
    latex_str += f"\\begin{{tabu}} {{ | {col_spec} | }}\n"
    latex_str += "\\hline\n"

    # Add header
    latex_str += " & ".join(columns.values()) + " \\\\\n"
    latex_str += "\\hline\n"

    # Add data rows
    for _, row in df_selected.iterrows():
        latex_str += " & ".join(str(val) for val in row.values) + " \\\\\n"
        latex_str += "\\hline\n"

    # Add mean±std row with correct ordering
    mean_std_strs = ["Mean"]
    cols_order = [
        "final_auroc",
        "final_auprc",
        "top_0.1pct_anomalies_found",
        "top_0.1pct_precision",
        "top_1.0pct_anomalies_found",
        "top_1.0pct_precision",
    ]

    for col in cols_order:
        mean_std_strs.append(f"{means[col]:.2f} $\\pm$ {stds[col]:.2f}")

    latex_str += " & ".join(mean_std_strs) + " \\\\\n"
    latex_str += "\\hline\n"

    # Close the table
    latex_str += "\\end{tabu}}\n"
    latex_str += "\\end{center}\n"
    latex_str += "\\end{table*}"

    return latex_str


def analyze_results(results_dir):
    """Analyze results from all CSV files in the summary directory."""
    # Convert to Path object if string
    results_dir = Path(results_dir)
    results_path = results_dir / "summary"

    # Ensure directory exists
    if not results_path.exists():
        raise ValueError(f"Results directory not found: {results_path}")

    # Dictionary to store all results files
    result_files = {
        "miniimagenet": results_path / "miniimagenet_results.csv",
        "galaxymnist": results_path / "galaxymnist_results.csv",
        "training_iterations": results_path / "training_iterations_results.csv",
        "active_learning": results_path / "active_learning_results.csv",
        "n_samples_ablation": results_path / "n_samples_ablation_results.csv",
    }

    # Process each results file
    for name, filepath in result_files.items():
        if filepath.exists():
            df = pd.read_csv(filepath)

            # Create LaTeX table
            caption = f"Results for {name.replace('_', ' ').title()}"
            label = f"tab:{name}_results"
            latex_table = create_latex_table(df, caption, label)

            # Save to .tex file
            output_file = results_path / f"{name}_table.tex"
            with open(output_file, "w") as f:
                f.write(latex_table)

            print(f"Created LaTeX table for {name} at {output_file}")


def main():
    """Main function to analyze results."""
    # Use absolute path from current file location
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "benchmark_results" / "results_20250321_124008"

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Please provide the path to results directory as argument")
        return

    analyze_results(results_dir)


if __name__ == "__main__":
    main()
