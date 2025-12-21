#!/usr/bin/env python3
"""
GREEN Score Plotting

Regenerates plots from green_scores_by_version_full.csv without recomputing GREEN scores.
Supports multiple correctness modes and acceptance rate visualization.

Usage:
    python plot_green.py
    python plot_green.py --input /path/to/green_scores_by_version_full.csv
    python plot_green.py --split-by-acceptance
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate GREEN score plots from existing CSV"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="green_scores_by_version_full.csv",
        help="Path to input CSV file (output from analyze_green.py)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Directory for output plots (default: pngs subfolder next to input file)",
    )
    parser.add_argument(
        "--split-by-acceptance",
        action="store_true",
        help="Generate separate plots for accepted and rejected samples (default: False)",
    )
    return parser.parse_args()


def normalize_boolean_column(series: pd.Series) -> pd.Series:
    """Normalize boolean-like string values to True/False/None.

    Handles: 'true', 'True', 'false', 'False' -> bool
    Removes: 'ERROR', nan -> None
    """

    def normalize(x):
        if pd.isna(x):
            return None
        x_str = str(x).lower()
        if x_str == "true":
            return True
        elif x_str == "false":
            return False
        else:  # 'ERROR' or other invalid values
            return None

    return series.apply(normalize)


def parse_body_part(value) -> str | None:
    """Parse body part from JSON-like string.

    Examples:
        '["Leg"]' -> 'Leg'
        '["Pelvis", "Pelvis"]' -> 'Pelvis'
        '["Extremities", "Pelvis"]' -> 'Extremities' (takes first)
        nan -> None
    """
    if pd.isna(value):
        return None
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed[0]  # Take first body part
        return None
    except (json.JSONDecodeError, TypeError):
        return None


def generate_plot(
    results_df: pd.DataFrame,
    output_path: str,
    title_suffix: str = "",
    file_suffix: str = "",
    show_acceptance_rate: bool = True,
    correctness_column: str = "is_correct_normalized",
):
    """Generate the GREEN scores visualization plot.

    Args:
        results_df: DataFrame with results
        output_path: Base output path for the plot
        title_suffix: Suffix to add to plot title
        file_suffix: Suffix to add to filename
        show_acceptance_rate: Whether to show acceptance rate line
        correctness_column: Column to use for acceptance rate calculation
    """
    if len(results_df) == 0:
        print(f"No data to plot{title_suffix}")
        return

    fine_grained = (
        results_df.groupby("generation_version")
        .agg(
            {
                "vlm_green_score": ["mean", "std", "count"],
            }
        )
        .round(4)
    )

    # Calculate acceptance rate per version based on correctness_column
    if correctness_column in results_df.columns:
        acceptance_rates = (
            results_df.groupby("generation_version")[correctness_column]
            .mean()
            .reindex(fine_grained.index)
            .values
        )
    else:
        acceptance_rates = None

    versions = fine_grained.index.tolist()
    vlm_means = fine_grained[("vlm_green_score", "mean")].values
    vlm_stds = fine_grained[("vlm_green_score", "std")].values
    counts = fine_grained[("vlm_green_score", "count")].values

    # Calculate Standard Error of the Mean (SEM = std / sqrt(n))
    vlm_sems = vlm_stds / np.sqrt(counts)

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    x = np.arange(len(versions))
    bar_width = 0.35

    # Plot count bars on secondary axis
    bars_count = ax2.bar(
        x,
        counts,
        bar_width,
        color="lightgray",
        alpha=0.7,
        label="Sample Count",
    )

    # Plot VLM scores as dots with SEM error bars on primary axis
    ax1.errorbar(
        x,
        vlm_means,
        yerr=vlm_sems,
        fmt="o",
        markersize=8,
        capsize=5,
        color="steelblue",
        ecolor="steelblue",
        alpha=0.8,
        label="GREEN Score (VLM vs Reference)",
        zorder=5,
    )

    # Plot acceptance rate as a line on primary axis
    if show_acceptance_rate and acceptance_rates is not None:
        ax1.plot(
            x,
            acceptance_rates,
            "s-",
            markersize=6,
            color="darkorange",
            alpha=0.8,
            label="Acceptance Rate",
            zorder=4,
        )
        # Add acceptance rate labels
        for i, rate in enumerate(acceptance_rates):
            if not np.isnan(rate):
                ax1.text(
                    x[i] - 0.15,
                    rate,
                    f"{rate:.0%}",
                    ha="right",
                    va="center",
                    fontsize=8,
                    color="darkorange",
                )

    # Primary axis (GREEN score and acceptance rate share 0-1 scale)
    ax1.set_xlabel("Version")
    ax1.set_ylabel("GREEN Score / Acceptance Rate", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax1.grid(True, alpha=0.3, zorder=0)

    # Secondary axis (count)
    ax2.set_ylabel("Sample Count", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    ax1.set_title(f"GREEN Scores by Version (mean ± SEM){title_suffix}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(versions, rotation=45, ha="right")

    # Add value labels next to dots
    for i, mean in enumerate(vlm_means):
        ax1.text(
            x[i] + 0.15,
            mean,
            f"{mean:.2f}",
            ha="left",
            va="center",
            fontsize=8,
            color="steelblue",
        )

    # Add count labels on bars
    for i, n in enumerate(counts):
        ax2.text(
            x[i],
            n + max(counts) * 0.02,
            f"{int(n)}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="gray",
        )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    plt.close()


def generate_plots_for_subset(
    data: pd.DataFrame,
    output_dir: Path,
    subset_name: str,
    file_prefix: str,
    split_by_acceptance: bool = False,
):
    """Generate plots for a subset of data.

    Args:
        data: DataFrame with results
        output_dir: Directory for output plots
        subset_name: Name for plot titles
        file_prefix: Prefix for output filenames
        split_by_acceptance: If True, generate accepted/rejected plots
    """
    print(f"\n{'='*60}")
    print(f"Generating plots for: {subset_name}")
    print(f"Total samples: {len(data)}")
    print(f"{'='*60}")

    if len(data) == 0:
        print(f"No data for {subset_name}, skipping.")
        return

    # Define correctness modes to plot
    correctness_modes = [
        ("is_correct", "is_correct_normalized", ""),
        ("is_findings_correct", "is_findings_correct_normalized", "_findings"),
        ("is_impressions_correct", "is_impressions_correct_normalized", "_impressions"),
        ("is_both_correct", "is_both_correct_normalized", "_both"),
    ]

    for mode_name, col_name, mode_suffix in correctness_modes:
        if col_name not in data.columns:
            print(f"  Skipping {mode_name}: column not found")
            continue

        # Plot with acceptance rate for this mode
        mode_title = f" [{mode_name}]" if mode_name != "is_correct" else ""
        generate_plot(
            data,
            str(output_dir / f"{file_prefix}_plot{mode_suffix}.png"),
            title_suffix=f" - {subset_name}{mode_title}",
            show_acceptance_rate=True,
            correctness_column=col_name,
        )

        # Plot accepted/rejected splits (optional)
        if split_by_acceptance:
            accepted_df = data[data[col_name] == True]
            if len(accepted_df) > 0:
                print(f"  {mode_name} accepted samples: {len(accepted_df)}")
                generate_plot(
                    accepted_df,
                    str(output_dir / f"{file_prefix}_plot{mode_suffix}_accepted.png"),
                    title_suffix=f" - {subset_name} (Accepted Only){mode_title}",
                    show_acceptance_rate=False,
                    correctness_column=col_name,
                )

            rejected_df = data[data[col_name] == False]
            if len(rejected_df) > 0:
                print(f"  {mode_name} rejected samples: {len(rejected_df)}")
                generate_plot(
                    rejected_df,
                    str(output_dir / f"{file_prefix}_plot{mode_suffix}_rejected.png"),
                    title_suffix=f" - {subset_name} (Rejected Only){mode_title}",
                    show_acceptance_rate=False,
                    correctness_column=col_name,
                )


def main():
    args = parse_args()

    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Loading data from: {input_path}")
    data = pd.read_csv(input_path)
    print(f"Loaded {len(data)} rows")

    # Determine output directory (default: pngs subfolder)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent / "pngs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to only rows that were kept for GREEN scoring and have scores
    data = data[data["kept_for_green"] == True].copy()
    data = data[data["vlm_green_score"].notna()].copy()
    print(f"Rows with GREEN scores: {len(data)}")

    # Normalize is_correct column
    if "is_correct" in data.columns:
        data["is_correct_normalized"] = normalize_boolean_column(data["is_correct"])
    else:
        data["is_correct_normalized"] = None

    # Normalize is_findings_correct column
    if "is_findings_correct" in data.columns:
        data["is_findings_correct_normalized"] = normalize_boolean_column(
            data["is_findings_correct"]
        )

    # Normalize is_impressions_correct column
    if "is_impressions_correct" in data.columns:
        data["is_impressions_correct_normalized"] = normalize_boolean_column(
            data["is_impressions_correct"]
        )

    # Create combined column (is_findings_correct AND is_impressions_correct)
    if (
        "is_findings_correct_normalized" in data.columns
        and "is_impressions_correct_normalized" in data.columns
    ):
        data["is_both_correct_normalized"] = data.apply(
            lambda row: (
                True
                if (
                    row.get("is_findings_correct_normalized") == True
                    and row.get("is_impressions_correct_normalized") == True
                )
                else (
                    False
                    if (
                        row.get("is_findings_correct_normalized") == False
                        or row.get("is_impressions_correct_normalized") == False
                    )
                    else None
                )
            ),
            axis=1,
        )

    # Normalize is_rad_report_normal column
    if "is_rad_report_normal" not in data.columns:
        print("Error: is_rad_report_normal column not found")
        sys.exit(1)

    data["is_normal_normalized"] = normalize_boolean_column(
        data["is_rad_report_normal"]
    )

    # Show distribution
    print("\nis_rad_report_normal distribution (after normalization):")
    print(data["is_normal_normalized"].value_counts(dropna=False))

    # Filter out None values (ERROR, nan)
    valid_data = data[data["is_normal_normalized"].notna()].copy()
    print(f"\nRows with valid is_rad_report_normal: {len(valid_data)}")

    # Split by normal/abnormal
    normal_data = valid_data[valid_data["is_normal_normalized"] == True]
    abnormal_data = valid_data[valid_data["is_normal_normalized"] == False]

    print(f"Normal reports: {len(normal_data)}")
    print(f"Abnormal reports: {len(abnormal_data)}")

    # Generate plots for all data (before normal/abnormal split)
    generate_plots_for_subset(
        valid_data,
        output_dir,
        subset_name="All Reports",
        file_prefix="green_all",
        split_by_acceptance=args.split_by_acceptance,
    )

    # Generate plots for normal reports
    generate_plots_for_subset(
        normal_data,
        output_dir,
        subset_name="Normal Reports",
        file_prefix="green_normal",
        split_by_acceptance=args.split_by_acceptance,
    )

    # Generate plots for abnormal reports
    generate_plots_for_subset(
        abnormal_data,
        output_dir,
        subset_name="Abnormal Reports",
        file_prefix="green_abnormal",
        split_by_acceptance=args.split_by_acceptance,
    )

    # Generate plots by body part
    print("\n" + "=" * 60)
    print("Generating plots by body part...")
    print("=" * 60)

    if "parsed_body_part" in valid_data.columns:
        # Parse body part column
        valid_data["body_part_normalized"] = valid_data["parsed_body_part"].apply(
            parse_body_part
        )

        # Count by body part
        body_part_counts = valid_data["body_part_normalized"].value_counts()
        print("\nBody part distribution:")
        print(body_part_counts)

        # Filter to body parts with >100 entries
        body_parts_to_plot = body_part_counts[body_part_counts > 100].index.tolist()
        print(f"\nBody parts with >100 entries: {body_parts_to_plot}")

        body_part_dirs = []
        for body_part in body_parts_to_plot:
            if body_part is None:
                continue

            # Create subfolder for this body part
            body_part_dir = output_dir / body_part
            body_part_dir.mkdir(parents=True, exist_ok=True)
            body_part_dirs.append(body_part)

            # Filter data for this body part
            bp_data = valid_data[valid_data["body_part_normalized"] == body_part]
            bp_prefix = f"green_{body_part.lower().replace('-', '_')}"

            # Generate plots for all data in this body part
            generate_plots_for_subset(
                bp_data,
                body_part_dir,
                subset_name=f"{body_part}",
                file_prefix=bp_prefix,
                split_by_acceptance=args.split_by_acceptance,
            )

            # Generate plots for normal reports in this body part
            bp_normal = bp_data[bp_data["is_normal_normalized"] == True]
            if len(bp_normal) > 0:
                generate_plots_for_subset(
                    bp_normal,
                    body_part_dir,
                    subset_name=f"{body_part} - Normal",
                    file_prefix=f"{bp_prefix}_normal",
                    split_by_acceptance=args.split_by_acceptance,
                )

            # Generate plots for abnormal reports in this body part
            bp_abnormal = bp_data[bp_data["is_normal_normalized"] == False]
            if len(bp_abnormal) > 0:
                generate_plots_for_subset(
                    bp_abnormal,
                    body_part_dir,
                    subset_name=f"{body_part} - Abnormal",
                    file_prefix=f"{bp_prefix}_abnormal",
                    split_by_acceptance=args.split_by_acceptance,
                )
    else:
        print("Warning: parsed_body_part column not found, skipping body part plots")
        body_part_dirs = []

    print("\n" + "=" * 60)
    print("All plots generated!")
    print("=" * 60)
    print(f"\nOutput files in: {output_dir}")
    print("  - green_all_plot.png (all reports)")
    print("  - green_normal_plot.png (normal reports)")
    print("  - green_abnormal_plot.png (abnormal reports)")
    if args.split_by_acceptance:
        print("  - green_all_plot_accepted.png (all, accepted)")
        print("  - green_all_plot_rejected.png (all, rejected)")
        print("  - green_normal_plot_accepted.png (normal, accepted)")
        print("  - green_normal_plot_rejected.png (normal, rejected)")
        print("  - green_abnormal_plot_accepted.png (abnormal, accepted)")
        print("  - green_abnormal_plot_rejected.png (abnormal, rejected)")
    if body_part_dirs:
        print("\nBody part plots in subfolders:")
        for bp in body_part_dirs:
            print(f"  - {output_dir}/{bp}/")


if __name__ == "__main__":
    main()
