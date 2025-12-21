#!/usr/bin/env python3
"""
GREEN Score Analysis Script

Combines data processing and visualization for GREEN score analysis.
Processes analytics CSV files, computes GREEN scores by version, and generates plots.

Usage:
    python analyze_green.py                           # Default: sample 3 per version
    python analyze_green.py --input /path/to/file.csv
    python analyze_green.py --samples all             # Run all samples
    python analyze_green.py --samples 50              # Sample 50 per version
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from packaging import version as pkg_version

# Add parent directory to path for GREEN import
sys.path.insert(0, str(Path(__file__).parent.parent))
from green_score import GREEN


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze GREEN scores from analytics CSV files"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="/Users/ruian/Downloads/ANALYTICS_MOST_RECENT.csv",
        help="Path to input analytics CSV file",
    )
    parser.add_argument(
        "--samples",
        "-s",
        type=str,
        default="3",
        help="Number of samples per version, or 'all' for all samples (default: 3)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output CSV file (default: green_scores_by_version.csv in same dir as input)",
    )
    parser.add_argument(
        "--min-version",
        type=str,
        default="0.1.2",
        help="Minimum version to process (default: 0.1.2)",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip generating the plot"
    )
    parser.add_argument(
        "--split-by-acceptance",
        action="store_true",
        help="Generate separate plots for accepted and rejected samples (default: False)",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Perform self-check (report_text vs report_text) sanity check (default: False)",
    )
    return parser.parse_args()


def load_and_clean_data(input_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean the analytics data.

    Returns:
        tuple: (original_data with kept_for_green column, clean_data for processing)
    """
    print(f"Loading data from: {input_file}")
    data = pd.read_csv(input_file)
    print(f"Loaded {len(data)} rows")

    # Add row_number as original index
    data["row_number"] = data.index

    # Create kept_for_green column based on filtering criteria
    # Exclude: addendums, Justin radiologist, NaN radiologist, NaN is_correct
    is_correct_valid = (
        data["is_correct"].notna() if "is_correct" in data.columns else True
    )
    data["kept_for_green"] = (
        (data["report_is_addendum"] != True)
        & (data["report_radiologist_first_name"] != "Justin")
        & (data["report_radiologist_first_name"].notna())
        & is_correct_valid
    )

    # Get clean data for processing
    clean_data = data[data["kept_for_green"] == True].copy()

    print(
        f"Rows kept for GREEN scoring: {len(clean_data)} (rejected {len(data) - len(clean_data)})"
    )
    print(f"  - Addendums rejected: {(data['report_is_addendum'] == True).sum()}")
    print(
        f"  - Justin/NaN radiologists rejected: {((data['report_radiologist_first_name'] == 'Justin') | (data['report_radiologist_first_name'].isna())).sum()}"
    )
    if "is_correct" in data.columns:
        print(f"  - NaN is_correct rejected: {data['is_correct'].isna().sum()}")

    return data, clean_data


def get_versions_to_process(clean_data: pd.DataFrame, min_version: str) -> list:
    """Get sorted list of versions >= min_version."""
    versions = clean_data["generation_version"].dropna().unique()
    versions = sorted(
        [
            v
            for v in versions
            if pkg_version.parse(str(v)) >= pkg_version.parse(min_version)
        ]
    )

    print(f"\nVersions to process (>= {min_version}):")
    for v in versions:
        count = len(clean_data[clean_data["generation_version"] == v])
        print(f"  {v}: {count} rows")

    return versions


def compute_green_scores(
    clean_data: pd.DataFrame,
    versions: list,
    samples_per_version: int | None,
    perform_self_check: bool = False,
) -> pd.DataFrame:
    """Compute GREEN scores for all versions."""
    print("\nInitializing GREEN scorer...")
    green_scorer = GREEN(use_azure=True, verbose=False)
    print("GREEN scorer initialized.")

    all_results = []

    for ver in versions:
        print(f"\n{'='*60}")
        print(f"Processing version: {ver}")
        print(f"{'='*60}")

        ver_data = clean_data[clean_data["generation_version"] == ver].copy()

        # Apply sampling if configured
        if samples_per_version is not None:
            ver_data = ver_data.head(samples_per_version)
            print(f"Sampling {len(ver_data)} rows")
        else:
            print(f"Processing all {len(ver_data)} rows")

        refs = ver_data["report_text"].fillna("").astype(str).tolist()
        hyps_vlm = ver_data["generated_vlm_output_text"].fillna("").astype(str).tolist()
        row_numbers = ver_data["row_number"].tolist()
        is_correct_values = (
            ver_data["is_correct"].tolist()
            if "is_correct" in ver_data.columns
            else [None] * len(ver_data)
        )

        # (1) Optional sanity check: report_text vs report_text
        if perform_self_check:
            hyps_self = ver_data["report_text"].fillna("").astype(str).tolist()
            print(f"\n--- (1) Sanity check: report_text vs report_text ---")
            mean1, std1, scores1, summary1, df1 = green_scorer(refs, hyps_self)
            print(f"Mean score: {mean1:.4f} (should be ~1.0)")

        # (2) Actual comparison: report_text vs generated_vlm_output_text
        print(f"\n--- Comparison: report_text vs generated_vlm_output_text ---")
        mean2, std2, scores2, summary2, df2 = green_scorer(refs, hyps_vlm)
        print(f"Mean score: {mean2:.4f}")

        # Combine results for this version
        for i, row_num in enumerate(row_numbers):
            result = {
                "generation_version": ver,
                "row_number": row_num,
                "is_correct": is_correct_values[i],
                "vlm_green_score": scores2[i],
                "vlm_green_analysis": df2["green_analysis"].iloc[i],
            }
            if perform_self_check:
                result["self_green_score"] = scores1[i]
                result["self_green_analysis"] = df1["green_analysis"].iloc[i]
            # Add error counts for VLM comparison
            for col in df2.columns:
                if col.startswith("(") or col == "Matched Findings":
                    result[f"vlm_{col}"] = df2[col].iloc[i]
            all_results.append(result)

    print(f"\n{'='*60}")
    print(
        f"Done! Processed {len(all_results)} total rows across {len(versions)} versions"
    )
    print(f"{'='*60}")

    results_df = pd.DataFrame(all_results)

    # Add major.minor version for grouping
    results_df["version_major_minor"] = results_df["generation_version"].apply(
        lambda v: ".".join(str(v).split(".")[:2])
    )

    return results_df


def print_summary(results_df: pd.DataFrame):
    """Print summary statistics."""
    has_self_check = "self_green_score" in results_df.columns

    if has_self_check:
        fine_grained = (
            results_df.groupby("generation_version")
            .agg(
                {
                    "self_green_score": ["mean", "std", "count"],
                    "vlm_green_score": ["mean", "std"],
                }
            )
            .round(4)
        )
        consolidated = (
            results_df.groupby("version_major_minor")
            .agg(
                {
                    "self_green_score": ["mean", "std", "count"],
                    "vlm_green_score": ["mean", "std"],
                }
            )
            .round(4)
        )
    else:
        fine_grained = (
            results_df.groupby("generation_version")
            .agg(
                {
                    "vlm_green_score": ["mean", "std", "count"],
                }
            )
            .round(4)
        )
        consolidated = (
            results_df.groupby("version_major_minor")
            .agg(
                {
                    "vlm_green_score": ["mean", "std", "count"],
                }
            )
            .round(4)
        )

    print("\n" + "=" * 80)
    print("FINE-GRAINED SCORES (by exact version)")
    print("=" * 80)
    print(fine_grained)

    print("\n" + "=" * 80)
    print("CONSOLIDATED SCORES (by major.minor version)")
    print("=" * 80)
    print(consolidated)

    comparison = (
        results_df.groupby(["version_major_minor", "generation_version"])
        .agg({"vlm_green_score": ["mean", "std", "count"]})
        .round(4)
    )

    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(comparison)

    return fine_grained, consolidated


def generate_plot(
    results_df: pd.DataFrame,
    output_path: str,
    title_suffix: str = "",
    file_suffix: str = "",
    show_acceptance_rate: bool = True,
    correctness_column: str = "is_correct",
):
    """Generate the GREEN scores visualization plot.

    Args:
        results_df: DataFrame with results
        output_path: Base output path for the plot
        title_suffix: Suffix to add to plot title (e.g., " - Accepted Only")
        file_suffix: Suffix to add to filename (e.g., "_accepted")
        show_acceptance_rate: Whether to show acceptance rate bars (default True)
        correctness_column: Column to use for acceptance rate calculation (default: is_correct)
    """
    if len(results_df) == 0:
        print(f"No data to plot{title_suffix}")
        return

    fine_grained = (
        results_df.groupby("generation_version")
        .agg(
            {
                "self_green_score": ["mean", "std", "count"],
                "vlm_green_score": ["mean", "std"],
            }
        )
        .round(4)
    )

    # Calculate acceptance rate per version based on correctness_column
    if correctness_column in results_df.columns:
        # Handle string values: 'true' -> True, everything else ('false', 'ERROR', etc.) -> False
        is_correct_bool = results_df[correctness_column].apply(
            lambda x: str(x).lower() == "true" if pd.notna(x) else False
        )
        acceptance_rates = (
            is_correct_bool.groupby(results_df["generation_version"])
            .mean()
            .reindex(fine_grained.index)
            .values
        )
    else:
        acceptance_rates = None

    versions = fine_grained.index.tolist()
    vlm_means = fine_grained[("vlm_green_score", "mean")].values
    vlm_stds = fine_grained[("vlm_green_score", "std")].values
    counts = fine_grained[("self_green_score", "count")].values

    # Calculate Standard Error of the Mean (SEM = std / sqrt(n))
    vlm_sems = vlm_stds / np.sqrt(counts)

    print("\nComparison of std vs SEM:")
    for v, std, sem, n in zip(versions, vlm_stds, vlm_sems, counts):
        print(f"  {v}: std={std:.4f}, SEM={sem:.4f}, n={n}")

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

    # Plot acceptance rate as a line on primary axis (same 0-1 scale as GREEN score)
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
    plot_path = output_path.replace(".csv", f"_plot{file_suffix}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {plot_path}")

    plt.close()


def main():
    args = parse_args()

    # Parse samples argument
    if args.samples.lower() == "all":
        samples_per_version = None
        print("Mode: FULL (all samples)")
    else:
        samples_per_version = int(args.samples)
        print(f"Mode: SAMPLING ({samples_per_version} per version)")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = "green_scores_by_version.csv"

    # Load and clean data
    original_data, clean_data = load_and_clean_data(args.input)

    # Get versions to process
    versions = get_versions_to_process(clean_data, args.min_version)

    if not versions:
        print(f"No versions found >= {args.min_version}")
        sys.exit(1)

    # Compute GREEN scores
    results_df = compute_green_scores(
        clean_data, versions, samples_per_version, args.self_check
    )

    # Print summary
    print_summary(results_df)

    # Merge GREEN scores back to original dataframe
    score_columns = [
        "vlm_green_score",
        "vlm_green_analysis",
    ]
    if args.self_check:
        score_columns.extend(["self_green_score", "self_green_analysis"])
    # Also include error count columns
    for col in results_df.columns:
        if col.startswith("vlm_(") or col == "vlm_Matched Findings":
            score_columns.append(col)

    merge_cols = ["row_number"] + score_columns
    original_with_scores = original_data.merge(
        results_df[merge_cols], on="row_number", how="left"
    )

    # Save full results (original data + GREEN scores)
    full_output_path = output_path.replace(".csv", "_full.csv")
    original_with_scores.to_csv(full_output_path, index=False)
    print(f"\nFull results (with rejected rows) saved to: {full_output_path}")
    print(
        f"Total rows: {len(original_with_scores)} (kept: {original_with_scores['kept_for_green'].sum()}, rejected: {(~original_with_scores['kept_for_green']).sum()})"
    )

    # Save processed results only
    results_df.to_csv(output_path, index=False)
    print(f"Processed results only saved to: {output_path}")
    print(f"Processed rows: {len(results_df)}")

    # Generate plots
    if not args.no_plot:
        # Create pngs subdirectory
        pngs_dir = Path(output_path).parent / "pngs"
        pngs_dir.mkdir(parents=True, exist_ok=True)
        plot_output_path = str(pngs_dir / Path(output_path).name)

        print("\n--- Generating plots ---")

        # Define correctness modes to plot
        correctness_modes = [
            ("is_correct", "is_correct", ""),
            ("is_findings_correct", "is_findings_correct", "_findings"),
            ("is_impressions_correct", "is_impressions_correct", "_impressions"),
            ("is_both_correct", None, "_both"),  # Special case: combined
        ]

        # Add combined column (is_findings_correct AND is_impressions_correct)
        if (
            "is_findings_correct" in results_df.columns
            and "is_impressions_correct" in results_df.columns
        ):
            results_df["is_both_correct"] = results_df.apply(
                lambda row: (
                    "true"
                    if (
                        str(row.get("is_findings_correct", "")).lower() == "true"
                        and str(row.get("is_impressions_correct", "")).lower() == "true"
                    )
                    else "false"
                ),
                axis=1,
            )

        for mode_name, col_name, file_suffix in correctness_modes:
            # Use mode_name as column if col_name is None (for combined mode)
            actual_col = col_name if col_name else mode_name

            if actual_col not in results_df.columns:
                print(f"  Skipping {mode_name}: column not found")
                continue

            # Plot with acceptance rate for this mode
            title_suffix = f" ({mode_name})" if mode_name != "is_correct" else ""
            generate_plot(
                results_df,
                plot_output_path,
                title_suffix=title_suffix,
                file_suffix=file_suffix,
                show_acceptance_rate=True,
                correctness_column=actual_col,
            )

            # Plot accepted/rejected splits (optional)
            if args.split_by_acceptance:
                is_true = results_df[actual_col].apply(
                    lambda x: str(x).lower() == "true" if pd.notna(x) else False
                )
                accepted_df = results_df[is_true]
                rejected_df = results_df[~is_true]

                if len(accepted_df) > 0:
                    generate_plot(
                        accepted_df,
                        plot_output_path,
                        title_suffix=f" - Accepted Only ({mode_name})",
                        file_suffix=f"{file_suffix}_accepted",
                        show_acceptance_rate=False,
                        correctness_column=actual_col,
                    )

                if len(rejected_df) > 0:
                    generate_plot(
                        rejected_df,
                        plot_output_path,
                        title_suffix=f" - Rejected Only ({mode_name})",
                        file_suffix=f"{file_suffix}_rejected",
                        show_acceptance_rate=False,
                        correctness_column=actual_col,
                    )

    print("\nAll done!")


if __name__ == "__main__":
    main()
