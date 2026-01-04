#!/usr/bin/env python3
"""
GREEN Score Analysis Script

Processes analytics CSV files and computes GREEN scores by version.
For plotting, use plot_green.py after running this script.

Usage:
    python analyze_green.py                           # Default: sample 3 per version
    python analyze_green.py --input /path/to/file.csv
    python analyze_green.py --samples all             # Run all samples
    python analyze_green.py --samples 50              # Sample 50 per version
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from packaging import version as pkg_version

# Add parent directory to path for GREEN import
sys.path.insert(0, str(Path(__file__).parent.parent))
from green_score import GREEN

import warnings

warnings.filterwarnings("ignore", message=".*resume_download.*")


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
        "--self-check",
        action="store_true",
        help="Perform self-check (report_text vs report_text) sanity check (default: False)",
    )
    parser.add_argument(
        "--use-full-text",
        action="store_true",
        help="Compare against generated_report_full_text (with extraction) instead of generated_vlm_output_text (default: False)",
    )
    parser.add_argument(
        "--ref-column",
        type=str,
        default=None,
        help="Column name for reference text (default: report_text with FINDINGS/IMPRESSION extraction)",
    )
    parser.add_argument(
        "--hyp-column",
        type=str,
        default=None,
        help="Column name for hypothesis text (default: generated_vlm_output_text, or generated_report_full_text with --use-full-text)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Use columns directly without FINDINGS/IMPRESSION extraction (default: False)",
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


def extract_findings_impression_text(report_text: str) -> str:
    if not isinstance(report_text, str):
        return ""

    pattern = re.compile(
        r"FINDINGS:\s*(.*?)\s*IMPRESSION:\s*(.*?)(?:\n{2,}|$)",
        re.DOTALL | re.IGNORECASE,
    )

    match = pattern.search(report_text)
    if not match:
        return ""

    findings = match.group(1).strip()
    impression = re.split(
        r"Electronically Signed|Signed By|Authored By",
        match.group(2),
        flags=re.IGNORECASE,
    )[0].strip()

    return f"FINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}"


def compute_green_scores(
    clean_data: pd.DataFrame,
    versions: list,
    samples_per_version: int | None,
    perform_self_check: bool = False,
    use_full_text: bool = False,
    ref_column: str | None = None,
    hyp_column: str | None = None,
    no_extract: bool = False,
) -> pd.DataFrame:
    """Compute GREEN scores for all versions.

    Args:
        ref_column: Custom column for reference text (default: report_text)
        hyp_column: Custom column for hypothesis text (default: generated_vlm_output_text)
        no_extract: If True, use columns directly without FINDINGS/IMPRESSION extraction
    """
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

        # Extract reference text
        ref_col = ref_column or "report_text"
        if no_extract:
            ver_data["ref_extracted"] = ver_data[ref_col].fillna("").astype(str)
        else:
            ver_data["ref_extracted"] = (
                ver_data[ref_col]
                .fillna("")
                .astype(str)
                .apply(extract_findings_impression_text)
            )
        refs = ver_data["ref_extracted"].tolist()

        # Choose hypothesis source based on configuration
        if hyp_column:
            # Use custom hypothesis column
            if no_extract:
                ver_data["hyp_extracted"] = ver_data[hyp_column].fillna("").astype(str)
            else:
                ver_data["hyp_extracted"] = (
                    ver_data[hyp_column]
                    .fillna("")
                    .astype(str)
                    .apply(extract_findings_impression_text)
                )
            hyps_vlm = ver_data["hyp_extracted"].tolist()
            comparison_source = f"{hyp_column}{'' if no_extract else ' (extracted)'}"
        elif use_full_text:
            # Extract from generated_report_full_text
            ver_data["hyp_extracted"] = (
                ver_data["generated_report_full_text"]
                .fillna("")
                .astype(str)
                .apply(extract_findings_impression_text)
            )
            hyps_vlm = ver_data["hyp_extracted"].tolist()
            comparison_source = "generated_report_full_text (extracted)"
        else:
            # Use generated_vlm_output_text directly (already contains just findings/impression)
            hyps_vlm = (
                ver_data["generated_vlm_output_text"].fillna("").astype(str).tolist()
            )
            ver_data["hyp_extracted"] = hyps_vlm  # Store for CSV output
            comparison_source = "generated_vlm_output_text"

        row_numbers = ver_data["row_number"].tolist()
        ref_extracted_values = ver_data["ref_extracted"].tolist()
        hyp_extracted_values = ver_data["hyp_extracted"].tolist()

        # print(f"Comparing against: {comparison_source}")
        # print(ref_extracted_values[:3])
        # print(ver_data["generated_report_full_text"].tolist()[:3])
        # print(hyp_extracted_values[:3])

        is_correct_values = (
            ver_data["is_correct"].tolist()
            if "is_correct" in ver_data.columns
            else [None] * len(ver_data)
        )

        # (1) Optional sanity check: report_text vs report_text
        if perform_self_check:
            hyps_self = refs  # Use extracted refs for self-check
            print(f"\n--- (1) Sanity check: ref_extracted vs ref_extracted ---")
            mean1, std1, scores1, summary1, df1 = green_scorer(refs, hyps_self)
            print(f"Mean score: {mean1:.4f} (should be ~1.0)")

        # (2) Actual comparison
        print(f"\n--- Comparison: ref_extracted vs {comparison_source} ---")
        mean2, std2, scores2, summary2, df2 = green_scorer(refs, hyps_vlm)
        print(f"Mean score: {mean2:.4f}")

        # Combine results for this version
        for i, row_num in enumerate(row_numbers):
            result = {
                "generation_version": ver,
                "row_number": row_num,
                "is_correct": is_correct_values[i],
                "ref_extracted": ref_extracted_values[i],
                "hyp_extracted": hyp_extracted_values[i],
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
        clean_data,
        versions,
        samples_per_version,
        args.self_check,
        args.use_full_text,
        args.ref_column,
        args.hyp_column,
        args.no_extract,
    )

    # Print summary
    print_summary(results_df)

    # Merge GREEN scores back to original dataframe
    score_columns = [
        "ref_extracted",
        "hyp_extracted",
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

    print("\nAll done! Use plot_green.py to generate plots from the CSV.")


if __name__ == "__main__":
    main()
