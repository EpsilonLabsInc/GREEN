#!/usr/bin/env python3
"""
Simple GREEN Score Computation

Computes GREEN scores for any CSV file with reference and hypothesis columns.
No special filtering or version handling - just straightforward scoring.

Usage:
    python compute_green.py --input data.csv --ref-column reference --hyp-column hypothesis
    python compute_green.py -i data.csv -r ground_truth -h generated_text -o scores.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path for GREEN import
sys.path.insert(0, str(Path(__file__).parent.parent))
from green_score import GREEN

import warnings

warnings.filterwarnings("ignore", message=".*resume_download.*")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute GREEN scores for any CSV with ref/hyp columns"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--ref-column",
        "-r",
        type=str,
        required=True,
        help="Column name for reference text",
    )
    parser.add_argument(
        "--hyp-column",
        "-y",
        type=str,
        required=True,
        help="Column name for hypothesis text",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output CSV file (default: input_with_green_scores.csv)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=None,
        help="Process in batches of this size (for large files)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output",
    )
    return parser.parse_args()


def compute_scores(
    data: pd.DataFrame,
    ref_column: str,
    hyp_column: str,
    batch_size: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Compute GREEN scores for all rows.

    Args:
        data: DataFrame with ref and hyp columns
        ref_column: Column name for reference text
        hyp_column: Column name for hypothesis text
        batch_size: If set, process in batches
        verbose: Show progress

    Returns:
        DataFrame with added GREEN score columns
    """
    print(f"\nInitializing GREEN scorer...")
    green_scorer = GREEN(use_azure=True, verbose=verbose)
    print("GREEN scorer initialized.")

    # Get reference and hypothesis texts
    refs = data[ref_column].fillna("").astype(str).tolist()
    hyps = data[hyp_column].fillna("").astype(str).tolist()

    total_rows = len(refs)
    print(f"\nProcessing {total_rows} rows...")

    if batch_size and batch_size < total_rows:
        # Process in batches
        all_scores = []
        all_analyses = []
        all_error_dfs = []

        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            print(f"  Processing rows {start + 1} to {end}...")

            batch_refs = refs[start:end]
            batch_hyps = hyps[start:end]

            mean, std, scores, summary, result_df = green_scorer(batch_refs, batch_hyps)
            all_scores.extend(scores)
            all_analyses.extend(result_df["green_analysis"].tolist())
            all_error_dfs.append(result_df)

        # Combine error DataFrames
        error_df = pd.concat(all_error_dfs, ignore_index=True)
        scores = all_scores
        analyses = all_analyses
    else:
        # Process all at once
        mean, std, scores, summary, error_df = green_scorer(refs, hyps)
        analyses = error_df["green_analysis"].tolist()

    # Add scores to original DataFrame
    result = data.copy()
    result["green_score"] = scores
    result["green_analysis"] = analyses

    # Add error counts
    for col in error_df.columns:
        if col.startswith("(") or col == "Matched Findings":
            result[f"green_{col}"] = error_df[col].values

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total rows: {len(result)}")
    print(f"Mean GREEN score: {result['green_score'].mean():.4f}")
    print(f"Std GREEN score: {result['green_score'].std():.4f}")
    print(f"Min: {result['green_score'].min():.4f}")
    print(f"Max: {result['green_score'].max():.4f}")

    return result


def main():
    args = parse_args()

    # Load data
    print(f"Loading data from: {args.input}")
    data = pd.read_csv(args.input)
    print(f"Loaded {len(data)} rows")

    # Validate columns exist
    if args.ref_column not in data.columns:
        print(f"Error: Reference column '{args.ref_column}' not found in CSV")
        print(f"Available columns: {list(data.columns)}")
        sys.exit(1)

    if args.hyp_column not in data.columns:
        print(f"Error: Hypothesis column '{args.hyp_column}' not found in CSV")
        print(f"Available columns: {list(data.columns)}")
        sys.exit(1)

    # Compute scores
    result = compute_scores(
        data,
        args.ref_column,
        args.hyp_column,
        args.batch_size,
        args.verbose,
    )

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input)
        output_path = str(
            input_path.parent / f"{input_path.stem}_with_green_scores.csv"
        )

    # Save results
    result.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
