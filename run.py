import argparse
import pandas as pd
from green_score import GREEN


def main():
    parser = argparse.ArgumentParser(description="Run GREEN score analysis on CSV data")
    parser.add_argument("--input-csv", required=True, help="Path to input CSV file")
    parser.add_argument(
        "--gt-column", required=True, help="Column name for ground truth (reference)"
    )
    parser.add_argument(
        "--comparison-columns",
        required=True,
        help="Comma-separated list of column names to compare against ground truth",
    )
    parser.add_argument(
        "--output-csv",
        default="green_scores_output.csv",
        help="Path to output CSV file",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    # Parse comparison columns
    comparison_cols = [col.strip() for col in args.comparison_columns.split(",")]

    # Load input CSV
    print(f"Loading input CSV: {args.input_csv}")
    input_df = pd.read_csv(args.input_csv)

    # Validate columns exist
    if args.gt_column not in input_df.columns:
        raise ValueError(f"Ground truth column '{args.gt_column}' not found in CSV")
    for col in comparison_cols:
        if col not in input_df.columns:
            raise ValueError(f"Comparison column '{col}' not found in CSV")

    # Get references (ground truth)
    refs = input_df[args.gt_column].fillna("").astype(str).tolist()

    # Initialize GREEN scorer
    print("Initializing GREEN scorer...")
    green_scorer = GREEN(use_azure=True, verbose=args.verbose)

    # Start with the original dataframe
    output_df = input_df.copy()

    # Run analysis for each comparison column
    for col in comparison_cols:
        print(f"\n{'='*60}")
        print(f"Analyzing column: {col}")
        print(f"{'='*60}")

        hyps = input_df[col].fillna("").astype(str).tolist()

        mean, std, green_score_list, summary, result_df = green_scorer(refs, hyps)

        print(f"\n[{col}] Mean GREEN score: {mean:.4f} ± {std:.4f}")
        if summary:
            print(summary)

        # Add results with column prefix
        output_df[f"{col}_green_score"] = green_score_list
        output_df[f"{col}_green_analysis"] = result_df["green_analysis"].values

        # Add error counts
        error_columns = [
            "(a) False report of a finding in the candidate",
            "(b) Missing a finding present in the reference",
            "(c) Misidentification of a finding's anatomic location/position",
            "(d) Misassessment of the severity of a finding",
            "(e) Mentioning a comparison that isn't in the reference",
            "(f) Omitting a comparison detailing a change from a prior study",
            "Matched Findings",
        ]
        for error_col in error_columns:
            if error_col in result_df.columns:
                # Shorten column names for readability
                short_name = (
                    error_col.split(")")[0] + ")" if ")" in error_col else error_col
                )
                output_df[f"{col}_{short_name}"] = result_df[error_col].values

    # Save combined results
    print(f"\nSaving results to: {args.output_csv}")
    output_df.to_csv(args.output_csv, index=False)
    print("Done!")

    # Print summary statistics
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    for col in comparison_cols:
        scores = output_df[f"{col}_green_score"].dropna()
        print(
            f"{col}: mean={scores.mean():.4f}, std={scores.std():.4f}, n={len(scores)}"
        )


if __name__ == "__main__":
    main()
