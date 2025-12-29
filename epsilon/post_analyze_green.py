#!/usr/bin/env python3
"""
Post-Analysis for GREEN Scores

Adds soft_correct column using Azure GPT to determine if differences between
reference and hypothesis are minor (punctuation, "No comparison", etc.).

Usage:
    python post_analyze_green.py                                    # Default input
    python post_analyze_green.py --input green_scores_by_version_full.csv
    python post_analyze_green.py --samples 10                       # Test with 10 samples
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

# Add parent directory to path for config loading
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add soft_correct column to GREEN analysis results"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="green_scores_by_version_full.csv",
        help="Path to input CSV file (output from analyze_green.py)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output CSV file (default: adds _soft_correct suffix)",
    )
    parser.add_argument(
        "--samples",
        "-s",
        type=str,
        default=None,
        help="Number of samples to process, or 'all' for all samples (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of parallel API calls (default: 50)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output",
    )
    return parser.parse_args()


def load_azure_config(config_path=None):
    """Load Azure OpenAI configuration from config files."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.json"

    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    azure_config = config.get("azure_openai", {})

    # Load secrets from config.secret.json
    secret_path = Path(config_path).parent / "config.secret.json"
    if secret_path.exists():
        with open(secret_path, "r") as f:
            secret_config = json.load(f)
        if "azure_openai" in secret_config:
            azure_config.update(secret_config["azure_openai"])

    return azure_config


def init_azure_client(verbose=False):
    """Initialize Azure OpenAI client."""
    try:
        from openai import AzureOpenAI
    except ImportError:
        raise ImportError(
            "openai package is required. Install with: pip install openai"
        )

    azure_config = load_azure_config()

    api_key = os.environ.get("AZURE_OPENAI_API_KEY", azure_config.get("api_key"))
    if api_key == "<your-api-key>" or not api_key:
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Azure OpenAI API key not configured. Set AZURE_OPENAI_API_KEY "
                "environment variable or update config.secret.json."
            )

    model_name = azure_config.get("model_name", "gpt-5.2")
    deployment = azure_config.get("deployment", model_name)

    client = AzureOpenAI(
        api_version=azure_config.get("api_version", "2024-12-01-preview"),
        azure_endpoint=azure_config.get("endpoint"),
        api_key=api_key,
    )

    if verbose:
        print(f"Initialized Azure OpenAI client with model: {model_name}")

    return client, deployment


def extract_findings_impression_text(report_text: str) -> str:
    """Extract FINDINGS and IMPRESSION sections from report text."""
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


SOFT_CORRECT_PROMPT = """You are a medical report comparison expert. Compare the REFERENCE (ground truth) radiology report with the HYPOTHESIS (generated) report.

Determine if the HYPOTHESIS should be considered a "soft accept" - meaning the differences are MINOR and clinically insignificant.

SOFT ACCEPT criteria (return "SOFT_ACCEPT"):
S1. Only punctuation differences (commas, periods, colons, semicolons)
S2. Only whitespace/formatting differences
S3. Only capitalization differences
S4. Only minor word order changes that don't change meaning
S5. Hypothesis adds generic/boilerplate phrases like:
   - "No comparison available"
   - "No prior comparison"
   - "No previous study for comparison"
   - "Clinical correlation recommended"
   - "No acute findings"
   - "No significant change" (when reference is empty or similar)
S6. Hypothesis omits generic phrases that add no clinical value
S7. Minor synonym substitutions (e.g., "normal" vs "unremarkable", "no" vs "none")
S8. Number formatting differences (e.g., "1" vs "one")

REJECT criteria (return "REJECT"):
R1. Missing clinically significant findings
R2. Added signifcant findings not present in reference
R3. Different anatomical locations mentioned
R4. Different severity/size descriptions
R5. Different diagnoses or impressions
R6. Contradictory statements
R7. Missing or different measurements



REFERENCE:
{reference}

HYPOTHESIS:
{hypothesis}

Respond with ONLY a JSON object in this exact format:
{{"decision": "SOFT_ACCEPT" or "REJECT",
"reason_list": "list reasons here, 'S1', 'R2', 'S1, S2' etc. If new reason, explain briefly.",
"reason_text": "brief explanation of the decision"}}
"""


def check_soft_correct(
    client,
    deployment: str,
    reference: str,
    hypothesis: str,
    verbose: bool = False,
) -> tuple[bool, str]:
    """
    Use Azure GPT to determine if differences are minor enough for soft accept.

    Returns:
        tuple: (is_soft_correct: bool, reason: str)
    """
    # Handle empty/missing texts
    if not reference and not hypothesis:
        return True, "Both empty"
    if not reference:
        return False, "Reference is empty but hypothesis is not"
    if not hypothesis:
        return False, "Hypothesis is empty but reference is not"

    # Quick check: if texts are identical, it's definitely a soft accept
    if reference.strip() == hypothesis.strip():
        return True, "Texts are identical"

    prompt = SOFT_CORRECT_PROMPT.format(reference=reference, hypothesis=hypothesis)

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical report comparison expert. Respond only with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_completion_tokens=200,
        )

        result_text = response.choices[0].message.content.strip()

        # Parse JSON response
        # Handle potential markdown code blocks
        if result_text.startswith("```"):
            result_text = re.sub(r"```json?\n?", "", result_text)
            result_text = re.sub(r"\n?```", "", result_text)

        result = json.loads(result_text)
        decision = result.get("decision", "REJECT").upper()
        reason = result.get("reason", "No reason provided")

        is_soft_correct = decision == "SOFT_ACCEPT"

        if verbose:
            print(f"  Decision: {decision}, Reason: {reason}")

        return is_soft_correct, reason

    except json.JSONDecodeError as e:
        if verbose:
            print(f"  JSON parse error: {e}, Response: {result_text}")
        return False, f"JSON parse error: {result_text}"
    except Exception as e:
        if verbose:
            print(f"  API error: {e}")
        return False, f"API error: {str(e)}"


def process_single_row(args):
    """Process a single row for soft_correct (for parallel execution)."""
    idx, row, client, deployment, verbose = args

    ref = row.get("ref_extracted", "")
    hyp = row.get("hyp_extracted", "")

    # If already marked correct, it's definitely soft correct
    is_correct = row.get("is_correct")
    if is_correct == True or str(is_correct).lower() == "true":
        return idx, True, "Already marked as correct"

    is_soft_correct, reason = check_soft_correct(
        client, deployment, ref, hyp, verbose=verbose
    )

    return idx, is_soft_correct, reason


def compute_soft_correct(
    data: pd.DataFrame,
    client,
    deployment: str,
    batch_size: int = 50,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Add soft_correct column to the dataframe using Azure GPT.

    Args:
        data: DataFrame with ref_extracted and hyp_extracted columns
        client: Azure OpenAI client
        deployment: Azure deployment name
        batch_size: Number of parallel API calls
        verbose: Print verbose output

    Returns:
        DataFrame with soft_correct and soft_correct_reason columns added
    """
    # Initialize new columns
    data = data.copy()
    data["soft_correct"] = None
    data["soft_correct_reason"] = None

    # Filter to rows that need processing (have ref and hyp extracted)
    mask = data["ref_extracted"].notna() & data["hyp_extracted"].notna()
    rows_to_process = data[mask].copy()

    print(f"Processing {len(rows_to_process)} rows with ref/hyp data...")

    # Prepare arguments for parallel processing
    args_list = [
        (idx, row, client, deployment, verbose)
        for idx, row in rows_to_process.iterrows()
    ]

    # Process in parallel with progress bar
    results = {}
    error_count = 0
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {
            executor.submit(process_single_row, args): args[0] for args in args_list
        }

        with tqdm(total=len(futures), desc="Computing soft_correct") as pbar:
            for future in as_completed(futures):
                idx, is_soft_correct, reason = future.result()
                results[idx] = (is_soft_correct, reason)
                # Count and show errors
                if reason.startswith("API error:") or reason.startswith(
                    "JSON parse error:"
                ):
                    error_count += 1
                    if error_count == 1:
                        # Print first error as warning
                        tqdm.write(f"⚠️  First error encountered: {reason}...")
                pbar.update(1)

    if error_count > 0:
        print(f"\n⚠️  Total API errors: {error_count}/{len(results)}")

    # Update dataframe with results
    for idx, (is_soft_correct, reason) in results.items():
        data.at[idx, "soft_correct"] = is_soft_correct
        data.at[idx, "soft_correct_reason"] = reason

    return data


def print_summary(data: pd.DataFrame):
    """Print summary statistics for soft_correct."""
    print("\n" + "=" * 60)
    print("SOFT CORRECT SUMMARY")
    print("=" * 60)

    # Filter to rows with soft_correct values
    valid = data[data["soft_correct"].notna()].copy()

    if len(valid) == 0:
        print("No rows with soft_correct values")
        return

    # Normalize boolean helper
    def normalize_bool(x):
        if pd.isna(x):
            return None
        if isinstance(x, bool):
            return x
        return str(x).lower() == "true"

    # Convert soft_correct to numeric (1/0) for proper aggregation
    valid["soft_correct_int"] = valid["soft_correct"].astype(int)

    total = len(valid)
    soft_correct_count = valid["soft_correct_int"].sum()
    soft_correct_rate = soft_correct_count / total

    print(f"Total processed: {total}")
    print(f"Soft correct: {soft_correct_count} ({soft_correct_rate:.1%})")
    print(f"Rejected: {total - soft_correct_count} ({1 - soft_correct_rate:.1%})")

    # Show average GREEN score
    if "vlm_green_score" in valid.columns:
        avg_green = valid["vlm_green_score"].mean()
        std_green = valid["vlm_green_score"].std()
        print(f"\nAvg GREEN score (all): {avg_green:.3f} ± {std_green:.3f}")

        # GREEN score for soft_correct vs rejected
        soft_green = valid[valid["soft_correct_int"] == 1]["vlm_green_score"]
        reject_green = valid[valid["soft_correct_int"] == 0]["vlm_green_score"]
        if len(soft_green) > 0:
            print(
                f"Avg GREEN score (soft_correct=True): {soft_green.mean():.3f} ± {soft_green.std():.3f}"
            )
        if len(reject_green) > 0:
            print(
                f"Avg GREEN score (soft_correct=False): {reject_green.mean():.3f} ± {reject_green.std():.3f}"
            )

    # Compare with is_correct if available
    if "is_correct" in valid.columns:
        valid["is_correct_bool"] = valid["is_correct"].apply(normalize_bool)
        correct_mask = valid["is_correct_bool"] == True

        originally_correct = correct_mask.sum()
        newly_accepted = (
            (valid["soft_correct_int"] == 1) & (valid["is_correct_bool"] == False)
        ).sum()

        print(f"\nOriginally correct (is_correct=True): {originally_correct}")
        print(f"Newly accepted by soft_correct: {newly_accepted}")
        print(
            f"Total soft accepted: {soft_correct_count} (original + {newly_accepted} new)"
        )

    # Create is_correct_both column (findings AND impressions correct)
    if (
        "is_findings_correct" in valid.columns
        and "is_impressions_correct" in valid.columns
    ):
        valid["is_findings_correct_bool"] = valid["is_findings_correct"].apply(
            normalize_bool
        )
        valid["is_impressions_correct_bool"] = valid["is_impressions_correct"].apply(
            normalize_bool
        )

        # is_correct_both = True only if BOTH are True
        valid["is_correct_both"] = valid.apply(
            lambda row: (
                True
                if (
                    row["is_findings_correct_bool"] == True
                    and row["is_impressions_correct_bool"] == True
                )
                else (
                    False
                    if (
                        row["is_findings_correct_bool"] == False
                        or row["is_impressions_correct_bool"] == False
                    )
                    else None
                )
            ),
            axis=1,
        )

        print("\n" + "-" * 60)
        print("SOFT CORRECT BY is_correct_both (findings AND impressions)")
        print("-" * 60)

        # Group by is_correct_both
        for status, label in [
            (True, "is_correct_both=True"),
            (False, "is_correct_both=False"),
            (None, "is_correct_both=None"),
        ]:
            if status is None:
                subset = valid[valid["is_correct_both"].isna()]
            else:
                subset = valid[valid["is_correct_both"] == status]

            if len(subset) == 0:
                continue

            subset_total = len(subset)
            subset_soft = subset["soft_correct_int"].sum()
            subset_rate = subset_soft / subset_total if subset_total > 0 else 0

            print(f"\n{label}:")
            print(f"  Total: {subset_total}")
            print(f"  Soft correct: {subset_soft} ({subset_rate:.1%})")
            print(f"  Rejected: {subset_total - subset_soft} ({1 - subset_rate:.1%})")
            if "vlm_green_score" in subset.columns:
                avg_green = subset["vlm_green_score"].mean()
                std_green = subset["vlm_green_score"].std()
                print(f"  Avg GREEN score: {avg_green:.3f} ± {std_green:.3f}")

        # Also show breakdown by individual columns
        print("\n" + "-" * 60)
        print("SOFT CORRECT BY is_findings_correct")
        print("-" * 60)
        for status in [True, False]:
            subset = valid[valid["is_findings_correct_bool"] == status]
            if len(subset) == 0:
                continue
            subset_total = len(subset)
            subset_soft = subset["soft_correct_int"].sum()
            subset_rate = subset_soft / subset_total if subset_total > 0 else 0
            avg_green = (
                subset["vlm_green_score"].mean()
                if "vlm_green_score" in subset.columns
                else 0
            )
            print(
                f"  is_findings_correct={status}: {subset_soft}/{subset_total} soft correct ({subset_rate:.1%}), avg GREEN: {avg_green:.3f}"
            )

        print("\n" + "-" * 60)
        print("SOFT CORRECT BY is_impressions_correct")
        print("-" * 60)
        for status in [True, False]:
            subset = valid[valid["is_impressions_correct_bool"] == status]
            if len(subset) == 0:
                continue
            subset_total = len(subset)
            subset_soft = subset["soft_correct_int"].sum()
            subset_rate = subset_soft / subset_total if subset_total > 0 else 0
            avg_green = (
                subset["vlm_green_score"].mean()
                if "vlm_green_score" in subset.columns
                else 0
            )
            print(
                f"  is_impressions_correct={status}: {subset_soft}/{subset_total} soft correct ({subset_rate:.1%}), avg GREEN: {avg_green:.3f}"
            )

    # Breakdown by version if available
    if "generation_version" in valid.columns:
        print("\n" + "-" * 60)
        print("By version:")
        print("-" * 60)
        version_stats = (
            valid.groupby("generation_version")
            .agg(
                {
                    "soft_correct_int": ["sum", "count", "mean"],
                }
            )
            .round(3)
        )
        version_stats.columns = ["soft_correct_count", "total", "soft_correct_rate"]
        print(version_stats)


def main():
    args = parse_args()

    # Parse samples argument
    samples_count = None
    if args.samples:
        if args.samples.lower() == "all":
            samples_count = None
            print("Mode: FULL (all samples)")
        else:
            samples_count = int(args.samples)
            print(f"Mode: SAMPLING ({samples_count} samples)")

    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Loading data from: {input_path}")
    data = pd.read_csv(input_path)
    print(f"Loaded {len(data)} rows")

    # Apply sampling if specified
    if samples_count:
        # Filter to rows with ref/hyp first, then sample
        valid_mask = data["ref_extracted"].notna() & data["hyp_extracted"].notna()
        valid_indices = data[valid_mask].index[:samples_count]
        # Keep all rows but only process sampled ones
        process_mask = data.index.isin(valid_indices)
        data_to_process = data[process_mask].copy()
        print(f"Sampling {len(data_to_process)} rows for processing")
    else:
        data_to_process = data

    # Initialize Azure client
    print("\nInitializing Azure OpenAI client...")
    client, deployment = init_azure_client(verbose=args.verbose)

    # Compute soft_correct
    result_data = compute_soft_correct(
        data_to_process,
        client,
        deployment,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    # If we sampled, merge back to full data
    if samples_count:
        data.loc[result_data.index, "soft_correct"] = result_data["soft_correct"]
        data.loc[result_data.index, "soft_correct_reason"] = result_data[
            "soft_correct_reason"
        ]
        result_data = data

    # Print summary
    print_summary(result_data)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = str(input_path).replace(".csv", "_soft_correct.csv")

    # Save results
    result_data.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
