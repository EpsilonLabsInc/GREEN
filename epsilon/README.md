# GREEN Score Analysis Tool

Analyze GREEN scores from analytics CSV files, compute scores by version, and generate visualizations.

## GREEN Score Formula

$$\text{GREEN Score} = \frac{\text{Matched Findings}}{\text{Matched Findings} + \sum(\text{Significant Errors})}$$

### Score Components

| Component | Description |
|-----------|-------------|
| **Matched Findings** | Clinically relevant findings present in both reference and hypothesis |
| **Significant Errors** | Sum of 6 error subcategories (see below) |
| **Score Range** | 0 to 1 (1.0 = perfect match, 0 = no matched findings) |

### Significant Error Subcategories

| Code | Error Type |
|------|------------|
| (a) | False report of a finding in the candidate |
| (b) | Missing a finding present in the reference |
| (c) | Misidentification of a finding's anatomic location/position |
| (d) | Misassessment of the severity of a finding |
| (e) | Mentioning a comparison that isn't in the reference |
| (f) | Omitting a comparison detailing a change from a prior study |

> **Note:** Clinically Insignificant Errors are tracked but **do NOT affect** the GREEN score calculation.

---

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `compute_green_analytics.py` | Compute GREEN scores from raw analytics CSV (with version filtering, addendum exclusion) |
| `compute_green_general.py` | Compute GREEN scores from any CSV with ref/hyp columns (no special filtering) |
| `plot_green.py` | Regenerate plots from existing CSV (no recomputation) |
| `plot_green_by_normal.py` | Generate plots split by normal/abnormal reports and body part |

---

## 1. compute_green_analytics.py - Compute GREEN Scores (Analytics CSV)

Processes analytics CSV files with special filtering (excludes addendums, specific radiologists, etc.), computes GREEN scores by version.

### Usage

```bash
# From the epsilon directory
cd /Users/ruian/projects/epsilonlabs/GREEN/epsilon

# Run with defaults (3 samples per version for testing)
python compute_green_analytics.py

# Run all samples (full analysis)
python compute_green_analytics.py --samples all

# Use a specific input file
python compute_green_analytics.py --input /path/to/analytics.csv
```

### Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input` | `-i` | `ANALYTICS_MOST_RECENT.csv` | Path to input analytics CSV file |
| `--samples` | `-s` | `3` | Number of samples per version, or `all` for all samples |
| `--output` | `-o` | `green_scores_by_version.csv` | Path to output CSV file |
| `--min-version` | | `0.1.2` | Minimum version to process |
| `--self-check` | | | Perform self-check (report vs itself) sanity check |
| `--use-full-text` | | | Compare against `generated_report_full_text` (with extraction) instead of `generated_vlm_output_text` |
| `--ref-column` | | `report_text` | Column name for reference text |
| `--hyp-column` | | `generated_vlm_output_text` | Column name for hypothesis text |
| `--no-extract` | | | Use columns directly without FINDINGS/IMPRESSION extraction |

### Examples

```bash
# Sample 50 rows per version
python compute_green_analytics.py --samples 50

# Run all samples with custom output
python compute_green_analytics.py --samples all --output /tmp/results.csv

# Process only versions >= 0.2.0
python compute_green_analytics.py --min-version 0.2.0

# Run with self-check
python compute_green_analytics.py --samples all --self-check

# Use custom columns for comparison
python compute_green_analytics.py --ref-column my_reference_col --hyp-column my_hypothesis_col

# Use columns directly without FINDINGS/IMPRESSION extraction
python compute_green_analytics.py --ref-column col1 --hyp-column col2 --no-extract

# Run in background with all samples
nohup python compute_green_analytics.py --samples all > output.log 2>&1 &
```

### Output Files

#### CSV Files
| File | Description |
|------|-------------|
| `green_scores_by_version.csv` | Processed results only (rows that passed filtering) |
| `green_scores_by_version_full.csv` | All original rows with `kept_for_green` column and GREEN scores merged |

### Data Filtering

Rows are **excluded** from GREEN scoring if:
- `report_is_addendum == True` (addendum reports)
- `report_radiologist_first_name == 'Justin'` (specific radiologist)
- `report_radiologist_first_name` is NaN
- `is_correct` is NaN

The `kept_for_green` column in the full output indicates whether each row was included.

---

## 2. compute_green_general.py - Compute GREEN Scores (Any CSV)

Simple GREEN scorer for any CSV file with reference and hypothesis columns. No special filtering or version handling.

### Usage

```bash
# Basic usage with analytics CSV (same columns as compute_green_analytics.py)
python compute_green_general.py -i /Users/ruian/Downloads/ANALYTICS_MOST_RECENT.csv \
    -r report_text -y generated_vlm_output_text

# With custom output path
python compute_green_general.py -i /Users/ruian/Downloads/ANALYTICS_MOST_RECENT.csv \
    -r report_text -y generated_vlm_output_text -o scores.csv

# With batching for large files
python compute_green_general.py -i /Users/ruian/Downloads/ANALYTICS_MOST_RECENT.csv \
    -r report_text -y generated_vlm_output_text --batch-size 100

# Verbose mode
python compute_green_general.py -i /Users/ruian/Downloads/ANALYTICS_MOST_RECENT.csv \
    -r report_text -y generated_vlm_output_text -v
```

### Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input` | `-i` | (required) | Path to input CSV file |
| `--ref-column` | `-r` | (required) | Column name for reference text |
| `--hyp-column` | `-y` | (required) | Column name for hypothesis text |
| `--output` | `-o` | `{input}_with_green_scores.csv` | Path to output CSV file |
| `--batch-size` | `-b` | None | Process in batches (for large files) |
| `--verbose` | `-v` | | Show verbose output |

### Output Columns Added

| Column | Description |
|--------|-------------|
| `green_score` | GREEN score (0 to 1) |
| `green_analysis` | Full analysis text from GREEN model |
| `green_(a)` to `green_(f)` | Error counts by category |
| `green_Matched Findings` | Number of matched findings |

---

## 3. plot_green.py - Regenerate Plots

Regenerates plots from an existing `green_scores_by_version_full.csv` without recomputing GREEN scores.

### Usage

```bash
# Default: uses green_scores_by_version_full.csv in current directory
python plot_green.py

# With specific input file
python plot_green.py --input /path/to/green_scores_by_version_full.csv

# Generate accepted/rejected split plots
python plot_green.py --split-by-acceptance
```

### Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input` | `-i` | `green_scores_by_version_full.csv` | Path to input CSV file |
| `--output-dir` | `-o` | `pngs/` | Directory for output plots |
| `--split-by-acceptance` | | | Generate accepted/rejected split plots |

---

## 3. plot_green_by_normal.py - Plots by Report Type

Generates plots split by normal/abnormal reports and by body part.

### Usage

```bash
# Default: uses green_scores_by_version_full.csv
python plot_green_by_normal.py

# With acceptance splits
python plot_green_by_normal.py --split-by-acceptance
```

### Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input` | `-i` | `green_scores_by_version_full.csv` | Path to input CSV file |
| `--output-dir` | `-o` | `pngs/` | Directory for output plots |
| `--split-by-acceptance` | | | Generate accepted/rejected split plots |

### Output Structure

```
pngs/
├── green_all_plot.png              # All reports
├── green_all_plot_findings.png     # All reports (is_findings_correct)
├── green_all_plot_impressions.png  # All reports (is_impressions_correct)
├── green_all_plot_both.png         # All reports (both correct)
├── green_normal_plot.png           # Normal reports only
├── green_abnormal_plot.png         # Abnormal reports only
├── Chest/                          # Body part subfolder
│   ├── green_chest_plot.png
│   ├── green_chest_normal_plot.png
│   └── green_chest_abnormal_plot.png
├── Abdomen/
├── Hand/
└── ...                             # Other body parts (>100 samples)
```

---

## Correctness Modes

All plotting scripts support 4 correctness modes for acceptance rate calculation:

| Mode | Column | Description |
|------|--------|-------------|
| `is_correct` | `is_correct` | Overall report correctness |
| `is_findings_correct` | `is_findings_correct` | Findings section correctness |
| `is_impressions_correct` | `is_impressions_correct` | Impressions section correctness |
| `is_both_correct` | Computed | `is_findings_correct AND is_impressions_correct` |

Each plot shows:
- **Blue dots with error bars**: GREEN score mean ± SEM (Standard Error of Mean)
- **Orange line with squares**: Acceptance rate for the selected correctness mode
- **Gray bars**: Sample count per version (secondary y-axis)

---

## Input CSV Requirements

The input CSV must contain these columns:

| Column | Type | Description |
|--------|------|-------------|
| `report_text` | string | Ground truth radiology report |
| `generated_vlm_output_text` | string | VLM-generated report to evaluate |
| `generation_version` | string | Version string (e.g., "0.1.2", "0.2.0") |
| `report_is_addendum` | boolean | Used to filter out addendums |
| `report_radiologist_first_name` | string | Used to filter specific radiologists |
| `is_correct` | string | Values: `'true'`, `'false'`, `'ERROR'` |
| `is_findings_correct` | string | Values: `'true'`, `'false'`, `'ERROR'` |
| `is_impressions_correct` | string | Values: `'true'`, `'false'`, `'ERROR'` |
| `is_rad_report_normal` | string | Used for normal/abnormal splits |
| `parsed_body_part` | JSON string | e.g., `'["Chest"]'` for body part plots |

---

## Dependencies

```
pandas
numpy
matplotlib
packaging
green_score (from parent directory)
```

---

## Workflow Example

```bash
# Step 1: Compute GREEN scores (takes time due to LLM inference)
python compute_green_analytics.py --samples all

# Step 2: Regenerate plots with different settings (fast, no recomputation)
python plot_green.py --split-by-acceptance

# Step 3: Generate body part and normal/abnormal plots
python plot_green_by_normal.py --split-by-acceptance
```
