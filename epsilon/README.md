# GREEN Score Analysis Tool

Analyze GREEN scores from analytics CSV files, computing scores by version and generating visualizations.

## GREEN Score Formula

$$\text{GREEN Score} = \frac{\text{Matched Findings}}{\text{Matched Findings} + \sum(\text{Significant Errors})}$$

- **Matched Findings**: Clinically relevant findings in both reference and hypothesis
- **Significant Errors**: Weighted sum of omissions, hallucinations, and other discrepancies
- **Range**: 0 to 1 (1.0 = perfect match)

## Quick Start

```bash
# From the GREEN project root directory
cd /Users/ruian/projects/epsilonlabs/GREEN

# Run with defaults (3 samples per version for testing)
python epsilon/analyze_green.py

# Run all samples (full analysis)
python epsilon/analyze_green.py --samples all

# Use a specific input file
python epsilon/analyze_green.py --input /path/to/analytics.csv
```

## Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input` | `-i` | `/Users/ruian/Downloads/ANALYTICS_MOST_RECENT.csv` | Path to input analytics CSV file |
| `--samples` | `-s` | `3` | Number of samples per version, or `all` for all samples |
| `--output` | `-o` | `green_scores_by_version.csv` | Path to output CSV file |
| `--min-version` | | `0.1.2` | Minimum version to process |
| `--no-plot` | | | Skip generating plots |

## Examples

```bash
# Sample 50 rows per version
python epsilon/analyze_green.py --samples 50

# Run all samples with custom output
python epsilon/analyze_green.py --samples all --output /tmp/results.csv

# Process only versions >= 0.2.0
python epsilon/analyze_green.py --min-version 0.2.0

# Run without plots (headless mode)
python epsilon/analyze_green.py --no-plot

# Run in background with all samples
nohup python epsilon/analyze_green.py --samples all > output.log 2>&1 &
```

## Output

The script produces:

### CSV Files
1. **`green_scores_by_version.csv`** - Processed results only (rows that passed filtering)
2. **`green_scores_by_version_full.csv`** - All original rows with `kept_for_green` column and GREEN scores merged

### Plots (3 PNG files)
1. **`green_scores_by_version_plot.png`** - All data with acceptance rate line
2. **`green_scores_by_version_plot_accepted.png`** - Accepted samples only (`is_correct == 'true'`)
3. **`green_scores_by_version_plot_rejected.png`** - Rejected samples only (`is_correct == 'false'`)

Each plot shows:
- GREEN scores as blue dots with SEM error bars
- Acceptance rate as orange line (on all-data plot only)
- Sample counts as gray bars (secondary y-axis)

### Console Output
- Summary statistics (fine-grained by exact version, consolidated by major.minor)
- Comparison of std vs SEM for each version

## Data Filtering

Rows are excluded from GREEN scoring if:
- `report_is_addendum == True` (addendum reports)
- `report_radiologist_first_name == 'Justin'` (specific radiologist)
- `report_radiologist_first_name` is NaN
- `is_correct` is NaN

The `kept_for_green` column in the full output indicates whether each row was included.

## Input CSV Requirements

The input CSV must contain these columns:
- `report_text` - Ground truth radiology report
- `generated_vlm_output_text` - VLM-generated report to evaluate
- `generation_version` - Version string (e.g., "0.1.2", "0.2.0")
- `report_is_addendum` - Boolean to filter out addendums
- `report_radiologist_first_name` - Used to filter specific radiologists
- `is_correct` - String values: `'true'`, `'false'`, `'ERROR'` (used for acceptance rate)

## Dependencies

- pandas
- numpy
- matplotlib
- packaging
- green_score (from parent directory)
