# GREEN Score Analysis Tool

Analyze GREEN scores from analytics CSV files, computing scores by version and generating visualizations.

## Quick Start

```bash
# From the GREEN project root directory
cd /Users/ruian/projects/epsilonlabs/GREEN

# Run with defaults (10 samples per version, default input file)
python epsilon/analyze_green.py

# Run all samples
python epsilon/analyze_green.py --samples all

# Use a specific input file
python epsilon/analyze_green.py --input /path/to/analytics.csv
```

## Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input` | `-i` | `/Users/ruian/Downloads/ANALYTICS_MOST_RECENT.csv` | Path to input analytics CSV file |
| `--samples` | `-s` | `10` | Number of samples per version, or `all` for all samples |
| `--output` | `-o` | `<input_dir>/green_scores_by_version.csv` | Path to output CSV file |
| `--min-version` | | `0.1.2` | Minimum version to process |
| `--no-plot` | | | Skip generating the plot |

## Examples

```bash
# Sample 50 rows per version
python epsilon/analyze_green.py --samples 50

# Run all samples with custom output
python epsilon/analyze_green.py --samples all --output /tmp/results.csv

# Process only versions >= 0.2.0
python epsilon/analyze_green.py --min-version 0.2.0

# Run without showing plot (headless mode)
python epsilon/analyze_green.py --no-plot
```

## Output

The script produces:

1. **Console output**: Summary statistics (fine-grained by exact version, consolidated by major.minor)
2. **CSV file**: Detailed results with all GREEN scores and error breakdowns
3. **PNG plot**: Visualization showing GREEN scores by version with error bars (SEM) and sample counts

## Input CSV Requirements

The input CSV must contain these columns:
- `report_text` - Ground truth radiology report
- `generated_vlm_output_text` - VLM-generated report to evaluate
- `generation_version` - Version string (e.g., "0.1.2", "0.2.0")
- `report_is_addendum` - Boolean to filter out addendums
- `report_radiologist_first_name` - Used to filter specific radiologists

## Dependencies

- pandas
- numpy
- matplotlib
- packaging
- green_score (from parent directory)
