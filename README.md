# Adaptive Attention-based SResdRVFL for Time-Series Forecasting

This project implements an adaptive stacked residual Deep RVFL model for univariate time-series forecasting with:

- Multi-resolution decomposition (trend/seasonal/residual)
- Adaptive residual scaling (`s`)
- Attention-based block aggregation
- 95% uncertainty interval
- Online update via Recursive Least Squares (RLS)
- Visualization of predictions, attention evolution, and error distribution

## Files

- `adaptive_sresdrvfl.py`: main model + interactive CLI
- `visualize_results.py`: visualization script for `results.csv`
- `aemo_qld1_2022_01.csv`: sample AEMO QLD dataset
- `results.csv`: generated logging output during interactive runs

## Setup

Install dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn statsmodels
```

## Run Forecasting (Interactive)

```bash
python adaptive_sresdrvfl.py --csv-path ".\aemo_qld1_2022_01.csv" --region QLD --window-size 50
```

Interactive flow:

1. Enter 50 comma-separated historical values.
2. Review forecast report (point forecast, CI, attention, learned scaling factor).
3. Enter actual next value to update online (or press Enter to skip).

`results.csv` is saved every 10 successful updates and on exit.

## Visualize Results

```bash
python visualize_results.py --input results.csv --output-dir plots --show
```

Generated plots:

- `plots/forecast_vs_actual_ci.png`
- `plots/attention_evolution.png`
- `plots/error_distribution.png`

## `results.csv` Columns

- `Actual`
- `Predicted`
- `Lower_Bound`
- `Upper_Bound`
- `Block1_Attn`
- `Block2_Attn`
- `Block3_Attn`
- `Block4_Attn`
- `Block5_Attn`

