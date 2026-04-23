# 10-701 Project

## Overview

This project studies whether causal discovery can improve gold forecasting from macro-financial time-series data. The target is the **next-day COMEX gold log return**, predicted from a 20-day window of stationary-transformed financial features such as the U.S. Dollar Index, USD/JPY, equity-market and volatility measures, commodity prices, and Treasury-rate variables.

The main question is whether a causally filtered input set can match or outperform a full-feature LSTM while remaining smaller and easier to interpret. To test that, the repository compares:

- `PCMCI` for causal feature discovery on the training split
- `Granger` as a simpler bivariate comparison baseline
- `VAR` as a linear forecasting baseline
- several `LSTM` variants trained under the same splits and shared hyperparameters

The default chronological splits are:

- train: `2006-01-03` to `2018-12-31`
- validation: `2019-01-01` to `2021-12-31`
- test: `2022-01-01` to `2024-12-31`

## Quickstart

From the project root:

```bash
python -m pip install -r requirements.txt
```

Then run the pipeline in this order:

```bash
cd data
python yfinance_data.py
cd ..

cd PCMCI
python pcmci_discovery.py
cd ..

python granger_feature_select.py

cd lstm
python train.py
```

Notes:

- `data/yfinance_data.py` is the required data pipeline for the current project workflow.
- `lstm/train.py` loads both `PCMCI/results/pcmci_output/ontrain/selected_features.json` and `granger_outputs/granger_selected_features.json` by default, so run both feature-selection steps before training.
- `VAR` is independent of the LSTM pipeline once the data files in `data/results/` exist.

## Installation

Use any Python environment you prefer, then install:

```bash
python -m pip install -r requirements.txt
```

The main dependencies used in this repository include `yfinance`, `pandas`, `statsmodels`, `torch`, `scikit-learn`, `matplotlib`, `optuna`, and `tigramite`.

## Optional FRED Setup

You do **not** need a `.env` file for the default Yahoo Finance workflow described above.

If you later run the FRED pipeline, create `data/.env` with:

- `FRED_API_KEY=...`

You can request a key from the [Federal Reserve Bank of St. Louis](https://fred.stlouisfed.org).

## Data Pipeline

Run the Yahoo Finance data download and preprocessing pipeline from the `data` directory:

```bash
cd data
python yfinance_data.py
```

This script downloads raw daily closes, applies the preprocessing pipeline, and writes:

- intermediate provenance files to `data/pipeline_steps/`
- downstream model inputs to `data/results/`

The most important outputs for later stages are:

- `data/results/gold_base_stationary_dropna.csv`
- `data/results/split_definition.json`
- `data/results/feature_dictionary.csv`
- `data/results/redundancy_check_report.csv`
- `data/results/no_scaling_policy.json`

## PCMCI

`PCMCI/pcmci_discovery.py` runs PCMCI (PC + momentary conditional independence) with linear partial correlation (`ParCorr`) on the **training split only** of `data/results/gold_base_stationary_dropna.csv`. It is used to identify lagged exogenous variables with statistically significant links to the gold log-return target.

Run from `PCMCI/`:

```bash
python pcmci_discovery.py
```

Outputs are written under `PCMCI/results/pcmci_output/ontrain/`:

- `pcmci_results.pkl`
- `causal_features.csv`
- `self_links.csv`
- `pvalue_heatmap.png`
- `causal_graph.png` when available
- `selected_features.json`

For the LSTM pipeline, the key artifact is `selected_features.json`, which supplies the PCMCI-selected feature set.

## Granger

`granger_feature_select.py` runs pairwise Granger causality tests for each candidate feature against the gold log-return target on the **training split only**. Unlike PCMCI, this is a bivariate screen and does not condition on the rest of the feature panel.

Run from the project root:

```bash
python granger_feature_select.py
```

If PCMCI output already exists, the script also reports overlap between the Granger and PCMCI selections and writes a comparison JSON.

Outputs are written under `granger_outputs/`:

- `granger_selected_features.json`
- `granger_full_results.json`
- `granger_vs_pcmci.json` when PCMCI results are present

For the default LSTM workflow, `granger_selected_features.json` is the required downstream artifact.

## VAR Baseline

`VAR/var_model.py` fits a vector autoregression baseline with `statsmodels` on a fixed set of endogenous series (`USE_COLS` in `VAR/var_model.py`). Lag order is chosen by **AIC** on the training period, and forecasts are evaluated by split using MAE, RMSE, MAPE, and directional accuracy.

Run from `VAR/`:

```bash
python var_model.py
```

Outputs are written under `VAR/var_outputs/`:

- `var_predictions_h1.csv`, `var_predictions_h5.csv`, ...
- `var_metrics.csv`
- `var_config.json`
- `var_summary.txt`

## LSTM Pipeline

`lstm/train.py` trains multiple LSTM variants with shared architecture and multi-seed evaluation so that comparisons reflect both average performance and seed variance. The implemented variants include:

- `LSTM-all`
- `LSTM-causal`
- `LSTM-granger`
- `LSTM-pca`
- `LSTM-reg-l1`
- `LSTM-reg-l2`

Run from `lstm/` after the data pipeline, PCMCI, and Granger steps have completed:

```bash
python train.py
```

Main outputs:

- checkpoints under `lstm/checkpoints/`
- per-run artifacts under `lstm/runs/`
- summary CSV / JSON outputs under `lstm/experiment_outputs/`

## Optional LSTM Utilities

Run these commands from `lstm/`.

- Generate test-split predicted vs. actual returns for checkpoints:

```bash
python predict.py
python predict.py --label LSTM-all_seed42
python predict.py --checkpoint checkpoints/LSTM-all_seed42.pt
```

- Recover absolute gold prices from predicted log returns:

```bash
python recover_prices.py
python recover_prices.py --label LSTM-all_seed42
python recover_prices.py --plot-start 2023-01-01 --plot-end 2023-12-31
```

`recover_prices.py` reads each run's `returns.csv` and writes `prices.csv`, price plots, return plots, and aggregate plots. It reports both:

- **one-step** prices: true prior close times `exp(predicted return)`
- **rolling** prices: prior close taken from the model's own predicted path

- Tune the locked LSTM hyperparameter configuration with Optuna:

```bash
python tune_lstm_mse.py
python tune_lstm.py
```

Both tuning scripts write `tune_lstm_best_config.json` for manual incorporation into `train.py`.
