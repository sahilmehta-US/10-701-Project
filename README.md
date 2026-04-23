# 10-701-Project

## Overview

## Install

To install the required dependencies, first make sure python is installed in your preferred environment. Then run the following command in the root directory:

```
python -m pip install -r requirements.txt
```

Secondly, write a `.env` file in the data folder with the following API keys:

- `FRED_API_KEY`: an API Key to fetch Federal Reserve Economic Data from the [Federal Reserve Bank of St. Louis](fred.stlouisfed.org)

Note that currently we only use the yfinance data for the project. If time permits, we will add the fred data to the project.

## Get the yfinance data

To get the yfinance data, run the following command in the data folder:

```
python yfinance_data.py
```

This will download the yfinance data and run all preprocessing steps. The intermediate files will be saved in the pipeline_steps folder. The files to be used by downstream tasks will be saved in the results folder.

## PCMCI

PCMCI (PC algorithm + momentary conditional independence) with linear partial correlation (`ParCorr`) runs on the **training split** of `data/results/gold_base_stationary_dropna.csv` to discover exogenous variables with a significant causal link to the gold log-return target, subject to the assumptions documented in `PCMCI/pcmci_discovery.py` (linearity, approximate stationarity, no latent confounders, lag truncation, FDR for multiple tests).

**Run** (from the `PCMCI` directory):

```
python pcmci_discovery.py
```

**Dependencies** (in addition to the project’s main stack): `tigramite` (and its usual `numpy` / `matplotlib` requirements).

**Outputs** (under `PCMCI/results/pcmci_output/ontrain/`): raw `pcmci_results.pkl`, `causal_features.csv`, `self_links.csv`, plots (`pvalue_heatmap.png`, `causal_graph.png` when available), and **`selected_features.json`**, which lists `exogenous_features` for the LSTM pipeline (`lstm/train.py` reads this file by default).

## Granger

Pairwise **Granger causality** tests every other column against the gold log-return on the **training split** only. This is a bivariate screen (it does not condition on other features), so it is useful as a comparison baseline for PCMCI’s multivariate, conditioning-based MCI step.

**Run** (from the project root, same level as the `data` folder):

```
python granger_feature_select.py
```

If `PCMCI/results/pcmci_output/ontrain/selected_features.json` exists, the script also prints a Granger–PCMCI overlap analysis and writes `granger_outputs/granger_vs_pcmci.json`.

**Outputs** (under `granger_outputs/`): `granger_selected_features.json` (same *role* as PCMCI’s selected-features JSON for swapping in training), `granger_full_results.json`, and optional `granger_vs_pcmci.json`.

## VAR

A **vector autoregression** baseline (`statsmodels`) is fit on a fixed set of endogenous series (see `USE_COLS` in `VAR/var_model.py`), with lag order chosen by **AIC** on the training period. Expanding-window multi-step forecasts are produced for configured horizons; metrics (MAE, RMSE, MAPE, directional accuracy) are reported by split (train / validation / test on the `target_date` of each forecast).

**Run** (from the `VAR` directory):

```
python var_model.py
```

**Outputs** (under `VAR/var_outputs/`): per-horizon `var_predictions_h{1,5,...}.csv`, `var_metrics.csv`, `var_config.json`, and `var_summary.txt`.

## LSTM

`lstm/train.py` trains and evaluates several LSTM variants (e.g. all features, **PCMCI-selected** features, **Granger-selected** features, PCA, and regularized “all features” with emphasis on non-causal inputs) with shared architecture and **multi-seed** runs so comparisons reflect both average performance and seed variance. Checkpoints, metrics, and optional summaries (pairwise and/or tabular) are written under the paths defined in that script (`CHECKPOINT_DIR`, `OUTPUT_DIR` / `paths.py`).

**Run** (from the `lstm` directory, after `gold_base_stationary_dropna.csv`, `split_definition.json`, and the feature JSONs you intend to use exist):

```
python train.py
```

**Optional** (same `lstm` directory):

- `python predict.py` — load checkpoints and write test-split predicted vs. actual returns (`lstm/runs/.../returns.csv` by default; supports `--label` or `--checkpoint`).
- `python recover_prices.py` — after `predict.py`, rebuild **absolute gold prices** from predicted log-returns using realized closes from `data/pipeline_steps/step_4_cleaned_close_prices.csv`. Reads each run’s `returns.csv` and writes `prices.csv`, price/return plots, and cross-seed aggregates. Computes **one-step** prices (true prior close × exp(predicted return), aligned with return evaluation) and **rolling** prices (prior close from the model’s own price path, so errors compound). Optional `--label`, `--plot-start` / `--plot-end` to restrict plot windows; see the script docstring for full output layout under `lstm/runs/`.
- `python tune_lstm_mse.py` — Optuna search on **validation MSE** for the LSTM-all variant; writes `tune_lstm_best_config.json` to copy into `train.py` (all variants use the same locked config).
- `python tune_lstm.py` — Optuna search on **validation directional accuracy** (configurable; see that file); also writes a best config JSON to merge into `train.py` by hand.