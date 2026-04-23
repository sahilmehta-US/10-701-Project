"""
recover_prices.py
-----------------
Reconstruct predicted absolute gold prices from predicted log-returns.

Inverse of:   r_t = log(P_t) - log(P_{t-1})
So:           P_hat_t = P_{t-1} * exp(r_hat_t)

Two modes are computed for each test date:
  * one-step-ahead:  uses the TRUE previous close P_{t-1}^true
                     (matches how the LSTM is trained/evaluated)
  * rolling:         uses the PREDICTED previous close P_hat_{t-1}
                     (errors compound multiplicatively)

Input:  lstm/runs/<arch_label>/seed<N>/returns.csv   (from predict.py)

Output (always):
    lstm/runs/<arch_label>/seed<N>/prices.csv
    lstm/runs/<arch_label>/seed<N>/prices.png
    lstm/runs/<arch_label>/seed<N>/returns_plot.png
    lstm/runs/<arch_label>/_aggregate/prices.png        (mean ± std across seeds)
    lstm/runs/<arch_label>/_aggregate/returns_plot.png  (mean ± std across seeds)
    lstm/runs/_all_models/one_step.png                  (when multiple models)
    lstm/runs/_all_models/returns.png                   (when multiple models)

Output (when --plot-start / --plot-end given):
    lstm/runs/<arch_label>/seed<N>/<start>_<end>/prices.png
    lstm/runs/<arch_label>/seed<N>/<start>_<end>/returns_plot.png
    lstm/runs/_all_models/<start>_<end>/one_step.png
    lstm/runs/_all_models/<start>_<end>/returns.png

Metrics in the range plots are recomputed over the visible window.

Usage:
    python recover_prices.py
    python recover_prices.py --label LSTM-all
    python recover_prices.py --plot-start 2023-01-01 --plot-end 2023-12-31
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paths

RUNS_DIR = paths.RUNS_DIR
RAW_PRICES_CSV = "../data/pipeline_steps/step_4_cleaned_close_prices.csv"
GOLD_PRICE_COL = "Gold Futures (COMEX)"


def _load_raw_prices(csv_path=RAW_PRICES_CSV, price_col=GOLD_PRICE_COL):
    """Return (Date, price) dataframe indexed chronologically on trading days."""
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df[["Date", price_col]].dropna().sort_values("Date").reset_index(drop=True)
    df = df.rename(columns={price_col: "price_true"})
    return df


def _compute_metrics(merged):
    """Return dict of price-space metrics for a merged predictions dataframe."""
    if len(merged) == 0:
        return {
            "rmse_one_step": float("nan"),
            "rmse_rolling": float("nan"),
            "mape_one_step": float("nan"),
            "mape_rolling": float("nan"),
            "direction_accuracy": float("nan"),
            "n_rows": 0,
        }
    err_one = merged["price_true"] - merged["price_pred_one_step"]
    err_roll = merged["price_true"] - merged["price_pred_rolling"]
    return {
        "rmse_one_step": float(np.sqrt((err_one ** 2).mean())),
        "rmse_rolling": float(np.sqrt((err_roll ** 2).mean())),
        "mape_one_step": float((err_one.abs() / merged["price_true"]).mean() * 100),
        "mape_rolling": float((err_roll.abs() / merged["price_true"]).mean() * 100),
        "direction_accuracy": float(
            (np.sign(merged["y_pred"]) == np.sign(merged["y_true"])).mean() * 100
        ),
        "n_rows": len(merged),
    }


def _slice_range(merged, start, end):
    """Return the merged rows within [start, end] inclusive."""
    if start is None and end is None:
        return merged
    mask = pd.Series(True, index=merged.index)
    if start is not None:
        mask &= merged["Date"] >= pd.Timestamp(start)
    if end is not None:
        mask &= merged["Date"] <= pd.Timestamp(end)
    return merged.loc[mask].reset_index(drop=True)


def _plot_model(merged, label, out_path, metrics, range_suffix=""):
    """Plot actual + one-step + rolling for a single model."""
    if len(merged) == 0:
        print(f"[{label}] no rows in range — skipping plot {out_path}")
        return None

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(merged["Date"], merged["price_true"], color="black",
            linewidth=1.2, label="Actual")
    ax.plot(merged["Date"], merged["price_pred_one_step"], color="tomato",
            linewidth=1.0, alpha=0.9, label="Predicted (one-step)")
    ax.plot(merged["Date"], merged["price_pred_rolling"], color="steelblue",
            linewidth=1.0, alpha=0.7, label="Predicted (rolling)")
    title = f"Gold Price Recovery — {label}{range_suffix}\n" \
            f"one-step RMSE={metrics['rmse_one_step']:.2f} " \
            f"MAPE={metrics['mape_one_step']:.2f}%  |  " \
            f"rolling RMSE={metrics['rmse_rolling']:.2f} " \
            f"MAPE={metrics['mape_rolling']:.2f}%  |  " \
            f"dir-acc={metrics['direction_accuracy']:.1f}%"
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Gold Futures (COMEX) close")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[{label}] Wrote {out_path}")
    return out_path


def _return_diagnostics(merged):
    """Correlation and shrinkage stats for predicted vs true log-returns."""
    if len(merged) < 2:
        return {"corr": float("nan"), "shrinkage": float("nan"),
                "std_true": float("nan"), "std_pred": float("nan"),
                "mean_true": float("nan"), "mean_pred": float("nan")}
    yt = merged["y_true"].to_numpy()
    yp = merged["y_pred"].to_numpy()
    std_t, std_p = float(yt.std(ddof=0)), float(yp.std(ddof=0))
    corr = float(np.corrcoef(yt, yp)[0, 1]) if std_t > 0 and std_p > 0 else float("nan")
    return {
        "corr": corr,
        "shrinkage": (std_p / std_t) if std_t > 0 else float("nan"),
        "std_true": std_t, "std_pred": std_p,
        "mean_true": float(yt.mean()), "mean_pred": float(yp.mean()),
    }


def _plot_model_returns(merged, label, out_path, metrics, range_suffix=""):
    """Plot predicted vs true log-returns: time series + scatter.

    Makes the shrinkage-toward-zero behavior of MSE-optimal forecasts
    on low-SNR targets visually obvious.
    """
    if len(merged) == 0:
        print(f"[{label}] no rows in range — skipping {out_path}")
        return None

    diag = _return_diagnostics(merged)

    fig, (ax_ts, ax_sc) = plt.subplots(
        1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2.2, 1]}
    )

    # Time series
    ax_ts.axhline(0.0, color="gray", linewidth=0.7, alpha=0.6)
    ax_ts.plot(merged["Date"], merged["y_true"], color="black",
               linewidth=0.9, alpha=0.85, label="Actual return")
    ax_ts.plot(merged["Date"], merged["y_pred"], color="tomato",
               linewidth=0.9, alpha=0.85, label="Predicted return")
    ax_ts.set_xlabel("Date")
    ax_ts.set_ylabel("log-return")
    ax_ts.grid(alpha=0.3)
    ax_ts.legend(loc="best", fontsize=9)
    ax_ts.set_title("Return time series")

    # Scatter (y_pred vs y_true) with y=x reference and OLS fit line
    ax_sc.axhline(0.0, color="gray", linewidth=0.7, alpha=0.6)
    ax_sc.axvline(0.0, color="gray", linewidth=0.7, alpha=0.6)
    ax_sc.scatter(merged["y_true"], merged["y_pred"],
                  s=8, alpha=0.45, color="tomato", edgecolor="none")

    lo = float(min(merged["y_true"].min(), merged["y_pred"].min()))
    hi = float(max(merged["y_true"].max(), merged["y_pred"].max()))
    pad = 0.05 * (hi - lo) if hi > lo else 0.01
    lims = (lo - pad, hi + pad)
    ax_sc.plot(lims, lims, color="black", linewidth=1.0,
               linestyle="--", label="y = x (perfect)")

    if np.isfinite(diag["corr"]) and merged["y_true"].std() > 0:
        # OLS slope y_pred = a + b * y_true; fitted line illustrates shrinkage.
        b, a = np.polyfit(merged["y_true"], merged["y_pred"], 1)
        xs = np.array(lims)
        ax_sc.plot(xs, a + b * xs, color="steelblue", linewidth=1.2,
                   label=f"OLS fit (slope={b:.2f})")

    ax_sc.set_xlim(lims)
    ax_sc.set_ylim(lims)
    ax_sc.set_xlabel("Actual log-return  y_true")
    ax_sc.set_ylabel("Predicted log-return  y_pred")
    ax_sc.grid(alpha=0.3)
    ax_sc.legend(loc="best", fontsize=8)
    ax_sc.set_title("Predicted vs actual")

    fig.suptitle(
        f"Return Diagnostics — {label}{range_suffix}    "
        f"corr={diag['corr']:.3f}  "
        f"shrinkage σ(pred)/σ(true)={diag['shrinkage']:.2f}  "
        f"dir-acc={metrics['direction_accuracy']:.1f}%",
        y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{label}] Wrote {out_path}")
    return out_path


def _label_from_returns_path(predictions_csv, runs_dir=RUNS_DIR):
    """Recover (label, arch_label, seed) from a returns.csv path.

    Expected path shape: ``<runs_dir>/<arch_label>/seed<N>/returns.csv``.
    Falls back to parsing the filename for older/non-standard inputs.
    """
    norm_runs = os.path.normpath(runs_dir)
    rel = os.path.relpath(predictions_csv, norm_runs)
    parts = rel.split(os.sep)
    if (len(parts) == 3 and parts[-1] == paths.RETURNS_CSV_FILENAME
            and parts[1].startswith("seed")):
        arch_label = parts[0]
        try:
            seed = int(parts[1][len("seed"):])
        except ValueError:
            seed = None
        return paths.make_label(arch_label, seed), arch_label, seed

    # Fallback: predictions_csv was passed explicitly and does not live in
    # the standard tree. Parse the old-style "<label>_returns.csv" name.
    base = os.path.basename(predictions_csv)
    label = base.replace("_returns.csv", "").replace(".csv", "")
    arch_label, seed = paths.parse_label(label)
    return label, arch_label, seed


def recover_from_predictions(predictions_csv, raw_prices_csv=RAW_PRICES_CSV,
                             price_col=GOLD_PRICE_COL, runs_dir=RUNS_DIR,
                             label=None, plot_start=None, plot_end=None):
    """Recover absolute prices from a predictions CSV.

    Always writes the full-range CSV and PNGs into the run folder
    (``<runs_dir>/<arch>/seed<N>/``).  If plot_start/plot_end are given,
    also writes a range-limited copy into a nested subfolder named for
    the range, with metrics recomputed over the visible window.
    """
    parsed_label, arch_label, seed = _label_from_returns_path(
        predictions_csv, runs_dir=runs_dir
    )
    if label is None:
        label = parsed_label
    out_dir = paths.run_dir(arch_label, seed, root=runs_dir)

    preds = pd.read_csv(predictions_csv, parse_dates=["Date"])
    preds = preds.sort_values("Date").reset_index(drop=True)

    raw = _load_raw_prices(raw_prices_csv, price_col)

    # Merge predicted returns with the true price series on Date.
    merged = preds.merge(raw, on="Date", how="inner")
    if len(merged) != len(preds):
        missing = set(preds["Date"]) - set(merged["Date"])
        print(f"[WARN] {len(missing)} prediction dates had no matching raw "
              f"price and were dropped: e.g. {sorted(missing)[:3]}")

    # Attach the previous *trading-day* close. Using the raw price series
    # (rather than merged.shift) keeps the "t-1" anchor correct even when
    # a test date lands adjacent to a gap in the prediction set.
    raw["price_prev_true"] = raw["price_true"].shift(1)
    merged = merged.drop(columns=["price_true"]).merge(
        raw[["Date", "price_true", "price_prev_true"]], on="Date", how="inner"
    )
    merged = merged.dropna(subset=["price_prev_true"]).reset_index(drop=True)

    # One-step-ahead: anchor to true previous close every step.
    merged["price_pred_one_step"] = (
        merged["price_prev_true"] * np.exp(merged["y_pred"])
    )

    # Rolling: feed predicted price back in. Anchor on the first true prev
    # close, then compound with predicted returns thereafter.
    rolling = np.empty(len(merged), dtype=np.float64)
    anchor = merged["price_prev_true"].iloc[0]
    for i, r in enumerate(merged["y_pred"].to_numpy()):
        anchor = anchor * np.exp(r)
        rolling[i] = anchor
    merged["price_pred_rolling"] = rolling

    # Sanity sign-check: the "true" recovered price using y_true should
    # reproduce the raw price series exactly (up to float error).
    merged["price_true_from_return"] = (
        merged["price_prev_true"] * np.exp(merged["y_true"])
    )
    max_recon_err = (merged["price_true"] - merged["price_true_from_return"]).abs().max()

    # Full-range metrics + CSV (always saved).
    metrics_full = _compute_metrics(merged)
    print(f"[{label}] rows={len(merged)}  "
          f"true-recon-max-err={max_recon_err:.2e}")
    print(f"[{label}] one-step : RMSE={metrics_full['rmse_one_step']:8.4f}  "
          f"MAPE={metrics_full['mape_one_step']:5.3f}%")
    print(f"[{label}] rolling  : RMSE={metrics_full['rmse_rolling']:8.4f}  "
          f"MAPE={metrics_full['mape_rolling']:5.3f}%")
    print(f"[{label}] direction accuracy = "
          f"{metrics_full['direction_accuracy']:5.2f}%")

    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, paths.PRICES_CSV_FILENAME)
    merged.to_csv(out_csv, index=False)
    print(f"[{label}] Wrote {out_csv}")

    out_png = os.path.join(out_dir, paths.PRICES_PLOT_FILENAME)
    _plot_model(merged, label, out_png, metrics_full)
    out_returns_png = os.path.join(out_dir, paths.RETURNS_PLOT_FILENAME)
    _plot_model_returns(merged, label, out_returns_png, metrics_full)

    # Optional range-limited plots — nested one level under the run dir,
    # so a single arch/seed keeps all its views co-located.
    range_png = None
    range_returns_png = None
    metrics_range = None
    rname = paths.range_name(plot_start, plot_end)
    if rname is not None:
        sub = _slice_range(merged, plot_start, plot_end)
        metrics_range = _compute_metrics(sub)
        suffix = f" [{plot_start or '…'} → {plot_end or '…'}]"
        range_dir = paths.range_subdir(out_dir, rname)
        range_png = _plot_model(
            sub, label,
            os.path.join(range_dir, paths.PRICES_PLOT_FILENAME),
            metrics_range,
            range_suffix=suffix,
        )
        range_returns_png = _plot_model_returns(
            sub, label,
            os.path.join(range_dir, paths.RETURNS_PLOT_FILENAME),
            metrics_range,
            range_suffix=suffix,
        )
        if sub.empty:
            print(f"[{label}] range produced 0 rows — plot skipped")

    return {
        "label": label,
        "arch_label": arch_label,
        "seed": seed,
        "out_csv": out_csv,
        "out_png": out_png,
        "out_returns_png": out_returns_png,
        "range_png": range_png,
        "range_returns_png": range_returns_png,
        **{k: metrics_full[k] for k in (
            "rmse_one_step", "rmse_rolling",
            "mape_one_step", "mape_rolling",
            "direction_accuracy", "n_rows",
        )},
        "metrics_range": metrics_range,
        "merged": merged,
    }


def _stack_across_seeds(results, col):
    """Inner-join per-seed frames on Date and stack `col` into [n_seeds, T].

    Returns (dates, stacked).  Only dates present in every seed's test
    slice are kept; in practice they're identical because every run sees
    the same test split, but the merge keeps us honest if a seed ever
    drops rows upstream.
    """
    if not results:
        return None, None
    combined = results[0]["merged"][["Date", col]].rename(columns={col: "s0"})
    for i, r in enumerate(results[1:], start=1):
        df = r["merged"][["Date", col]].rename(columns={col: f"s{i}"})
        combined = combined.merge(df, on="Date", how="inner")
    combined = combined.sort_values("Date").reset_index(drop=True)
    dates = combined["Date"]
    stacked = combined.drop(columns="Date").to_numpy().T
    return dates, stacked


def _mean_std(stacked):
    """Return (mean, std) along axis 0. std=0 if only one row."""
    if stacked.shape[0] > 1:
        return stacked.mean(0), stacked.std(0, ddof=1)
    return stacked[0], np.zeros_like(stacked[0])


def _slice_results(results, plot_start, plot_end):
    """Return a shallow copy of each result dict with `merged` clipped to
    [plot_start, plot_end]. Runs that become empty are dropped."""
    sliced = []
    for r in results:
        sub = _slice_range(r["merged"], plot_start, plot_end)
        if sub.empty:
            continue
        sliced.append({**r, "merged": sub})
    return sliced


def _aggregate_out_path(arch_label, filename, runs_dir, range_name):
    """Aggregate output path, nested under `<range_name>/` when given."""
    base = paths.aggregate_dir(arch_label, root=runs_dir)
    if range_name is not None:
        base = os.path.join(base, range_name)
    return os.path.join(base, filename)


def plot_aggregate_prices(results, arch_label, runs_dir=RUNS_DIR,
                          plot_start=None, plot_end=None):
    """Mean ± std of price predictions across seeds, for one architecture.

    Per-seed prices.png files in each run folder stay untouched; this
    plot is the one that survives into the report, where seed noise
    should be a band rather than N overlapping lines.

    If plot_start / plot_end are given, each seed's frame is clipped to
    the window before stacking and the output is nested under
    ``_aggregate/<start>_<end>/`` so full-range and range-clipped views
    can coexist without overwriting each other.
    """
    if not results:
        return None
    working = _slice_results(results, plot_start, plot_end)
    if not working:
        print(f"[{arch_label}] aggregate prices: no rows in range, skipped")
        return None

    dates, one_step = _stack_across_seeds(working, "price_pred_one_step")
    _, rolling = _stack_across_seeds(working, "price_pred_rolling")
    if dates is None or one_step is None or len(dates) == 0:
        print(f"[{arch_label}] aggregate prices: no overlapping dates, skipped")
        return None

    # `price_true` is seed-independent — take from first run, aligned to
    # the shared dates from the stack.
    truth = (working[0]["merged"][["Date", "price_true"]]
             .merge(pd.DataFrame({"Date": dates}), on="Date", how="inner")
             .sort_values("Date").reset_index(drop=True))

    one_mean, one_std = _mean_std(one_step)
    roll_mean, roll_std = _mean_std(rolling)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(truth["Date"], truth["price_true"], color="black",
            linewidth=1.2, label="Actual", zorder=5)
    ax.plot(dates, one_mean, color="tomato", linewidth=1.0,
            label=f"One-step (mean, N={len(working)})")
    ax.fill_between(dates, one_mean - one_std, one_mean + one_std,
                    color="tomato", alpha=0.25, label="One-step ±1σ")
    ax.plot(dates, roll_mean, color="steelblue", linewidth=1.0, alpha=0.8,
            label=f"Rolling (mean, N={len(working)})")
    ax.fill_between(dates, roll_mean - roll_std, roll_mean + roll_std,
                    color="steelblue", alpha=0.20, label="Rolling ±1σ")
    title = (f"Gold Price Recovery — {arch_label} "
             f"(aggregated over {len(working)} seeds)")
    if plot_start is not None or plot_end is not None:
        title += f"\n[{plot_start or '…'} → {plot_end or '…'}]"
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Gold Futures (COMEX) close")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    rname = paths.range_name(plot_start, plot_end)
    out_path = _aggregate_out_path(arch_label, paths.PRICES_PLOT_FILENAME,
                                   runs_dir, rname)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[{arch_label}] Aggregate price plot -> {out_path}")
    return out_path


def plot_aggregate_returns(results, arch_label, runs_dir=RUNS_DIR,
                           plot_start=None, plot_end=None):
    """Mean ± std of predicted log-returns across seeds, for one architecture.

    Range clipping mirrors `plot_aggregate_prices`: when a window is
    given, each seed's frame is sliced before stacking and the output
    lands in ``_aggregate/<start>_<end>/``.
    """
    if not results:
        return None
    working = _slice_results(results, plot_start, plot_end)
    if not working:
        print(f"[{arch_label}] aggregate returns: no rows in range, skipped")
        return None

    dates, y_pred = _stack_across_seeds(working, "y_pred")
    if dates is None or y_pred is None or len(dates) == 0:
        print(f"[{arch_label}] aggregate returns: no overlapping dates, skipped")
        return None

    truth = (working[0]["merged"][["Date", "y_true"]]
             .merge(pd.DataFrame({"Date": dates}), on="Date", how="inner")
             .sort_values("Date").reset_index(drop=True))
    mean_p, std_p = _mean_std(y_pred)

    # Correlation of the *mean* prediction with truth is a cleaner summary
    # than averaging per-seed correlations (which would understate it).
    yt = truth["y_true"].to_numpy()
    corr = (float(np.corrcoef(yt, mean_p)[0, 1])
            if yt.std() > 0 and mean_p.std() > 0 else float("nan"))
    shrink = (float(mean_p.std() / yt.std()) if yt.std() > 0 else float("nan"))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhline(0.0, color="gray", linewidth=0.7, alpha=0.6)
    ax.plot(truth["Date"], truth["y_true"], color="black",
            linewidth=0.9, alpha=0.85, label="Actual return", zorder=5)
    ax.plot(dates, mean_p, color="tomato", linewidth=0.9,
            label=f"Predicted (mean, N={len(working)})")
    ax.fill_between(dates, mean_p - std_p, mean_p + std_p,
                    color="tomato", alpha=0.25, label="Predicted ±1σ")
    title = (f"Predicted vs Actual Log-Returns — {arch_label} "
             f"(aggregated over {len(working)} seeds)    "
             f"corr(mean, truth)={corr:.3f}  "
             f"shrinkage σ(mean)/σ(true)={shrink:.2f}")
    if plot_start is not None or plot_end is not None:
        title += f"\n[{plot_start or '…'} → {plot_end or '…'}]"
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("log-return")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    rname = paths.range_name(plot_start, plot_end)
    out_path = _aggregate_out_path(arch_label, paths.RETURNS_PLOT_FILENAME,
                                   runs_dir, rname)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[{arch_label}] Aggregate returns plot -> {out_path}")
    return out_path


def plot_combined_one_step(results, out_path, plot_start=None, plot_end=None):
    """Overlay all models' one-step predictions on a single chart.

    If plot_start / plot_end are set, the view is clipped and legend
    metrics are recomputed over the visible window.
    """
    if not results:
        return None

    # Slice each model's merged frame to the requested window (if any)
    # and recompute metrics for the legend.
    sliced = []
    for r in results:
        sub = _slice_range(r["merged"], plot_start, plot_end)
        if sub.empty:
            continue
        m = _compute_metrics(sub)
        sliced.append({"label": r["label"], "merged": sub, "metrics": m})

    if not sliced:
        print(f"[combined] no rows in range — skipping {out_path}")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    # Ground truth — use the longest sliced series as the reference line.
    truth_ref = max(sliced, key=lambda s: len(s["merged"]))["merged"]
    ax.plot(truth_ref["Date"], truth_ref["price_true"],
            color="black", linewidth=1.6, label="Actual", zorder=5)

    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(sliced):
        m = s["merged"]
        ax.plot(
            m["Date"], m["price_pred_one_step"],
            color=cmap(i % 10), linewidth=1.0, alpha=0.85,
            label=f"{s['label']}  "
                  f"(RMSE={s['metrics']['rmse_one_step']:.2f}, "
                  f"MAPE={s['metrics']['mape_one_step']:.2f}%)",
        )

    title = "Gold Price Recovery — One-Step Predictions (all models)"
    if plot_start is not None or plot_end is not None:
        title += f"\n[{plot_start or '…'} → {plot_end or '…'}]"
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Gold Futures (COMEX) close")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[combined] Wrote {out_path}")
    return out_path


def plot_combined_returns(results, out_path, plot_start=None, plot_end=None):
    """Overlay all models' predicted log-returns on one time-series chart.

    Exposes shrinkage toward zero side-by-side across models: the tighter
    a model's predicted-return line hugs zero relative to the actual line,
    the closer its implied price forecast is to a naive random walk.
    """
    if not results:
        return None

    sliced = []
    for r in results:
        sub = _slice_range(r["merged"], plot_start, plot_end)
        if sub.empty:
            continue
        diag = _return_diagnostics(sub)
        metrics = _compute_metrics(sub)
        sliced.append({"label": r["label"], "merged": sub,
                       "diag": diag, "metrics": metrics})

    if not sliced:
        print(f"[combined] no rows in range — skipping {out_path}")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(0.0, color="gray", linewidth=0.7, alpha=0.6)

    truth_ref = max(sliced, key=lambda s: len(s["merged"]))["merged"]
    ax.plot(truth_ref["Date"], truth_ref["y_true"], color="black",
            linewidth=1.0, alpha=0.75, label="Actual return", zorder=5)

    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(sliced):
        m = s["merged"]
        ax.plot(
            m["Date"], m["y_pred"],
            color=cmap(i % 10), linewidth=0.9, alpha=0.8,
            label=f"{s['label']}  "
                  f"(corr={s['diag']['corr']:.2f}, "
                  f"shrink={s['diag']['shrinkage']:.2f}, "
                  f"dir={s['metrics']['direction_accuracy']:.1f}%)",
        )

    title = "Predicted vs Actual Log-Returns (all models)"
    if plot_start is not None or plot_end is not None:
        title += f"\n[{plot_start or '…'} → {plot_end or '…'}]"
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("log-return")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[combined] Wrote {out_path}")
    return out_path


def compute_persistence_baseline(reference_merged, plot_start=None, plot_end=None):
    """Synthetic 'predict y_pred = 0' (random-walk) baseline.

    Uses the same dates and true prices as `reference_merged`, so the
    baseline metrics are directly comparable with any model row in the
    summary table.  A persistence forecaster says "tomorrow = today",
    i.e. `P_hat_t = P_{t-1}^true`, which is exactly what `exp(0)` gives.

    Note: rolling under persistence collapses to a constant (P_0 forever),
    so rolling metrics for this baseline are not informative.
    """
    fake = reference_merged.copy()
    fake["y_pred"] = 0.0
    fake["price_pred_one_step"] = fake["price_prev_true"]
    fake["price_pred_rolling"] = fake["price_prev_true"].iloc[0]

    metrics_full = _compute_metrics(fake)
    metrics_range = None
    if plot_start is not None or plot_end is not None:
        metrics_range = _compute_metrics(_slice_range(fake, plot_start, plot_end))

    return {
        "label": "persistence (y_pred=0)",
        "out_csv": None,
        "out_png": None,
        "out_returns_png": None,
        "range_png": None,
        "range_returns_png": None,
        **{k: metrics_full[k] for k in (
            "rmse_one_step", "rmse_rolling",
            "mape_one_step", "mape_rolling",
            "direction_accuracy", "n_rows",
        )},
        "metrics_range": metrics_range,
        "merged": fake,
        "is_baseline": True,
    }


def _format_rel(model_val, baseline_val):
    """Format a signed %-improvement of model vs baseline on the same metric.
    Negative = model worse than baseline."""
    if baseline_val == 0 or not np.isfinite(baseline_val):
        return "    n/a"
    pct = (1.0 - model_val / baseline_val) * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:5.2f}%"


def _validate_date(s, name):
    if s is None:
        return None
    try:
        return pd.Timestamp(s).strftime("%Y-%m-%d")
    except Exception as exc:
        raise SystemExit(f"Invalid --{name} value: {s!r} ({exc})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=str, default=None,
                        help="Path to a returns.csv. Overrides --label.")
    parser.add_argument("--label", type=str, default=None,
                        help="Full label (e.g. LSTM-all_seed42). Loads "
                             "runs/<arch>/seed<N>/returns.csv.")
    parser.add_argument("--runs-dir", type=str, default=RUNS_DIR,
                        help="Root folder of per-run output "
                             "(<runs-dir>/<arch>/seed<N>/returns.csv).")
    parser.add_argument("--raw-prices", type=str, default=RAW_PRICES_CSV)
    parser.add_argument("--price-col", type=str, default=GOLD_PRICE_COL)
    parser.add_argument("--plot-start", type=str, default=None,
                        help="Inclusive start date for range plots "
                             "(YYYY-MM-DD). Full-range plots are always saved.")
    parser.add_argument("--plot-end", type=str, default=None,
                        help="Inclusive end date for range plots "
                             "(YYYY-MM-DD). Full-range plots are always saved.")
    args = parser.parse_args()

    plot_start = _validate_date(args.plot_start, "plot-start")
    plot_end = _validate_date(args.plot_end, "plot-end")
    if plot_start is not None and plot_end is not None and plot_start > plot_end:
        raise SystemExit(
            f"--plot-start ({plot_start}) must not be after --plot-end "
            f"({plot_end})."
        )

    rname = paths.range_name(plot_start, plot_end)

    if args.predictions is not None:
        files = [args.predictions]
    elif args.label is not None:
        arch_label, seed = paths.parse_label(args.label)
        files = [paths.returns_csv(arch_label, seed, root=args.runs_dir)]
    else:
        # New layout: <runs>/<arch>/seed<N>/returns.csv
        files = sorted(glob.glob(os.path.join(
            args.runs_dir, "*", "seed*", paths.RETURNS_CSV_FILENAME
        )))
        if not files:
            raise SystemExit(
                f"No predictions found under {args.runs_dir}/*/seed*/. "
                f"Run predict.py first."
            )

    summary = []
    for f in files:
        summary.append(
            recover_from_predictions(
                f,
                raw_prices_csv=args.raw_prices,
                price_col=args.price_col,
                runs_dir=args.runs_dir,
                plot_start=plot_start,
                plot_end=plot_end,
            )
        )

    # Per-arch aggregate plots (mean ± std across seeds). Per-seed plots
    # stay untouched in each run folder for reference. Full-range views
    # are always emitted; the range-clipped views are nested under
    # `_aggregate/<start>_<end>/` so the two coexist.
    by_arch = {}
    for r in summary:
        by_arch.setdefault(r["arch_label"], []).append(r)
    for arch_label, runs in by_arch.items():
        plot_aggregate_prices(runs, arch_label, runs_dir=args.runs_dir)
        plot_aggregate_returns(runs, arch_label, runs_dir=args.runs_dir)
        if rname is not None:
            plot_aggregate_prices(
                runs, arch_label, runs_dir=args.runs_dir,
                plot_start=plot_start, plot_end=plot_end,
            )
            plot_aggregate_returns(
                runs, arch_label, runs_dir=args.runs_dir,
                plot_start=plot_start, plot_end=plot_end,
            )

    if len(summary) > 1:
        # Always: full-range combined plots in runs/_all_models/
        combined_dir = paths.all_models_dir(root=args.runs_dir)
        plot_combined_one_step(
            summary,
            os.path.join(combined_dir, paths.COMBINED_ONE_STEP_FILENAME),
        )
        plot_combined_returns(
            summary,
            os.path.join(combined_dir, paths.COMBINED_RETURNS_FILENAME),
        )
        # Optional: range-limited combined plots in runs/_all_models/<range>/
        if rname is not None:
            combined_range_dir = paths.all_models_dir(
                range_name=rname, root=args.runs_dir
            )
            plot_combined_one_step(
                summary,
                os.path.join(combined_range_dir, paths.COMBINED_ONE_STEP_FILENAME),
                plot_start=plot_start, plot_end=plot_end,
            )
            plot_combined_returns(
                summary,
                os.path.join(combined_range_dir, paths.COMBINED_RETURNS_FILENAME),
                plot_start=plot_start, plot_end=plot_end,
            )

    # Persistence baseline ("y_pred = 0"): predict tomorrow's price equals
    # today's price, evaluated on the same dates as the models.  Added to
    # the tables as a reference row but intentionally excluded from the
    # combined plots (where it would collapse to the y=0 axis already drawn).
    baseline = None
    if summary:
        baseline = compute_persistence_baseline(
            summary[0]["merged"],
            plot_start=plot_start, plot_end=plot_end,
        )

    print("\n" + "=" * 95)
    print("SUMMARY (full range)")
    print("=" * 95)
    header = (f"{'Model':<24} {'RMSE(1-step)':>13} {'MAPE(1-step)':>13} "
              f"{'RMSE(roll)':>12} {'MAPE(roll)':>12} {'DirAcc%':>8} "
              f"{'vs base':>9}")
    print(header)
    print("-" * 95)

    def _print_row(r, base_rmse):
        rel = _format_rel(r["rmse_one_step"], base_rmse) if base_rmse else "    —"
        print(f"{r['label']:<24} {r['rmse_one_step']:>13.4f} "
              f"{r['mape_one_step']:>12.3f}% {r['rmse_rolling']:>12.4f} "
              f"{r['mape_rolling']:>11.3f}% "
              f"{r['direction_accuracy']:>7.2f}% {rel:>9}")

    base_rmse_full = baseline["rmse_one_step"] if baseline else None
    if baseline:
        _print_row(baseline, base_rmse=None)
        print("-" * 95)
    for r in summary:
        _print_row(r, base_rmse=base_rmse_full)

    if rname is not None:
        print("\n" + "=" * 103)
        print(f"SUMMARY (range {plot_start or 'min'} → {plot_end or 'max'})")
        print("=" * 103)
        print(f"{'Model':<24} {'RMSE(1-step)':>13} {'MAPE(1-step)':>13} "
              f"{'RMSE(roll)':>12} {'MAPE(roll)':>12} {'DirAcc%':>8} "
              f"{'vs base':>9} {'n':>5}")
        print("-" * 103)

        def _print_row_range(r, base_rmse):
            m = r["metrics_range"] or {}
            if not m or m.get("n_rows", 0) == 0:
                print(f"{r['label']:<24}  (no rows in range)")
                return
            rel = _format_rel(m["rmse_one_step"], base_rmse) if base_rmse else "    —"
            print(f"{r['label']:<24} {m['rmse_one_step']:>13.4f} "
                  f"{m['mape_one_step']:>12.3f}% {m['rmse_rolling']:>12.4f} "
                  f"{m['mape_rolling']:>11.3f}% "
                  f"{m['direction_accuracy']:>7.2f}% {rel:>9} {m['n_rows']:>5d}")

        base_rmse_range = (
            baseline["metrics_range"]["rmse_one_step"]
            if baseline and baseline["metrics_range"] else None
        )
        if baseline:
            _print_row_range(baseline, base_rmse=None)
            print("-" * 103)
        for r in summary:
            _print_row_range(r, base_rmse=base_rmse_range)


if __name__ == "__main__":
    main()
