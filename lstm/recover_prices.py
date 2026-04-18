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

Input:  lstm/predictions/<label>_returns.csv  (from predict.py)

Output (always):
    lstm/predictions/<label>_prices.csv
    lstm/predictions/<label>_prices.png
    lstm/predictions/_all_models_one_step.png     (when multiple models)

Output (when --plot-start / --plot-end given):
    lstm/predictions/<start>_<end>/<label>_prices.png
    lstm/predictions/<start>_<end>/_all_models_one_step.png

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

PREDICTIONS_DIR = "predictions"
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


def recover_from_predictions(predictions_csv, raw_prices_csv=RAW_PRICES_CSV,
                             price_col=GOLD_PRICE_COL, out_dir=None,
                             label=None, plot_start=None, plot_end=None,
                             range_dir=None):
    """Recover absolute prices from a predictions CSV.

    Always writes the full-range CSV and PNG to `out_dir`.
    If plot_start/plot_end are given, also writes a range-limited PNG
    (with metrics recomputed over the visible window) to `range_dir`.
    """
    if out_dir is None:
        out_dir = os.path.dirname(predictions_csv) or "."
    if label is None:
        base = os.path.basename(predictions_csv)
        label = base.replace("_returns.csv", "")

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
    out_csv = os.path.join(out_dir, f"{label}_prices.csv")
    merged.to_csv(out_csv, index=False)
    print(f"[{label}] Wrote {out_csv}")

    out_png = os.path.join(out_dir, f"{label}_prices.png")
    _plot_model(merged, label, out_png, metrics_full)

    # Optional range-limited plot.
    range_png = None
    metrics_range = None
    if (plot_start is not None or plot_end is not None) and range_dir is not None:
        sub = _slice_range(merged, plot_start, plot_end)
        metrics_range = _compute_metrics(sub)
        suffix = f" [{plot_start or '…'} → {plot_end or '…'}]"
        range_png = _plot_model(
            sub, label,
            os.path.join(range_dir, f"{label}_prices.png"),
            metrics_range,
            range_suffix=suffix,
        )
        if sub.empty:
            print(f"[{label}] range produced 0 rows — plot skipped")

    return {
        "label": label,
        "out_csv": out_csv,
        "out_png": out_png,
        "range_png": range_png,
        **{k: metrics_full[k] for k in (
            "rmse_one_step", "rmse_rolling",
            "mape_one_step", "mape_rolling",
            "direction_accuracy", "n_rows",
        )},
        "metrics_range": metrics_range,
        "merged": merged,
    }


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
                        help="Path to a predictions CSV. Overrides --label.")
    parser.add_argument("--label", type=str, default=None,
                        help="Load predictions/<label>_returns.csv.")
    parser.add_argument("--predictions-dir", type=str, default=PREDICTIONS_DIR)
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

    range_dir = None
    if plot_start is not None or plot_end is not None:
        range_name = f"{plot_start or 'min'}_{plot_end or 'max'}"
        range_dir = os.path.join(args.predictions_dir, range_name)
        os.makedirs(range_dir, exist_ok=True)
        print(f"[range] Saving range plots into {range_dir}")

    if args.predictions is not None:
        files = [args.predictions]
    elif args.label is not None:
        files = [os.path.join(args.predictions_dir, f"{args.label}_returns.csv")]
    else:
        files = sorted(
            glob.glob(os.path.join(args.predictions_dir, "*_returns.csv"))
        )
        if not files:
            raise SystemExit(
                f"No predictions found in {args.predictions_dir}/. "
                f"Run predict.py first."
            )

    summary = []
    for f in files:
        summary.append(
            recover_from_predictions(
                f,
                raw_prices_csv=args.raw_prices,
                price_col=args.price_col,
                plot_start=plot_start,
                plot_end=plot_end,
                range_dir=range_dir,
            )
        )

    if len(summary) > 1:
        # Always: full-range combined plot in predictions/
        plot_combined_one_step(
            summary,
            os.path.join(args.predictions_dir, "_all_models_one_step.png"),
        )
        # Optional: range-limited combined plot in predictions/<range>/
        if range_dir is not None:
            plot_combined_one_step(
                summary,
                os.path.join(range_dir, "_all_models_one_step.png"),
                plot_start=plot_start, plot_end=plot_end,
            )

    print("\n" + "=" * 72)
    print("SUMMARY (full range)")
    print("=" * 72)
    print(f"{'Model':<20} {'RMSE(1-step)':>13} {'MAPE(1-step)':>13} "
          f"{'RMSE(roll)':>12} {'MAPE(roll)':>12} {'DirAcc%':>8}")
    print("-" * 80)
    for r in summary:
        print(f"{r['label']:<20} {r['rmse_one_step']:>13.4f} "
              f"{r['mape_one_step']:>12.3f}% {r['rmse_rolling']:>12.4f} "
              f"{r['mape_rolling']:>11.3f}% {r['direction_accuracy']:>7.2f}%")

    if range_dir is not None:
        print("\n" + "=" * 72)
        print(f"SUMMARY (range {plot_start or 'min'} → {plot_end or 'max'})")
        print("=" * 72)
        print(f"{'Model':<20} {'RMSE(1-step)':>13} {'MAPE(1-step)':>13} "
              f"{'RMSE(roll)':>12} {'MAPE(roll)':>12} {'DirAcc%':>8} "
              f"{'n':>5}")
        print("-" * 88)
        for r in summary:
            m = r["metrics_range"] or {}
            if not m or m.get("n_rows", 0) == 0:
                print(f"{r['label']:<20}  (no rows in range)")
                continue
            print(f"{r['label']:<20} {m['rmse_one_step']:>13.4f} "
                  f"{m['mape_one_step']:>12.3f}% {m['rmse_rolling']:>12.4f} "
                  f"{m['mape_rolling']:>11.3f}% "
                  f"{m['direction_accuracy']:>7.2f}% {m['n_rows']:>5d}")


if __name__ == "__main__":
    main()
