"""
paths.py
--------
Centralised label <-> filesystem-path helpers for the LSTM pipeline.

All three scripts (train.py, predict.py, recover_prices.py) route their
plot / CSV outputs through here so the on-disk layout is consistent:

    lstm/runs/<arch_label>/seed<N>/             per-run artifacts (losses, prices, returns)
    lstm/runs/<arch_label>/seed<N>/<range>/     range-limited per-run plots
    lstm/runs/<arch_label>/_aggregate/          mean +/- std across seeds
    lstm/runs/_all_models/                      cross-architecture combined plots
    lstm/runs/_all_models/<range>/              range-limited combined plots
    lstm/checkpoints/<label>.pt                 flat registry keyed by full label

Because the directory path already encodes `arch_label` and `seed`, the
per-run filenames are kept short ("losses.png", "prices.csv", ...).
Checkpoints are the exception: they live in a flat folder keyed by the
full label so `predict.py` can glob them without scanning the runs tree.
"""

import os

RUNS_DIR = "runs"
CHECKPOINT_DIR = "checkpoints"
ALL_MODELS_SUBDIR = "_all_models"
AGGREGATE_SUBDIR = "_aggregate"

# Per-run filenames (short; arch + seed already encoded in the path).
LOSSES_FILENAME = "losses.png"
RETURNS_CSV_FILENAME = "returns.csv"
RETURNS_PLOT_FILENAME = "returns_plot.png"
PRICES_CSV_FILENAME = "prices.csv"
PRICES_PLOT_FILENAME = "prices.png"

# Combined-plot filenames in _all_models/.
COMBINED_ONE_STEP_FILENAME = "one_step.png"
COMBINED_RETURNS_FILENAME = "returns.png"


def parse_label(label):
    """Split 'LSTM-all-reg-l1-attn_seed42' -> ('LSTM-all-reg-l1-attn', 42).

    Returns (arch_label, seed). `seed` is None for legacy labels without a
    `_seed<N>` suffix (older checkpoints trained before the seed sweep).
    """
    if "_seed" in label:
        arch, seed_part = label.rsplit("_seed", 1)
        try:
            return arch, int(seed_part)
        except ValueError:
            pass
    return label, None


def make_label(arch_label, seed):
    """Inverse of `parse_label`."""
    if seed is None:
        return arch_label
    return f"{arch_label}_seed{seed}"


def _seed_folder(seed):
    return "no_seed" if seed is None else f"seed{seed}"


def run_dir(arch_label, seed, root=RUNS_DIR):
    """Per-seed run folder, e.g. runs/LSTM-all-attn/seed42/."""
    return os.path.join(root, arch_label, _seed_folder(seed))


def aggregate_dir(arch_label, root=RUNS_DIR):
    """Per-architecture aggregate folder, e.g. runs/LSTM-all-attn/_aggregate/."""
    return os.path.join(root, arch_label, AGGREGATE_SUBDIR)


def all_models_dir(range_name=None, root=RUNS_DIR):
    """Combined-plot folder. If range_name is set, nest one level deeper."""
    d = os.path.join(root, ALL_MODELS_SUBDIR)
    if range_name is not None:
        d = os.path.join(d, range_name)
    return d


def range_subdir(run_dir_path, range_name):
    """Range-limited plot folder inside a specific run."""
    return os.path.join(run_dir_path, range_name)


def range_name(plot_start, plot_end):
    """Canonical name for a range subfolder, or None if both bounds are None."""
    if plot_start is None and plot_end is None:
        return None
    return f"{plot_start or 'min'}_{plot_end or 'max'}"


def checkpoint_path(label, root=CHECKPOINT_DIR):
    """Flat .pt registry keyed by full label (arch + seed)."""
    return os.path.join(root, f"{label}.pt")


# ---------- Convenience accessors ----------------------------------------

def losses_png(arch_label, seed, root=RUNS_DIR):
    return os.path.join(run_dir(arch_label, seed, root=root), LOSSES_FILENAME)


def returns_csv(arch_label, seed, root=RUNS_DIR):
    return os.path.join(run_dir(arch_label, seed, root=root), RETURNS_CSV_FILENAME)


def returns_plot_png(arch_label, seed, root=RUNS_DIR):
    return os.path.join(run_dir(arch_label, seed, root=root), RETURNS_PLOT_FILENAME)


def prices_csv(arch_label, seed, root=RUNS_DIR):
    return os.path.join(run_dir(arch_label, seed, root=root), PRICES_CSV_FILENAME)


def prices_plot_png(arch_label, seed, root=RUNS_DIR):
    return os.path.join(run_dir(arch_label, seed, root=root), PRICES_PLOT_FILENAME)
