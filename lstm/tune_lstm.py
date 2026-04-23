"""
tune_lstm.py
------------
Automated hyperparameter tuning for LSTM using Optuna.

Tunes on LSTM-directional variant (since directional objective is where we
hope to see real DirAcc gain). Once best config is found, it's printed
and saved; you then update train.py's CONFIG section manually and rerun
train.py with all variants using the locked config.

CRITICAL: objective uses VAL DirAcc (not test), following rubric's
"hyperparameter tuning on held-out validation set, not test".

Usage:
    pip install optuna
    python tune_lstm.py

Outputs:
    optuna_study.pkl           - full study object (for resume)
    tune_lstm_best_config.json - best config to copy into train.py
"""

import json
import pickle
import numpy as np
import optuna

# Import everything we need from train.py
from train import (
    run, load_causal_features,
    CSV, SPLIT_JSON, SEQ_LEN, BATCH_SIZE, TARGET_COL, EPOCHS, LOSS_SCALE,
)

# ════════════════════════════════════════════════════════════════
#  TUNING CONFIG
# ════════════════════════════════════════════════════════════════
N_TRIALS = 30                    # how many hyperparameter configs to try
SEEDS_PER_TRIAL = 3              # multi-seed to reduce noise per config
TUNE_VARIANT = "directional"     # "directional" or "mse"
N_EPOCHS_TUNE = 30               # fewer epochs for speed during tuning
PATIENCE_TUNE = 5                # tighter early stopping during tuning


def objective(trial):
    """Sample a config, train LSTM-directional, return mean val-DirAcc
    (negated since Optuna minimizes)."""

    # Sample hyperparameters
    config = {
        "lr":           trial.suggest_float("lr", 1e-4, 2e-3, log=True),
        "hidden":       trial.suggest_categorical("hidden", [16, 32, 64]),
        "num_layers":   trial.suggest_categorical("num_layers", [1, 2]),
        "dropout":      trial.suggest_float("dropout", 0.1, 0.4),
        "lambda_dir":   trial.suggest_float("lambda_dir", 0.05, 2.0, log=True),
    }

    causal_cols = load_causal_features()

    val_diraccs = []
    val_mses = []
    for seed in range(SEEDS_PER_TRIAL):
        try:
            result = run(
                CSV, SPLIT_JSON, SEQ_LEN, BATCH_SIZE, TARGET_COL,
                N_EPOCHS_TUNE, config["lr"], config["hidden"],
                config["num_layers"], config["dropout"], LOSS_SCALE,
                feature_cols=None,
                label=f"tune-trial{trial.number}-seed{seed}",
                loss_type=TUNE_VARIANT, lambda_dir=config["lambda_dir"],
                early_stop=PATIENCE_TUNE,
                seed=seed, verbose=False,
            )
            val_diraccs.append(result["val_metrics"]["directional_acc"])
            val_mses.append(result["val_metrics"]["mse"])
        except Exception as e:
            print(f"[trial {trial.number} seed {seed}] failed: {e}")
            val_diraccs.append(0.5)  # worst case
            val_mses.append(1.0)

    mean_val_diracc = float(np.mean(val_diraccs))
    mean_val_mse = float(np.mean(val_mses))

    # Record both metrics in trial (so we can inspect trade-offs later)
    trial.set_user_attr("mean_val_diracc", mean_val_diracc)
    trial.set_user_attr("mean_val_mse", mean_val_mse)

    print(f"[trial {trial.number}] config={config}")
    print(f"                  val_diracc={mean_val_diracc:.4f}, val_mse={mean_val_mse*LOSS_SCALE:.2f}")

    # We maximize val_diracc ⇔ minimize (-val_diracc)
    return -mean_val_diracc


def main():
    print(f"Starting Optuna tuning: {N_TRIALS} trials, {SEEDS_PER_TRIAL} seeds/trial")
    print(f"Objective: maximize val DirAcc on LSTM-{TUNE_VARIANT}")
    print(f"Tuning epochs: {N_EPOCHS_TUNE}, early stopping patience: {PATIENCE_TUNE}")
    print()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=N_TRIALS)

    # Best config
    best_trial = study.best_trial
    print("\n" + "=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)
    print(f"Trial #{best_trial.number}")
    print(f"Val DirAcc: {-best_trial.value:.4f}")
    print(f"Val MSE:    {best_trial.user_attrs['mean_val_mse']*LOSS_SCALE:.2f}")
    print("Params:")
    for key, val in best_trial.params.items():
        print(f"  {key:<15} = {val}")

    # Save
    with open("optuna_study.pkl", "wb") as f:
        pickle.dump(study, f)

    best_config = {
        "trial_number": best_trial.number,
        "val_diracc": -best_trial.value,
        "val_mse_scaled": best_trial.user_attrs["mean_val_mse"] * LOSS_SCALE,
        "params": best_trial.params,
        "tune_variant": TUNE_VARIANT,
        "n_trials": N_TRIALS,
        "seeds_per_trial": SEEDS_PER_TRIAL,
    }
    with open("tune_lstm_best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\n[SAVED] optuna_study.pkl (resume-able)")
    print(f"[SAVED] tune_lstm_best_config.json (copy into train.py)")

    # Show top 5 trials for comparison
    print("\n" + "=" * 70)
    print("TOP 5 TRIALS (by val DirAcc)")
    print("=" * 70)
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else 0)[:5]
    for t in sorted_trials:
        print(f"  Trial {t.number:3d}: val_diracc={-t.value:.4f}  "
              f"lr={t.params['lr']:.1e} hidden={t.params['hidden']} "
              f"layers={t.params['num_layers']} dropout={t.params['dropout']:.2f} "
              f"λ_dir={t.params['lambda_dir']:.2f}")


if __name__ == "__main__":
    main()
