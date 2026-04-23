"""

tune_lstm_mse.py: automated hyperparameter tuning for the LSTM gold-return predictor.

Objective:
Minimise mean validation MSE (averaged across seeds) on the LSTM-all
variant. The resulting configuration is then *locked* and applied to every
feature-selection variant in train.py, so that model-vs-model differences
reflect the feature set rather than per-variant hyperparameter tuning.

Directional accuracy is recorded as an auxiliary diagnostic only — the
search direction is MSE.

Methodology notes
1. No data leakage: Tuning signal comes from the validation split only.
The test split is never touched here.
2. Each trial is evaluated over len(TUNING_SEEDS) random seeds to reduce
single-seed noise in the objective.
3. Tuning seeds are completely different from the final-training seeds in
train.py, so the selected config is not overfit to a specific seed set.
4. Uses Optuna's TPE sampler (Akiba et al., 2019) with the MedianPruner to
terminate clearly-underperforming trials before all seeds are evaluated.

References
Optuna         : Akiba et al. (2019), "Optuna: A Next-generation
                Hyperparameter Optimization Framework". KDD 2019.
                Docs: https://optuna.readthedocs.io/
TPE sampler    : Bergstra et al. (2011), "Algorithms for Hyper-Parameter
                Optimization". NeurIPS 2011.
                API:  https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html
MedianPruner   : https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html

Usage: python tune_lstm_mse.py

outputs: optuna_study.pkl
        tune_lstm_best_config.json  - best config to copy into train.py
"""


from dataclasses import dataclass
import json
import pickle

import numpy as np
import optuna

# Re-use the single-run entry point and shared constants from train.py so the
# tuning and training pipelines cannot drift apart.
from train import (
    run,
    CSV, SPLIT_JSON, SEQ_LEN, BATCH_SIZE, TARGET_COL, LOSS_SCALE,
)



#  TUNING CONFIG
# 60 trials * 5 seeds = 180 run() invocations nominally, reduced
# in practice by MedianPruner (see below). Enough trials for TPE's informed
# phase (trial>10) to explore a wider search space than the earlier 30-trial
# run. See report method (hyperparam and other experiment settings) for the
# design rationale.
N_TRIALS = 60 # 10 TPE warm-up + 50 informed
N_EPOCHS_TUNE = 50 # same cap as final training
PATIENCE_TUNE = 10 # tighter than final training (patience=10),
                # speeds tuning at the cost of a slight bias toward fast-converging configs

TUNING_SEEDS     = [0, 1, 2, 3, 4]
SEEDS_PER_TRIAL  = len(TUNING_SEEDS)

# MedianPruner: kill a trial if its objective is worse than the median of
# previously completed trials. We evaluate seeds sequentially and report
# each seed's val MSE as an "intermediate value", so Optuna can prune a
# clearly losing configuration before running all SEEDS_PER_TRIAL seeds.
# See: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html
PRUNER_WARMUP_TRIALS = 10 # no pruning for the first N trials (TPE warm-up)
PRUNER_WARMUP_STEPS  = 1 # report at least 1 seed before pruning is allowed

OUTPUT_STUDY_PKL = "optuna_study.pkl"
OUTPUT_BEST_JSON = "tune_lstm_best_config.json"


@dataclass
class TrialConfig:
    """
One point in hyperparameter space that run() consumes.

    Keeping this as a dataclass (rather than a dict) makes the set of tunable
    knobs explicit and lets both objective() and main() format the result
    with the same field names.
    """

    lr: float
    hidden: int
    num_layers: int
    dropout: float

    @classmethod
    def from_trial(cls, trial: optuna.Trial) -> "TrialConfig":
        """
        Sample a configuration from an Optuna trial.

        Search space (widened from the original 30-trial run):
        lr: log-uniform in [5e-4, 1e-3] (before: [1e-4, 2e-3])
        hidden: {16, 32, 64} (before: {16, 32, 64})
        num_layers: {1, 2, 3} (before: {1, 2})
        dropout: uniform in [0.0, 0.5] (before [0.1, 0.4])

        Note: previous setting done a 30-trial search clustered its
        top configs at the interior of the space (hidden=32, layers=2,
        dropout approx 0.36)
        Widening lets TPE confirm whether that cluster is a true optimum
        or just a boundary effect (turns out to have a better convergence)
        """

        return cls(
            lr=trial.suggest_float("lr", 5e-4, 1e-3, log=True),
            hidden=trial.suggest_categorical("hidden", [16, 32, 64]),
            num_layers=trial.suggest_categorical("num_layers", [1, 2, 3]),
            dropout=trial.suggest_float("dropout", 0.0, 0.5),
        )


@dataclass
class TrialResult:
    """
    Aggregated metrics for a single hyperparameter configuration
    val_mse is the search objective. val_diracc is recorded for context only
    """

    val_mse: float # mean val MSE across seeds  (OBJECTIVE)
    val_diracc: float # mean val directional acc   (diagnostic)



#  OBJECTIVE
def evaluate_config(cfg: TrialConfig, seeds: list[int],
                    trial: optuna.Trial | None = None) -> TrialResult:
    """
    Train one model per seed with cfg, return averaged val metrics.

    If trial is provided, each seed's val MSE is reported as an
    intermediate value, and the trial is pruned mid-way if Optuna's pruner
    decides this config is clearly worse than the median of completed
    trials. This saves compute on obviously-bad configs.
    """

    val_mses = []
    val_diraccs = []
    trial_num = trial.number if trial is not None else -1

    for step, seed in enumerate(seeds):
        result = run(
            CSV, SPLIT_JSON, SEQ_LEN, BATCH_SIZE, TARGET_COL,N_EPOCHS_TUNE,
            cfg.lr, cfg.hidden, cfg.num_layers, cfg.dropout,False, 1, 1,LOSS_SCALE,
            feature_cols=None,label=f"tune-trial{trial_num}-seed{seed}",
            use_early_stop=PATIENCE_TUNE,seed=seed,verbose=False,
        )
        val_mses.append(result["val_metrics"]["mse"])
        val_diraccs.append(result["val_metrics"]["directional_acc"])

        # Report running mean to Optuna as intermediate value; may prune.
        if trial is not None:
            running_mean = float(np.mean(val_mses))
            trial.report(running_mean, step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return TrialResult(
        val_mse=float(np.mean(val_mses)),
        val_diracc=float(np.mean(val_diraccs)),
    )


def objective(trial: optuna.Trial) -> float:
    """ Optuna objective: mean validation MSE over TUNING_SEEDS"""
    cfg = TrialConfig.from_trial(trial)
    res = evaluate_config(cfg, TUNING_SEEDS, trial=trial)

    # Attach both metrics so either can be inspected downstream; only
    # val_mse drives the search.
    trial.set_user_attr("mean_val_mse", res.val_mse)
    trial.set_user_attr("mean_val_diracc", res.val_diracc)

    print(f"[trial {trial.number:2d}] "
        f"val_mse={res.val_mse * LOSS_SCALE:7.2f}  "
        f"(val_diracc={res.val_diracc:.4f})")

    return res.val_mse

def print_best(study: optuna.Study) -> None:
    best = study.best_trial
    print("\n" + "=" * 30)
    print("BEST CONFIGURATION")
    print("=" * 30)
    print(f"Trial #{best.number}")
    print(f"  val MSE (scaled)   : {best.user_attrs['mean_val_mse'] * LOSS_SCALE:.2f}")
    print(f"  val DirAcc (diag.) : {best.user_attrs['mean_val_diracc']:.4f}")
    print("Params:")
    for key, val in best.params.items():
        print(f"  {key:<12} = {val}")


def print_top_k(study: optuna.Study, k: int = 5) -> None:
    """
    Show the top-k completed trials by val MSE.

    Tight clustering here is the main robustness signal for the search —
    if the top-k all share hidden/num_layers, the search found a stable
    region rather than a single lucky point.
    """

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    top = sorted(completed, key=lambda t: t.value)[:k]

    print("\n" + "=" * 30)
    print(f"TOP {k} TRIALS (by val MSE)  —  "
        f"{len(completed)} completed, {len(pruned)} pruned")
    print("=" * 30)
    for t in top:
        p = t.params
        print(f"Trial {t.number:3d}: "
                f"val_mse={t.value * LOSS_SCALE:7.2f}  "
                f"lr={p['lr']:.1e}  "
                f"hidden={p['hidden']}  "
                f"layers={p['num_layers']}  "
                f"dropout={p['dropout']:.2f}")


def save_outputs(study: optuna.Study) -> None:
    # Full study — load with pickle.load(open(OUTPUT_STUDY_PKL, "rb"))
    # to resume, inspect trial history, or re-plot.
    with open(OUTPUT_STUDY_PKL, "wb") as f:
        pickle.dump(study, f)

    best = study.best_trial
    best_config = {
        "trial_number":     best.number,
        "val_mse_scaled":   best.user_attrs["mean_val_mse"] * LOSS_SCALE,
        "val_diracc":       best.user_attrs["mean_val_diracc"],
        "params":           best.params,
        "n_trials":         N_TRIALS,
        "seeds_per_trial":  SEEDS_PER_TRIAL,
        "tuning_seeds":     TUNING_SEEDS,
        "n_epochs_tune":    N_EPOCHS_TUNE,
        "patience_tune":    PATIENCE_TUNE,
        "objective":        "mean_val_mse",
        "pruner":           "MedianPruner",
    }
    with open(OUTPUT_BEST_JSON, "w") as f:
        json.dump(best_config, f, indent=2)

    print(f"\n[SAVED] {OUTPUT_STUDY_PKL} (resumable study)")
    print(f"[SAVED] {OUTPUT_BEST_JSON} (copy params into train.py CONFIG)")

def main():
    print(f"Optuna tuning: {N_TRIALS} trials × {len(TUNING_SEEDS)} seeds/trial")
    print(f"Tuning seeds : {TUNING_SEEDS}  (disjoint from final-training seeds)")
    print(f"Objective    : minimise mean validation MSE on LSTM-all")
    print(f"Tuning regime: {N_EPOCHS_TUNE} max epochs, early-stop patience={PATIENCE_TUNE}")
    print(f"Pruner       : MedianPruner (warmup={PRUNER_WARMUP_TRIALS} trials)")
    print()

    # TPE sampler seeded for reproducibility of the search path itself (seed=42
    # for Optuna not equal the seeds used for model training inside each trial).
    # MedianPruner kills obviously-bad trials mid-evaluation: if a config's
    # running mean val MSE (after reporting at least one seed) is worse than
    # the median of completed trials, the remaining seeds are skipped.
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=PRUNER_WARMUP_TRIALS,
            n_warmup_steps=PRUNER_WARMUP_STEPS,
        ),
    )
    study.optimize(objective, n_trials=N_TRIALS)

    print_best(study)
    save_outputs(study)
    print_top_k(study, k=5)


if __name__ == "__main__":
    main()