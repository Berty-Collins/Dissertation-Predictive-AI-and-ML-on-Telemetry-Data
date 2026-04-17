"""
optimisation/bayesian_opt.py
=============================
Uses trained ML models as surrogate functions to find optimal vehicle
configurations WITHOUT running BeamNG — orders of magnitude faster than
brute-force simulation search.

This addresses Research Question 2: "How can outputs guide optimisation
of vehicle parameters for speed, efficiency, or handling?"

Two strategies:
1. Single-objective: minimise launch_time_0_60_s (fastest 0-60)
2. Multi-objective (Pareto front): launch_time_0_60_s vs circle_max_lat_g
   (find setups that are both fast in a straight line and grippy in corners)

Parameter space and defaults are imported directly from scenario_runner.py
so this file stays in sync automatically as the sweep design changes.

Libraries:
    pip install optuna
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import MODELS_DIR, RESULTS_DIR, RANDOM_STATE

# Import parameter space directly from the runner — single source of truth.
# PARAM_RANGES: {name: (min, max, default)}
from data_collection.scenario_runner import PARAM_RANGES, JBEAM_KEYS
from data_processing.preprocess import engineer_features, get_feature_columns

log = logging.getLogger(__name__)

FIGURES_DIR = RESULTS_DIR / "figures" / "optimisation"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Surrogate loading
# ---------------------------------------------------------------------------

def load_surrogate(target: str):
    """
    Load a trained model, its scaler, and its feature name list.

    Feature names are saved by train_evaluate.py alongside the model as
    features_{target}.pkl — they define the exact column order the scaler
    and model expect.
    """
    model_path    = MODELS_DIR / f"best_{target}.pkl"
    scaler_path   = MODELS_DIR / f"scaler_{target}.pkl"
    features_path = MODELS_DIR / f"features_{target}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model for '{target}'. Run models/train_evaluate.py first."
        )
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"No scaler for '{target}'. Run models/train_evaluate.py first."
        )

    model    = joblib.load(model_path)
    scaler   = joblib.load(scaler_path)
    features = joblib.load(features_path) if features_path.exists() else None

    if features is None:
        log.warning(
            "features_%s.pkl not found — re-run train_evaluate.py to generate it. "
            "Falling back to column-order inference (may be incorrect).", target
        )

    return model, scaler, features


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def config_to_feature_vector(config: dict, features: Optional[List[str]]) -> np.ndarray:
    """
    Convert a parameter dict → 1-row array ready for scaler.transform().

    1. Build a single-row DataFrame from the config.
    2. Run engineer_features() to add all feat_* columns.
    3. Select columns in exactly the order the scaler was fitted on.
    """
    df = pd.DataFrame([config])
    df = engineer_features(df)

    if features is not None:
        # Use the saved feature list — guaranteed to match scaler column order.
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(
                f"config_to_feature_vector: engineered DataFrame is missing "
                f"columns that the scaler expects: {missing}"
            )
        return df[features].values.astype(float)
    else:
        # Fallback: use the same logic as preprocess.get_feature_columns.
        feat_cols = get_feature_columns(df)
        return df[feat_cols].values.astype(float)


def predict_target(config: dict, target: str,
                   model=None, scaler=None, features=None) -> float:
    """Predict a single target value for a given parameter configuration."""
    if model is None or scaler is None:
        model, scaler, features = load_surrogate(target)

    X = config_to_feature_vector(config, features)
    X_scaled = scaler.transform(X)
    return float(model.predict(X_scaled)[0])


# ---------------------------------------------------------------------------
# Parameter space helpers (built from PARAM_RANGES)
# ---------------------------------------------------------------------------

def _build_default_config() -> dict:
    """Return a config dict at JBeam defaults for all parameters."""
    return {k: PARAM_RANGES[k][2] for k in PARAM_RANGES}


def _suggest_param(trial, name: str):
    """Suggest a value for one parameter using the correct Optuna suggest method."""
    lo, hi, _ = PARAM_RANGES[name]
    # Integer-step parameters
    if name in ("spring_F", "spring_R"):
        return trial.suggest_int(name, int(lo), int(hi), step=500)
    if name in ("arb_spring_F", "arb_spring_R",
                "damp_bump_F", "damp_bump_R",
                "damp_rebound_F", "damp_rebound_R",
                "lsdpreload_R", "lsdpreload_F"):
        return trial.suggest_int(name, int(lo), int(hi), step=100)
    # Continuous parameters
    return trial.suggest_float(name, lo, hi)


def _trial_to_config(trial) -> dict:
    """Build a full parameter config from an Optuna trial."""
    return {name: _suggest_param(trial, name) for name in PARAM_RANGES}


# ---------------------------------------------------------------------------
# Single-objective Bayesian optimisation
# ---------------------------------------------------------------------------

def optimise_single_objective(
        target: str = "launch_time_0_60_s",
        n_trials: int = 500,
        direction: str = "minimize",
        fixed_params: Optional[Dict] = None,
) -> Tuple[dict, float]:
    """
    Find the vehicle configuration that optimises `target` using
    Bayesian optimisation via Optuna (TPE sampler).

    Parameters
    ----------
    target : str
        ML KPI to optimise (e.g., "launch_time_0_60_s", "circle_max_lat_g")
    n_trials : int
        Number of surrogate evaluations (each is instant — no BeamNG needed)
    direction : str
        "minimize" or "maximize"
    fixed_params : dict, optional
        Parameters to hold fixed (e.g., {"brakebias": 0.62})

    Returns
    -------
    best_config : dict
    best_value : float
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("pip install optuna")

    model, scaler, features = load_surrogate(target)
    fixed = fixed_params or {}

    def objective(trial) -> float:
        config = {}
        for name in PARAM_RANGES:
            config[name] = fixed[name] if name in fixed else _suggest_param(trial, name)
        return predict_target(config, target, model, scaler, features)

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_config = study.best_params
    best_value  = study.best_value

    log.info("Optimisation complete:")
    log.info("  Target: %s (%s)", target, direction)
    log.info("  Best value: %.4f", best_value)
    log.info("  Best config: %s", best_config)

    _plot_optimisation_history(study, target, direction)

    return best_config, best_value


def _plot_optimisation_history(study, target: str, direction: str):
    values = [t.value for t in study.trials if t.value is not None]
    best_so_far = (np.minimum.accumulate(values) if direction == "minimize"
                   else np.maximum.accumulate(values))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(range(len(values)), values, alpha=0.4, s=10, label="Trial value")
    axes[0].plot(range(len(best_so_far)), best_so_far, "r-", lw=2, label="Best so far")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel(target)
    axes[0].set_title(f"Optimisation History — {target}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    try:
        import optuna.importance as oi
        importances = oi.get_param_importances(study)
        params = list(importances.keys())[:15]
        imps   = [importances[p] for p in params]
        axes[1].barh(params[::-1], imps[::-1], color="#2196F3")
        axes[1].set_xlabel("Importance")
        axes[1].set_title("Parameter Importance (Optuna FAnova)")
    except Exception:
        axes[1].text(0.5, 0.5, "Importance not\navailable",
                     ha="center", va="center", transform=axes[1].transAxes)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"optimisation_{target}.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Optimisation plot saved.")


# ---------------------------------------------------------------------------
# Multi-objective: acceleration vs cornering grip (Pareto front)
# ---------------------------------------------------------------------------

def pareto_front_analysis(
        obj1: str = "launch_time_0_60_s",   # minimise — faster 0-60
        obj2: str = "circle_max_lat_g",      # maximise — more cornering grip
        n_trials: int = 1000,
) -> pd.DataFrame:
    """
    Find the Pareto front between two objectives.

    obj1 (launch_time_0_60_s) is minimised; obj2 (circle_max_lat_g) is
    maximised by negating it for the NSGA-II solver (both directions set
    to "minimize").

    Returns a DataFrame of non-dominated solutions sorted by obj1.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("pip install optuna")

    model1, scaler1, features1 = load_surrogate(obj1)
    model2, scaler2, features2 = load_surrogate(obj2)

    def objective(trial):
        config = _trial_to_config(trial)
        v1 =  predict_target(config, obj1, model1, scaler1, features1)   # minimise
        v2 = -predict_target(config, obj2, model2, scaler2, features2)   # negate to minimise
        return v1, v2

    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    pareto_data = []
    for t in study.best_trials:
        row = t.params.copy()
        row[obj1] =  t.values[0]           # actual launch time
        row[obj2] = -t.values[1]           # undo negation → actual lat g
        pareto_data.append(row)

    pareto_df = pd.DataFrame(pareto_data).sort_values(obj1)

    # Plot
    all_v1 = [ t.values[0] for t in study.trials if t.values]
    all_v2 = [-t.values[1] for t in study.trials if t.values]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(all_v1, all_v2, alpha=0.2, s=10, c="grey", label="All solutions")
    ax.scatter(pareto_df[obj1], pareto_df[obj2],
               c="red", s=40, zorder=5, label="Pareto front")
    ax.plot(pareto_df[obj1], pareto_df[obj2], "r-", alpha=0.5)
    ax.set_xlabel(f"{obj1} (s)  ← lower is faster")
    ax.set_ylabel(f"{obj2} (g)  ← higher is more grip")
    ax.set_title(f"Pareto Front: Acceleration vs Cornering Grip")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"pareto_{obj1}_vs_{obj2}.png", dpi=150, bbox_inches="tight")
    plt.close()

    pareto_df.to_csv(RESULTS_DIR / f"pareto_{obj1}_vs_{obj2}.csv", index=False)
    log.info("Pareto front: %d non-dominated solutions.", len(pareto_df))
    return pareto_df


# ---------------------------------------------------------------------------
# Sensitivity analysis — one-at-a-time (OAT)
# ---------------------------------------------------------------------------

def sensitivity_analysis(
        target: str = "launch_time_0_60_s",
        base_config: Optional[dict] = None,
        n_points: int = 50,
) -> pd.DataFrame:
    """
    Vary each parameter individually across its full range (holding all
    others at default) and record the predicted change in `target`.

    Produces easily interpretable sensitivity plots for the dissertation.
    """
    model, scaler, features = load_surrogate(target)
    base = base_config or _build_default_config()

    param_list = list(PARAM_RANGES.keys())
    n_params   = len(param_list)
    ncols      = 4
    nrows      = (n_params + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.5))
    axes = axes.flatten()

    sensitivities = []
    base_pred = predict_target(base, target, model, scaler, features)

    for i, name in enumerate(param_list):
        lo, hi, _ = PARAM_RANGES[name]
        values = np.linspace(lo, hi, n_points)
        preds  = []
        for v in values:
            cfg = base.copy()
            cfg[name] = v
            preds.append(predict_target(cfg, target, model, scaler, features))

        preds = np.array(preds)
        sens_range = preds.max() - preds.min()
        sensitivities.append({
            "parameter":   name,
            "min_pred":    preds.min(),
            "max_pred":    preds.max(),
            "range":       sens_range,
            "rel_range_%": sens_range / (abs(base_pred) + 1e-6) * 100,
        })

        ax = axes[i]
        ax.plot(values, preds, color="#2196F3", lw=1.5)
        ax.axhline(base_pred, color="red", ls="--", lw=1, alpha=0.7, label="Baseline")
        ax.set_xlabel(name, fontsize=8)
        ax.set_ylabel(target, fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

    for j in range(n_params, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"One-at-a-Time Sensitivity Analysis — {target}", fontsize=13)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"sensitivity_{target}.png", dpi=150, bbox_inches="tight")
    plt.close()

    sens_df = pd.DataFrame(sensitivities).sort_values("range", ascending=False)
    sens_df.to_csv(RESULTS_DIR / f"sensitivity_{target}.csv", index=False)
    log.info("Top 5 most sensitive parameters for %s:", target)
    log.info(sens_df.head(5).to_string(index=False))
    return sens_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    print("\n=== Single-objective: Minimise 0-60 Time ===")
    best_config, best_time = optimise_single_objective(
        target="launch_time_0_60_s", n_trials=500, direction="minimize"
    )
    print(f"Best 0-60 time: {best_time:.3f} s")
    print(f"Config: {best_config}")

    print("\n=== Single-objective: Maximise Cornering Grip ===")
    best_config_c, best_lat_g = optimise_single_objective(
        target="circle_max_lat_g", n_trials=500, direction="maximize"
    )
    print(f"Best max lateral g: {best_lat_g:.3f} g")

    print("\n=== Multi-objective: Acceleration vs Cornering Grip ===")
    pareto = pareto_front_analysis(
        obj1="launch_time_0_60_s", obj2="circle_max_lat_g", n_trials=800
    )
    print(f"Pareto front: {len(pareto)} non-dominated solutions")

    print("\n=== Sensitivity Analysis: 0-60 Time ===")
    sens = sensitivity_analysis(target="launch_time_0_60_s")
    print(sens.head(10).to_string(index=False))
