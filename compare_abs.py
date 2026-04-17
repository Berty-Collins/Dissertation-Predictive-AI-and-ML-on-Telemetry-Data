"""
compare_abs.py
==============
Train models on the ABS-on (baseline) and ABS-off datasets and compare R2.
Shows whether disabling ABS reveals stronger parameter -> KPI relationships.

Usage:
    python compare_abs.py

Requires:
    results/sweep_results.csv         (ABS on  -- standard run)
    results/sweep_results_no_abs.csv  (ABS off -- run with --abs-off flag)
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("compare_abs")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.settings import RESULTS_DIR, PARAM_COLS, ML_TARGETS
from data_processing.preprocess import clean_data, engineer_features, get_feature_columns

FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ABS_ON_CSV  = RESULTS_DIR / "sweep_results_rb.csv"
ABS_OFF_CSV = RESULTS_DIR / "sweep_results_no_abs_rb.csv"


def reconstruct_from_jsons(runs_dir: Path) -> pd.DataFrame:
    """Flatten summary JSONs into a DataFrame (same logic as scenario_runner main)."""
    rows = []
    for sp in sorted(runs_dir.glob("*_summary.json")):
        d = json.loads(sp.read_text())
        row = {
            "run_id":      d["run_id"],
            "config_name": d["config"]["name"],
            "_source":     d["config"].get("_source", ""),
            "_oat_param":  d["config"].get("_oat_param", ""),
            "_oat_level":  d["config"].get("_oat_level", ""),
        }
        row.update(d["config"].get("params", {}))
        row.update(d.get("summary", {}))
        rows.append(row)
    return pd.DataFrame(rows)


def load_dataset(csv_path: Path, runs_dir: Path = None) -> pd.DataFrame:
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        log.info("Loaded %d rows from %s", len(df), csv_path.name)
        return df
    if runs_dir and runs_dir.exists():
        log.info("CSV not found, reconstructing from %s", runs_dir)
        df = reconstruct_from_jsons(runs_dir)
        df.to_csv(csv_path, index=False)
        log.info("Saved %d rows to %s", len(df), csv_path.name)
        return df
    raise FileNotFoundError(f"Neither {csv_path} nor {runs_dir} found")


def cv_r2_for_dataset(df: pd.DataFrame, targets, label: str):
    """Return a dict {target: {ridge_r2, rf_r2}} for the given dataset."""
    df = clean_data(df, use_flagged=False)
    df = engineer_features(df)
    feat_cols = get_feature_columns(df)
    log.info("%s: %d rows, %d features after cleaning", label, len(df), len(feat_cols))

    results = {}
    for target in targets:
        if target not in df.columns:
            continue
        y = df[target].values
        X = df[feat_cols].values

        ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]))
        rf    = RandomForestRegressor(n_estimators=150, max_depth=6,
                                      min_samples_leaf=8, random_state=42, n_jobs=-1)
        try:
            cv_r = cross_val_score(ridge, X, y, cv=5, scoring="r2").mean()
            cv_rf = cross_val_score(rf, X, y, cv=5, scoring="r2").mean()
        except Exception as e:
            log.warning("  %s / %s failed: %s", label, target, e)
            cv_r = cv_rf = float("nan")

        results[target] = {"ridge": cv_r, "rf": cv_rf}
        log.info("  %-35s  Ridge=%+.4f  RF=%+.4f", target, cv_r, cv_rf)
    return results


def plot_comparison(abs_on: dict, abs_off: dict, targets: list, save_path: Path):
    targets_present = [t for t in targets if t in abs_on and t in abs_off]
    n = len(targets_present)
    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, n * 0.5 + 1)))

    for ax, model_key, title in zip(axes, ["ridge", "rf"],
                                    ["Ridge (scaled)", "Random Forest"]):
        on_vals  = [abs_on[t][model_key]  for t in targets_present]
        off_vals = [abs_off[t][model_key] for t in targets_present]
        y_pos = np.arange(n)

        ax.barh(y_pos - 0.2, on_vals,  0.35, label="ABS on",  color="#2196F3", alpha=0.85)
        ax.barh(y_pos + 0.2, off_vals, 0.35, label="ABS off", color="#F44336", alpha=0.85)
        ax.axvline(0, color="black", lw=0.8)
        ax.axvline(0.5, color="green", lw=0.8, ls="--", alpha=0.5, label="R2=0.5")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(targets_present, fontsize=8)
        ax.set_xlabel("CV R2 (5-fold)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlim(-1.0, 1.1)

    fig.suptitle("ABS On vs ABS Off — CV R2 by KPI", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Comparison plot saved: %s", save_path)


def main():
    log.info("Loading ABS-on dataset...")
    df_on = load_dataset(ABS_ON_CSV, Path("data/runs_rb"))

    log.info("Loading ABS-off dataset...")
    df_off = load_dataset(ABS_OFF_CSV, Path("data/no_abs_rb"))

    targets = [t for t in ML_TARGETS if t in df_on.columns and t in df_off.columns]
    log.info("Targets to compare: %d", len(targets))

    log.info("\n--- ABS ON ---")
    abs_on = cv_r2_for_dataset(df_on, targets, "ABS-on")

    log.info("\n--- ABS OFF ---")
    abs_off = cv_r2_for_dataset(df_off, targets, "ABS-off")

    # Print delta table
    print("\n" + "=" * 80)
    print(f"{'Target':<35} {'Ridge ON':>10} {'Ridge OFF':>10} {'Delta':>8}  "
          f"{'RF ON':>8} {'RF OFF':>8} {'Delta':>8}")
    print("-" * 80)
    for t in targets:
        if t not in abs_on or t not in abs_off:
            continue
        r_on  = abs_on[t]["ridge"];  r_off = abs_off[t]["ridge"]
        rf_on = abs_on[t]["rf"];    rf_off = abs_off[t]["rf"]
        print(f"{t:<35} {r_on:>+10.4f} {r_off:>+10.4f} {r_off-r_on:>+8.4f}  "
              f"{rf_on:>+8.4f} {rf_off:>+8.4f} {rf_off-rf_on:>+8.4f}")

    # Save results CSV
    rows = []
    for t in targets:
        if t not in abs_on or t not in abs_off:
            continue
        rows.append({
            "target": t,
            "abs_on_ridge":  abs_on[t]["ridge"],
            "abs_on_rf":     abs_on[t]["rf"],
            "abs_off_ridge": abs_off[t]["ridge"],
            "abs_off_rf":    abs_off[t]["rf"],
            "delta_ridge":   abs_off[t]["ridge"] - abs_on[t]["ridge"],
            "delta_rf":      abs_off[t]["rf"]    - abs_on[t]["rf"],
        })
    out = RESULTS_DIR / "abs_comparison.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("Results saved: %s", out)

    plot_comparison(abs_on, abs_off, targets, FIGURES_DIR / "abs_on_vs_off.png")


if __name__ == "__main__":
    main()
