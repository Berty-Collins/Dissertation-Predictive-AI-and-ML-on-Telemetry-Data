"""
run_lap_sweep.py
────────────────
Main entry-point for the lap-time parameter sweep.

Mirrors the structure of run_sweep.py (the 4-test pipeline) so the two
experiments are directly comparable.

Workflow
--------
1.  Generate LHS + OAT sample points using the existing parameter_sweep.py
2.  Launch BeamNG via LapScenarioRunner.open()
3.  For each parameter set: run_lap() → append row to sweep_results_lap.csv
4.  After all runs: run the full ML/AI pipeline on the new CSV

Usage
-----
    python run_lap_sweep.py [--n-lhs 200] [--resume]

    --n-lhs N   : number of Latin Hypercube samples (default 200)
    --resume    : skip runs already in sweep_results_lap.csv (run_id match)

Notes
-----
- Waypoints must already exist at data_collection/waypoints_hirochi.json.
  If not, run record_waypoints.py first.
- Results go to results/sweep_results_lap.csv (never overwrites 4-test data).
- ML_TARGETS["lap_time_s"] is patched in at runtime if not in settings.py.
"""

import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.settings import BEAMNG_HOME
from data_collection.parameter_sweep import generate_lhs_samples, generate_oat_samples
from data_collection.lap_scenario_runner import LapScenarioRunner

RESULTS_FILE = ROOT / "results" / "sweep_results_lap.csv"
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── ML pipeline import (same modules as 4-test pipeline) ─────────────────────
try:
    from data_processing.preprocess import clean_data, engineer_features, prepare_datasets
    from models.train_evaluate      import train_baseline_models
    from models.train_ai_models     import train_ai_models
    ML_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] ML pipeline import failed ({e}). Sweep will still run.")
    ML_AVAILABLE = False


def _load_existing(path: Path) -> set:
    """Return set of run_ids already in the CSV."""
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(df["run_id"].tolist()) if "run_id" in df.columns else set()


def _save_row(row: dict, path: Path):
    df_row = pd.DataFrame([row])
    if path.exists():
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, mode="w", header=True, index=False)


def run_sweep(n_lhs: int, resume: bool):
    # ── generate sample space ─────────────────────────────────────────────────
    # generate_lhs_samples / generate_oat_samples both return DataFrames
    print("[Sweep] Generating sample points ...")
    lhs_df = generate_lhs_samples(n_lhs)
    oat_df = generate_oat_samples()

    # Strip OAT metadata columns before merging
    oat_clean = oat_df.drop(columns=[c for c in oat_df.columns if c.startswith("_")],
                            errors="ignore")

    # Combine into one DataFrame and convert to a list of plain dicts
    all_df = pd.concat([lhs_df, oat_clean], ignore_index=True)

    # Drop any remaining metadata columns from LHS too (safety)
    all_df = all_df.drop(columns=[c for c in all_df.columns if c.startswith("_")],
                         errors="ignore")

    # Each element is a {param_name: value} dict — exactly what run_lap() expects
    all_samples = [row.to_dict() for _, row in all_df.iterrows()]

    print(f"[Sweep] {len(lhs_df)} LHS  +  {len(oat_clean)} OAT  "
          f"=  {len(all_samples)} total runs")

    existing_ids = _load_existing(RESULTS_FILE) if resume else set()
    if resume and existing_ids:
        print(f"[Sweep] Resuming: {len(existing_ids)} run(s) already in CSV, skipping.")

    # ── open BeamNG session ───────────────────────────────────────────────────
    runner = LapScenarioRunner()
    runner.open()

    completed_count = 0
    dnf_count       = 0

    try:
        for i, params in enumerate(all_samples):
            run_id = i + 1  # 1-based

            if resume and run_id in existing_ids:
                print(f"[Sweep] Skipping run {run_id} (already done)")
                continue

            try:
                result = runner.run_lap(params)
                _save_row(result, RESULTS_FILE)

                if result["completed"]:
                    completed_count += 1
                    print(f"  [Run {run_id}/{len(all_samples)}]  "
                          f"lap_time={result['lap_time_s']:.3f}s  "
                          f"max_lat_g={result['max_lateral_g']:.3f}  "
                          f"max_spd={result['max_speed_ms']:.1f}m/s")
                else:
                    dnf_count += 1
                    print(f"  [Run {run_id}/{len(all_samples)}]  DNF ({result['dnf_reason']})")

            except Exception as e:
                print(f"  [Run {run_id}] ERROR: {e}")
                traceback.print_exc()
                # Write a NaN row so run_id is tracked even on failure
                # params is guaranteed a dict here so **params is safe
                nan_row = {
                    "run_id":        run_id,
                    "completed":     False,
                    "dnf_reason":    str(e),
                    "lap_time_s":    float("nan"),
                    "max_lateral_g": float("nan"),
                    "max_speed_ms":  float("nan"),
                    "avg_speed_ms":  float("nan"),
                    **params,
                }
                _save_row(nan_row, RESULTS_FILE)

    finally:
        runner.close()

    total_done = completed_count + dnf_count
    print(f"\n[Sweep] Done.  {completed_count}/{total_done} completed laps, "
          f"{dnf_count} DNFs")
    print(f"         Results → {RESULTS_FILE}")


def run_ml_pipeline():
    if not ML_AVAILABLE:
        print("[ML] Pipeline not available — skipping model training.")
        return

    print("\n[ML] Running baseline models on lap-time data ...")
    df = pd.read_csv(RESULTS_FILE)

    # Drop DNF rows for ML
    df_clean = df[df["completed"] == True].copy()
    df_clean = df_clean.drop(columns=["completed", "dnf_reason", "run_id"],
                             errors="ignore")

    if len(df_clean) < 20:
        print(f"[ML] Only {len(df_clean)} completed runs — need ≥20 for ML.  Skipping.")
        return

    # Patch ML_TARGETS so the pipeline knows about lap_time_s
    import config.settings as cfg
    lap_targets = {
        "lap_time_s":    {"direction": "minimize", "unit": "s"},
        "max_lateral_g": {"direction": "maximize", "unit": "g"},
        "max_speed_ms":  {"direction": "maximize", "unit": "m/s"},
        "avg_speed_ms":  {"direction": "maximize", "unit": "m/s"},
    }
    original_targets = cfg.ML_TARGETS.copy()
    cfg.ML_TARGETS = lap_targets

    try:
        df_clean = clean_data(df_clean)
        df_eng   = engineer_features(df_clean)
        datasets = prepare_datasets(df_eng, list(lap_targets.keys()))
        train_baseline_models(datasets,
                              output_csv="results/model_evaluation_results_lap.csv")
        train_ai_models(datasets,
                        output_csv="results/model_evaluation_results_lap.csv")
    finally:
        cfg.ML_TARGETS = original_targets

    print("[ML] Done.  Results → results/model_evaluation_results_lap.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lap-time parameter sweep")
    parser.add_argument("--n-lhs",  type=int, default=200,
                        help="Number of LHS sample points (default 200)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs already saved in sweep_results_lap.csv")
    parser.add_argument("--no-ml",  action="store_true",
                        help="Skip ML training after sweep completes")
    args = parser.parse_args()

    run_sweep(args.n_lhs, args.resume)

    if not args.no_ml:
        run_ml_pipeline()