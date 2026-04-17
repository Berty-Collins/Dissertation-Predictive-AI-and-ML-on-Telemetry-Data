"""
run_oat_sweep.py
================
Runs only the OAT (One-At-a-Time) rows from the existing sample plan
through BeamNG and appends results to sweep_results.csv.

The 300 LHS runs already exist.  This script adds the 80 OAT rows
(indices 301-380 in sample_plan.csv) which are essential for giving
the ML/AI models structured signal to learn from.

Usage:
    python run_oat_sweep.py            # run all 80 OAT rows
    python run_oat_sweep.py --test     # run first 2 rows only (smoke test)
    python run_oat_sweep.py --resume   # skip any OAT rows already in results

Results are appended to results/sweep_results.csv with _source='oat'.
After this completes, re-run:
    python validate_data.py
    python data_processing/preprocess.py
    python models/train_evaluate.py
    python models/train_ai_models.py
"""
import argparse
import logging
import sys
import subprocess
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("oat_sweep.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

from config.settings import DATA_DIR, RESULTS_DIR
from data_collection.parameter_sweep import load_sample_plan, config_row_to_dict
from data_collection.scenario_runner import ScenarioRunner

PLAN_PATH    = DATA_DIR / "sample_plan.csv"
RESULTS_PATH = RESULTS_DIR / "sweep_results.csv"


def run_oat_sweep(test_mode: bool = False, resume: bool = True):
    # ── Load plan ─────────────────────────────────────────────────────────────
    if not PLAN_PATH.exists():
        log.error(f"sample_plan.csv not found at {PLAN_PATH}")
        log.error("Run: python run_sweep.py --n 300  to regenerate the plan first")
        sys.exit(1)

    plan = load_sample_plan(PLAN_PATH)
    log.info(f"Loaded plan: {len(plan)} total rows")

    # Select only OAT rows
    if "_source" in plan.columns:
        oat_rows = plan[plan["_source"] == "oat"].copy()
    else:
        # Fallback: OAT rows are always indices 301-380 in a standard plan
        oat_rows = plan.iloc[301:381].copy()
        log.warning("No _source column in plan — using rows 301-380 as OAT")

    log.info(f"OAT rows found: {len(oat_rows)}")

    if test_mode:
        oat_rows = oat_rows.head(2)
        log.info("TEST MODE — running first 2 OAT rows only")

    # ── Load existing results ─────────────────────────────────────────────────
    existing_results = []
    done_run_ids = set()

    if RESULTS_PATH.exists():
        existing_df = pd.read_csv(RESULTS_PATH)
        existing_results = existing_df.to_dict("records")

        # Find any OAT rows already completed (for --resume)
        if resume and "_source" in existing_df.columns:
            done_run_ids = set(
                existing_df[existing_df["_source"] == "oat"]["run_id"].tolist()
            )
            if done_run_ids:
                log.info(f"Resume: {len(done_run_ids)} OAT rows already complete, skipping")

    log.info(f"Existing results: {len(existing_results)} runs")

    # ── Run OAT rows ──────────────────────────────────────────────────────────
    new_results = []
    runner = ScenarioRunner(run_id_offset=301)

    try:
        runner.connect()

        for idx, (plan_idx, row) in enumerate(oat_rows.iterrows()):
            run_id = int(plan_idx)

            if run_id in done_run_ids:
                log.info(f"  Skipping run {run_id} (already done)")
                continue

            config = config_row_to_dict(row)
            log.info(f"  Running OAT row {run_id}  "
                     f"({idx+1}/{len(oat_rows)})  "
                     f"param={row.get('_oat_param', '?')}  "
                     f"level={row.get('_oat_level', '?')}")

            try:
                result = runner.run_single(config, idx)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                log.error(f"  Run {run_id} failed: {e}", exc_info=True)
                result = None

            if result:
                flat = {"run_id": run_id}
                flat.update({f"p_{k}": v for k, v in result["config"].items()})
                s = result["summary"]
                s.pop("config", None)
                s.pop("run_id", None)
                flat.update(s)

                # Tag as OAT — critical for downstream scripts
                flat["_source"]    = "oat"
                flat["_oat_param"] = row.get("_oat_param", "")
                flat["_oat_level"] = row.get("_oat_level", "")

                new_results.append(flat)

                # Save after every run so a crash doesn't lose progress
                all_results = existing_results + new_results
                # Ensure _source column exists on old rows too
                for r in all_results:
                    if "_source" not in r:
                        r["_source"] = "lhs"

                pd.DataFrame(all_results).to_csv(RESULTS_PATH, index=False)
                log.info(f"  Saved. Total in file: {len(all_results)}")

    except KeyboardInterrupt:
        log.info("Sweep interrupted — progress saved.")
    finally:
        runner.disconnect()
        subprocess.run(
            ["taskkill", "/F", "/IM", "BeamNG.tech.x64.exe"],
            capture_output=True
        )

    log.info(f"\nOAT sweep complete.")
    log.info(f"  New OAT runs completed : {len(new_results)}")
    log.info(f"  Total rows in CSV      : {len(existing_results) + len(new_results)}")
    log.info(f"\nNext steps:")
    log.info(f"  1. python validate_data.py")
    log.info(f"  2. python models/train_evaluate.py")
    log.info(f"  3. python models/train_ai_models.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run OAT rows through BeamNG and append to sweep_results.csv"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode — run first 2 OAT rows only"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Re-run all OAT rows even if already in results"
    )
    args = parser.parse_args()

    run_oat_sweep(test_mode=args.test, resume=not args.no_resume)