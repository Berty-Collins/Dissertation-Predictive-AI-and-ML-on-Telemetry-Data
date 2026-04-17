"""
run_sweep.py — Full dissertation data collection sweep (LHS + OAT).

Runs the complete v2 parameter sweep in one go:
  1. 1 baseline + 300 LHS runs
  2. 85 OAT runs (one parameter at a time, 5 levels each)

Total: 386 runs saved to results/sweep_results.csv

Usage:
    python run_sweep.py              # full sweep (386 runs)
    python run_sweep.py --fresh      # delete old results and start from scratch
    python run_sweep.py --lhs-only   # run LHS only (301 runs), skip OAT
    python run_sweep.py --oat-only   # run OAT only (skip if LHS already done)
    python run_sweep.py --n 5        # quick test: 5 LHS runs + OAT
"""
import argparse
import logging
import sys
import subprocess
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sweep.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

from config.settings import DATA_DIR, RESULTS_DIR
from data_collection.parameter_sweep import (
    build_full_plan, save_sample_plan, load_sample_plan, config_row_to_dict,
)
from data_collection.scenario_runner import ScenarioRunner

PLAN_PATH    = DATA_DIR / "sample_plan.csv"
RESULTS_PATH = RESULTS_DIR / "sweep_results.csv"

# How many times to attempt reconnecting after a BeamNG crash
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_WAIT_S       = 15   # seconds to wait before reconnecting


def _kill_beamng():
    subprocess.run(["taskkill", "/F", "/IM", "BeamNG.tech.x64.exe"],
                   capture_output=True)
    time.sleep(3)


def _is_connection_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(x in msg for x in [
        "10061", "10057", "connection refused", "broken pipe",
        "socket", "winerror", "not connected"
    ])


def _reconnect(runner: ScenarioRunner) -> bool:
    """Kill BeamNG, restart, rebuild scenario. Returns True on success."""
    log.warning("BeamNG connection lost -- attempting to reconnect ...")
    _kill_beamng()
    for attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
        try:
            log.info(f"  Reconnect attempt {attempt}/{MAX_RECONNECT_ATTEMPTS} "
                     f"(waiting {RECONNECT_WAIT_S}s) ...")
            time.sleep(RECONNECT_WAIT_S)
            runner.connect()
            log.info("  Reconnected successfully.")
            return True
        except Exception as e:
            log.warning(f"  Reconnect attempt {attempt} failed: {e}")
            _kill_beamng()
    log.error("All reconnect attempts failed.")
    return False


# ── LHS sweep ─────────────────────────────────────────────────────────────────

def run_lhs_sweep(n_runs: int, runner: ScenarioRunner,
                  plan: pd.DataFrame, results: list, done_ids: set) -> list:
    """Run the baseline + LHS portion of the plan (indices 0 to n_runs)."""
    lhs_plan = plan.iloc[:n_runs + 1]   # +1 for baseline at index 0
    log.info(f"LHS phase: {len(lhs_plan)} runs (1 baseline + {n_runs} LHS)")

    for i, (_, row) in enumerate(lhs_plan.iterrows()):
        run_id = i
        if run_id in done_ids:
            log.info(f"  Skipping run {run_id:04d} (already done)")
            continue

        config = config_row_to_dict(row)
        try:
            result = runner.run_single(config, i)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if _is_connection_error(e):
                log.error(f"Run {run_id:04d} connection error: {e}")
                if not _reconnect(runner):
                    log.error("Could not reconnect -- stopping sweep.")
                    return results
                result = None  # skip this run, resume from next
            else:
                log.error(f"Run {run_id:04d} failed: {e}", exc_info=True)
                result = None

        if result:
            flat = {"run_id": run_id,
                    "_source": row.get("_source", "lhs")}
            flat.update({f"p_{k}": v for k, v in result["config"].items()})
            s = result["summary"]
            s.pop("config", None); s.pop("run_id", None)
            flat.update(s)
            results.append(flat)
            _save(results)
            done_ids.add(run_id)
            log.info(f"  Saved. Total: {len(results)} runs.")

    return results


# ── OAT sweep ─────────────────────────────────────────────────────────────────

def run_oat_sweep(runner: ScenarioRunner, plan: pd.DataFrame,
                  results: list, done_ids: set) -> list:
    """Run the OAT portion of the plan (rows tagged _source=='oat')."""
    if "_source" in plan.columns:
        oat_plan = plan[plan["_source"] == "oat"].copy()
    else:
        oat_plan = plan.iloc[301:].copy()
        log.warning("No _source column — using rows 301+ as OAT")

    log.info(f"OAT phase: {len(oat_plan)} runs")

    for idx, (plan_idx, row) in enumerate(oat_plan.iterrows()):
        run_id = int(plan_idx)
        if run_id in done_ids:
            log.info(f"  Skipping OAT run {run_id} (already done)")
            continue

        config = config_row_to_dict(row)
        log.info(f"  OAT run {run_id} ({idx+1}/{len(oat_plan)})  "
                 f"param={row.get('_oat_param', '?')}  "
                 f"level={row.get('_oat_level', '?')}")
        try:
            result = runner.run_single(config, idx)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if _is_connection_error(e):
                log.error(f"OAT run {run_id} connection error: {e}")
                if not _reconnect(runner):
                    log.error("Could not reconnect -- stopping sweep.")
                    return results
                result = None  # skip this run, resume from next
            else:
                log.error(f"OAT run {run_id} failed: {e}", exc_info=True)
                result = None

        if result:
            flat = {"run_id":      run_id,
                    "_source":     "oat",
                    "_oat_param":  row.get("_oat_param", ""),
                    "_oat_level":  row.get("_oat_level", "")}
            flat.update({f"p_{k}": v for k, v in result["config"].items()})
            s = result["summary"]
            s.pop("config", None); s.pop("run_id", None)
            flat.update(s)
            results.append(flat)
            _save(results)
            done_ids.add(run_id)
            log.info(f"  Saved. Total: {len(results)} runs.")

    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(results: list):
    """Save results to CSV, tagging any untagged rows as lhs."""
    df = pd.DataFrame(results)
    if "_source" not in df.columns:
        df["_source"] = "lhs"
    df.to_csv(RESULTS_PATH, index=False)


def _load_existing() -> tuple:
    """Load existing results and return (results_list, done_ids_set)."""
    if RESULTS_PATH.exists():
        df = pd.read_csv(RESULTS_PATH)
        results  = df.to_dict("records")
        done_ids = set(df["run_id"].tolist())
        log.info(f"Resuming — {len(done_ids)} runs already complete.")
        return results, done_ids
    return [], set()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full dissertation sweep: LHS + OAT runs")
    parser.add_argument("--n",        type=int, default=300,
                        help="Number of LHS runs (default 300)")
    parser.add_argument("--fresh",    action="store_true",
                        help="Delete existing results and restart from scratch")
    parser.add_argument("--lhs-only", action="store_true",
                        help="Run LHS phase only, skip OAT")
    parser.add_argument("--oat-only", action="store_true",
                        help="Run OAT phase only, skip LHS")
    args = parser.parse_args()

    if args.fresh and RESULTS_PATH.exists():
        RESULTS_PATH.unlink()
        if PLAN_PATH.exists():
            PLAN_PATH.unlink()
        log.info("--fresh: deleted existing results and plan")

    # Generate or load sample plan
    if PLAN_PATH.exists() and not args.fresh:
        log.info(f"Loading existing plan: {PLAN_PATH}")
        plan = load_sample_plan(PLAN_PATH)
    else:
        log.info(f"Generating v2 plan: {args.n} LHS + OAT runs ...")
        plan = build_full_plan(n_lhs=args.n, oat_levels=5)
        log.info(f"  Total runs planned: {len(plan)}")
        save_sample_plan(plan, PLAN_PATH)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results, done_ids = _load_existing()

    runner = ScenarioRunner(run_id_offset=0)
    try:
        runner.connect()

        # ── Phase 1: LHS ──────────────────────────────────────────────────
        if not args.oat_only:
            log.info("\n" + "="*60)
            log.info("PHASE 1: LHS SWEEP")
            log.info("="*60)
            results = run_lhs_sweep(args.n, runner, plan, results, done_ids)
            log.info(f"LHS phase complete. {len(results)} total runs saved.")
        else:
            log.info("Skipping LHS phase (--oat-only)")

        # ── Phase 2: OAT ──────────────────────────────────────────────────
        if not args.lhs_only:
            log.info("\n" + "="*60)
            log.info("PHASE 2: OAT SWEEP")
            log.info("="*60)
            results = run_oat_sweep(runner, plan, results, done_ids)
            log.info(f"OAT phase complete. {len(results)} total runs saved.")
        else:
            log.info("Skipping OAT phase (--lhs-only)")

    except KeyboardInterrupt:
        log.info("Sweep interrupted — progress saved.")
    finally:
        runner.disconnect()
        subprocess.run(["taskkill", "/F", "/IM", "BeamNG.tech.x64.exe"],
                       capture_output=True)

    log.info("\n" + "="*60)
    log.info(f"SWEEP COMPLETE — {len(results)} runs in {RESULTS_PATH}")
    log.info("Next steps:")
    log.info("  python validate_data.py")
    log.info("  python models/train_evaluate.py")
    log.info("  python models/train_ai_models.py")
    log.info("="*60)


if __name__ == "__main__":
    main()