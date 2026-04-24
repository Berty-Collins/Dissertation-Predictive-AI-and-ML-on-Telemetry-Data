"""
run_sweep.py -- Entry point for the dissertation data collection sweep.

Delegates to data_collection/scenario_runner.py (the canonical runner).

Usage:
    python run_sweep.py                    # ABS-on sweep (rev-build)
    python run_sweep.py --abs-off          # ABS-off sweep
    python run_sweep.py --fresh            # wipe existing results and restart
    python run_sweep.py --abs-off --fresh  # fresh ABS-off sweep

For both datasets in one go, use run_both_datasets.bat instead.
"""
from data_collection.scenario_runner import main

if __name__ == "__main__":
    # Pass all command-line arguments straight through to scenario_runner.main()
    main()
