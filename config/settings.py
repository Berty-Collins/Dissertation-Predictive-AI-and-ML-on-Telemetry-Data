"""
config/settings.py
Central configuration for the dissertation ML pipeline.
All paths and constants derived from project root.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"
MODELS_DIR   = PROJECT_ROOT / "models" / "saved"
LOG_DIR      = PROJECT_ROOT / "logs"

SWEEP_CSV       = RESULTS_DIR / "sweep_results_no_abs_rb.csv"  # primary: ABS-off rev-build (better signal)
SWEEP_CSV_ABS   = RESULTS_DIR / "sweep_results_rb.csv"         # secondary: ABS-on rev-build
CLEAN_CSV       = RESULTS_DIR / "clean_data.csv"               # preprocess.py output

# ── ML settings ───────────────────────────────────────────────────────────────
TEST_SIZE    = 0.20
RANDOM_STATE = 42
CV_FOLDS     = 5

# Prefix used on parameter columns in sweep_results.csv
# Runner saves params directly (no prefix): spring_F, brakebias, etc.
PARAM_PREFIX = ""

# ── Target KPIs ───────────────────────────────────────────────────────────────
# Exact column names produced by scenario_runner.py → sweep_results.csv
ML_TARGETS = [
    # Launch/Brake combined test
    "launch_time_0_60_s",         # 0–60 mph time (s) — lower is faster
    "launch_peak_lon_g",          # peak longitudinal g during acceleration
    "launch_dist_3s_m",           # distance covered in first 3 s of acceleration
    "brake_stopping_distance_m",  # braking distance from 100 mph (m)
    "brake_peak_brake_g",         # peak longitudinal deceleration (g)
    # brake_yaw_rate_variance dropped: always 0 on flat gridmap (no yaw in straight-line braking)

    # Circle test
    "circle_max_lat_g",           # max lateral g — cornering grip limit
    "circle_avg_lat_g",           # average lateral g — sustained cornering
    # circle_entry_speed_ms dropped: CV=0.1%, constant across all setups
    # circle_speed_loss_ms dropped: constant/degenerate (R²=1.0 — all runs same speed loss)
    # circle_understeer_proxy dropped: derived from speed_loss, same degeneracy

    # Slalom test — ABS-on only; ABS-off slalom is unreliable (wheels lock during brake-to-stop)
    "slalom_max_lat_g",           # max lateral g through gates
    "slalom_avg_speed_ms",        # average speed through slalom (m/s)
    # slalom_max_yaw_rate / slalom_yaw_rate_variance dropped from ABS-off comparison:
    # chaotic yaw from wheel lockup under braking makes these near-random without ABS

    # Step steer test
    "step_steer_peak_yaw_rate",   # peak yaw rate during steer phase (rad/s)
    "step_steer_time_to_peak_s",  # time from steer onset to peak yaw rate (s) — response speed
    "step_steer_peak_lat_g",      # peak lateral g during steer phase
    "step_steer_yaw_overshoot",   # peak/steady-state yaw ratio — damping quality (>1 = underdamped)
    # step_steer_settle_time_s dropped: constant at 1.5 s cap (car never fully settles in window)
]

# ── Parameter column names as saved by runner ─────────────────────────────────
# These are the exact keys in PARAM_RANGES in scenario_runner.py
PARAM_COLS = [
    "spring_F",            # front spring rate (N/m)
    "spring_R",            # rear spring rate (N/m)
    "arb_spring_F",        # front anti-roll bar stiffness
    "arb_spring_R",        # rear anti-roll bar stiffness
    "camber_FR",           # front camber multiplier
    "camber_RR",           # rear camber multiplier
    "toe_FR",              # front toe multiplier
    "toe_RR",              # rear toe multiplier
    "damp_bump_F",         # front bump damping (N/m/s)
    "damp_bump_R",         # rear bump damping (N/m/s)
    "damp_rebound_F",      # front rebound damping (N/m/s)
    "damp_rebound_R",      # rear rebound damping (N/m/s)
    "brakebias",           # brake bias (higher = more front)
    "brakestrength",       # brake strength multiplier
    "lsdpreload_R",        # rear LSD preload (Nm)
    "lsdlockcoef_R",       # rear LSD accel lock coeff
    "lsdlockcoefrev_R",    # rear LSD coast lock coeff
    "lsdpreload_F",        # front LSD preload (Nm)
    "lsdlockcoef_F",       # front LSD accel lock coeff
    "tyre_pressure_F",     # front tyre PSI
    "tyre_pressure_R",     # rear tyre PSI
    # Gearing
    "gear_1",              # 1st gear ratio
    "gear_2",              # 2nd gear ratio
    "gear_3",              # 3rd gear ratio
    "gear_4",              # 4th gear ratio
    "gear_5",              # 5th gear ratio
    "gear_6",              # 6th gear ratio
]

# Expected run counts (1 baseline + 500 LHS + 54 OAT = 555 total planned)
# 27 params × 2 edges (min/max) = 54 OAT runs
# Collected so far: 1 baseline + 406 LHS = 407 (OAT pending)
EXPECTED_RUNS    = 407
EXPECTED_SOURCES = {"baseline": 1, "lhs": 406, "oat": 0}