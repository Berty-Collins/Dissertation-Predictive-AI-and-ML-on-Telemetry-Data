"""
parameter_sweep.py  (v2 — redesigned after v1 showed near-zero parameter sensitivity)
======================================================================================
16-parameter BeamNG etk800 space redesigned around parameters the AI driver
cannot compensate for.

v1 finding: suspension parameters (springs, dampers, ARB, ride height) showed
max |r| = 0.047 with all KPI targets — effectively zero correlation.  Root cause:
BeamNG's AI driver adapts its inputs to compensate for handling changes, masking
any setup effect.  ESC was also active, further neutralising parameter differences.

v2 design principles:
  1. PHYSICAL LIMITERS first — brake strength and differential lock directly
     constrain what the vehicle can physically do regardless of AI behaviour.
  2. CORRECT RANGES — v1 had camber and toe at min=max=1.0 (never varied).
     All ranges are now taken directly from the JBeam "range" definitions.
  3. CORRECT DEFAULTS — taken from JBeam defaults, not midpoints.
  4. ESC DISABLED — must be disabled in scenario_runner before each run.
     See scenario_runner.py: add controller.setESC(False) after vehicle load.

Parameter sources (verified from etk800.zip JBeam files):
  etk800.jbeam              : $brakestrength
  etk800_brakes.jbeam       : $brakebias
  etk800_differential_F/R   : $lsdpreload, $lsdlockcoef, $lsdlockcoefrev
  etk800_suspension_F/R     : $spring, $arb_spring, $camber, $toe_RR
  etk800_suspension_F       : $toe_FR (corrected default 0.9997)

Units match JBeam $-variable values directly.
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc
from pathlib import Path

# (name, min, max, unit, category, jbeam_var, default)
# All ranges and defaults verified directly from JBeam "range" definitions
PARAM_SPACE = [
    # ── Physical limiters (AI cannot compensate) ───────────────────────────
    # Brake strength multiplier — scales ALL brake torque.
    # Lower = weaker brakes = longer stopping distance regardless of AI inputs.
    ("brake_strength",      0.60,   1.00,  "",      "BRAKES",        "brakestrength",     1.00),

    # Front/rear brake bias — physical balance of brake torque distribution.
    # Affects directional stability under braking (yaw at brake application).
    ("brake_bias",          0.52,   0.78,  "frac",  "BRAKES",        "brakebias",         0.68),

    # Rear LSD preload — initial locking torque regardless of throttle.
    # Higher = more locked diff = more oversteer tendency on corner exit.
    ("lsd_preload_r",       0.0,  500.0,   "N/m",   "DIFFERENTIAL",  "lsdpreload_R",     80.0),

    # Rear LSD power lock coefficient — additional lock proportional to engine torque.
    # Higher = more locking under acceleration.
    ("lsd_lock_r",          0.00,   0.50,  "",      "DIFFERENTIAL",  "lsdlockcoef_R",     0.15),

    # Rear LSD coast lock coefficient — locking under engine braking.
    # Affects rotation on corner entry under trailing throttle.
    ("lsd_lock_coast_r",    0.00,   0.50,  "",      "DIFFERENTIAL",  "lsdlockcoefrev_R",  0.01),

    # Front LSD preload — affects front traction and steering feel.
    ("lsd_preload_f",       0.0,  500.0,   "N/m",   "DIFFERENTIAL",  "lsdpreload_F",     25.0),

    # Front LSD power lock coefficient.
    ("lsd_lock_f",          0.00,   0.50,  "",      "DIFFERENTIAL",  "lsdlockcoef_F",     0.10),

    # ── Suspension (kept — affects vehicle dynamics, corrected ranges) ──────
    # Spring rates — use JBeam-verified ranges and defaults.
    ("spring_front",    15000, 160000,  "N/m",   "SUSPENSION",    "spring_F",          80000),
    ("spring_rear",     15000, 140000,  "N/m",   "SUSPENSION",    "spring_R",          70000),

    # Anti-roll bars — resist body roll in corners.
    ("arb_front",        5000, 100000,  "N/m",   "SUSPENSION",    "arb_spring_F",      45000),
    ("arb_rear",         2000,  50000,  "N/m",   "SUSPENSION",    "arb_spring_R",      25000),

    # ── Wheel alignment (corrected — v1 had these stuck at 1.0) ────────────
    # Front camber — JBeam default 1.002, range 0.95-1.05
    ("camber_front",     0.95,   1.05,  "mult",  "ALIGNMENT",     "camber_FR",          1.002),

    # Rear camber — JBeam default 0.984, range 0.95-1.05
    ("camber_rear",      0.95,   1.05,  "mult",  "ALIGNMENT",     "camber_RR",          0.984),

    # Front toe — JBeam default 0.9997, range 0.98-1.02
    ("toe_front",        0.98,   1.02,  "mult",  "ALIGNMENT",     "toe_FR",             0.9997),

    # Rear toe — NEW in v2. JBeam default 0.9975, range 0.99-1.01
    # Rear toe-in increases straight-line stability; toe-out increases rotation.
    ("toe_rear",         0.99,   1.01,  "mult",  "ALIGNMENT",     "toe_RR",             0.9975),

    # Tyre pressure — applied via Lua, affects grip and response.
    ("tyre_pressure_front",  24.0,  34.0, "PSI",  "TYRES",        "lua_only",           29.0),
    ("tyre_pressure_rear",   24.0,  34.0, "PSI",  "TYRES",        "lua_only",           29.0),
]

PARAM_NAMES    = [p[0] for p in PARAM_SPACE]
PARAM_MINS     = np.array([p[1] for p in PARAM_SPACE])
PARAM_MAXS     = np.array([p[2] for p in PARAM_SPACE])
PARAM_DEFAULTS = np.array([p[6] for p in PARAM_SPACE])
N_PARAMS       = len(PARAM_SPACE)
PARAM_META     = {p[0]: {"unit": p[3], "category": p[4], "jbeam_var": p[5], "default": p[6]}
                  for p in PARAM_SPACE}

# Baseline uses JBeam defaults, not midpoints
BASELINE_CONFIG = {name: float(default)
                   for name, default in zip(PARAM_NAMES, PARAM_DEFAULTS)}


def generate_lhs_samples(n_samples: int = 300, seed: int = 42) -> pd.DataFrame:
    sampler = qmc.LatinHypercube(d=N_PARAMS, seed=seed)
    scaled  = qmc.scale(sampler.random(n=n_samples), PARAM_MINS, PARAM_MAXS)
    df = pd.DataFrame(scaled, columns=PARAM_NAMES)

    # Round to sensible precision
    for col in ["spring_front", "spring_rear", "arb_front", "arb_rear"]:
        df[col] = df[col].round(0).astype(int)
    for col in ["lsd_preload_f", "lsd_preload_r"]:
        df[col] = df[col].round(1)
    for col in ["lsd_lock_f", "lsd_lock_r", "lsd_lock_coast_r"]:
        df[col] = df[col].round(3)
    for col in ["tyre_pressure_front", "tyre_pressure_rear"]:
        df[col] = df[col].round(1)
    for col in ["camber_front", "camber_rear", "toe_front", "toe_rear"]:
        df[col] = df[col].round(3)
    df["brake_strength"] = df["brake_strength"].round(3)
    df["brake_bias"]     = df["brake_bias"].round(3)

    return df


def generate_oat_samples(n_levels: int = 5) -> pd.DataFrame:
    """
    One-At-a-Time sweep: vary each parameter across n_levels evenly-spaced
    values (min to max) while all others stay at JBeam defaults.
    Total runs = N_PARAMS x n_levels = 17 x 5 = 85.
    """
    rows   = []
    levels = np.linspace(0.0, 1.0, n_levels)

    for i, name in enumerate(PARAM_NAMES):
        for frac in levels:
            row = BASELINE_CONFIG.copy()
            row[name]         = float(PARAM_MINS[i] + frac * (PARAM_MAXS[i] - PARAM_MINS[i]))
            row["_oat_param"] = name
            row["_oat_level"] = round(frac, 3)
            rows.append(row)

    df = pd.DataFrame(rows)

    for col in ["spring_front", "spring_rear", "arb_front", "arb_rear"]:
        df[col] = df[col].round(0).astype(int)
    for col in ["lsd_preload_f", "lsd_preload_r"]:
        df[col] = df[col].round(1)
    for col in ["lsd_lock_f", "lsd_lock_r", "lsd_lock_coast_r"]:
        df[col] = df[col].round(3)
    for col in ["tyre_pressure_front", "tyre_pressure_rear"]:
        df[col] = df[col].round(1)
    for col in ["camber_front", "camber_rear", "toe_front", "toe_rear"]:
        df[col] = df[col].round(3)
    df["brake_strength"] = df["brake_strength"].round(3)
    df["brake_bias"]     = df["brake_bias"].round(3)

    return df


def build_full_plan(n_lhs: int = 300, oat_levels: int = 5) -> pd.DataFrame:
    """
    Combine: 1 baseline + n_lhs LHS + N_PARAMS x oat_levels OAT.
    Default: 1 + 300 + 85 = 386 runs.
    """
    baseline  = pd.DataFrame([BASELINE_CONFIG])
    lhs       = generate_lhs_samples(n_lhs)
    oat       = generate_oat_samples(oat_levels)
    oat_clean = oat.drop(columns=["_oat_param", "_oat_level"])

    plan = pd.concat([baseline, lhs, oat_clean], ignore_index=True)
    plan["_source"] = (
            ["baseline"] + ["lhs"] * len(lhs) + ["oat"] * len(oat_clean)
    )
    return plan


def save_sample_plan(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="run_index")
    print(f"Sample plan saved: {path}  ({len(df)} runs x {N_PARAMS} parameters)")


def load_sample_plan(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="run_index")


def config_row_to_dict(row: pd.Series) -> dict:
    """Strip internal metadata columns before passing to the runner."""
    return {k: v for k, v in row.to_dict().items() if not k.startswith("_")}


def discrepancy_check(df: pd.DataFrame) -> float:
    cols = [c for c in PARAM_NAMES if c in df.columns]
    mins = np.array([PARAM_META[c]["default"] for c in cols])
    unit = (df[cols].values - PARAM_MINS[:len(cols)]) / (
            PARAM_MAXS[:len(cols)] - PARAM_MINS[:len(cols)] + 1e-9)
    return qmc.discrepancy(unit)


if __name__ == "__main__":
    plan = build_full_plan(n_lhs=300, oat_levels=5)
    print(f"\nPlan summary (v2):")
    print(f"  Parameters : {N_PARAMS}")
    print(f"  Baseline   : 1  (JBeam defaults)")
    print(f"  LHS        : {(plan['_source']=='lhs').sum()}")
    print(f"  OAT        : {(plan['_source']=='oat').sum()}  "
          f"({N_PARAMS} params x 5 levels)")
    print(f"  Total      : {len(plan)}")
    print(f"\nParameter space:")
    print(f"  {'Name':<25} {'Default':>10} {'Min':>10} {'Max':>10}  JBeam var")
    print(f"  {'-'*70}")
    for p in PARAM_SPACE:
        name, mn, mx, unit, cat, jbeam, default = p
        print(f"  {name:<25} {default:>10.4g} {mn:>10.4g} {mx:>10.4g}  {jbeam}")

    save_sample_plan(plan, Path("data/sample_plan_v2.csv"))
    print(f"\nNOTE: ESC must be disabled in scenario_runner.py before each run.")
    print(f"      Add: vehicle.queue_lua_command('controller.mainController.setESCActive(false)')")
    print(f"      after the vehicle is loaded and before the test starts.")