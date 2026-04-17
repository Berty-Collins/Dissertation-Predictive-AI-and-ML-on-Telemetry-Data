"""
scenario_runner.py  --  complete rewrite
BeamNG.tech v0.38.3.0  |  beamngpy v1.35  |  Python 3.11

Confirmed from sensor inspection:
  - vehicle.state has no 'speed' key; speed = sqrt(vx²+vy²+vz²) from 'vel'
  - gforces sensor returns m/s², divide by 9.81 for g units
  - electrics keys: throttle_input, brake_input, steering_input, esc_active, tcs_active, abs_active
  - vehicle starts with freezeState=True and parkingbrake=1 -- must be cleared via Lua + control()
  - resume() is the correct unpause call; sleep 0.5 s before calling it

Test design: ALL tests are purely TIME-BASED -- no speed-threshold polling loops.
  LAUNCH+BRAKE : 0->60mph full throttle, then full brake to stop
  CIRCLE       : 0->80mph, full lock circle for 15s, brake to stop
  SLALOM       : 0->60mph, 6 steering gates × 1.2s each, brake to stop
  STEP STEER   : 0->60mph, steady speed, 0.40 steer step held 2s, settle 1.5s, brake

KPIs are extracted from collected telemetry after each test.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import qmc

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Damage, Electrics, GForces, Timer

# ===============================================================
# PATHS
# ===============================================================
import argparse as _argparse
_ap = _argparse.ArgumentParser(add_help=False)
_ap.add_argument("--abs-off", action="store_true",
                 help="Disable ABS via wheels.setABSBehavior('off')")
_ap.add_argument("--rev-build", action="store_true",
                 help="Force parts reload on every run (removes+restores ABS slot) "
                      "to trigger the transmission rev-build launch procedure. "
                      "Use for BOTH ABS-on and ABS-off datasets so launch conditions match.")
_ap.add_argument("--n-runs", type=int, default=None,
                 help="Only run the first N configs (default: all)")
_ap.add_argument("--fresh", action="store_true",
                 help="Ignore any existing summary JSONs and re-run everything "
                      "(does not delete old files, just overwrites them)")
_CLI, _ = _ap.parse_known_args()

ABS_DISABLED = _CLI.abs_off        # True when --abs-off is passed
REV_BUILD    = _CLI.rev_build      # True when --rev-build is passed
N_RUNS_LIMIT = _CLI.n_runs         # None = all configs
FRESH_RUN    = _CLI.fresh          # True = ignore completed_ids, overwrite all

BEAMNG_HOME  = r"P:\BeamNG.tech.v0.38.3.0"
PROJECT_ROOT = Path(r"P:\ComputerScience\Dissertation\Dissertation")

# Output directory and CSV path encode both flags so datasets never collide:
#   (no flags)              -> data/runs/              sweep_results.csv
#   --rev-build             -> data/runs_rb/           sweep_results_rb.csv
#   --abs-off --rev-build   -> data/no_abs_rb/         sweep_results_no_abs_rb.csv
#   --abs-off               -> data/no_abs/            sweep_results_no_abs.csv
if REV_BUILD and ABS_DISABLED:
    _run_suffix  = "no_abs_rb"
    _csv_name    = "sweep_results_no_abs_rb.csv"
elif REV_BUILD:
    _run_suffix  = "runs_rb"
    _csv_name    = "sweep_results_rb.csv"
elif ABS_DISABLED:
    _run_suffix  = "no_abs"
    _csv_name    = "sweep_results_no_abs.csv"
else:
    _run_suffix  = "runs"
    _csv_name    = "sweep_results.csv"

OUTPUT_DIR   = PROJECT_ROOT / "data" / _run_suffix
RESULTS_CSV  = PROJECT_ROOT / "results" / _csv_name
RESULTS_DIR  = PROJECT_ROOT / "results"
LOG_DIR      = PROJECT_ROOT / "logs"
BEAMNG_HOST  = "localhost"
BEAMNG_PORT  = 64256

VEHICLE_MODEL = "etk800"
VEHICLE_ID    = "ego"
MAP_NAME      = "smallgrid"
SPAWN_POS     = (0.0, 0.0, 0.5)
SPAWN_ROT     = (0.0, 0.0, 0.0, 1.0)

# ===============================================================
# TEST DURATIONS  (seconds -- all time-based, no speed polling)
# ===============================================================
POLL_HZ       = 25
POLL_DT       = 1.0 / POLL_HZ

# speed-based tests -- durations are in _accel_to / _brake_to_stop
CIRCLE_DURATION = 15.0   # sustained cornering
GATE_DURATION   = 1.2    # per slalom gate
SLALOM_GATES    = 6
SETTLE_TIME     = 3.0    # between tests

CIRCLE_THROTTLE = 0.55   # was 0.35 -- car was decelerating in circle at low throttle
CIRCLE_STEER    = 1.0
SLALOM_THROTTLE = 0.50   # maintain speed through gates
SLALOM_STEER    = 0.55   # reduced -- full lock spins car without ESC

# Step steer test parameters
# Standard vehicle dynamics test: steady speed -> step steer held -> return to straight
STEP_STEER_AMOUNT    = 0.40   # steering fraction (0–1); 0.40 is ~40% lock -- aggressive enough
                               # to excite suspension dynamics without spinning the car
STEP_STEER_THROTTLE  = 0.30   # light throttle to hold ~60 mph during cornering
STEP_STEER_PRE_S     = 0.5    # straight-ahead settle before applying steer
STEP_STEER_HOLD_S    = 2.0    # duration of held steer input (reach steady state)
STEP_STEER_SETTLE_S  = 1.5    # post-steer settle phase to measure decay

G = 9.81  # gforces sensor returns m/s²

# ===============================================================
# PARAMETER SPACE
# ===============================================================
# Parameter ranges defined in PARAM_RANGES below

# -- Parameter ranges (min, max, default) -------------------------------------
# All JBeam vars go into set_part_config({'vars': {...}}).
# Tyre pressures are applied via vehicle.deflate_tire() workaround
# or set directly; kept as config metadata for the pipeline.
PARAM_RANGES = {
    # JBeam variable         (    min,      max,  default)   description
    # -- Suspension ---------------------------------------------------------
    "spring_F":              (  15000,   160000,   80000),   # front spring rate (N/m)
    "spring_R":              (  15000,   140000,   70000),   # rear spring rate (N/m)
    "arb_spring_F":          (   5000,   100000,   45000),   # front anti-roll bar stiffness
    "arb_spring_R":          (   2000,    50000,   25000),   # rear anti-roll bar stiffness
    # -- Geometry -----------------------------------------------------------
    "camber_FR":             (   0.95,     1.05,   1.002),   # front camber multiplier
    "camber_RR":             (   0.95,     1.05,   0.984),   # rear camber multiplier
    "toe_FR":                (   0.98,     1.02,  0.9997),   # front toe multiplier
    "toe_RR":                (   0.99,     1.01,  0.9975),   # rear toe multiplier
    # -- Dampers (confirmed from etk800_suspension_F/R.jbeam) ---------------
    "damp_bump_F":           (    500,    12500,    6000),   # front bump damping (N/m/s)
    "damp_bump_R":           (    500,    10000,    6000),   # rear bump damping (N/m/s)
    "damp_rebound_F":        (    500,    25000,   18000),   # front rebound damping (N/m/s)
    "damp_rebound_R":        (    500,    20000,   14000),   # rear rebound damping (N/m/s)
    # -- Brakes -------------------------------------------------------------
    "brakebias":             (   0.20,     0.90,    0.68),   # brake bias -- full JBeam range
    "brakestrength":         (   0.60,     1.00,    1.00),   # brake strength multiplier
    # -- Rear LSD -----------------------------------------------------------
    "lsdpreload_R":          (      0,      500,      80),   # rear LSD preload (Nm)
    "lsdlockcoef_R":         (   0.00,     0.80,    0.15),   # rear LSD accel lock coeff
    "lsdlockcoefrev_R":      (   0.00,     0.50,    0.01),   # rear LSD coast lock coeff
    # -- Front LSD ----------------------------------------------------------
    "lsdpreload_F":          (      0,      500,      25),   # front LSD preload (Nm)
    "lsdlockcoef_F":         (   0.00,     0.50,    0.10),   # front LSD accel lock coeff
    # -- Gearing (confirmed JBeam variables in etk_transmission.jbeam) --------
    # set_part_config({'vars': {'gear_1': ...}}) -- no $ prefix in the key
    # Defaults from JBeam; full JBeam-defined range 0.5–5.0 for all gears.
    # Note: engine power is NOT sweepable -- the base etk_engine_i6_3.0 has
    # no exposed variables; power variants require slot-swapping, not var-setting.
    "gear_1":                (    2.0,      5.0,     4.41),   # 1st gear ratio (launch)
    "gear_2":                (    1.5,      3.5,     2.31),   # 2nd gear ratio
    "gear_3":                (    1.0,      2.5,     1.54),   # 3rd gear ratio
    "gear_4":                (    0.7,      1.8,     1.18),   # 4th gear ratio
    "gear_5":                (    0.5,      1.4,     1.00),   # 5th gear ratio
    "gear_6":                (    0.5,      1.0,     0.84),   # 6th gear ratio (top speed)
    # -- Tyre Pressure (metadata only -- no JBeam var) -----------------------
    "tyre_pressure_F":       (   20.0,     40.0,    29.0),   # front tyre PSI
    "tyre_pressure_R":       (   20.0,     40.0,    29.0),   # rear tyre PSI
}

JBEAM_KEYS  = [k for k in PARAM_RANGES if not k.startswith("tyre_pressure")]
N_LHS       = 500  # Latin Hypercube runs -> 500 LHS + 1 baseline + 54 OAT = 555 total

# ── Tyre pressure ────────────────────────────────────────────────────────────
PSI_TO_KPA  = 6.89476  # 1 PSI = 6.89476 kPa (BeamNG uses kPa internally)



def _default_vars() -> Dict:
    return {k: PARAM_RANGES[k][2] for k in JBEAM_KEYS}


def _sample_to_config(sample: Dict, run_id: int, label: str) -> Dict:
    """Build a config dict from a parameter sample."""
    jbeam = {k: v for k, v in sample.items() if k in JBEAM_KEYS}
    return {
        "name":              label,
        "run_id":            run_id,
        "vars":              jbeam,
        "tyre_pressure_F":   sample.get("tyre_pressure_F", 29.0),
        "tyre_pressure_R":   sample.get("tyre_pressure_R", 29.0),
        "params":            dict(sample),   # full param record for analysis
    }


def build_configs() -> List[Dict]:
    """
    Build the full sweep:
      Run 0        : baseline (all defaults)
      Runs 1-500   : Latin Hypercube Sampling -- 500 points across all 27 params
                     LHS guarantees even coverage with no clustering or gaps
      Runs 501-554 : One-At-a-Time edge runs -- min and max of each param vs default
                     (27 params × 2 edges = 54 OAT runs)
      Total: 555 runs
    """
    configs: List[Dict] = []
    param_keys = list(PARAM_RANGES.keys())

    # -- Baseline --------------------------------------------------------------
    defaults = {k: PARAM_RANGES[k][2] for k in param_keys}
    baseline_cfg = _sample_to_config(defaults, 0, "baseline")
    baseline_cfg["_source"] = "baseline"
    baseline_cfg["_oat_param"] = ""
    configs.append(baseline_cfg)

    # -- Latin Hypercube Sampling -----------------------------------------------
    n_params = len(param_keys)
    sampler  = qmc.LatinHypercube(d=n_params, seed=42)
    samples  = sampler.random(n=N_LHS)                 # (50, n_params) in [0,1]
    lo  = np.array([PARAM_RANGES[k][0] for k in param_keys])
    hi  = np.array([PARAM_RANGES[k][1] for k in param_keys])
    scaled = qmc.scale(samples, lo, hi)

    for i, row in enumerate(scaled):
        sample = {k: float(row[j]) for j, k in enumerate(param_keys)}
        cfg = _sample_to_config(sample, i + 1, f"lhs_{i+1:03d}")
        cfg["_source"]    = "lhs"
        cfg["_oat_param"] = ""
        configs.append(cfg)

    # -- One-At-a-Time edges ---------------------------------------------------
    run_id = N_LHS + 1
    for k in param_keys:
        lo_val, hi_val, _ = PARAM_RANGES[k]
        for edge_val, edge_name in [(lo_val, "min"), (hi_val, "max")]:
            sample = dict(defaults)
            sample[k] = edge_val
            cfg = _sample_to_config(sample, run_id, f"oat_{k}_{edge_name}")
            cfg["_source"]    = "oat"
            cfg["_oat_param"] = k
            cfg["_oat_level"] = edge_name
            configs.append(cfg)
            run_id += 1

    return configs


# Generate the sweep at import time so run_single can index into it
VEHICLE_CONFIGS: List[Dict] = build_configs()

# ===============================================================
# LOGGING
# ===============================================================
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "scenario_runner.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("scenario_runner")
logging.getLogger("beamngpy").setLevel(logging.WARNING)

# ===============================================================
# TELEMETRY COLUMNS
# ===============================================================
COLUMNS = [
    # Time
    "t_elapsed", "sim_time",
    "px", "py", "pz",
    # Velocity
    "vx", "vy", "vz", "speed_ms", "speed_mph",
    # Orientation
    "qx", "qy", "qz", "qw",
    "yaw_rate_rads",
    # G-forces (in g, from GForces sensor)
    "gf_lon_g", "gf_lat_g", "gf_vert_g",
    # Powertrain
    "rpm", "gear", "wheel_speed_ms",
    # Driver inputs
    "throttle_in", "brake_in", "steering_in",
    # Safety systems (all should be 0 with ESC off)
    "esc_active", "abs_active", "tcs_active",
    # Engine & thermal
    "engine_load", "oil_temp", "water_temp",
    # Wheel & tyre
    "avg_wheel_av",
    # Suspension (from accSmooth channels)
    "acc_x_smooth", "acc_y_smooth", "acc_z_smooth",
    # Damage
    "damage",
]

# ===============================================================
# QUATERNION UTILS
# ===============================================================
def _quat_conj(q):
    return (-q[0], -q[1], -q[2], q[3])

def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    )

def yaw_rate(q_prev, q_curr, dt):
    if q_prev is None or dt < 1e-9:
        return 0.0
    dq = _quat_mul(q_curr, _quat_conj(q_prev))
    if dq[3] < 0:
        dq = tuple(-x for x in dq)
    return 2.0 * dq[2] / dt

_GEAR_MAP = {"P":0,"N":0,"D":1,"R":-1,
             "1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8}

def parse_gear(raw):
    if isinstance(raw, (int, float)): return int(raw)
    if isinstance(raw, str): return _GEAR_MAP.get(raw.strip().upper(), 0)
    return 0

# ===============================================================
# SENSOR POLLING
# ===============================================================
def poll(vehicle: Vehicle, q_prev, t_prev: float, t_start: float) -> Tuple[Dict, tuple, float]:
    vehicle.sensors.poll()
    sd  = vehicle.state
    ed  = vehicle.sensors["electrics"].data
    gfd = vehicle.sensors["gforces"].data
    dmd = vehicle.sensors["damage"].data
    tmd = vehicle.sensors["timer"].data

    t_now = time.monotonic()
    dt    = (t_now - t_prev) if t_prev > 0 else POLL_DT

    pos = sd.get("pos", [0,0,0])
    vel = sd.get("vel", [0,0,0])
    rot = sd.get("rotation", [0,0,0,1])

    # airspeed from the electrics sensor (m/s) -- more reliable for standstill
    # detection than wheelspeed which can creep on a stationary vehicle.
    spd  = float(ed.get("airspeed", 0.0))
    quat = (float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]))
    yr   = yaw_rate(q_prev, quat, dt)

    row = {
        "t_elapsed":    t_now - t_start,
        "px": pos[0], "py": pos[1], "pz": pos[2],
        "vx": vel[0], "vy": vel[1], "vz": vel[2],
        "speed_ms":     spd,
        "speed_mph":    spd / MPH_TO_MS,
        "qx": quat[0], "qy": quat[1], "qz": quat[2], "qw": quat[3],
        "yaw_rate_rads": yr,
        "gf_lon_g":     gfd.get("gx",  0.0) / G,
        "gf_lat_g":     gfd.get("gy",  0.0) / G,
        "gf_vert_g":    gfd.get("gz",  0.0) / G,
        "rpm":          float(ed.get("rpm", 0)),
        "gear":         parse_gear(ed.get("gear", 0)),
        "throttle_in":  float(ed.get("throttle_input", 0)),
        "brake_in":     float(ed.get("brake_input", 0)),
        "steering_in":  float(ed.get("steering_input", 0)),
        "esc_active":   int(bool(ed.get("esc_active",   False))),
        "abs_active":   int(bool(ed.get("abs_active",   False))),
        "tcs_active":   int(bool(ed.get("tcs_active",   False))),
        "engine_load":  float(ed.get("engine_load",     0.0)),
        "oil_temp":     float(ed.get("oil_temperature", 0.0)),
        "water_temp":   float(ed.get("water_temperature", 0.0)),
        "wheel_speed_ms": float(ed.get("wheelspeed",    0.0)),
        "avg_wheel_av": float(ed.get("avg_wheel_av",    0.0)),
        "acc_x_smooth": float(ed.get("accXSmooth",      0.0)),
        "acc_y_smooth": float(ed.get("accYSmooth",      0.0)),
        "acc_z_smooth": float(ed.get("accZSmooth",      0.0)),
        "sim_time":     float(tmd.get("time", 0.0)),
        "damage":       float(dmd.get("damage", 0)),
    }
    return row, quat, t_now

def collect(vehicle: Vehicle, duration: float, t_start: float,
            q_prev=None, t_prev: float = 0.0) -> List[Dict]:
    """Poll sensors for `duration` seconds, return list of rows."""
    rows = []
    end  = time.monotonic() + duration
    while time.monotonic() < end:
        row, q_prev, t_prev = poll(vehicle, q_prev, t_prev, t_start)
        rows.append(row)
        time.sleep(POLL_DT)
    return rows

def ctrl(vehicle: Vehicle, throttle=0.0, brake=0.0, steering=0.0):
    """Send control with parking brake always released."""
    vehicle.control(throttle=throttle, brake=brake,
                    steering=steering, parkingbrake=0)


# -- speed / timeout constants (must be defined before _accel_to default args) --
MPH_TO_MS     = 0.44704
V_60_MS       =  60.0 * MPH_TO_MS
V_80_MS       =  80.0 * MPH_TO_MS
V_100_MS      = 100.0 * MPH_TO_MS
V_120_MS      = 120.0 * MPH_TO_MS
TIMEOUT_ACCEL = 35.0
TIMEOUT_BRAKE = 25.0


def _accel_to(vehicle: Vehicle, target_ms: float,
              t_start: float, timeout: float = TIMEOUT_ACCEL
              ) -> Tuple[List[Dict], bool]:
    """
    Full throttle until `target_ms` is reached or `timeout` seconds elapse.

    Returns (rows, reached) where `reached` is False if the speed target was
    never hit within the timeout.  Always logs a warning on timeout so that
    slow / unusual setups are visible in the log without hanging the sim.
    """
    rows: List[Dict] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row, _, _ = poll(vehicle, None, 0.0, t_start)
        rows.append(row)
        if row["speed_ms"] >= target_ms:
            return rows, True
        time.sleep(POLL_DT)
    peak = max((r["speed_ms"] for r in rows), default=0.0)
    log.warning("  [TIMEOUT] target %.1f m/s (%.0f mph) not reached in %.0f s "
                "(peak %.1f m/s / %.0f mph) -- continuing with current speed",
                target_ms, target_ms / MPH_TO_MS,
                timeout, peak, peak / MPH_TO_MS)
    return rows, False


def _brake_to_stop(vehicle: Vehicle, t_start: float,
                   timeout: float = TIMEOUT_BRAKE) -> List[Dict]:
    """
    Full brake until speed < 0.2 m/s or `timeout` seconds elapse.
    Logs a warning if the car doesn't stop in time.
    """
    rows: List[Dict] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row, _, _ = poll(vehicle, None, 0.0, t_start)
        rows.append(row)
        if row["speed_ms"] < 0.2:
            # Re-send brake to prevent arcade gearbox engaging reverse
            # on a stationary car with no active inputs.
            ctrl(vehicle, brake=1.0)
            return rows
        time.sleep(POLL_DT)
    residual = rows[-1]["speed_ms"] if rows else 0.0
    log.warning("  [TIMEOUT] vehicle did not stop within %.0f s "
                "(residual %.1f m/s) -- continuing", timeout, residual)
    ctrl(vehicle, brake=1.0)
    return rows


def apply_tyre_pressure(vehicle: Vehicle, psi_front: float, psi_rear: float):
    """
    Set tyre pressure via Lua — BeamNG has no direct API for this.
    Wheels 0-1 = front axle, 2-3 = rear axle (standard 4-wheel layout).
    pressure is in kPa internally; we convert from PSI.
    """
    kpa_f = psi_front * PSI_TO_KPA
    kpa_r = psi_rear  * PSI_TO_KPA
    lua = (
        f"for i=0,1 do if wheels.wheels[i] then "
        f"wheels.wheels[i].pressure={kpa_f:.2f} end end "
        f"for i=2,3 do if wheels.wheels[i] then "
        f"wheels.wheels[i].pressure={kpa_r:.2f} end end"
    )
    vehicle.queue_lua_command(lua, False)
    log.info("  Tyre pressure set: front=%.1f PSI (%.1f kPa), rear=%.1f PSI (%.1f kPa)",
             psi_front, kpa_f, psi_rear, kpa_r)


def teleport_reset(vehicle: Vehicle, bng: BeamNGpy,
                   pos=SPAWN_POS, rot=SPAWN_ROT):
    """Teleport vehicle to pos/rot with velocity reset, then hold brake.

    If ABS_DISABLED, re-applies wheels.setABSBehavior("off") after each
    teleport as a belt-and-braces guard (in case reset restores defaults).
    """
    bng.vehicles.teleport(vehicle, pos, rot, reset=True)
    # Send brake immediately -- teleport with reset=True clears all inputs.
    # Without this, the arcade gearbox sees throttle=0/brake=0 at rest
    # during the sleep and engages reverse.
    ctrl(vehicle, brake=1.0)
    time.sleep(0.5)
    if ABS_DISABLED:
        lua_kill_abs = (
            'wheels.setABSBehavior("off"); '
            'for i=0,wheels.wheelRotatorCount-1 do '
            '  local wd=wheels.wheelRotators[i]; '
            '  if wd and wd.updateBrakeNoABS then '
            '    wd.updateBrake=wd.updateBrakeNoABS '
            '  end '
            'end; '
            'electrics.values.abs_active=false'
        )
        vehicle.queue_lua_command(lua_kill_abs, False)
    elif REV_BUILD:
        vehicle.queue_lua_command('wheels.setABSBehavior("realistic")', False)
    ctrl(vehicle, brake=1.0)


# ===============================================================
# VEHICLE INIT
# ===============================================================
def init_vehicle(vehicle: Vehicle, bng: BeamNGpy):
    """Disable ESC/TCS and optionally ABS; release parking brake.

    set_esc_mode('off') suppresses ESC and TCS via the vehicle controller.
    It pauses BeamNG 0.38, so resume() is called immediately after.

    ABS is disabled by calling wheels.setABSBehavior("off") via Lua.
    This sets the module-level absBehavior variable that gates the ABS
    branch at line 666 of wheels.lua:
        if (wd.hasABS and absBehavior ~= "off") or absBehavior == "arcade"
    Setting absBehavior="off" makes this condition always false, regardless
    of the per-wheel hasABS flag -- the correct runtime approach.

    Previous attempts that failed:
      - set_part_config({"parts": {"etk800_ABS": ""}}) -- wrong slot;
        the real ABS slot is etk_DSE_ABS (in common/etk/etk_dse_abs.jbeam)
      - wheels.wheels[i].enableABS = false -- sets JBeam source data;
        hasABS is copied from enableABS only at initWheels() time, not live
    """
    vehicle.set_esc_mode('off')
    bng.control.resume()
    # Hold brake immediately after resume -- set_esc_mode resets inputs and
    # the arcade gearbox selects reverse if throttle=0 brake=0 at rest.
    ctrl(vehicle, brake=1.0)
    log.info('  ESC/TCS disabled via set_esc_mode')

    if ABS_DISABLED:
        # Three-layer ABS kill:
        #   1. setABSBehavior("off") sets the module-level absBehavior variable so
        #      any future updateWheelBrakeMethods() call (e.g. from resetWheels on
        #      teleport) also uses "off".
        #   2. Direct per-wheel patch: set wd.updateBrake = wd.updateBrakeNoABS on
        #      every wheel, bypassing absBehavior entirely for the current frame.
        #   3. Zero electrics.values.abs_active so telemetry confirms ABS is off.
        lua_kill_abs = (
            'wheels.setABSBehavior("off"); '
            'for i=0,wheels.wheelRotatorCount-1 do '
            '  local wd=wheels.wheelRotators[i]; '
            '  if wd and wd.updateBrakeNoABS then '
            '    wd.updateBrake=wd.updateBrakeNoABS '
            '  end '
            'end; '
            'electrics.values.abs_active=false'
        )
        vehicle.queue_lua_command(lua_kill_abs, False)
        time.sleep(0.1)
        log.info('  ABS disabled: setABSBehavior("off") + direct wheel patch applied')
    elif REV_BUILD:
        vehicle.queue_lua_command('wheels.setABSBehavior("realistic")', False)
        time.sleep(0.1)
        log.info('  ABS restored via wheels.setABSBehavior("realistic")')

    # Pulse throttle to engage forward gear, then hold brake.
    vehicle.control(throttle=1.0, brake=0.0, steering=0.0, parkingbrake=0)
    time.sleep(0.3)
    ctrl(vehicle, brake=1.0)
    time.sleep(0.3)
    log.info('  Vehicle ready, brake held')

# ===============================================================
# FOUR TESTS  --  all time-based
# ===============================================================


def test_launch_brake(vehicle: Vehicle) -> Tuple[List[Dict], Dict]:
    """ESC off. 0->60mph full throttle, full brake to rest."""
    log.info("  [LAUNCH+BRAKE] 0 -> 60 mph -> 0")
    t_start = time.monotonic()

    ctrl(vehicle, throttle=1.0)
    accel_rows, reached_60 = _accel_to(vehicle, V_60_MS, t_start)

    # t_60: use actual elapsed time if 60 mph was reached, else record the
    # timeout duration so the ML pipeline sees a numeric "slow" value, not NaN.
    t_60 = next((r["t_elapsed"] for r in accel_rows if r["speed_ms"] >= V_60_MS),
                accel_rows[-1]["t_elapsed"] if accel_rows else TIMEOUT_ACCEL)

    peak_lon = max((abs(r["gf_lon_g"]) for r in accel_rows), default=0.0)
    brake_px = accel_rows[-1]["px"] if accel_rows else 0.0
    brake_py = accel_rows[-1]["py"] if accel_rows else 0.0

    # launch_dist_3s_m: distance covered in first 3 s of acceleration
    t_3s_ref = (accel_rows[0]["t_elapsed"] if accel_rows else 0.0) + 3.0
    row_3s   = next((r for r in accel_rows if r["t_elapsed"] >= t_3s_ref),
                    accel_rows[-1] if accel_rows else None)
    dist_3s  = math.hypot(
        (row_3s["px"] if row_3s else 0.0) - (accel_rows[0]["px"] if accel_rows else 0.0),
        (row_3s["py"] if row_3s else 0.0) - (accel_rows[0]["py"] if accel_rows else 0.0),
    ) if accel_rows else 0.0

    ctrl(vehicle, brake=1.0)
    brake_rows = _brake_to_stop(vehicle, t_start)

    dist   = math.hypot((brake_rows[-1]["px"] if brake_rows else 0.0) - brake_px,
                        (brake_rows[-1]["py"] if brake_rows else 0.0) - brake_py)
    peak_b = max((abs(r["gf_lon_g"]) for r in brake_rows), default=0.0)
    yr_var = float(np.var([r["yaw_rate_rads"] for r in brake_rows])) if brake_rows else 0.0

    kpis = {
        "launch_time_0_60_s":        t_60,
        "launch_peak_lon_g":         peak_lon,
        "launch_dist_3s_m":          dist_3s,
        "brake_stopping_distance_m": dist,
        "brake_peak_brake_g":        peak_b,
        "brake_yaw_rate_variance":   yr_var,
    }
    log.info("  [LAUNCH+BRAKE] KPIs: %s", kpis)
    return accel_rows + brake_rows, kpis


def test_circle(vehicle: Vehicle) -> Tuple[List[Dict], Dict]:
    """ESC off. 0->80mph, full lock 15s, brake to rest."""
    log.info("  [CIRCLE] 0 -> 80 mph -> full lock 15 s -> 0")
    t_start = time.monotonic()

    ctrl(vehicle, throttle=1.0)
    accel_rows, _ = _accel_to(vehicle, V_80_MS, t_start)
    entry_speed = accel_rows[-1]["speed_ms"] if accel_rows else 0.0

    ctrl(vehicle, throttle=CIRCLE_THROTTLE, steering=CIRCLE_STEER)
    circle_rows = collect(vehicle, CIRCLE_DURATION, t_start)

    ctrl(vehicle, brake=1.0)
    brake_rows = _brake_to_stop(vehicle, t_start)

    lat_gs = [abs(r["gf_lat_g"]) for r in circle_rows]
    # Steady-state speed = mean of last 5 rows (settled cornering phase)
    steady_speed = float(np.mean([r["speed_ms"] for r in circle_rows[-5:]])) if len(circle_rows) >= 5 else (
        float(np.mean([r["speed_ms"] for r in circle_rows])) if circle_rows else 0.0
    )
    max_lat_g    = max(lat_gs, default=0.0)
    speed_loss   = max(0.0, entry_speed - steady_speed)

    kpis = {
        "circle_max_lat_g":        max_lat_g,
        "circle_avg_lat_g":        float(np.mean(lat_gs)) if lat_gs else 0.0,
        "circle_entry_speed_ms":   entry_speed,
        "circle_speed_loss_ms":    speed_loss,
        "circle_understeer_proxy": speed_loss / (max_lat_g + 1e-6),
    }
    log.info("  [CIRCLE] KPIs: %s", kpis)
    # Keep brake held
    return accel_rows + circle_rows + brake_rows, kpis


def test_slalom(vehicle: Vehicle, bng: BeamNGpy) -> Tuple[List[Dict], Dict]:
    """ESC on. Accelerate to 60mph then weave through 6 gates."""
    log.info("  [SLALOM] enabling ESC...")
    vehicle.set_esc_mode("on")
    bng.control.resume()
    # Hold brake immediately -- set_esc_mode resets inputs and arcade gearbox
    # will select reverse if throttle=0 brake=0 at rest.
    ctrl(vehicle, brake=1.0)
    time.sleep(0.3)

    t_start = time.monotonic()

    ctrl(vehicle, throttle=1.0)
    accel_rows, _ = _accel_to(vehicle, V_60_MS, t_start)

    slalom_rows: List[Dict] = []
    for i in range(SLALOM_GATES):
        steer = SLALOM_STEER if i % 2 == 0 else -SLALOM_STEER
        ctrl(vehicle, throttle=SLALOM_THROTTLE, steering=steer)
        gate_rows = collect(vehicle, GATE_DURATION, t_start)
        slalom_rows.extend(gate_rows)

    ctrl(vehicle, brake=1.0)
    brake_rows = _brake_to_stop(vehicle, t_start)

    vehicle.set_esc_mode("off")
    bng.control.resume()
    ctrl(vehicle, brake=1.0)
    time.sleep(0.3)

    lat_gs    = [abs(r["gf_lat_g"]) for r in slalom_rows]
    yaw_rates = [r["yaw_rate_rads"] for r in slalom_rows]
    speeds    = [r["speed_ms"] for r in slalom_rows]
    kpis = {
        "slalom_max_lat_g":         max(lat_gs, default=0.0),
        "slalom_max_yaw_rate":      max((abs(y) for y in yaw_rates), default=0.0),
        "slalom_yaw_rate_variance": float(np.var(yaw_rates)) if yaw_rates else 0.0,
        "slalom_avg_speed_ms":      float(np.mean(speeds)) if speeds else 0.0,
    }
    log.info("  [SLALOM] KPIs: %s", kpis)
    # Keep brake held until run_single moves to next config
    return accel_rows + slalom_rows + brake_rows, kpis


def test_step_steer(vehicle: Vehicle) -> Tuple[List[Dict], Dict]:
    """ESC off. 0->60 mph, steady speed, step steer held 2s, return straight, brake.

    Standard vehicle dynamics step steer manoeuvre.  Measures the yaw rate
    transient response to a step steering input at constant speed.  Highly
    sensitive to spring rates, dampers, ARB stiffness, camber and toe because
    these all determine the vehicle's yaw natural frequency, damping ratio and
    peak lateral force.

    Procedure:
      1. Accelerate to 60 mph (V_60_MS) full throttle
      2. Hold straight at light throttle for STEP_STEER_PRE_S (settle)
      3. Apply STEP_STEER_AMOUNT steering step, hold for STEP_STEER_HOLD_S
      4. Return steering to 0, hold STEP_STEER_SETTLE_S (measure decay)
      5. Full brake to stop

    KPIs:
      step_steer_peak_yaw_rate  -- peak |yaw rate| during steer phase (rad/s)
      step_steer_time_to_peak_s -- time from steer onset to peak yaw rate (s)
      step_steer_peak_lat_g     -- peak |lateral g| during steer phase
      step_steer_yaw_overshoot  -- peak / steady-state yaw rate (>1 = underdamped)
      step_steer_settle_time_s  -- time after steer removal until |yaw| < 10% of peak
    """
    log.info("  [STEP STEER] 0 -> 60 mph -> step steer -> settle -> 0")
    t_start = time.monotonic()

    ctrl(vehicle, throttle=1.0)
    accel_rows, _ = _accel_to(vehicle, V_60_MS, t_start)

    # Pre-steer settle: hold straight at light throttle
    ctrl(vehicle, throttle=STEP_STEER_THROTTLE, steering=0.0)
    pre_rows = collect(vehicle, STEP_STEER_PRE_S, t_start)

    # Step steer: apply and hold
    ctrl(vehicle, throttle=STEP_STEER_THROTTLE, steering=STEP_STEER_AMOUNT)
    steer_rows = collect(vehicle, STEP_STEER_HOLD_S, t_start)

    # Post-steer settle: return to straight, measure yaw decay
    ctrl(vehicle, throttle=STEP_STEER_THROTTLE, steering=0.0)
    settle_rows = collect(vehicle, STEP_STEER_SETTLE_S, t_start)

    ctrl(vehicle, brake=1.0)
    brake_rows = _brake_to_stop(vehicle, t_start)

    # -- KPIs --
    yaw_abs = [abs(r["yaw_rate_rads"]) for r in steer_rows]
    lat_gs  = [abs(r["gf_lat_g"])      for r in steer_rows]

    peak_yaw = max(yaw_abs, default=0.0)
    peak_lat = max(lat_gs,  default=0.0)

    # Time from steer onset to peak yaw rate
    if steer_rows and peak_yaw > 1e-6:
        t_steer_start = steer_rows[0]["t_elapsed"]
        t_peak = next(
            (r["t_elapsed"] for r in steer_rows
             if abs(r["yaw_rate_rads"]) >= peak_yaw * 0.999),
            steer_rows[-1]["t_elapsed"]
        )
        time_to_peak = t_peak - t_steer_start
    else:
        time_to_peak = STEP_STEER_HOLD_S

    # Steady-state yaw = mean of last 5 rows of steer phase
    steady_yaw = float(np.mean([abs(r["yaw_rate_rads"]) for r in steer_rows[-5:]])) \
        if len(steer_rows) >= 5 else peak_yaw
    yaw_overshoot = peak_yaw / (steady_yaw + 1e-6)

    # Settle time: from steer removal to |yaw| < 10% of peak
    settle_time = STEP_STEER_SETTLE_S  # worst case: never settled within window
    if settle_rows and peak_yaw > 1e-6:
        threshold = 0.10 * peak_yaw
        t_settle_start = settle_rows[0]["t_elapsed"]
        for r in settle_rows:
            if abs(r["yaw_rate_rads"]) < threshold:
                settle_time = r["t_elapsed"] - t_settle_start
                break

    kpis = {
        "step_steer_peak_yaw_rate":  peak_yaw,
        "step_steer_time_to_peak_s": time_to_peak,
        "step_steer_peak_lat_g":     peak_lat,
        "step_steer_yaw_overshoot":  yaw_overshoot,
        "step_steer_settle_time_s":  settle_time,
    }
    log.info("  [STEP STEER] KPIs: %s", kpis)
    return accel_rows + pre_rows + steer_rows + settle_rows + brake_rows, kpis


# ===============================================================
# SAVE
# ===============================================================
def save_csv(rows: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log.info("    Saved %d rows -> %s", len(rows), path.name)

# ===============================================================
# RESUMABILITY
# ===============================================================
def completed_ids(out_dir: Path) -> set:
    ids = set()
    for p in out_dir.glob("run_*_summary.json"):
        try:
            with open(p, encoding="utf-8") as f:
                ids.add(int(json.load(f)["run_id"]))
        except Exception:
            pass
    return ids


# =======================================================================
# CONFIG VERIFICATION
# =======================================================================
def verify_config(vehicle: Vehicle, config: Dict) -> bool:
    """Read back the active part config from the vehicle and compare
    against what we tried to set. Logs any mismatches.

    Returns True if all JBeam vars match (within float tolerance),
    False if any mismatch is detected.
    """
    intended = config.get("vars", {})
    if not intended:
        log.info("  [VERIFY] No JBeam vars to verify for '%s'", config["name"])
        return True

    try:
        active = vehicle.get_part_config()
    except Exception as e:
        log.warning("  [VERIFY] get_part_config() failed: %s", e)
        return False

    # In beamngpy v1.35 get_part_config returns a nested tree.
    # The 'vars' key contains the JBeam variable overrides.
    active_vars = active.get("vars", {})
    if not active_vars:
        log.warning("  [VERIFY] Active config has no 'vars' key -- cannot verify")
        log.debug("  [VERIFY] Full active config: %s", active)
        return False

    all_ok = True
    for key, intended_val in intended.items():
        active_val = active_vars.get(key)
        if active_val is None:
            log.warning("  [VERIFY] MISSING  %-25s  intended=%-12s  active=NOT FOUND",
                        key, intended_val)
            all_ok = False
        else:
            # Float comparison with tolerance
            try:
                diff = abs(float(active_val) - float(intended_val))
                rel  = diff / (abs(float(intended_val)) + 1e-9)
                if rel > 0.001:   # >0.1% difference -> mismatch
                    log.warning(
                        "  [VERIFY] MISMATCH %-25s  intended=%-12.4g  active=%-12.4g  diff=%.4g%%",
                        key, intended_val, active_val, rel * 100)
                    all_ok = False
                else:
                    log.info(
                        "  [VERIFY] OK       %-25s  intended=%-12.4g  active=%-12.4g",
                        key, intended_val, active_val)
            except (TypeError, ValueError):
                if str(active_val) != str(intended_val):
                    log.warning("  [VERIFY] MISMATCH %-25s  intended=%s  active=%s",
                                key, intended_val, active_val)
                    all_ok = False
                else:
                    log.info("  [VERIFY] OK       %-25s  %s", key, active_val)

    if all_ok:
        log.info("  [VERIFY] All %d vars verified OK for config '%s'",
                 len(intended), config["name"])
    else:
        log.warning("  [VERIFY] Config '%s' has mismatches -- check JBeam var names",
                    config["name"])
    return all_ok

# ===============================================================
# run_single
# ===============================================================
def run_single(vehicle: Vehicle, bng: BeamNGpy,
               config: Dict, run_id: int) -> Dict:
    log.info("=" * 56)
    log.info("run_id=%04d  config='%s'", run_id, config["name"])
    log.info("=" * 56)

    # Apply JBeam config.
    # REV_BUILD mode: always remove the etk800_ABS slot to force a full parts
    # reload, which triggers the transmission rev-build launch procedure.
    # ABS behavior is then restored or killed via wheels.setABSBehavior() in
    # init_vehicle() -- so both ABS-on and ABS-off datasets share the same
    # reload / rev-build, differing only in ABS behavior.
    #
    # Normal mode (no --rev-build): vars only, no parts reload.
    # ABS is still set via wheels.setABSBehavior() in init_vehicle().
    part_cfg: Dict = {"vars": config.get("vars", _default_vars())}
    if REV_BUILD:
        # Removing the DSE ABS slot forces a full vehicle reload (rev-build)
        # AND disables the enableABS:true flag on all pressure wheels.
        # etk_DSE_ABS (in common/etk/etk_dse_abs.jbeam) is the real ABS slot
        # that sets enableABS:true -- NOT etk800_ABS (which is a different,
        # smaller slot in etk800_brakes.jbeam that was a red herring).
        # etk_DSE_ABS is a child of etk_DSE -> etk_DSE_ABS.
        # Setting it to "" removes it and leaves hasABS=false on all wheels.
        part_cfg["parts"] = {"etk_DSE_ABS": ""}
    vehicle.set_part_config(part_cfg)
    time.sleep(0.5)
    bng.control.resume()
    # Hold brake immediately -- set_part_config resets inputs to 0,0,0 and the
    # arcade gearbox selects reverse if throttle=0 brake=0 at rest.
    ctrl(vehicle, brake=1.0)
    time.sleep(1.0)

    # Verify the config was actually applied
    verify_config(vehicle, config)

    # Apply tyre pressure (no JBeam var -- must go via Lua)
    apply_tyre_pressure(
        vehicle,
        psi_front=config.get("tyre_pressure_F", 29.0),
        psi_rear=config.get("tyre_pressure_R",  29.0),
    )

    # Release parking brake and freeze state
    init_vehicle(vehicle, bng)

    # Camera
    vehicle.switch()
    bng.camera.set_player_mode(vehicle, "orbit", {})

    run_dir = OUTPUT_DIR / f"run_{run_id:04d}"

    log.info("  Test 1/4: LAUNCH + BRAKE (0->60->0)")
    teleport_reset(vehicle, bng)
    rows_lb, kpis_lb = test_launch_brake(vehicle)
    save_csv(rows_lb, run_dir / "launch_brake.csv")

    log.info("  Test 2/4: CIRCLE")
    teleport_reset(vehicle, bng)
    rows_c, kpis_c = test_circle(vehicle)
    save_csv(rows_c, run_dir / "circle.csv")

    log.info("  Test 3/4: SLALOM")
    teleport_reset(vehicle, bng)
    rows_s, kpis_s = test_slalom(vehicle, bng)
    save_csv(rows_s, run_dir / "slalom.csv")

    log.info("  Test 4/4: STEP STEER (0->60mph -> step -> settle -> 0)")
    teleport_reset(vehicle, bng)
    rows_ss, kpis_ss = test_step_steer(vehicle)
    save_csv(rows_ss, run_dir / "step_steer.csv")

    summary = {}
    for d in (kpis_lb, kpis_c, kpis_s, kpis_ss):
        summary.update(d)

    # Diagnostic: count ABS-active frames across all tests so we can verify
    # whether ABS disable is actually working (should be 0 when --abs-off).
    all_rows = rows_lb + rows_c + rows_s + rows_ss
    abs_frames = sum(1 for r in all_rows if r.get("abs_active", 0))
    summary["_diag_abs_active_frames"] = abs_frames
    if ABS_DISABLED and abs_frames > 0:
        log.warning("  [ABS CHECK] ABS still active: %d frames with abs_active=1 "
                    "(expected 0 with --abs-off)", abs_frames)
    else:
        log.info("  [ABS CHECK] abs_active frames: %d", abs_frames)

    result = {"run_id": run_id, "config": config,
              "summary": summary, "telemetry_path": str(run_dir)}

    sp = OUTPUT_DIR / f"run_{run_id:04d}_summary.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    log.info("  Summary -> %s", sp.name)
    return result

# ===============================================================
# MAIN
# ===============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if FRESH_RUN:
        done = set()
        log.info("--fresh: ignoring existing summary JSONs, all runs will execute")
    else:
        done = completed_ids(OUTPUT_DIR)
        if done:
            log.info("Resuming: skipping %d completed run IDs", len(done))

    bng = BeamNGpy(BEAMNG_HOST, BEAMNG_PORT, home=BEAMNG_HOME)
    bng.open(launch=True)
    log.info("Connected to BeamNG")

    scenario = Scenario(MAP_NAME, "dissertation_runner")
    vehicle  = Vehicle(VEHICLE_ID, model=VEHICLE_MODEL, license="DISS")
    vehicle.attach_sensor("electrics", Electrics())
    vehicle.attach_sensor("gforces",   GForces())
    vehicle.attach_sensor("damage",    Damage())
    vehicle.attach_sensor("timer",     Timer())
    scenario.add_vehicle(vehicle, pos=SPAWN_POS, rot_quat=SPAWN_ROT)
    scenario.make(bng)

    log.info("Loading scenario...")
    bng.scenario.load(scenario)   # connect() fires here, bumps are placed
    log.info("Starting scenario...")
    bng.scenario.start()

    time.sleep(0.5)
    bng.control.resume()
    log.info("Resumed. Settling %.0f s...", 2.0)
    time.sleep(2.0)

    # Initial vehicle setup
    init_vehicle(vehicle, bng)

    configs_to_run = VEHICLE_CONFIGS
    if N_RUNS_LIMIT is not None:
        configs_to_run = VEHICLE_CONFIGS[:N_RUNS_LIMIT]
        log.info("N_RUNS_LIMIT=%d: running first %d configs only", N_RUNS_LIMIT, len(configs_to_run))

    if ABS_DISABLED:
        log.info('ABS DISABLED (wheels.setABSBehavior("off") called each run)')
    if REV_BUILD:
        log.info('REV_BUILD ON (parts reload forces rev-build launch; ABS restored via setABSBehavior)')

    results = []
    for i, config in enumerate(configs_to_run):
        if i in done:
            log.info("Skipping run %04d (%s)", i, config["name"])
            continue
        try:
            results.append(run_single(vehicle, bng, config, i))
        except Exception as e:
            log.exception("run %04d FAILED: %s", i, e)
            try:
                ctrl(vehicle)
            except Exception:
                pass

    master = OUTPUT_DIR / "all_results.json"
    with open(master, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Done. Results -> %s", master)

    # Write a flat CSV of all parameter values + KPIs for easy ML ingestion
    if results:
        flat_rows = []
        for r in results:
            row = {
                "run_id":      r["run_id"],
                "config_name": r["config"]["name"],
                "_source":     r["config"].get("_source", ""),
                "_oat_param":  r["config"].get("_oat_param", ""),
                "_oat_level":  r["config"].get("_oat_level", ""),
                "abs_disabled": ABS_DISABLED,
                "rev_build":    REV_BUILD,
            }
            row.update(r["config"].get("params", {}))
            row.update(r.get("summary", {}))
            flat_rows.append(row)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        flat_keys = list(flat_rows[0].keys()) if flat_rows else []
        with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=flat_keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(flat_rows)
        log.info("Flat sweep CSV -> %s", RESULTS_CSV)

    bng.close()

if __name__ == "__main__":
    main()