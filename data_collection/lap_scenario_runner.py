"""
lap_scenario_runner.py
──────────────────────
Lap-time data collection for the dissertation parameter sweep.

Driving approach
----------------
Uses vehicle.ai.set_mode("span") with set_aggression(1.0) — the built-in
BeamNG road-network AI driving flat-out around the circuit.

Why NOT set_script():
  • Script waypoints use 't' timestamps that pace the AI to a reference speed,
    meaning setups aren't actually driving as fast as possible.
  • Recording a good reference lap is fiddly and the AI still doesn't
    faithfully hit every apex.
  • Span AI at aggression=1.0 drives the road network at maximum capability
    for whatever vehicle setup is loaded — which is exactly what lap time
    as a performance metric requires.

Lap detection
-------------
Start-finish line = spawn position on Hirochi Raceway pit-lane.
  1. Wait for vehicle to leave the LAUNCH_RADIUS bubble → start clock.
  2. Once MIN_PROGRESS_M has been covered, watch for return to FINISH_RADIUS.
  3. On return → stop clock, record lap_time_s.
  4. If MAX_LAP_TIME_S exceeded → DNF, NaN in results.

ESC
---
set_esc_mode('off') called both before and after set_part_config() because
part resets in BeamNG 0.38 can silently re-enable stability control.

KPIs
----
  lap_time_s      — primary metric (seconds)
  max_lateral_g   — peak lateral g from Electrics sensor
  max_speed_ms    — peak speed (m/s)
  avg_speed_ms    — mean speed over full lap (m/s)
"""

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics  # State is auto-attached; Electrics is not

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import BEAMNG_HOME

# ── circuit ───────────────────────────────────────────────────────────────────
MAP_NAME      = "automation_test_track"  # handling circuit 1
VEHICLE_MODEL = "etk800"
# Start/finish = Checkpoint_StartFinish from handlingcircuit1.prefab
SPAWN_POS: tuple = (-299.039795, 10.3676872, 118.655609)

# ── lap detection ─────────────────────────────────────────────────────────────
LAUNCH_RADIUS  = 20.0   # m  — leave this bubble to START the clock
FINISH_RADIUS  = 18.0   # m  — enter this bubble to STOP the clock
MIN_PROGRESS_M = 300.0  # m  — minimum distance before finish is checked
MAX_LAP_TIME_S = 300    # s  — DNF if lap isn't completed within this

# ── physics / polling ─────────────────────────────────────────────────────────
PHYSICS_HZ   = 40
TELEMETRY_HZ = 20
SETTLE_STEPS = 80   # steps after recover() before releasing the AI
GRAVITY_MS2  = 9.81

# ── 17-parameter v2 space → JBeam variable names ─────────────────────────────
# (lua_only=True means applied via part config vars rather than direct JBeam)
PARAM_MAP: Dict[str, tuple] = {
    "brake_strength":       ("brakestrength",   False),
    "brake_bias":           ("brakebias",        False),
    "lsd_preload_r":        ("lsdpreload_R",     False),
    "lsd_lock_r":           ("lsdlockcoef_R",    False),
    "lsd_lock_coast_r":     ("lsdlockcoefrev_R", False),
    "lsd_preload_f":        ("lsdpreload_F",     False),
    "lsd_lock_f":           ("lsdlockcoef_F",    False),
    "spring_front":         ("spring_F",         False),
    "spring_rear":          ("spring_R",         False),
    "arb_front":            ("arb_spring_F",     False),
    "arb_rear":             ("arb_spring_R",     False),
    "camber_front":         ("camber_FR",        False),
    "camber_rear":          ("camber_RR",        False),
    "toe_front":            ("toe_FR",           False),
    "toe_rear":             ("toe_RR",           False),
    "tyre_pressure_front":  ("tirePressure_F",   True),
    "tyre_pressure_rear":   ("tirePressure_R",   True),
}


def _dist(a: tuple, b: tuple) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


class LapScenarioRunner:
    """
    Persistent BeamNG session that runs one vehicle configuration per call.

    Usage
    -----
        runner = LapScenarioRunner()
        runner.open()
        try:
            for params in sweep:
                result = runner.run_lap(params)
                save(result)
        finally:
            runner.close()
    """

    def __init__(self):
        self._bng: Optional[BeamNGpy]    = None
        self._vehicle: Optional[Vehicle] = None
        self._run_index = 0

    # ── session lifecycle ─────────────────────────────────────────────────────

    def open(self):
        """Launch BeamNG and load the circuit.  Call once before the sweep."""
        print(f"[LapRunner] Launching BeamNG from {BEAMNG_HOME} ...")
        self._bng = BeamNGpy("localhost", 64256, home=str(BEAMNG_HOME))
        self._bng.open(launch=True)

        scenario = Scenario(MAP_NAME, "lap_sweep")

        self._vehicle = Vehicle("ego", model=VEHICLE_MODEL, license="LAP")

        # Spawn position found by find_spawn.py — confirmed above ground
        scenario.add_vehicle(
            self._vehicle,
            pos      = (-296.944921, 10.3903273, 119.54186),  # from defaultStartPosition
            rot_quat = (-0.00564688444, 0.00564688584, 0.707084298, 0.707084239),
        )
        scenario.make(self._bng)

        self._bng.load_scenario(scenario)
        self._bng.start_scenario()

        # Load handling circuit prefab via Lua so named waypoints exist in world
        self._bng.queue_lua_command(
            'scenetree.loadPrefabIntoScene('
            '"levels/automation_test_track/quickrace/handlingcircuit1.prefab", true)'
        )


        # Attach Electrics explicitly — it is NOT auto-attached in beamngpy 1.35
        # (State IS auto-attached, so do NOT attach it again or you get a duplicate error)
        self._vehicle.attach_sensor("electrics", Electrics())

        # Let physics settle then record actual position as start/finish
        self._bng.step(80)
        self._vehicle.sensors.poll()

        print(f"[LapRunner] Start/finish line at: "
              f"({SPAWN_POS[0]:.1f}, {SPAWN_POS[1]:.1f}, {SPAWN_POS[2]:.1f})")
        print("[LapRunner] Session ready.")

    def close(self):
        if self._bng:
            self._bng.close()
            self._bng = None
        print("[LapRunner] Closed.")

    # ── parameter injection ───────────────────────────────────────────────────

    def _apply_params(self, params: Dict[str, float]):
        """Push all 17 parameters into the vehicle."""
        jbeam_vars: Dict[str, float] = {}
        psi_f = params.get("tyre_pressure_front", 29.0)
        psi_r = params.get("tyre_pressure_rear",  29.0)

        for name, value in params.items():
            entry = PARAM_MAP.get(name)
            if entry is None:
                continue
            jvar, lua_only = entry
            if not lua_only:
                jbeam_vars[jvar] = value

        # Apply chassis / suspension / brake vars
        if jbeam_vars:
            self._vehicle.set_part_config({"vars": jbeam_vars})

        # Apply tyre pressure (PSI → bar)
        self._vehicle.set_part_config({"vars": {
            "tirePressure_F": round(psi_f * 0.0689476, 4),
            "tirePressure_R": round(psi_r * 0.0689476, 4),
        }})

    # ── core run ──────────────────────────────────────────────────────────────

    def run_lap(self, params: Dict[str, float]) -> Dict[str, Any]:
        """
        Apply setup, drive one flat-out lap with span AI, return KPIs.

        Returns
        -------
        dict with keys:
            run_id, completed, dnf_reason,
            lap_time_s, max_lateral_g, max_speed_ms, avg_speed_ms,
            <all 17 input params>
        """
        self._run_index += 1
        run_id = self._run_index
        print(f"\n[LapRunner] ── Run {run_id} ──────────────────────────────────")

        # Apply params — set_part_config() reloads the vehicle back to spawn
        self._apply_params(params)
        self._bng.step(SETTLE_STEPS)

        # Disable ESC after part reload
        self._vehicle.set_esc_mode("off")

        # ── waypoint-following AI at max aggression ───────────────────────────
        # Navigate to the three circuit checkpoints in order, looping.
        # This guarantees the correct route even at road junctions.
        self._vehicle.ai.set_mode("span")
        self._vehicle.ai.set_aggression(1.0)
        self._vehicle.ai.set_route([
            "Split1",
            "Split2",
            "Checkpoint_StartFinish",
        ])

        # ── telemetry loop ────────────────────────────────────────────────────
        spawn          = SPAWN_POS
        in_zone        = True    # starts True — sitting at spawn
        timing         = False
        lap_start      = None
        lap_time_s     = None
        total_dist     = 0.0
        last_pos: Optional[tuple] = None
        speed_samples: List[float] = []
        max_speed_ms   = 0.0
        max_lateral_g  = 0.0
        completed      = False
        dnf_reason     = None

        poll_steps   = max(1, int(PHYSICS_HZ / TELEMETRY_HZ))
        wall_deadline = time.time() + MAX_LAP_TIME_S + 20  # safety ceiling

        while time.time() < wall_deadline:
            self._bng.step(poll_steps)
            self._vehicle.sensors.poll()

            s    = self._vehicle.sensors["state"].data
            elec = self._vehicle.sensors["electrics"].data

            pos     = tuple(s["pos"])
            vel     = s["vel"]
            speed   = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
            d_spawn = _dist(pos, spawn)

            # Lateral g from Electrics sensor (m/s² smoothed → g)
            # accYSmooth is the lateral axis in BeamNG 0.38 electrics data
            raw_lat = (elec.get("accYSmooth")
                       or elec.get("accY")
                       or 0.0)
            lat_g = abs(float(raw_lat)) / GRAVITY_MS2

            # Accumulate telemetry
            speed_samples.append(speed)
            if speed > max_speed_ms:
                max_speed_ms = speed
            if lat_g > max_lateral_g:
                max_lateral_g = lat_g
            if last_pos:
                total_dist += _dist(pos, last_pos)
            last_pos = pos

            # ── state machine ─────────────────────────────────────────────────
            if in_zone:
                # Waiting for the car to pull away from the start line
                if d_spawn > LAUNCH_RADIUS:
                    in_zone   = False
                    timing    = True
                    lap_start = time.time()
                    print(f"  Clock started — left start zone "
                          f"(dist so far: {total_dist:.0f} m)")

            elif timing:
                elapsed = time.time() - lap_start

                if elapsed > MAX_LAP_TIME_S:
                    dnf_reason = f"timeout"
                    print(f"  ✗ DNF — {elapsed:.0f} s exceeded {MAX_LAP_TIME_S} s limit")
                    break

                # Lap complete once we've done enough distance AND returned
                if total_dist > MIN_PROGRESS_M and d_spawn < FINISH_RADIUS:
                    lap_time_s = elapsed
                    completed  = True
                    print(f"  ✓  {lap_time_s:.3f} s  |  "
                          f"max_lat_g={max_lateral_g:.3f}  |  "
                          f"max_spd={max_speed_ms:.1f} m/s  |  "
                          f"avg_spd={sum(speed_samples)/len(speed_samples):.1f} m/s")
                    break

        else:
            dnf_reason = "wall_clock_exceeded"
            print("  ✗ DNF — wall clock exceeded")

        # Stop AI immediately after lap
        self._vehicle.ai.set_mode("disabled")

        avg_speed = (sum(speed_samples) / len(speed_samples)
                     if speed_samples else 0.0)

        return {
            "run_id":         run_id,
            "completed":      completed,
            "dnf_reason":     dnf_reason,
            "lap_time_s":     lap_time_s if completed else float("nan"),
            "max_lateral_g":  round(max_lateral_g, 4),
            "max_speed_ms":   round(max_speed_ms, 4),
            "avg_speed_ms":   round(avg_speed, 4),
            **params,
        }