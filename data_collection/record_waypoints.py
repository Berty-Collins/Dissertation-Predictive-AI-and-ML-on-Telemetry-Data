"""
record_waypoints.py
───────────────────
Step 1 of the lap-time pipeline.

Run this ONCE before any sweep.  It spawns the ETK 800 on Hirochi Raceway,
lets BeamNG's built-in span AI drive one lap while this script records the
vehicle's x/y/z position every SAMPLE_INTERVAL seconds, then saves the
waypoints to data_collection/waypoints_hirochi.json.

The saved waypoints are used by LapScenarioRunner for every subsequent run
so every setup follows the identical racing line.

Usage
-----
    python record_waypoints.py [--laps 1] [--speed 60] [--out data_collection/waypoints_hirochi.json]

Requirements
------------
    beamngpy 1.35, BeamNG.tech 0.38.x
"""

import argparse
import json
import math
import time
from pathlib import Path

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics, State

# ── project root so we can import config ─────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import BEAMNG_HOME

# ── constants ─────────────────────────────────────────────────────────────────
MAP_NAME        = "hirochi_raceway"
VEHICLE_MODEL   = "etk800"
VEHICLE_CONFIG  = "vehicles/etk800/etk800_coupe.pc"   # adjust if needed
SPAWN_POS       = (-391.0, -206.0, 25.5)              # pit-lane start, Hirochi
SPAWN_ROT       = (0.0, 0.0, 0.69, 0.72)             # quaternion (facing track)
SAMPLE_INTERVAL = 0.25    # seconds between waypoint samples
MIN_DIST        = 1.0     # metres — skip duplicate points closer than this
LAP_COMPLETE_DIST = 15.0  # metres from start to trigger lap-complete
MIN_LAP_PROGRESS = 200.0  # metres driven before we look for lap complete

def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

def _cumulative_dist(pts):
    total = 0.0
    for i in range(1, len(pts)):
        total += _dist(pts[i-1], pts[i])
    return total


def record(n_laps: int, target_speed_kph: float, out_path: Path):
    bng = BeamNGpy("localhost", 64256, home=str(BEAMNG_HOME))
    bng.open(launch=True)

    scenario = Scenario(MAP_NAME, "waypoint_recorder")
    vehicle  = Vehicle("recorder", model=VEHICLE_MODEL, license="REC")
    scenario.add_vehicle(vehicle, pos=SPAWN_POS, rot_quat=SPAWN_ROT)
    scenario.make(bng)

    bng.load_scenario(scenario)
    bng.start_scenario()

    # Attach sensors
    state = State()
    vehicle.attach_sensor("state", state)

    vehicle.recover()
    bng.step(30)  # let physics settle

    # Disable ESC for recording run too (consistent vehicle behaviour)
    vehicle.set_esc_mode("off")

    # Let span AI drive: it follows the road network at the requested speed
    vehicle.ai.set_mode("span")
    vehicle.ai.set_speed(target_speed_kph / 3.6)   # m/s
    vehicle.ai.set_aggression(0.5)

    waypoints = []   # list of (x, y, z)
    start_pos = None
    laps_done = 0
    total_dist = 0.0
    last_pos   = None

    print(f"[recorder] Recording {n_laps} lap(s) on {MAP_NAME} ...")
    print("  Press Ctrl+C to stop early and save what was collected.")

    try:
        while laps_done < n_laps:
            bng.step(int(SAMPLE_INTERVAL * 40))  # ~40 Hz physics steps
            vehicle.sensors.poll()

            pos = (
                state.data["pos"][0],
                state.data["pos"][1],
                state.data["pos"][2],
            )

            if start_pos is None:
                start_pos = pos

            # Skip if we haven't moved far enough from last sample
            if last_pos is not None and _dist(pos, last_pos) < MIN_DIST:
                continue

            waypoints.append(pos)
            if last_pos:
                total_dist += _dist(pos, last_pos)
            last_pos = pos

            # Detect lap completion
            if total_dist > MIN_LAP_PROGRESS and _dist(pos, start_pos) < LAP_COMPLETE_DIST:
                laps_done += 1
                print(f"  Lap {laps_done} complete — {len(waypoints)} waypoints, "
                      f"{total_dist:.0f} m driven")
                if laps_done < n_laps:
                    # Reset counters but keep accumulating waypoints for multi-lap
                    pass

    except KeyboardInterrupt:
        print("[recorder] Stopped by user.")

    bng.close()

    if len(waypoints) < 10:
        print("[recorder] ERROR: too few waypoints collected.  Drive was too short.")
        return

    # ── Annotate with time field ──────────────────────────────────────────────
    # Derive 't' from distance assuming a smooth constant-speed reference.
    # The script runner will follow the spatial path; 't' controls pacing.
    ref_speed_ms = target_speed_kph / 3.6
    scripted = []
    elapsed  = 0.0
    for i, (x, y, z) in enumerate(waypoints):
        if i > 0:
            d = _dist(waypoints[i-1], waypoints[i])
            elapsed += d / ref_speed_ms
        scripted.append({"x": x, "y": y, "z": z, "t": round(elapsed, 4)})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(scripted, f, indent=2)

    total_len = _cumulative_dist(waypoints)
    print(f"\n[recorder] Saved {len(scripted)} waypoints → {out_path}")
    print(f"           Circuit length: {total_len:.1f} m  |  "
          f"Scripted duration: {elapsed:.1f} s at {target_speed_kph} kph")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record reference waypoints on Hirochi Raceway")
    parser.add_argument("--laps",  type=int,   default=1,    help="Number of laps to record")
    parser.add_argument("--speed", type=float, default=60.0, help="Reference speed in kph")
    parser.add_argument("--out",   type=str,
                        default="data_collection/waypoints_hirochi.json",
                        help="Output JSON path")
    args = parser.parse_args()
    record(args.laps, args.speed, Path(args.out))