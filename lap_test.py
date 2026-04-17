"""
lap_test.py
───────────
Uses set_script() with the recorded lap for guaranteed correct routing.

The 't' values assume 20 kph — so slow the AI is always ahead of schedule
and drives at maximum speed throughout. The t values control WHICH ROUTE
the AI takes, not how fast. Measured lap time comes from telemetry and
reflects true vehicle performance at full speed.

Run with:  python lap_test.py
"""
import json, math, time
from pathlib import Path
from beamngpy import BeamNGpy, Scenario, Vehicle

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.settings import BEAMNG_HOME

SPAWN_POS = (-296.944921, 10.3903273, 119.54186)
SPAWN_ROT = (-0.00564688444, 0.00564688584, 0.707084298, 0.707084239)
FINISH    = (-299.039795, 10.3676872, 118.655609)
FINISH_R  = 20.0
MIN_DIST  = 300.0

def dist3(a, b):
    return math.sqrt(sum((a[i]-b[i])**2 for i in range(3)))

# Load recorded lap and rebuild t values at 20 kph pace
# AI always drives faster than this so it's always ahead of schedule = full speed
with open("circuit_path.json") as f:
    raw = json.load(f)

PACE_MS = 20 / 3.6  # 20 kph in m/s
script = []
t = 0.0
for i, p in enumerate(raw):
    if i > 0:
        prev = raw[i-1]
        d = math.sqrt((p['x']-prev['x'])**2 + (p['y']-prev['y'])**2 + (p['z']-prev['z'])**2)
        t += d / PACE_MS
    script.append({"x": p['x'], "y": p['y'], "z": p['z'], "t": round(t, 3)})

print(f"Loaded {len(script)} waypoints, t range: 0 -> {script[-1]['t']:.0f}s at 20kph pace")
print("AI will drive flat-out (always ahead of the 20kph schedule)\n")

bng = BeamNGpy("localhost", 64256, home=str(BEAMNG_HOME))
bng.open(launch=True)

scenario = Scenario("automation_test_track", "lap_test")
vehicle  = Vehicle("ego", model="etk800")
scenario.add_vehicle(vehicle, pos=SPAWN_POS, rot_quat=SPAWN_ROT)
scenario.make(bng)
bng.load_scenario(scenario)
bng.start_scenario()
bng.step(80)

vehicle.set_esc_mode("off")
vehicle.ai.set_script(script, cling=True)

print(f"{'Time':>6}  {'Speed':>7}  {'Finish':>8}  {'Total':>8}")
print("-" * 40)

start      = time.time()
lap_start  = None
last_pos   = None
total_dist = 0.0
in_zone    = True

while time.time() - start < 300:
    bng.step(40)
    vehicle.sensors.poll()
    pos   = tuple(vehicle.sensors["state"].data["pos"])
    vel   = vehicle.sensors["state"].data["vel"]
    speed = math.sqrt(sum(v**2 for v in vel)) * 3.6
    d_fin = dist3(pos, FINISH)
    if last_pos:
        total_dist += dist3(pos, last_pos)
    last_pos = pos
    elapsed = time.time() - start
    print(f"  {elapsed:>5.1f}s  {speed:>6.1f}kph  {d_fin:>7.1f}m  {total_dist:>7.1f}m")
    if in_zone and total_dist > FINISH_R * 2:
        in_zone = False
        lap_start = time.time()
        print("  *** LAP TIMER STARTED ***\n")
    if not in_zone and total_dist > MIN_DIST and d_fin < FINISH_R:
        print(f"\n  *** LAP COMPLETE: {time.time()-lap_start:.3f}s ***\n")
        break

bng.close()