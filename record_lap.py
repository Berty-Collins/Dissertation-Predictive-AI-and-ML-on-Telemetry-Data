"""
record_lap.py
─────────────
YOU drive one lap of the handling circuit manually.
This script records your position every 0.25s.
The output is used for all AI runs — the AI follows your exact line
but at maximum speed (t values are set to force flat-out pace).

Steps:
  1. Run this script — BeamNG launches and you spawn at the start line
  2. Drive one full lap of the handling circuit manually
  3. When you cross the finish line, press Ctrl+C to stop recording
  4. circuit_path.json is saved automatically

Run with:  python record_lap.py
"""
import json, math, time, signal, sys
from pathlib import Path
from beamngpy import BeamNGpy, Scenario, Vehicle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.settings import BEAMNG_HOME

SPAWN_POS = (-296.944921, 10.3903273, 119.54186)
SPAWN_ROT = (-0.00564688444, 0.00564688584, 0.707084298, 0.707084239)
OUT_PATH  = Path("circuit_path.json")
MIN_DIST  = 1.5   # m — minimum distance between recorded points

def dist3(a, b):
    return math.sqrt(sum((a[i]-b[i])**2 for i in range(3)))

bng = BeamNGpy("localhost", 64256, home=str(BEAMNG_HOME))
bng.open(launch=True)

scenario = Scenario("automation_test_track", "record_lap")
vehicle  = Vehicle("ego", model="etk800", license="REC")
scenario.add_vehicle(vehicle, pos=SPAWN_POS, rot_quat=SPAWN_ROT)
scenario.make(bng)
bng.load_scenario(scenario)
bng.start_scenario()

# Hand control to the player
bng.set_deterministic(False)
bng.step(60)

print("\n" + "="*60)
print("  DRIVE ONE LAP OF THE HANDLING CIRCUIT MANUALLY")
print("  Recording will start automatically when you move.")
print("  Press Ctrl+C when you cross the finish line.")
print("="*60 + "\n")

recorded = []
last_pos  = None
started   = False

def save_and_exit(sig=None, frame=None):
    if len(recorded) < 10:
        print("\nToo few points recorded — did you drive?")
        bng.close()
        sys.exit(1)

    # Set t values assuming flat-out pace:
    # Use very aggressive timing (half the recorded time) so the AI
    # drives as fast as physically possible along the line
    total_dist = sum(
        dist3(recorded[i], recorded[i-1]) for i in range(1, len(recorded))
    )
    # Target: AI tries to cover the circuit in 60s regardless of length
    # This forces maximum speed — the AI will push the car to its limit
    TARGET_LAP_S = 60.0
    script = []
    t = 0.0
    for i, (x, y, z) in enumerate(recorded):
        if i > 0:
            seg = dist3(recorded[i-1], recorded[i])
            t  += (seg / total_dist) * TARGET_LAP_S
        script.append({"x": x, "y": y, "z": z, "t": round(t, 3)})

    with open(OUT_PATH, "w") as f:
        json.dump(script, f, indent=2)

    print(f"\nSaved {len(script)} waypoints to {OUT_PATH}")
    print(f"Circuit length: {total_dist:.0f}m")
    print(f"AI target lap time set to: {TARGET_LAP_S}s (forces flat-out)")
    print("\nNow run lap_test.py to verify the AI follows the correct route.")
    bng.close()
    sys.exit(0)

signal.signal(signal.SIGINT, save_and_exit)

while True:
    bng.step(10)  # ~0.25s of physics
    vehicle.sensors.poll()
    pos   = tuple(vehicle.sensors["state"].data["pos"])
    vel   = vehicle.sensors["state"].data["vel"]
    speed = math.sqrt(sum(v**2 for v in vel)) * 3.6

    if not started and speed > 2.0:
        started = True
        print("Recording started — drive the full lap then press Ctrl+C at finish.")

    if started:
        if last_pos is None or dist3(pos, last_pos) >= MIN_DIST:
            recorded.append(pos)
            last_pos = pos
            if len(recorded) % 20 == 0:
                print(f"  {len(recorded)} points recorded  |  "
                      f"pos=({pos[0]:.1f}, {pos[1]:.1f})  {speed:.0f}kph")