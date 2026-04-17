"""
build_path.py
─────────────
Builds a set_script() path by picking road positions from road_network.json
that lie along the handling circuit route, ordered from spawn to finish.

Run with:  python build_path.py  (no BeamNG needed)
Outputs:   circuit_path.json  — ready to pass to vehicle.ai.set_script()
"""
import json, math
from pathlib import Path

# All confirmed road positions near each circuit segment (from road_network.json output)
# Ordered: spawn → Split1 → Split2 → StartFinish

# We'll pick roads whose positions trace the route.
# From the data we know:
#   StartFinish area:  x≈-299, y≈10
#   Split1 area:       x≈-692, y≈-105
#   Split2 area:       x≈-398, y≈-73
# Route order: StartFinish -> Split1 -> Split2 -> StartFinish

ROUTE_SEED_POINTS = [
    # Start/finish and heading toward Split1
    (-299.039795,  10.3676872, 118.655609),   # StartFinish checkpoint
    (-304.72,      10.23,      118.58),        # road 39070
    (-320.71,      16.46,      118.48),        # road 39801
    (-323.93,      16.24,      118.36),        # road 39767
    (-332.77,      10.93,      117.97),        # road 39059
    # Heading toward Split1
    (-692.76,     -114.91,     118.25),        # road 39257 (near Split1)
    (-692.51709,  -105.293823, 118.559509),    # Split1 checkpoint
    (-687.51,     -129.34,     118.31),        # road 39479
    (-686.03,     -129.14,     118.25),        # road 39772
    # Heading toward Split2
    (-431.53,     -108.96,     118.12),        # road 39092
    (-433.89,     -108.47,     118.25),        # road 39089
    (-432.65,     -112.80,     118.20),        # road 39252
    (-398.480042,  -73.5977173, 116.873146),   # Split2 checkpoint
    (-400.61,      -62.02,     116.42),        # road 39237
    # Heading back to StartFinish
    (-402.59,      -31.83,     117.11),        # road 39259
    (-304.86,      -13.36,     116.59),        # road 39677
    (-298.82,       13.59,     118.40),        # road 39235
    (-298.95,        6.94,     118.41),        # road 39201
    (-299.039795,   10.3676872, 118.655609),   # StartFinish checkpoint
]

def dist3(a, b):
    return math.sqrt(sum((a[i]-b[i])**2 for i in range(3)))

# Build set_script() format: list of {x, y, z, t}
# t = cumulative time assuming 80 kph average
SPEED_MS = 80 / 3.6
script = []
t = 0.0
for i, (x, y, z) in enumerate(ROUTE_SEED_POINTS):
    if i > 0:
        d = dist3(ROUTE_SEED_POINTS[i-1], (x, y, z))
        t += d / SPEED_MS
    script.append({"x": x, "y": y, "z": z, "t": round(t, 3)})

out_path = Path("circuit_path.json")
with open(out_path, "w") as f:
    json.dump(script, f, indent=2)

print(f"Saved {len(script)} waypoints to {out_path}")
print(f"Estimated lap time at 80kph avg: {t:.1f}s")
print("\nPath preview:")
for wp in script:
    print(f"  ({wp['x']:.1f}, {wp['y']:.1f}, {wp['z']:.1f})  t={wp['t']:.1f}s")