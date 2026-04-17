"""
build_circuit_path.py
Extracts a dense, ordered circuit path from BeamNG's AI road network.
Traces from spawn around the handling circuit and saves as ai_script.json.
"""
import json, zipfile
import numpy as np
from pathlib import Path
from config.settings import DATA_DIR

ZIP = r"P:\BeamNG.tech.v0.37.6.0\content\levels\automation_test_track.zip"
SPAWN = np.array([-294.031, 10.407])

# Load all road nodes
roads = []
with zipfile.ZipFile(ZIP) as z:
    for fname in z.namelist():
        if 'decal_road/ai' in fname and fname.endswith('.json'):
            for line in z.read(fname).decode('utf-8', errors='ignore').splitlines():
                try:
                    obj = json.loads(line.strip())
                    if 'nodes' in obj:
                        roads.append(obj)
                except: pass

all_nodes = np.array([[n[0],n[1],n[2]] for r in roads for n in r['nodes']])
print(f"Total nodes: {len(all_nodes)}")

# Filter to handling circuit area only
mask = ((all_nodes[:,0] > -750) & (all_nodes[:,0] < -260) &
        (all_nodes[:,1] > -200) & (all_nodes[:,1] < 60))
circuit_nodes = all_nodes[mask]
print(f"Circuit area nodes: {len(circuit_nodes)}")

# Build ordered path using nearest-neighbour starting from spawn
# Start at the node closest to a point just ahead of spawn (westward)
start_target = np.array([-320.0, 8.0])
dists = np.linalg.norm(circuit_nodes[:,:2] - start_target, axis=1)
start_idx = np.argmin(dists)

used = np.zeros(len(circuit_nodes), dtype=bool)
ordered = [circuit_nodes[start_idx]]
used[start_idx] = True

# Greedy nearest-neighbour chain
for _ in range(len(circuit_nodes) - 1):
    current = ordered[-1][:2]
    remaining = np.where(~used)[0]
    if len(remaining) == 0:
        break
    dists = np.linalg.norm(circuit_nodes[remaining, :2] - current, axis=1)

    # Only consider nodes within 40m (prevents jumping across the circuit)
    close = remaining[dists < 40]
    if len(close) == 0:
        break

    best = close[np.argmin(dists[dists < 40])]
    ordered.append(circuit_nodes[best])
    used[best] = True

print(f"Ordered path: {len(ordered)} nodes")

# Check coverage
xs = [p[0] for p in ordered]
ys = [p[1] for p in ordered]
print(f"X: {min(xs):.0f} to {max(xs):.0f}")
print(f"Y: {min(ys):.0f} to {max(ys):.0f}")

# Check it reaches Split1 (-692, -105) and Split2 (-398, -73)
split1 = np.array([-692.5, -105.3])
split2 = np.array([-398.5, -73.6])
path_xy = np.array([[p[0],p[1]] for p in ordered])
d1 = np.min(np.linalg.norm(path_xy - split1, axis=1))
d2 = np.min(np.linalg.norm(path_xy - split2, axis=1))
print(f"Nearest to Split1: {d1:.1f}m  Split2: {d2:.1f}m")

# Save
script = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "speed": 60.0}
          for p in ordered]
DATA_DIR.mkdir(parents=True, exist_ok=True)
out = DATA_DIR / "ai_script.json"
with open(out, "w") as f:
    json.dump(script, f, indent=2)
print(f"\nSaved {len(script)} points to {out}")