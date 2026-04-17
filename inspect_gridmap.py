"""Inspect gm_suspension_race scenario and road nodes."""
import zipfile, json
from pathlib import Path

BEAMNG = Path(r"P:\BeamNG.tech.v0.37.6.0")
ZIP = BEAMNG / "content" / "levels" / "GridMap.zip"

with zipfile.ZipFile(ZIP) as z:
    # Print scenario file
    for f in z.namelist():
        if 'suspension_race' in f and f.endswith('.json'):
            print(f"=== {f} ===")
            print(z.read(f).decode('utf-8', errors='ignore')[:3000])
            print()