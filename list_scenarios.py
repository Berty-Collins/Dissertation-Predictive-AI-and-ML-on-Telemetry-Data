"""List all available scenarios and their levels."""
import zipfile, json
from pathlib import Path

BEAMNG = Path(r"P:\BeamNG.tech.v0.37.6.0")

# Check all level zips for scenarios
for z_path in sorted(BEAMNG.glob("content/levels/*.zip")):
    level = z_path.stem
    try:
        with zipfile.ZipFile(z_path) as z:
            scenarios = [f for f in z.namelist() if f.endswith('.json')
                         and '/scenarios/' in f and 'info' not in f]
            if scenarios:
                print(f"\n{level}:")
                for s in scenarios:
                    name = s.split('/')[-1].replace('.json','')
                    print(f"  {name}")
    except: pass