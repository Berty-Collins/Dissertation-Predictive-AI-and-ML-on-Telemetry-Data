"""
find_jbeam_vars.py — finds etk800 JBeam vars whether packed in zip or loose.

    python find_jbeam_vars.py
"""
import re, zipfile, io
from pathlib import Path
from config.settings import BEAMNG_HOME

vehicle_dir = Path(BEAMNG_HOME) / "content" / "vehicles" / "etk800"
print(f"Directory: {vehicle_dir}")
print(f"Exists: {vehicle_dir.exists()}")

# List everything in the directory
if vehicle_dir.exists():
    all_files = list(vehicle_dir.iterdir())
    print(f"Contents ({len(all_files)} items):")
    for f in sorted(all_files):
        print(f"  {f.name}  ({f.stat().st_size} bytes)" if f.is_file() else f"  {f.name}/")
else:
    # Maybe the path is slightly different — search parent
    content_vehicles = Path(BEAMNG_HOME) / "content" / "vehicles"
    print(f"\nListing {content_vehicles}:")
    for f in sorted(content_vehicles.iterdir())[:30]:
        print(f"  {f.name}")

print("\n--- Searching for .jbeam inside zip archives ---")

def search_text(text, source_name):
    found = {}
    dollar_re = re.compile(r'\$\w+')
    for m in dollar_re.finditer(text):
        v = m.group()
        if v not in found:
            found[v] = source_name
    return found

all_vars = {}

# Check zips in vehicle dir and parent content dir
search_dirs = [vehicle_dir, vehicle_dir.parent, Path(BEAMNG_HOME) / "content"]
for search_dir in search_dirs:
    if not search_dir.exists():
        continue
    for zip_path in search_dir.rglob("*.zip"):
        if "etk800" not in zip_path.name.lower() and "etk800" not in str(zip_path).lower():
            continue
        print(f"\nFound zip: {zip_path}")
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for name in zf.namelist():
                    if name.endswith(".jbeam"):
                        print(f"  {name}")
                        text = zf.read(name).decode("utf-8", errors="ignore")
                        if '"variables"' in text:
                            print(f"    *** HAS variables BLOCK ***")
                            lines = text.splitlines()
                            for i, line in enumerate(lines):
                                if '"variables"' in line:
                                    for l in lines[max(0,i-1):min(len(lines),i+50)]:
                                        print("    " + l)
                        vars_found = search_text(text, name)
                        all_vars.update(vars_found)
        except Exception as e:
            print(f"  Error: {e}")

# Also check .zip4, .pc, .vfs formats BeamNG sometimes uses
for ext in ["*.zip4", "*.pc", "*.vfs", "*.pak"]:
    for p in Path(BEAMNG_HOME).rglob(ext):
        if "etk800" in str(p).lower():
            print(f"Found {ext}: {p}")

print(f"\n--- All $ variables found across etk800 files ---")
for v in sorted(all_vars):
    print(f"  {v}")