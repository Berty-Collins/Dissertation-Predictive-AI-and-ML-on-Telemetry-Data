"""
scan_etk800_zip.py
==================
Reads the key JBeam files from etk800.zip and dumps all $variable names.
Run: python scan_etk800_zip.py
"""
import re
import zipfile
from pathlib import Path

# Find etk800.zip
SEARCH_ROOTS = [
    Path(r"P:\BeamNG.tech.v0.38.3.0"),
    Path.home() / "AppData" / "Local" / "BeamNG" / "BeamNG.tech",
    ]

KEY_FILES = [
    "etk800_suspension_F.jbeam",
    "etk800_suspension_R.jbeam",
    "etk800_brakes.jbeam",
    "etk800_differential_F.jbeam",
    "etk800_differential_R.jbeam",
    "etk800.jbeam",
]

KEYWORDS = ["spring","arb","brake","bias","lsd","lock","camber","toe",
            "damper","damp","pressure","preload","stiff","adjust"]

etk_zip = None
for root in SEARCH_ROOTS:
    candidates = list(root.rglob("etk800.zip")) if root.exists() else []
    if candidates:
        etk_zip = candidates[0]
        break

if not etk_zip:
    print("etk800.zip not found — searching all zips...")
    for root in SEARCH_ROOTS:
        if root.exists():
            for zp in root.rglob("*.zip"):
                try:
                    with zipfile.ZipFile(zp) as z:
                        names = z.namelist()
                        if any("etk800_suspension_F" in n for n in names):
                            etk_zip = zp
                            break
                except Exception:
                    pass
        if etk_zip:
            break

if not etk_zip:
    print("ERROR: Could not find etk800 zip")
    exit(1)

print(f"Reading from: {etk_zip}\n")

with zipfile.ZipFile(etk_zip) as z:
    all_entries = z.namelist()

    for key in KEY_FILES:
        # find the entry
        matches = [e for e in all_entries if e.endswith(key)]
        if not matches:
            print(f"  NOT FOUND: {key}")
            continue

        entry = matches[0]
        txt   = z.read(entry).decode("utf-8", errors="ignore")

        # Find all $variable names
        vars_found = set(re.findall(r'"(\$[\w]+)"', txt))
        relevant   = sorted(v for v in vars_found
                            if any(k in v.lower() for k in KEYWORDS))

        print(f"{'='*60}")
        print(f"FILE: {key}")
        print(f"{'='*60}")
        if relevant:
            for v in relevant:
                # Find where it's defined/used
                idx = txt.find(f'"{v}"')
                ctx = txt[max(0,idx-5):idx+80].replace("\n","  ").strip()
                print(f"  {v}")
                print(f"    → {ctx[:100]}")
        else:
            print("  (no relevant variables found)")
        print()