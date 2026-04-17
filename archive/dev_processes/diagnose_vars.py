"""
diagnose_vars.py — prints etk800 JBeam var names and tests set_part_config.
Launches BeamNG itself; no need to have it open first.

    python diagnose_vars.py
"""
import json, time
from beamngpy import BeamNGpy, Scenario, Vehicle
from config.settings import BEAMNG_HOME, BEAMNG_USER

bng = BeamNGpy("localhost", 25252, home=str(BEAMNG_HOME), user=str(BEAMNG_USER))
bng.open(launch=True)

scenario = Scenario("smallgrid", "diag")
vehicle  = Vehicle("ego", model="etk800", license="DIAG")
scenario.add_vehicle(vehicle, pos=(0, 0, 0.5), rot_quat=(0, 0, 0, 1))
scenario.make(bng)
bng.load_scenario(scenario)
bng.start_scenario()
vehicle.connect(bng)
time.sleep(4)

print("\n=== Current part config (vars section) ===")
try:
    cfg = vehicle.get_part_config()
    print(json.dumps(cfg.get("vars", {}), indent=2))
except Exception as e:
    print(f"get_part_config failed: {e}")
    cfg = {}

print("\n=== Testing set_part_config: springFront → 100000 ===")
try:
    vehicle.set_part_config({"vars": {"springFront": 100000}})
    time.sleep(2.0)
    cfg2 = vehicle.get_part_config()
    val  = cfg2.get("vars", {}).get("springFront", "NOT FOUND")
    print(f"springFront after set: {val}")
    if val == "NOT FOUND":
        print("\n!! 'springFront' not valid. All available var names:")
        for k, v in cfg2.get("vars", {}).items():
            print(f"  {k!r}: {v}")
    else:
        print("OK — var names are correct, set_part_config works.")
except Exception as e:
    print(f"set_part_config failed: {e}")
    print("All vars from initial config:")
    for k, v in cfg.get("vars", {}).items():
        print(f"  {k!r}: {v}")

bng.close()