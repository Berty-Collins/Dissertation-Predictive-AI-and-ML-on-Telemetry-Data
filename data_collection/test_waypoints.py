"""Print all waypoints in handlingcircuit1 and test drive_using_waypoints."""
import time
from beamngpy import BeamNGpy, Scenario, Vehicle
from config.settings import BEAMNG_HOME, BEAMNG_USER, BEAMNG_BINARY

SPAWN_POS = (-294.031372, 10.4073992, 120.0)
SPAWN_ROT = (0.0, 0.0, 0.7071, 0.7071)

bng = BeamNGpy("localhost", 64256, home=str(BEAMNG_HOME),
               user=str(BEAMNG_USER), binary=str(BEAMNG_BINARY))
bng.open(launch=True)

scenario = Scenario("automation_test_track", "handlingcircuit1")
vehicle  = Vehicle("ego", model="etk800", license="TEST")
scenario.add_vehicle(vehicle, pos=SPAWN_POS, rot_quat=SPAWN_ROT, cling=True)
scenario.make(bng)
bng.scenario.load(scenario)
bng.scenario.start()
time.sleep(5)

print("\n=== Waypoints in handlingcircuit1 ===")
wps = scenario.find_waypoints()
for w in wps:
    print(f"  name={w.name!r:40s}  pos={getattr(w,'pos',None)}")

print(f"\nTotal: {len(wps)} waypoints")
names = [w.name for w in wps]
print(f"Names: {names}")

# Try driving — use ALL waypoint names in whatever order they appear
print(f"\nTrying drive_using_waypoints with: {names}")
vehicle.ai.drive_using_waypoints(
    wp_target_list=names,
    no_of_laps=1,
    route_speed=200,
    route_speed_mode="limit",
    aggression=1.0,
)

print("Watching for 30s ...")
for i in range(30):
    time.sleep(1)
    sensors = vehicle.poll_sensors()
    spd = ((sensors or {}).get("electrics") or {}).get("wheelspeed", 0) or 0
    pos = (vehicle.state or {}).get("pos", [0,0,0])
    print(f"  t={i+1:2d}s  {spd*3.6:.1f} km/h  pos=({pos[0]:.0f},{pos[1]:.0f})")

bng.close()