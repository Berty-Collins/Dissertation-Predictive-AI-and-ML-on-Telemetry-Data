"""Find correct spawn rotation by checking which way car rolls after spawn."""
import time
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics
from config.settings import BEAMNG_HOME, BEAMNG_USER, BEAMNG_BINARY

SPAWN_POS = (-294.031372, 10.4073992, 120.0)
ROTATIONS = {
    "0deg":   (0.0,  0.0,  0.0,    1.0),
    "90CCW":  (0.0,  0.0,  0.7071, 0.7071),
    "180deg": (0.0,  0.0,  1.0,    0.0),
    "90CW":   (0.0,  0.0, -0.7071, 0.7071),
}

bng = BeamNGpy("localhost", 64256, home=str(BEAMNG_HOME),
               user=str(BEAMNG_USER), binary=str(BEAMNG_BINARY))
bng.open(launch=True)

scenario = Scenario("automation_test_track", "handlingcircuit1")
vehicle  = Vehicle("ego", model="etk800", license="TEST")
vehicle.attach_sensor("electrics", Electrics())
scenario.add_vehicle(vehicle, pos=SPAWN_POS, rot_quat=(0,0,0,1), cling=True)
scenario.make(bng)
bng.scenario.load(scenario)
bng.scenario.start()
time.sleep(6)
vehicle.sensors.poll()

for name, rot in ROTATIONS.items():
    vehicle.teleport(pos=SPAWN_POS, rot_quat=rot, reset=True)
    time.sleep(2)
    # Use a short set_line push toward -X (west) to see if car accepts it
    vehicle.ai.set_line([
        {"pos": (-310, 10, 119), "speed": 20},
        {"pos": (-330, 8,  118), "speed": 20},
    ], cling=True)
    time.sleep(3)
    vehicle.ai.set_mode("disabled")
    s = vehicle.state
    pos = s.get("pos", [0,0,0]) if s else [0,0,0]
    print(f"{name:8s}  pos=({pos[0]:.1f},{pos[1]:.1f})  "
          f"moved_west={'YES' if pos[0] < -296 else 'no'}")
    time.sleep(1)

bng.close()