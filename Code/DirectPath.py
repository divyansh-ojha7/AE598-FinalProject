from astropy import units as u
from astropy import time
from astropy.coordinates import solar_system_ephemeris
import numpy as np

from poliastro import iod
from poliastro.bodies import Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Sun
from poliastro.ephem import Ephem
from poliastro.maneuver import Maneuver
from poliastro.twobody import Orbit
from poliastro.util import time_range
from poliastro.twobody.sampling import TrueAnomalyBounds
from poliastro.plotting import OrbitPlotter3D

# Set up Plotly to open in browser
import plotly.io as pio
pio.renderers.default = "browser"

# Use JPL ephemeris
solar_system_ephemeris.set("jpl")

# Mission time window
date_launch = time.Time("2030-01-01 00:00", scale="utc").tdb
date_arrival = time.Time("2045-01-01 00:00", scale="utc").tdb
ephem_time = time_range(date_launch, end=date_arrival)

# Define planets and ephemerides
planets = {
    "Mercury": Mercury,
    "Venus": Venus,
    "Earth": Earth,
    "Mars": Mars,
    "Jupiter": Jupiter,
    "Saturn": Saturn,
    "Uranus": Uranus,
    "Neptune": Neptune,
}
ephemerides = {name: Ephem.from_body(body, ephem_time) for name, body in planets.items()}

# Create orbits from ephemerides
ss_earth = Orbit.from_ephem(Sun, ephemerides["Earth"], date_launch)
ss_neptune = Orbit.from_ephem(Sun, ephemerides["Neptune"], date_arrival)

# Solve Lambert's transfer
man_lambert = Maneuver.lambert(ss_earth, ss_neptune)
ss_trans, _ = ss_earth.apply_maneuver(man_lambert, intermediate=True)

# Propagate transfer orbit to arrival time
ss_trans_arrival = ss_trans.propagate(date_arrival - date_launch)

# Get velocity vectors
v_earth = ss_earth.v
v_trans_depart = ss_trans.v
v_neptune = ss_neptune.v
v_trans_arrive = ss_trans_arrival.v

# Compute delta-Vs
delta_v_depart = (v_trans_depart - v_earth).to(u.km / u.s)
delta_v_arrive = (v_trans_arrive - v_neptune).to(u.km / u.s)

print(f"Departure velocity relative to Earth:  {np.linalg.norm(delta_v_depart.value):.2f} km / s")
print(f"Arrival velocity relative to Neptune:  {np.linalg.norm(delta_v_arrive.value):.2f} km / s")

# Plotting
plotter = OrbitPlotter3D()
plotter.set_attractor(Sun)

for name, ephem in ephemerides.items():
    plotter.plot_ephem(ephem, date_launch, label=name)

plotter.plot_trajectory(
    ss_trans.to_ephem(strategy=TrueAnomalyBounds(0 * u.deg, 180 * u.deg)).sample(),
    color="black",
    label="Transfer Orbit"
)
plotter.set_view(30 * u.deg, 260 * u.deg, distance=4 * u.km)

fig = plotter._figure
fig.show()
