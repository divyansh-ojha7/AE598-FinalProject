from astropy import units as u
from astropy import time
from astropy.coordinates import solar_system_ephemeris
import numpy as np

from poliastro.bodies import Earth, Jupiter, Neptune, Sun
from poliastro.ephem import Ephem
from poliastro.maneuver import Maneuver
from poliastro.twobody import Orbit
from poliastro.util import time_range
from poliastro.twobody.sampling import EpochsArray
from poliastro.plotting import OrbitPlotter3D

import plotly.io as pio
pio.renderers.default = "browser"  # Show plot in browser

# Set high-accuracy ephemeris
solar_system_ephemeris.set("jpl")

# Define mission dates
date_launch = time.Time("2029-12-31 23:58:50.816", scale="utc").tdb
date_flyby = time.Time("2031-05-27 08:22:50.815", scale="utc").tdb
date_arrival = time.Time("2038-05-27 02:22:50.815", scale="utc").tdb

# Define ephemeris time ranges
ephem_ej = time_range(date_launch, end=date_flyby)
ephem_jn = time_range(date_flyby, end=date_arrival)

# Generate ephemerides
earth = Ephem.from_body(Earth, ephem_ej)
jupiter = Ephem.from_body(Jupiter, time_range(date_launch, end=date_arrival))  # Need both phases
neptune = Ephem.from_body(Neptune, ephem_jn)

# Create orbits
ss_earth = Orbit.from_ephem(Sun, earth, date_launch)
ss_jupiter_flyby = Orbit.from_ephem(Sun, jupiter, date_flyby)
ss_neptune = Orbit.from_ephem(Sun, neptune, date_arrival)

# Lambert transfer: Earth → Jupiter
man_ej = Maneuver.lambert(ss_earth, ss_jupiter_flyby)
ss_ej_trans, _ = ss_earth.apply_maneuver(man_ej, intermediate=True)

# Lambert transfer: Jupiter → Neptune
man_jn = Maneuver.lambert(ss_jupiter_flyby, ss_neptune)
ss_jn_trans, _ = ss_jupiter_flyby.apply_maneuver(man_jn, intermediate=True)

# Compute Δv
dv_depart = (ss_ej_trans.v - ss_earth.v).to(u.km / u.s)
dv_flyby = (ss_jn_trans.v - ss_jupiter_flyby.v).to(u.km / u.s)
dv_arrive = (ss_jn_trans.propagate(date_arrival - date_flyby).v - ss_neptune.v).to(u.km / u.s)

print(f"Δv Earth departure: {np.linalg.norm(dv_depart.value):.2f} km/s")
print(f"Δv at Jupiter flyby (injection): {np.linalg.norm(dv_flyby.value):.2f} km/s")
print(f"Δv Neptune arrival: {np.linalg.norm(dv_arrive.value):.2f} km/s")

# Set up 3D plot
plotter = OrbitPlotter3D()
plotter.set_attractor(Sun)

# Plot planetary positions
plotter.plot_ephem(earth, date_launch, label="Earth at Launch")
plotter.plot_ephem(jupiter, date_flyby, label="Jupiter at Flyby")
plotter.plot_ephem(neptune, date_arrival, label="Neptune at Arrival")

# Sample orbits over time (avoid anomaly errors)
ej_times = date_launch + (date_flyby - date_launch) * np.linspace(0, 1, 100)
jn_times = date_flyby + (date_arrival - date_flyby) * np.linspace(0, 1, 100)

ej_ephem = ss_ej_trans.to_ephem(strategy=EpochsArray(ej_times))
jn_ephem = ss_jn_trans.to_ephem(strategy=EpochsArray(jn_times))

# Plot transfer trajectories
plotter.plot_trajectory(ej_ephem.sample(), color="orange", label="Earth → Jupiter")
plotter.plot_trajectory(jn_ephem.sample(), color="green", label="Jupiter → Neptune")

plotter.set_view(30 * u.deg, 260 * u.deg, distance=6 * u.km)
plotter._figure.show()
