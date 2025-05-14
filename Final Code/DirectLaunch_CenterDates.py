from astropy.time import Time
from astropy import units as u
from astropy.coordinates import solar_system_ephemeris
import numpy as np
from poliastro.bodies import Earth, Neptune, Sun
from poliastro.ephem import Ephem
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from poliastro.util import time_range

# Use high-accuracy ephemeris
solar_system_ephemeris.set("jpl")

# Define launch dates: every 1 day from Sep 15, 2029 to Nov 15, 2040
start_date = Time("2029-09-15", scale="tdb")
end_date = Time("2040-11-15", scale="tdb")
launch_dates = start_date + np.arange(0, (end_date - start_date).to(u.day).value + 1, 1) * u.day

# TOF range: 13 to 17 years
TOFs = [13, 14, 15, 16, 17]  # in years

# Dict to store best transfer per launch year
results_by_year = {}

# Sweep
for launch in launch_dates:
    launch_year = launch.datetime.year

    for tof in TOFs:
        arrival = launch + tof * u.year

        try:
            # Get ephemerides
            ephem_time = time_range(launch, end=arrival)
            earth_ephem = Ephem.from_body(Earth, ephem_time)
            neptune_ephem = Ephem.from_body(Neptune, ephem_time)

            ss_earth = Orbit.from_ephem(Sun, earth_ephem, launch)
            ss_neptune = Orbit.from_ephem(Sun, neptune_ephem, arrival)

            # Lambert solution
            man_lambert = Maneuver.lambert(ss_earth, ss_neptune)
            ss_trans, _ = ss_earth.apply_maneuver(man_lambert, intermediate=True)
            ss_arrival = ss_trans.propagate(arrival - launch)

            # Earth departure v_inf and C3
            v_inf_vec = (ss_trans.v - ss_earth.v).to(u.km / u.s)
            v_inf = np.linalg.norm(v_inf_vec.value)
            C3 = v_inf ** 2

            if C3 > 200:  # Filter unlaunchable cases
                continue

            # Arrival Δv
            v_arrival_rel = (ss_arrival.v - ss_neptune.v).to(u.km / u.s)
            arrival_dv = np.linalg.norm(v_arrival_rel.value)

            # Inclination
            h_vec = np.cross(ss_arrival.v.to_value(u.km / u.s), ss_neptune.r.to_value(u.km))
            inc_rad = np.arccos(h_vec[2] / np.linalg.norm(h_vec))
            inc_deg = np.rad2deg(inc_rad)

            # Store best transfer for this launch year
            if launch_year not in results_by_year or v_inf < results_by_year[launch_year]["v_inf"]:
                results_by_year[launch_year] = {
                    "launch": launch.utc.iso,
                    "arrival": arrival.utc.iso,
                    "tof_years": tof,
                    "v_inf": v_inf,
                    "C3": C3,
                    "arrival_dv": arrival_dv,
                    "inclination": inc_deg
                }

        except Exception:
            continue

# --- Print Results Per Year ---
print("\nBest Earth → Neptune Transfers Per Launch Year (C3 ≤ 200):\n")
print("Year  | Launch Date       | Arrival Date      | TOF  | Δv_depart (km/s) | C3 (km²/s²) | Arrival Δv (km/s) | Incl (deg)")
print("------|-------------------|-------------------|------|--------------|--------------|-------------------|------------")
for year in sorted(results_by_year):
    res = results_by_year[year]
    print(f"{year} | {res['launch']} | {res['arrival']} | {res['tof_years']:>4} | "
          f"{res['v_inf']:>12.2f} | {res['C3']:>12.2f} | {res['arrival_dv']:>17.2f} | {res['inclination']:>10.2f}")
