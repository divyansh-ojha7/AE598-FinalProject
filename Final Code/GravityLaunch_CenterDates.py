from astropy.time import Time
from astropy import units as u
from astropy.coordinates import solar_system_ephemeris
import numpy as np
from poliastro.bodies import Earth, Jupiter, Neptune, Sun
from poliastro.ephem import Ephem
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from poliastro.util import time_range

# Use high-accuracy ephemeris
solar_system_ephemeris.set("jpl")

# Define launch dates (one every 10 days)
start_date = Time("2030-01-01", scale="tdb")
end_date = Time("2040-12-30", scale="tdb")
launch_dates = start_date + np.arange(0, (end_date - start_date).to(u.day).value + 1, 10) * u.day

# TOF ranges
TOF_EARTH_TO_JUPITER_YEARS = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4]  # Earth to Jupiter
TOF_JUPITER_TO_NEPTUNE_YEARS = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]  # Jupiter to Neptune

# Best results per year
results_by_year = {}

for launch in launch_dates:
    year = launch.datetime.year
    for tof_ej in TOF_EARTH_TO_JUPITER_YEARS:
        for tof_jn in TOF_JUPITER_TO_NEPTUNE_YEARS:
            flyby = launch + tof_ej * u.year
            arrival = flyby + tof_jn * u.year

            try:
                ephem_e = Ephem.from_body(Earth, time_range(launch, end=launch))
                ephem_j = Ephem.from_body(Jupiter, time_range(flyby, end=flyby))
                ephem_n = Ephem.from_body(Neptune, time_range(arrival, end=arrival))

                ss_e = Orbit.from_ephem(Sun, ephem_e, launch)
                ss_j = Orbit.from_ephem(Sun, ephem_j, flyby)
                ss_n = Orbit.from_ephem(Sun, ephem_n, arrival)

                # Earth → Jupiter leg
                m1 = Maneuver.lambert(ss_e, ss_j)
                ss_leg1, _ = ss_e.apply_maneuver(m1, intermediate=True)

                # Jupiter → Neptune leg
                m2 = Maneuver.lambert(ss_j, ss_n)
                ss_leg2, _ = ss_j.apply_maneuver(m2, intermediate=True)

                # Earth departure velocity
                vinf_vec = (ss_leg1.v - ss_e.v).to(u.km / u.s)
                vinf = np.linalg.norm(vinf_vec.value)
                C3 = vinf ** 2
                # print(vinf)

                if C3 > 200:
                    continue


                # Arrival Δv
                ss_arrival = ss_leg2.propagate(arrival - flyby)
                dv_arrival_vec = (ss_arrival.v - ss_n.v).to(u.km / u.s)
                dv_arrival = np.linalg.norm(dv_arrival_vec.value)
                print(dv_arrival)

                if dv_arrival > 11:
                    continue

                # Inclination
                h_vec = np.cross(ss_arrival.v.to_value(u.km / u.s), ss_n.r.to_value(u.km))
                inc_rad = np.arccos(h_vec[2] / np.linalg.norm(h_vec))
                inc_deg = np.rad2deg(inc_rad)

                # Save best result per year
                total_tof = tof_ej + tof_jn
                if year not in results_by_year or vinf < results_by_year[year]["v_inf"]:
                    results_by_year[year] = {
                        "launch": launch.utc.iso,
                        "flyby": flyby.utc.iso,
                        "arrival": arrival.utc.iso,
                        "tof_years": total_tof,
                        "v_inf": vinf,
                        "C3": C3,
                        "arrival_dv": dv_arrival,
                        "inclination": inc_deg,
                        "tof_ej": tof_ej,
                        "tof_jn": tof_jn
                    }

            except Exception:
                continue

# Output results
print("\nBest Earth → Jupiter → Neptune Transfers Per Launch Year (C3 ≤ 200):\n")
print("Year  | Launch Date       | Flyby Date        | Arrival Date      | TOF  | Δv_depart (km/s) | C3 (km²/s²) | Arrival Δv (km/s) | Incl (deg) | tof_ej | tof_jn")
print("------|-------------------|-------------------|-------------------|------|------------------|--------------|-------------------|------------|------------|------------")
for year in sorted(results_by_year):
    res = results_by_year[year]
    print(f"{year} | {res['launch']} | {res['flyby']} | {res['arrival']} | {res['tof_years']:>4} | "
          f"{res['v_inf']:>16.2f} | {res['C3']:>12.2f} | {res['arrival_dv']:>17.2f} | {res['inclination']:>10.2f} | {res['tof_ej']:>10.2f} | {res['tof_jn']:>10.2f}")
