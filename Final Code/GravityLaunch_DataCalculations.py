from astropy.time import Time
from astropy import units as u
from astropy.coordinates import solar_system_ephemeris
import numpy as np
from poliastro.bodies import Earth, Jupiter, Neptune, Sun
from poliastro.ephem import Ephem
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from poliastro.util import time_range
import pandas as pd

# Use JPL ephemerides
solar_system_ephemeris.set("jpl")

# “Best‐launch” centers
center_dates = [
    "2030-01-20T23:58:50.816", "2031-02-24T23:58:50.815", "2032-03-30T23:58:50.814",
    "2033-05-04T23:58:50.815", "2034-06-18T23:58:50.816", "2035-07-23T23:58:50.816",
    "2036-08-26T23:58:50.817", "2037-09-30T23:58:50.818", "2038-10-25T23:58:50.818",
    "2039-11-29T23:58:50.817", "2040-12-23T23:58:50.816",
]

# Build ±10 day windows 231 launches
launch_dates = []
for d in center_dates:
    c = Time(d, scale="tdb")
    launch_dates.extend(c + np.arange(-10, 11, 1) * u.day)
print(f"Swept {len(launch_dates)} launch dates total;")  # should be 231

# Fixed TOFs
TOF_EJ = 1.4 * u.year
TOF_JN = 10.0 * u.year

mu_neptune = 6.8369e6  # km^3/s^2
r_neptune = 24622      # km
alt_peri_range  = np.arange(-5000, 1001, 100)   # km – where you actually skim Neptune
alt_eval = 1000                                         # km – where you want the flight-path angle


# Parallel storage, including flight path angles
years                 = []
launch_arrival_pairs  = []  # (launch, flyby, arrival)
departure_dvs         = []
c3_values             = []
arrival_dvs           = []
inclinations          = []
tof_years             = []
flight_path_angles    = []  # flight path angle at Neptune arrival
arrival_speed_at_altitude = []
e_data = []
altitude_data = []

for launch in launch_dates:
    flyby   = launch + TOF_EJ
    arrival = flyby  + TOF_JN

    try:
        # Ephemerides
        eph_e = Ephem.from_body(Earth,   time_range(launch, end=launch))
        eph_j = Ephem.from_body(Jupiter, time_range(flyby,  end=flyby))
        eph_n = Ephem.from_body(Neptune, time_range(arrival,end=arrival))

        # Sun‑centered orbits
        ss_e = Orbit.from_ephem(Sun, eph_e, launch)
        ss_j = Orbit.from_ephem(Sun, eph_j, flyby)
        ss_n = Orbit.from_ephem(Sun, eph_n, arrival)

        # Leg1: Earth to Jupiter
        m1      = Maneuver.lambert(ss_e, ss_j)
        leg1, _ = ss_e.apply_maneuver(m1, intermediate=True)
        vinf    = np.linalg.norm((leg1.v - ss_e.v).to(u.km/u.s).value)
        C3      = vinf**2
        if C3 > 200:
            continue

        # Leg2: Jupiter to Neptune
        m2        = Maneuver.lambert(ss_j, ss_n)
        leg2, _   = ss_j.apply_maneuver(m2, intermediate=True)
        leg2_arr  = leg2.propagate(arrival - flyby)

        # Δv (departure & arrival)
        dv_dep = np.linalg.norm((leg1.v - ss_e.v).to(u.km/u.s).value)
        dv_arr = np.linalg.norm((leg2_arr.v - ss_n.v).to(u.km/u.s).value)

        # Inclination at arrival
        h_vec   = np.cross(leg2_arr.v.to_value(u.km/u.s), ss_n.r.to_value(u.km))
        inc_deg = np.rad2deg(np.arccos(h_vec[2] / np.linalg.norm(h_vec)))

        # Approach Calculations
        for alt_peri in alt_peri_range:
            rp = r_neptune + alt_peri  # periapsis radius
            r_eval = r_neptune + alt_eval  # evaluation radius

            # Arrival Speed at Altitude (speed at the evaluation point)
            speed_at_altitude = np.sqrt(dv_arr ** 2 + 2 * mu_neptune / r_eval)

            # Neptune-relative vectors
            # Get spacecraft state at arrival
            r_arrival = leg2_arr.r.to(u.km).value

            v_arrival = leg2_arr.v.to(u.km / u.s).value

            # hyperbola elements (use rp!)
            e = 1 + (rp * dv_arr ** 2) / mu_neptune
            h = np.sqrt(mu_neptune * rp * (1 + e))

            # flight-path angle at r_eval
            cos_gamma = np.clip(h / (r_eval * speed_at_altitude), -1.0, 1.0)
            gamma_deg = -np.degrees(np.arccos(cos_gamma))  # negative ⇒ inbound
            flight_path_angle = gamma_deg

            # if flight_path_angle >= -30 and flight_path_angle <= -15:
                # Store in parallel lists
            years.append(launch.datetime.year)
            launch_arrival_pairs.append((launch.utc.iso, flyby.utc.iso, arrival.utc.iso))
            departure_dvs.append(dv_dep)
            c3_values.append(C3)
            arrival_dvs.append(dv_arr)
            inclinations.append(inc_deg)
            tof_years.append((TOF_EJ + TOF_JN).to_value(u.year))
            flight_path_angles.append(flight_path_angle)
            arrival_speed_at_altitude.append(speed_at_altitude)
            e_data.append(e)
            altitude_data.append(rp)

    except Exception:
        continue

# Print exactly 231 lines, including Flight Path Angle:
print("\nAll Earth → Jupiter → Neptune Transfers (C3 ≤ 200):\n")
print("Year | Launch Date       | Flyby Date        | Arrival Date      | TOF  | Δv_depart (km/s) | C3 (km²/s²) | Arrival Δv (km/s) | Incl (deg) | Altitude (km) | Flight Path Angle (deg) | Speed at 1000km (km/s) | E")
print("-----|-------------------|-------------------|-------------------|------|------------------|--------------|-------------------|-----------|-----------|-----------")
for i in range(len(years)):
    yr = years[i]
    l, f, a = launch_arrival_pairs[i]
    print(f"{yr} | {l} | {f} | {a} | "
          f"{tof_years[i]:>4.1f} | {departure_dvs[i]:>16.2f} | {c3_values[i]:>12.2f} | "
          f"{arrival_dvs[i]:>17.2f} | {inclinations[i]:>10.2f} | {altitude_data[i]:>10.2f} | {flight_path_angles[i]:>10.2f} | {arrival_speed_at_altitude[i]:>10.2f} | {e_data[i]:>10.2f}")



import matplotlib.pyplot as plt
from datetime import datetime

# unpack launch dates
# Extract launch times (first element of each triple)
launch_date_objs = [
    datetime.fromisoformat(launch_iso)
    for launch_iso, _, _ in launch_arrival_pairs
]

# Extract arrival times   (third element of each triple)
arrival_date_objs = [
    datetime.fromisoformat(arrival_iso)
    for _, _, arrival_iso in launch_arrival_pairs
]

# 1) Δv_depart vs Arrival Inclination
plt.figure(figsize=(8,6))
plt.scatter(inclinations, departure_dvs, label="Departure Δv", marker="o")
plt.xlabel("Arrival Inclination (deg)")
plt.ylabel("Earth Departure Δv (km/s)")
plt.title("Earth Δv vs Arrival Inclination (Jupiter Assist)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# 2) Earth Departure Δv vs Launch Date
plt.figure(figsize=(10,4))
plt.scatter(launch_date_objs, departure_dvs, marker="o", label="Δv_dep")
plt.xlabel("Launch Date")
plt.ylabel("Earth Departure Δv (km/s)")
plt.title("Δv_depart vs Launch Date (Jupiter Assist)")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()

# 3) Neptune Arrival Δv vs Launch Date
plt.figure(figsize=(10,4))
plt.scatter(launch_date_objs, arrival_dvs, marker="o", color="green", label="Δv_arr")
plt.xlabel("Launch Date")
plt.ylabel("Arrival Δv (km/s)")
plt.title("Neptune Arrival Δv vs Launch Date")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()

# 4) Arrival Inclination vs Launch Date
plt.figure(figsize=(10,4))
plt.scatter(launch_date_objs, inclinations, marker="o", color="purple", label="Inclination")
plt.xlabel("Launch Date")
plt.ylabel("Arrival Inclination (deg)")
plt.title("Arrival Inclination vs Launch Date")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()

# 5) Flight‑Path Angle vs Launch Date
plt.figure(figsize=(10,4))
plt.scatter(launch_date_objs, flight_path_angles, marker="o", color="orange", label="FPA")
plt.xlabel("Launch Date")
plt.ylabel("Flight Path Angle (deg)")
plt.title("Neptune Flight‑Path Angle vs Launch Date")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()

# 6) C3 vs Launch Date
plt.figure(figsize=(10,4))
plt.scatter(launch_date_objs, c3_values, marker="o", color="red", label="C3")
plt.xlabel("Launch Date")
plt.ylabel("C3 (km²/s²)")
plt.title("C3 vs Launch Date")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()

import matplotlib.ticker as ticker

fig, ax = plt.subplots(figsize=(10, 5))
sc = ax.scatter(launch_date_objs, arrival_speed_at_altitude, c=c3_values, cmap="plasma", s=20)
ax.set_xlabel("Launch Date", fontsize=14)
ax.set_ylabel("Speed at 1000 km above Neptune (km/s)", fontsize=14)
ax.set_title("Arrival Speed at 1000 km Altitude vs Time of Flight", fontsize=16)
ax.grid(True)
ax.xaxis.set_tick_params(rotation=45)
ax.tick_params(axis='both', labelsize=12)
ax.yaxis.get_offset_text().set_visible(False)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
cbar = fig.colorbar(sc)
cbar.set_label("C3 (km²/s²)", fontsize=14)
cbar.ax.tick_params(labelsize=12)  # make colorbar tick labels bigger too
plt.tight_layout()
plt.savefig("Gravity_speed_vs_launch")
plt.show()



import matplotlib.pyplot as plt
from datetime import datetime

# Choose a specific launch date
target_date_str = "2033-05-04 23:57:41.630"
target_date = datetime.fromisoformat(target_date_str)

# Filter data for this launch date
filtered_indices = [i for i, pair in enumerate(launch_arrival_pairs) if pair[0] == target_date_str]

# Extract corresponding values
altitudes_for_date = [altitude_data[i] for i in filtered_indices]
flight_path_for_date = [flight_path_angles[i] for i in filtered_indices]

# Sort by altitude for a smooth line
sorted_pairs = sorted(zip(altitudes_for_date, flight_path_for_date))
altitudes_sorted, flight_path_sorted = zip(*sorted_pairs)

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(altitudes_sorted, flight_path_sorted, marker='o')
plt.xlabel("Periapsis Altitude (km)", fontsize=14)
plt.ylabel("Flight Path Angle (deg)", fontsize=14)
plt.title(f"Flight Path Angle vs. Periapsis Altitude\nLaunch Date: {target_date.strftime('%Y-%m-%d')}", fontsize=16)
plt.tick_params(axis='both', labelsize=12)  # Increase tick label size
plt.grid(True)
plt.tight_layout()
plt.savefig("Gravity_flightangle_vs_altitude")
plt.show()

# df = pd.DataFrame({
#     "Launch Date": [pair[0] for pair in launch_arrival_pairs],
#     "Arrival Date": [pair[1] for pair in launch_arrival_pairs],
#     "Periapsis Altitude (km)": altitude_data,
#     "Speed (km/s)": arrival_speed_at_altitude,
#     "Flight Path (deg)": flight_path_angles,
#     "Eccentricity": e_data
# })
# df.to_excel("neptune_results_gravity.xlsx", index=False)
# print("\n✅ Saved results to 'neptune_transfer_results.xlsx'")


# Build a table of all valid transfers
# df = pd.DataFrame({
#     "Launch Date": [pair[0] for pair in launch_arrival_pairs],
#     "Flyby Date": [pair[1] for pair in launch_arrival_pairs],
#     "Arrival Date": [pair[2] for pair in launch_arrival_pairs],
#     "Departure Δv (km/s)": departure_dvs,
#     "C3 (km²/s²)": c3_values,
#     "Arrival Speed (km/s)": arrival_dvs,
#     "Inclination (deg)": inclinations,
#     "Flight Path (deg)": flight_path_angles
# })
# df.to_excel("GravityTransferNeptune.xlsx", index=False)
# print("\n✅ Saved results to 'GravityTransferNeptune.xlsx'")

# Find the index of the minimum arrival speed
min_index = np.argmin(arrival_dvs)

# Extract all relevant data for that index
min_data = {
    "Launch Year": years[min_index],
    "Launch Date": launch_arrival_pairs[min_index][0],
    "Flyby Date": launch_arrival_pairs[min_index][1],
    "Arrival Date": launch_arrival_pairs[min_index][2],
    "Total TOF (yr)": tof_years[min_index],
    "Departure Δv (km/s)": departure_dvs[min_index],
    "C3 (km²/s²)": c3_values[min_index],
    "Arrival Δv (km/s)": arrival_dvs[min_index],
    "Arrival Inclination (deg)": inclinations[min_index],
    "Periapsis Altitude (km)": altitude_data[min_index],
    "Flight Path Angle (deg)": flight_path_angles[min_index],
    "Arrival Speed at 1000 km (km/s)": arrival_speed_at_altitude[min_index],
    "Eccentricity": e_data[min_index],
}

# Print the results neatly
print("\n Transfer with Minimum Arrival Speed at 1000 km:")
for k, v in min_data.items():
    print(f"{k:30}: {v}")



# Gravity‑Assist Trajectory Map
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import solar_system_ephemeris
import numpy as np

from poliastro.bodies import Earth, Jupiter, Neptune, Sun
from poliastro.ephem import Ephem
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from poliastro.util import time_range

# Use high‑accuracy JPL ephemerides
solar_system_ephemeris.set("jpl")

# Best launch dates (centers) from your previous results
center_dates = [
    "2030-05-29T23:58:50.815", "2031-05-31T23:58:50.815", "2032-06-02T23:58:50.815",
    "2033-06-04T23:58:50.815", "2034-06-07T23:58:50.815", "2035-06-09T23:58:50.815",
    "2036-06-11T23:58:50.815", "2037-06-13T23:58:50.815", "2038-06-16T23:58:50.816",
    "2039-06-18T23:58:50.816", "2040-06-19T23:58:50.816"
]

# Build ±10 day windows → 231 candidate launch dates
launch_dates = []
for ds in center_dates:
    c = Time(ds, scale="tdb")
    launch_dates.extend(c + np.arange(-10, 11) * u.day)

print(f"Swept {len(launch_dates)} launch dates total;")

# TOF ranges for the two legs
TOF_EJ = np.arange(0.9, 1.5, 0.1)    # Earth→Jupiter in years
TOF_JN = np.arange(6.0, 13.5, 0.5)   # Jupiter→Neptune in years

# Storage
arrival_vinf_list = []
tof_list          = []
c3_list           = []

# Run the two‑leg Lambert map
for launch in launch_dates:
    for tof_ej in TOF_EJ:
        # compute flyby epoch
        flyby = launch + tof_ej * u.year

        for tof_jn in TOF_JN:
            arrival = flyby + tof_jn * u.year

            try:
                # get ephemerides at each key time
                eph_e = Ephem.from_body(Earth,   time_range(launch, end=launch))
                eph_j = Ephem.from_body(Jupiter, time_range(flyby,  end=flyby))
                eph_n = Ephem.from_body(Neptune, time_range(arrival,end=arrival))

                # Sun-centered orbits
                ss_e = Orbit.from_ephem(Sun, eph_e, launch)
                ss_j = Orbit.from_ephem(Sun, eph_j, flyby)
                ss_n = Orbit.from_ephem(Sun, eph_n, arrival)

                # Leg 1: Earth → Jupiter
                m1       = Maneuver.lambert(ss_e, ss_j)
                leg1, _  = ss_e.apply_maneuver(m1, intermediate=True)
                vinf_vec = (leg1.v - ss_e.v).to(u.km/u.s)
                v_inf    = np.linalg.norm(vinf_vec.value)
                C3       = v_inf**2
                if C3 > 700:
                    continue   # skip too‐energetic departures

                # Leg 2: Jupiter → Neptune
                m2        = Maneuver.lambert(ss_j, ss_n)
                leg2, _   = ss_j.apply_maneuver(m2, intermediate=True)
                leg2_arr  = leg2.propagate(arrival - flyby)

                # Arrival v_inf at Neptune
                v_arr_vec   = (leg2_arr.v - ss_n.v).to(u.km/u.s)
                arrival_vinf = np.linalg.norm(v_arr_vec.value)

                # Store total TOF & metrics
                arrival_vinf_list.append(arrival_vinf)
                tof_list.append(tof_ej + tof_jn)
                c3_list.append(C3)

            except Exception:
                # any failure (e.g. ephemeris or Lambert) => skip
                continue

# Plot Earth to Jupiter to Neptune map
plt.figure(figsize=(8, 6))
sc = plt.scatter(arrival_vinf_list, tof_list, c=c3_list, cmap="jet", s=25)
cbar = plt.colorbar(sc)
cbar.set_label("C3 (km²/s²)", fontsize=14)      # Label font size
cbar.ax.tick_params(labelsize=12)               # Tick font size on colorbarplt.xlabel("Neptune Arrival $v_\\infty$ (km/s)", fontsize=14)
plt.xlabel("Neptune Arrival $v_\\infty$ (km/s)", fontsize=14)
plt.ylabel("Total Time of Flight (yr)", fontsize=14)
plt.title("Gravity‑Assist Trajectory Map: Earth To Jupiter To Neptune", fontsize=16)
plt.tick_params(axis='both', labelsize=12)  # Increase tick label size

# plt.axvline(x=12, linestyle="--", color="black", label="Propulsive Insertion Limit")
plt.grid(True)
# plt.legend()
plt.tight_layout()
plt.savefig("Gravity‑Assist Trajectory Map")
plt.show()
