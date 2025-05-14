from astropy import units as u
from astropy import time
from astropy.coordinates import solar_system_ephemeris
import numpy as np
import matplotlib.pyplot as plt

from poliastro.bodies import Earth, Neptune, Sun
from poliastro.ephem import Ephem
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from poliastro.util import time_range
from datetime import datetime
import pandas as pd

# Use JPL ephemerides
solar_system_ephemeris.set("jpl")

# Closest-approach center dates (1 per year, 2030–2040)
# Launch dates center points
center_dates = [
    "2030-05-29T23:58:50.815", "2031-05-31T23:58:50.815", "2032-06-02T23:58:50.815",
    "2033-06-04T23:58:50.815", "2034-06-07T23:58:50.815", "2035-06-09T23:58:50.815",
    "2036-06-11T23:58:50.815", "2037-06-13T23:58:50.815", "2038-06-16T23:58:50.816",
    "2039-06-18T23:58:50.816", "2040-06-19T23:58:50.816"
]

# Generate ±10 days around each center date (21 dates per year)
launch_dates = []
for date_str in center_dates:
    center = time.Time(date_str, scale="tdb")
    sweep_window = center + np.arange(-10, 11, 1) * u.day
    launch_dates.extend(sweep_window)

# Debug print: confirm sweep
print(f"Swept {len(launch_dates)} launch dates total;")
print("First five:", [ldt.utc.iso for ldt in launch_dates[:5]])
print("Last five: ", [ldt.utc.iso for ldt in launch_dates[-5:]])
print()

mu_neptune = 6.8369e6  # km^3/s^2
r_neptune = 24622      # km
alt_peri_range  = np.arange(-5000, 1001, 50)   # km – where you actually skim Neptune
alt_eval = 1000                                         # km – where you want the flight-path angle


# Storage for results
departure_dvs = []
arrival_dvs = []
inclinations = []
launch_arrival_pairs = []
c3_values = []
flight_path = []
arrival_speed_at_altitude = []
e_data = []
altitude_data = []


# Track lowest Δv
min_dep_dv = float("inf")
min_arrival_dv = float("inf")
best_dep_info = None
best_arrival_info = None

# Main loop
for launch in launch_dates:
    arrival = launch + 17 * u.year

    try:
        # Generate planetary ephemerides for the mission window
        ephem_time = time_range(launch, end=arrival)
        earth_ephem = Ephem.from_body(Earth, ephem_time)
        neptune_ephem = Ephem.from_body(Neptune, ephem_time)

        ss_earth = Orbit.from_ephem(Sun, earth_ephem, launch)
        ss_neptune = Orbit.from_ephem(Sun, neptune_ephem, arrival)

        # Lambert transfer
        man_lambert = Maneuver.lambert(ss_earth, ss_neptune)
        ss_trans, _ = ss_earth.apply_maneuver(man_lambert, intermediate=True)
        ss_trans_arrival = ss_trans.propagate(arrival - launch)

        # Departure velocity relative to Earth
        delta_v_depart = (ss_trans.v - ss_earth.v).to(u.km / u.s)
        dep_dv_mag = np.linalg.norm(delta_v_depart.value)
        C3 = dep_dv_mag ** 2

        # Arrival velocity relative to Neptune
        delta_v_arrive = (ss_trans_arrival.v - ss_neptune.v).to(u.km / u.s)
        arr_dv_mag = np.linalg.norm(delta_v_arrive.value)

        # Inclination at arrival
        h_vec = np.cross(ss_trans_arrival.v.to_value(u.km / u.s), ss_neptune.r.to_value(u.km))
        inc_rad = np.arccos(h_vec[2] / np.linalg.norm(h_vec))
        inc_deg = np.rad2deg(inc_rad)

        # Approach Calculations
        for alt_peri in alt_peri_range:
            rp = r_neptune + alt_peri  # periapsis radius
            r_eval = r_neptune + alt_eval  # evaluation radius

            # Arrival Speed at Altitude (speed at the evaluation point)
            speed_at_altitude = np.sqrt(arr_dv_mag ** 2 + 2 * mu_neptune / r_eval)

            # Neptune-relative vectors
            # Get spacecraft state at arrival
            r_arrival = ss_trans_arrival.r.to(u.km).value

            v_arrival = ss_trans_arrival.v.to(u.km / u.s).value

            # hyperbola elements (use rp!)
            e = 1 + (rp * arr_dv_mag ** 2) / mu_neptune
            h = np.sqrt(mu_neptune * rp * (1 + e))

            # flight-path angle at r_eval
            cos_gamma = np.clip(h / (r_eval * speed_at_altitude), -1.0, 1.0)
            gamma_deg = -np.degrees(np.arccos(cos_gamma))  # negative ⇒ inbound
            flight_path_angle = gamma_deg
            print(flight_path_angle)

            # Save data
            departure_dvs.append(dep_dv_mag)
            arrival_dvs.append(arr_dv_mag)
            inclinations.append(inc_deg)
            launch_arrival_pairs.append((launch.utc.iso, arrival.utc.iso))
            c3_values.append(C3)
            flight_path.append(flight_path_angle)
            arrival_speed_at_altitude.append(speed_at_altitude)
            e_data.append(e)
            altitude_data.append(rp)

    except Exception as e:
        print(f"Failed transfer: {launch.utc.iso} → {arrival.utc.iso} | Error: {e}")
        continue

# --- Output Results ---
print("\n All Valid Transfers:")
print("Launch Date       Arrival Date       v_depart (km/s)   C3 (km²/s²)   Δv_arrival (km/s)   Inclination (deg)  Altitude  Flight Path Angle (deg)  V at 1000 (km)  E  ")
for i in range(len(departure_dvs)):
    print(f"{launch_arrival_pairs[i][0]}   {launch_arrival_pairs[i][1]}   "
          f"{departure_dvs[i]:.2f}              {c3_values[i]:.2f}         {arrival_dvs[i]:.2f}               {inclinations[i]:.2f}         {altitude_data[i]:.2f}         {flight_path[i]:.2f}         {arrival_speed_at_altitude[i]:.2f}         {e_data[i]:.2f}")

# Build a table of all valid transfers
# df = pd.DataFrame({
#     "Launch Date": [pair[0] for pair in launch_arrival_pairs],
#     "Arrival Date": [pair[1] for pair in launch_arrival_pairs],
#     "Departure Δv (km/s)": departure_dvs,
#     "C3 (km²/s²)": c3_values,
#     "Arrival Speed (km/s)": arrival_dvs,
#     "Inclination (deg)": inclinations,
#     "Flight Path (deg)": flight_path
# })
# df.to_excel("neptune_transfer_results.xlsx", index=False)
# print("\n✅ Saved results to 'neptune_transfer_results.xlsx'")

# df = pd.DataFrame({
#     "Launch Date": [pair[0] for pair in launch_arrival_pairs],
#     "Arrival Date": [pair[1] for pair in launch_arrival_pairs],
#     "Periapsis Altitude (km)": altitude_data,
#     "Speed (km/s)": arrival_speed_at_altitude,
#     "Flight Path (deg)": flight_path,
#     "Eccentricity": e_data
# })
# df.to_excel("neptune_results.xlsx", index=False)
# print("\n✅ Saved results to 'neptune_transfer_results.xlsx'")


# --- Plotting ---
plt.figure(figsize=(10, 6))
# plt.scatter(inclinations, departure_dvs, label="Departure Δv (Earth)", color="blue")
plt.scatter(inclinations, arrival_dvs, label="Different Launch Dates", color="green")
plt.xlabel("Arrival Inclination (deg)")
plt.ylabel("Arrival v (km/s)")
plt.title("Arrival v vs Inclination for Earth To Neptune Transfers (TOF = 17 years)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Convert launch dates from string to datetime for plotting
launch_date_objs = [datetime.fromisoformat(pair[0]) for pair in launch_arrival_pairs]

# Plot 1: Earth Departure Δv vs Launch Date
plt.figure(figsize=(10, 5))
plt.scatter(launch_date_objs, departure_dvs, marker='o', color='blue', label="Different Launch Dates")
plt.xlabel("Launch Date")
plt.ylabel("Earth Departure v (km/s)")
plt.title("Earth Departure v vs Launch Date")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.xticks(rotation=45)

# Plot 2: Arrival Inclination vs Launch Date
plt.figure(figsize=(10, 5))
plt.scatter(launch_date_objs, inclinations, marker='o', color='purple', label="Different Launch Dates")
plt.xlabel("Launch Date")
plt.ylabel("Arrival Inclination (deg)")
plt.title("Arrival Inclination vs Launch Date")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.xticks(rotation=45)
plt.show()

import matplotlib.ticker as ticker

fig, ax = plt.subplots(figsize=(10, 5))
sc = ax.scatter(launch_date_objs, arrival_speed_at_altitude, c=c3_values, cmap="plasma", s=20)
ax.set_xlabel("Launch Date", fontsize=14)
ax.set_ylabel("Speed at 1000 km above Neptune (km/s)", fontsize=14)
ax.set_title("Arrival Speed at 1000 km Altitude vs Time of Flight", fontsize=16)
ax.tick_params(axis='both', labelsize=12)
ax.grid(True)
ax.xaxis.set_tick_params(rotation=45)
ax.yaxis.get_offset_text().set_visible(False)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
cbar = fig.colorbar(sc)
cbar.set_label("C3 (km²/s²)", fontsize=14)
cbar.ax.tick_params(labelsize=12)  # make colorbar tick labels bigger too
plt.tight_layout()
plt.savefig("Direct_speed_vs_launch")
plt.show()



# Get optimal data
import matplotlib.pyplot as plt
from datetime import datetime

# Choose a specific launch date
target_date_str = "2030-05-29 23:57:41.630"
target_date = datetime.fromisoformat(target_date_str)

# Filter data for this launch date
filtered_indices = [i for i, pair in enumerate(launch_arrival_pairs) if pair[0] == target_date_str]

# Extract corresponding values
altitudes_for_date = [altitude_data[i] for i in filtered_indices]
flight_path_for_date = [flight_path[i] for i in filtered_indices]

# Sort by altitude for a smooth line
sorted_pairs = sorted(zip(altitudes_for_date, flight_path_for_date))
altitudes_sorted, flight_path_sorted = zip(*sorted_pairs)

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(altitudes_sorted, flight_path_sorted, marker='o')
plt.xlabel("Periapsis Altitude (km)", fontsize=14)
plt.ylabel("Flight Path Angle (deg)", fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.title(f"Flight Path Angle vs. Periapsis Altitude\nLaunch Date: {target_date.strftime('%Y-%m-%d')}", fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig("Direct_flightangle_vs_altitude")
plt.show()


# Find index of minimum arrival speed at 1000 km altitude
min_index = np.argmin(arrival_speed_at_altitude)

# Extract relevant information
min_data = {
    "Launch Date": launch_arrival_pairs[min_index][0],
    "Arrival Date": launch_arrival_pairs[min_index][1],
    "Departure Δv (km/s)": departure_dvs[min_index],
    "C3 (km²/s²)": c3_values[min_index],
    "Arrival Δv (km/s)": arrival_dvs[min_index],
    "Arrival Inclination (deg)": inclinations[min_index],
    "Periapsis Altitude (km)": altitude_data[min_index],
    "Flight Path Angle (deg)": flight_path[min_index],
    "Arrival Speed at 1000 km (km/s)": arrival_speed_at_altitude[min_index],
    "Eccentricity": e_data[min_index],
}

# Print the result
print("\nTransfer with Minimum Arrival Speed at 1000 km:")
print("-" * 80)
for key, val in min_data.items():
    print(f"{key:30}: {val}")




# Earth To Neptune Trajectory Map
import matplotlib.pyplot as plt
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

# Best launch dates from previous results
center_dates = [
    "2030-05-29T23:58:50.815", "2031-05-31T23:58:50.815", "2032-06-02T23:58:50.815",
    "2033-06-04T23:58:50.815", "2034-06-07T23:58:50.815", "2035-06-09T23:58:50.815",
    "2036-06-11T23:58:50.815", "2037-06-13T23:58:50.815", "2038-06-16T23:58:50.816",
    "2039-06-18T23:58:50.816", "2040-06-19T23:58:50.816"
]

# Create ±10 day windows around each best date
launch_dates = []
for date_str in center_dates:
    center = Time(date_str, scale="tdb")
    sweep_window = center + np.arange(-10, 11, 1) * u.day
    launch_dates.extend(sweep_window)

# Storage
arrival_vinf_list = []
tof_list = []
c3_list = []

# Fixed time of flight
TOF_YEARS_RANGE = np.arange(6, 17, 0.5)  # 6 to 13 years, step 0.5

# Run simulations
for launch in launch_dates:
    for tof_years in TOF_YEARS_RANGE:
        arrival = launch + tof_years * u.year

        try:
            ephem_time = time_range(launch, end=arrival)
            earth_ephem = Ephem.from_body(Earth, ephem_time)
            neptune_ephem = Ephem.from_body(Neptune, ephem_time)

            ss_earth = Orbit.from_ephem(Sun, earth_ephem, launch)
            ss_neptune = Orbit.from_ephem(Sun, neptune_ephem, arrival)

            # Lambert transfer
            man_lambert = Maneuver.lambert(ss_earth, ss_neptune)
            ss_trans, _ = ss_earth.apply_maneuver(man_lambert, intermediate=True)
            ss_arrival = ss_trans.propagate(arrival - launch)

            # Launch velocity (v_inf)
            v_inf_vec = (ss_trans.v - ss_earth.v).to(u.km / u.s)
            v_inf = np.linalg.norm(v_inf_vec.value)
            C3 = v_inf ** 2

            # if C3 > 200:
            #     continue  # Filter out high-energy launches

            # Arrival velocity
            v_arrival_vec = (ss_arrival.v - ss_neptune.v).to(u.km / u.s)
            arrival_vinf = np.linalg.norm(v_arrival_vec.value)

            # Store values
            arrival_vinf_list.append(arrival_vinf)
            tof_list.append(tof_years)
            c3_list.append(C3)

        except Exception:
            continue

# Plot it
plt.figure(figsize=(8, 6))
sc = plt.scatter(arrival_vinf_list, tof_list, c=c3_list, cmap='jet', s=25)
cbar = plt.colorbar(sc)
cbar.set_label("C3 (km²/s²)", fontsize=14)      # Label font size
cbar.ax.tick_params(labelsize=12)               # Tick font size on colorbarplt.xlabel("Neptune Arrival $v_\\infty$ (km/s)", fontsize=14)plt.xlabel("Arrival $v_\\infty$ (km/s)")
plt.xlabel("Neptune Arrival $v_\\infty$ (km/s)", fontsize=14)
plt.ylabel("Time of Flight (years)", fontsize=14)
plt.title("Earth To Neptune Trajectory Map", fontsize=14)
plt.tick_params(axis='both', labelsize=12)
# plt.axvline(x=12, linestyle="--", color="black", label="Propulsive Insertion Limit")
plt.grid(True)
# plt.legend()
plt.tight_layout()
plt.savefig("Direct Trajectory Map")
plt.show()


