# -*- coding: utf-8 -*-
"""
@author: Quiri
"""

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import PowerNorm
import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, interp1d
from sklearn.preprocessing import MinMaxScaler
import math

# ─── 1) Load & clean data ─────────────────────────────────────────────
CSV_PATH = r'C:\Users\Quiri\OneDrive\Dokumente\UIUC\Corsework\25Spring\AE598 Planetary Entry\Final Project\2021-05-26 DMPA_compiled_data.csv'  # CHANGE TO YOUR OWN FILE LOCATION

df = pd.read_csv(CSV_PATH)

# Columns
dens_col = 'FreestreamH'
vel_col  = 'Freestream velocity, m/s'
B0_col   = 'Magnet strength, T'
rc_col   = 'Magnet radius, m'
cdA_col  = 'Effective CD*A, m^2'


df = df.dropna(subset=[dens_col, vel_col, B0_col, rc_col, cdA_col])
# ─── 2) Assemble full arrays ────────────────────────────────────────────────
X = df[[dens_col, vel_col, B0_col, rc_col]].values
Y = df[cdA_col].values

# ─── 3) Normalize on the *full* dataset ────────────────────────────────────
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)                  # fit on all rows
orig_min = scaler.data_min_    # true min per column
orig_max = scaler.data_max_    # true max per column

# ─── 4) Random subsample for interpolator ─────────────────────────────────
n_samples = 2000               # pick however many points you can afford
rng = np.random.default_rng(42)
idx = rng.choice(X.shape[0], size=n_samples, replace=False)

X_small = X[idx]
Y_small = Y[idx]

# scale your small cloud
X_small_scaled = scaler.transform(X_small)

# build interpolators
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
interp_lin = LinearNDInterpolator(X_small_scaled, Y_small, fill_value=np.nan)
interp_nn  = NearestNDInterpolator( X_small_scaled, Y_small)

# ─── 5) Estimator uses true min/max and scaler ─────────────────────────────
def estimate_cdA_clamped(density, velocity, B0, rc, c_D_sc, A_sc):
    if (B0 <= 0.0):
        return c_D_sc*A_sc
    # Convert Density to Number Density
    density = density / 3.347e-27
    pt = np.array([[density, velocity, B0, rc]])
    # clamp using *true* orig_min/orig_max
    pt_clamped = np.clip(pt, orig_min, orig_max)
    pt_scaled  = scaler.transform(pt_clamped)
    val = interp_lin(pt_scaled)[0]
    cdA = 0.0
    if np.isnan(val):
        cdA = float(interp_nn(pt_scaled)[0])
    else:
        cdA = float(val)
    return c_D_sc*A_sc + cdA

# ─── 6) Main estimator with extrapolation BELOW minimum B0 ─────────────
B0_min_in_data = df[B0_col].min()

def estimate_cdA(density, velocity, B0, rc, c_D_sc, A_sc):
    if B0 < B0_min_in_data:
        cdA_min = estimate_cdA_clamped(density, velocity, B0_min_in_data, rc, c_D_sc, A_sc)
        slope = cdA_min / B0_min_in_data  # Line from (0, 0) to (B0_min, cdA_min)
        return c_D_sc*A_sc + slope * B0
    else:
        return estimate_cdA_clamped(density, velocity, B0, rc, c_D_sc, A_sc)

def magnet_dipole(B, rc, material, dt_manuever):
    mu0 = 4 * np.pi * 1e-7

    # Power unit allocation factors
    alpha_ppu = 6
    alpha_bg = 250
    alpha_bV = 680

    if material == 'Cu':
        rho_B = 8.96e3
        cond_B = 59.6e6
        cp_B = 0.39
        T_melt = 1084
    elif material == 'Al':
        rho_B = 2700
        cond_B = 37.7e6
        cp_B = 0.897
        T_melt = 660
    else:
        raise ValueError('Material values not programmed')

    NI = B * 2 * rc / mu0

    # AWG wire table: AWG, diameter (mm), current capacity (A)
    AWG_WireTable = np.array([
        [0, 8.25246, 245],
        [1, 7.34822, 211],
        [2, 6.54304, 181],
        [3, 5.82676, 158],
        [4, 5.18922, 135],
        [5, 4.62026, 118],
        [6, 4.1148, 101],
        [7, 3.66522, 89],
        [8, 3.2639, 73],
        [9, 2.90576, 64],
        [10, 2.58826, 55],
        [11, 2.30378, 47],
        [12, 2.05232, 41],
        [13, 1.8288, 35],
        [14, 1.62814, 32],
        [15, 1.45034, 28],
        [16, 1.29032, 22],
        [17, 1.15062, 19],
        [18, 1.02362, 16],
        [19, 0.91186, 14],
        [20, 0.8128, 11],
        [21, 0.7239, 9],
        [22, 0.64516, 7],
        [23, 0.57404, 4.7],
        [24, 0.51054, 3.5],
        [25, 0.45466, 2.7]
    ])

    N_stor = np.zeros(len(AWG_WireTable))
    I_stor = np.zeros(len(AWG_WireTable))

    for ii in range(len(AWG_WireTable)):
        N_start = 100
        n = 0
        I = NI / (N_start + n * 100)
        I_limit = AWG_WireTable[ii, 2]
        nold = n

        while I > I_limit:
            nold = n
            n += 1
            I = NI / (N_start + n * 100)

        I *= 0.95
        Np = N_start + nold * 100

        for step in [100, 10, 1]:
            for _ in range(20 if step == 100 else 50 if step == 10 else 100):
                Bper = Np * I * mu0 / (2 * rc)
                Np += step if Bper < B else -step

        N = Np
        err = abs(B - Bper) / B
        tol = 1e-5

        while tol < err:
            I_old = I
            Bper = N * I * mu0 / (2 * rc)
            dB = B - Bper
            I = I_old + np.sign(dB) * abs(dB) * 2 * rc / (N * mu0)
            err = abs(I - I_old)

        N_stor[ii] = N
        I_stor[ii] = I

    diameters_m = AWG_WireTable[:, 1] / 1000
    Aw = (np.pi / 4) * diameters_m**2

    mc = 2 * np.pi * rc * N_stor * Aw * rho_B
    R = 2 * np.pi * rc * N_stor / (Aw * cond_B)
    Pm = I_stor**2 * R

    mb = 2 * (1 / alpha_bg) * Pm * dt_manuever / 3600
    Vb = 2 * (1 / alpha_bV) * 0.001 * Pm * dt_manuever / 3600
    mppu = alpha_ppu * Pm / 1000

    Tm_dot = Pm / (mc * cp_B * 1000)
    magnetPASS_all = Tm_dot * dt_manuever < T_melt

    valid_idx = np.where(magnetPASS_all)[0]
    if len(valid_idx) == 0:
        raise ValueError("No valid configurations: all magnets would melt over duration of maneuver.")

    total_mass = 1.5 * (mc + mb + mppu)
    best_idx = valid_idx[np.argmin(total_mass[valid_idx])]

    mass = total_mass[best_idx]
    volume = 1.2 * (mc[best_idx] / rho_B + Vb[best_idx])
    N = int(N_stor[best_idx])
    I = I_stor[best_idx]
    magnetPASS = True

    return mass, N, I, volume, magnetPASS

def sphere_cone_area(radius, half_angle_deg):
    θ = math.radians(half_angle_deg)
    A_cap = 2 * math.pi * radius**2 * (1 - math.cos(θ))
    l = radius / math.sin(θ)
    A_cone = math.pi * radius * l
    return A_cap + A_cone

def SecondOrderApprox (val, a, b, c):
    return a*val**2+b*val+c

def required_TPS_mass (gamma_entry, beta, density_FL, density_RL, dia=5, half_angle=70):

    gamma_entry = gamma_entry*180/np.pi
    
    m_FL = 0
    t_FL = 0
    m_RL = 0
    t_RL = 0
    
    FL_coeffs = [
    [36, -5.32e-06,     5.59e-03,     0.044333011],
    [32, -4.88e-06,     5.46e-03,     0.074707797],
    [28, -5.32e-06,     0.005923148,  0.022611755],
    [24, -5.32e-06,     0.006135957,  0.02394145],
    [20, -4.43e-06,     0.006180289,  0.028375221],
    [16, -4.43e-06,     0.007244348, -0.126799727]]

    # Format: [gamma, a, b, c]
    RL_coeffs = [
    [36, 2.63e-06,     -0.002628746,  1.599271672],
    [32, 2.33e-06,     -0.00252078,   1.644346041],
    [28, 2.33e-06,     -0.002614145,  1.73858451],
    [24, 2.04e-06,     -0.002599555,  1.861851747],
    [20, 1.75e-06,     -0.002643331,  2.072648413],
    [16, 2.04e-06,     -0.003183084,  2.655434847]]
    

    
    """
    #This is old, tries to estimate based on heat load instead of beta
    FL_coeffs = [
    [36, -9.71803149463071e-16, 1.16720467486952e-07, -1.40768185495427],
    [32, -4.57123451117847E-16, 8.09758844260935E-08, -1.07643685203732],
    [28, -2.74850138050129E-16, 5.80286401438562E-08, -0.951359856785116],
    [24,  7.7445326465627E-14, -4.67512142663092E-06, 71.3485120183841],
    [20,  3.96828452613269E-08, -0.322558222276882, 655471.878892937],
    [16,  0.000263613496953446, -520.637280272095, 257064966.131174]]
    """
    

    a1 = next(row[1] for row in FL_coeffs if row[0] == 24)
    b1 = next(row[2] for row in FL_coeffs if row[0] == 24)
    c1 = next(row[3] for row in FL_coeffs if row[0] == 24)
    a2 = next(row[1] for row in FL_coeffs if row[0] == 28)
    b2 = next(row[2] for row in FL_coeffs if row[0] == 28)
    c2 = next(row[3] for row in FL_coeffs if row[0] == 28)
    a3 = next(row[1] for row in FL_coeffs if row[0] == 24)
    b3 = next(row[2] for row in RL_coeffs if row[0] == 24)
    c3 = next(row[3] for row in RL_coeffs if row[0] == 24)
    a4 = next(row[1] for row in RL_coeffs if row[0] == 28)
    b4 = next(row[2] for row in RL_coeffs if row[0] == 28)
    c4 = next(row[3] for row in RL_coeffs if row[0] == 28)
    

    y1_FL = SecondOrderApprox(beta, a1, b1, c1)
    y2_FL = SecondOrderApprox(beta, a2, b2, c2)
    y1_RL = SecondOrderApprox(beta, a3, b3, c3)
    y2_RL = SecondOrderApprox(beta, a4, b4, c4)
        
    m_FL, t_FL = np.polyfit([16, 20], [y1_FL, y2_FL], 1)
    m_RL, t_RL = np.polyfit([16, 20], [y1_RL, y2_RL], 1)

    FL_thickness = max(0, m_FL*gamma_entry+t_FL)
    RL_thickness = max(0, m_RL*gamma_entry+t_RL)
    
    A = sphere_cone_area(dia / 2.0, half_angle)
    
    V_FL = (FL_thickness/100) * A
    V_RL = (RL_thickness/100) * A

    return density_FL * V_FL + density_RL * V_RL


def rho(z):
    
    h_data = np.array([
    -85000, -80000, -75000, -70000, -65000, -60000, -55000, -50000,
    -45000, -40000, -35000, -30000, -25000, -20000, -15000, -10000,
    -5000, 0, 5000, 10000, 15000, 20000, 25000, 30000,
    35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000,
    75000, 80000, 85000, 90000, 95000, 100000, 105000, 110000,
    115000, 120000, 125000, 130000, 135000, 140000, 145000, 150000,
    155000, 160000, 165000, 170000, 175000, 180000, 185000, 190000,
    195000, 200000, 205000, 210000, 215000, 220000, 225000, 230000,
    235000, 240000, 245000, 250000, 255000, 260000, 265000, 270000,
    275000, 280000, 285000, 290000, 295000, 300000, 305000, 310000,
    315000, 320000, 325000, 330000, 335000, 340000, 345000, 350000,
    355000, 360000, 365000, 370000, 375000, 380000, 385000, 390000,
    395000, 400000, 405000, 410000, 415000, 420000, 425000, 430000,
    435000, 440000, 445000, 450000, 455000, 460000, 465000, 470000,
    475000, 480000, 485000, 490000, 495000, 500000, 505000, 510000,
    515000, 520000, 525000, 530000, 535000, 540000, 545000, 550000,
    555000, 560000, 565000, 570000, 575000, 580000, 585000, 590000,
    595000, 600000, 605000, 610000, 615000, 620000, 625000, 630000,
    635000, 640000, 645000, 650000, 655000, 660000, 665000, 670000,
    675000, 680000, 685000, 690000, 695000, 700000, 705000, 710000,
    715000, 720000, 725000, 730000, 735000, 740000, 745000, 750000,
    755000, 760000, 765000, 770000, 775000, 780000, 785000, 790000,
    795000, 800000, 805000, 810000, 815000, 820000, 825000, 830000,
    835000, 840000, 845000, 850000, 855000, 860000, 865000, 870000,
    875000, 880000, 885000, 890000, 895000, 900000, 905000, 910000,
    915000, 920000, 925000, 930000, 935000, 940000, 945000, 950000,
    955000, 960000, 965000, 970000, 975000, 980000, 985000, 990000,
    995000, 1000000, 1010000, 1020000, 1030000, 1040000, 1050000, 1060000,
    1070000, 1080000, 1090000, 1100000, 1110000, 1120000, 1130000, 1140000,
    1150000, 1160000, 1170000, 1180000, 1190000, 1200000, 1210000, 1220000,
    1230000, 1240000, 1250000, 1260000, 1270000, 1280000, 1290000, 1300000,
    1310000, 1320000, 1330000, 1340000, 1350000, 1360000, 1370000, 1380000,
    1390000, 1400000, 1410000, 1420000, 1430000, 1440000, 1450000, 1460000,
    1470000, 1480000, 1490000, 1500000, 1510000, 1520000, 1530000, 1540000,
    1550000, 1560000, 1570000, 1580000, 1590000, 1600000, 1610000, 1620000,
    1630000, 1640000, 1650000, 1660000, 1670000, 1680000, 1690000, 1700000,
    1710000, 1720000, 1730000, 1740000, 1750000, 1760000, 1770000, 1780000,
    1790000, 1800000, 1810000, 1820000, 1830000, 1840000, 1850000, 1860000,
    1870000, 1880000, 1890000, 1900000, 1910000, 1920000, 1930000, 1940000,
    1950000, 1960000, 1970000, 1980000, 1990000, 2000000, 2010000, 2020000,
    2030000, 2040000, 2050000, 2060000, 2070000, 2080000, 2090000, 2100000,
    2110000, 2120000, 2130000, 2140000, 2150000, 2160000, 2170000, 2180000,
    2190000, 2200000, 2210000, 2220000, 2230000, 2240000, 2250000, 2260000,
    2270000, 2280000, 2290000, 2300000, 2310000, 2320000, 2330000, 2340000,
    2350000, 2360000, 2370000, 2380000, 2390000, 2400000, 2410000, 2420000,
    2430000, 2440000, 2450000, 2460000, 2470000, 2480000, 2490000, 2500000,
    2510000, 2520000, 2530000, 2540000, 2550000, 2560000, 2570000, 2580000,
    2590000, 2600000, 2610000, 2620000, 2630000, 2640000, 2650000, 2660000,
    2670000, 2680000, 2690000, 2700000, 2710000, 2720000, 2730000, 2740000,
    2750000, 2760000, 2770000, 2780000, 2790000, 2800000, 2810000, 2820000,
    2830000, 2840000, 2850000, 2860000, 2870000, 2880000, 2890000, 2900000,
    2910000, 2920000, 2930000, 2940000, 2950000, 2960000, 2970000, 2980000,
    2990000, 3000000, 3010000, 3020000, 3030000, 3040000, 3050000, 3060000,
    3070000, 3080000, 3090000, 3100000, 3110000, 3120000, 3130000, 3140000,
    3150000, 3160000, 3170000, 3180000, 3190000, 3200000, 3210000, 3220000,
    3230000, 3240000, 3250000, 3260000, 3270000, 3280000, 3290000, 3300000,
    3310000, 3320000, 3330000, 3340000, 3350000, 3360000, 3370000, 3380000,
    3390000, 3400000, 3410000, 3420000, 3430000, 3440000, 3450000, 3460000,
    3470000, 3480000, 3490000, 3500000, 3510000, 3520000, 3530000, 3540000,
    3550000, 3560000, 3570000, 3580000, 3590000, 3600000, 3610000, 3620000,
    3630000, 3640000, 3650000, 3660000, 3670000, 3680000, 3690000, 3700000,
    3710000, 3720000, 3730000, 3740000, 3750000, 3760000, 3770000, 3780000,
    3790000, 3800000, 3810000, 3820000, 3830000, 3840000, 3850000, 3860000,
    3870000, 3880000, 3890000, 3900000, 3910000, 3920000, 3930000, 3940000,
    3950000, 3960000, 3970000, 3980000, 3990000, 4000000
    ])
    
    rho_data = np.array([
    2.5887e+00, 2.3576e+00, 2.1484e+00, 1.9591e+00, 1.7877e+00, 1.6325e+00,
    1.4855e+00, 1.3506e+00, 1.2274e+00, 1.1049e+00, 9.9682e-01, 9.0107e-01,
    8.0693e-01, 7.2107e-01, 6.4104e-01, 5.6172e-01, 4.9537e-01, 4.4021e-01,
    3.6520e-01, 3.0089e-01, 2.4125e-01, 1.8649e-01, 1.4057e-01, 1.0360e-01,
    7.5269e-02, 5.4321e-02, 3.8557e-02, 2.7114e-02, 1.9138e-02, 1.3586e-02,
    9.6695e-03, 6.8958e-03, 4.9258e-03, 3.4432e-03, 2.4264e-03, 1.7363e-03,
    1.3558e-03, 1.0648e-03, 8.4045e-04, 7.0913e-04, 5.9899e-04, 5.0870e-04,
    4.3849e-04, 3.7803e-04, 3.2597e-04, 2.8112e-04, 2.4166e-04, 2.0680e-04,
    1.7710e-04, 1.5179e-04, 1.3019e-04, 1.1174e-04, 9.8283e-05, 8.6537e-05,
    7.6213e-05, 6.7139e-05, 5.9160e-05, 5.2142e-05, 4.5969e-05, 4.0497e-05,
    3.6143e-05, 3.2263e-05, 2.8804e-05, 2.5720e-05, 2.2971e-05, 2.0519e-05,
    1.8366e-05, 1.6492e-05, 1.4812e-05, 1.3306e-05, 1.1955e-05, 1.0743e-05,
    9.6393e-06, 8.6794e-06, 7.8164e-06, 7.0406e-06, 6.3431e-06, 5.7158e-06,
    5.1517e-06, 4.6443e-06, 4.1879e-06, 3.7617e-06, 3.3919e-06, 3.0591e-06,
    2.7596e-06, 2.4900e-06, 2.2474e-06, 2.0289e-06, 1.8267e-06, 1.6501e-06,
    1.4910e-06, 1.3476e-06, 1.2184e-06, 1.1019e-06, 9.9684e-07, 8.9856e-07,
    8.1306e-07, 7.3593e-07, 6.6633e-07, 6.0350e-07, 5.4678e-07, 4.9555e-07,
    4.4927e-07, 4.0745e-07, 3.6964e-07, 3.3251e-07, 3.0144e-07, 2.7337e-07,
    2.4800e-07, 2.2506e-07, 2.0430e-07, 1.8552e-07, 1.6655e-07, 1.4999e-07,
    1.3514e-07, 1.2181e-07, 1.0985e-07, 9.9101e-08, 8.9442e-08, 8.0517e-08,
    7.3317e-08, 6.6787e-08, 6.0861e-08, 5.5479e-08, 5.0591e-08, 4.6147e-08,
    4.2107e-08, 3.8431e-08, 3.5085e-08, 3.2038e-08, 2.9262e-08, 2.6808e-08,
    2.4722e-08, 2.2804e-08, 2.1037e-08, 1.9411e-08, 1.7914e-08, 1.6534e-08,
    1.5263e-08, 1.4091e-08, 1.3005e-08, 1.2081e-08, 1.1223e-08, 1.0428e-08,
    9.6898e-09, 9.0048e-09, 8.3690e-09, 7.7788e-09, 7.2308e-09, 6.7220e-09,
    6.2496e-09, 5.8270e-09, 5.4467e-09, 5.0917e-09, 4.7603e-09, 4.4509e-09,
    4.1619e-09, 3.8921e-09, 3.6400e-09, 3.4046e-09, 3.1847e-09, 2.9792e-09,
    2.7872e-09, 2.6078e-09, 2.4402e-09, 2.2835e-09, 2.1381e-09, 2.0125e-09,
    1.8944e-09, 1.7834e-09, 1.6790e-09, 1.5808e-09, 1.4885e-09, 1.4016e-09,
    1.3200e-09, 1.2432e-09, 1.1709e-09, 1.1019e-09, 1.0390e-09, 9.7891e-10,
    9.2720e-10, 8.7828e-10, 8.3198e-10, 7.8816e-10, 7.4669e-10, 7.0744e-10,
    6.7029e-10, 6.3512e-10, 6.0182e-10, 5.7030e-10, 5.4046e-10, 5.1220e-10,
    4.8545e-10, 4.6012e-10, 4.3723e-10, 4.1592e-10, 3.9568e-10, 3.7644e-10,
    3.5816e-10, 3.4079e-10, 3.2428e-10, 3.0859e-10, 2.9367e-10, 2.7950e-10,
    2.6602e-10, 2.5320e-10, 2.4102e-10, 2.2944e-10, 2.1842e-10, 2.0794e-10,
    1.9798e-10, 1.8851e-10, 1.7949e-10, 1.7092e-10, 1.6277e-10, 1.5551e-10,
    1.4898e-10, 1.4273e-10, 1.2976e-10, 1.1823e-10, 1.0797e-10, 9.8796e-11,
    9.0581e-11, 8.3206e-11, 7.6567e-11, 7.0578e-11, 6.5163e-11, 6.0257e-11,
    5.5803e-11, 5.1751e-11, 4.8058e-11, 4.4686e-11, 4.1602e-11, 3.8777e-11,
    3.6185e-11, 3.3802e-11, 3.1610e-11, 2.9588e-11, 2.7723e-11, 2.5999e-11,
    2.4404e-11, 2.2926e-11, 2.1556e-11, 2.0283e-11, 1.9100e-11, 1.7999e-11,
    1.6973e-11, 1.6016e-11, 1.5124e-11, 1.4290e-11, 1.3510e-11, 1.2780e-11,
    1.2097e-11, 1.1457e-11, 1.0856e-11, 1.0292e-11, 9.7619e-12, 9.2638e-12,
    8.7952e-12, 8.3542e-12, 7.9388e-12, 7.5472e-12, 7.1780e-12, 6.8296e-12,
    6.5007e-12, 6.1899e-12, 5.8962e-12, 5.6185e-12, 5.3557e-12, 5.1070e-12,
    4.8715e-12, 4.6483e-12, 4.4368e-12, 4.2362e-12, 4.0459e-12, 3.8652e-12,
    3.6937e-12, 3.5308e-12, 3.3760e-12, 3.2289e-12, 3.0890e-12, 2.9559e-12,
    2.8292e-12, 2.7086e-12, 2.5938e-12, 2.4845e-12, 2.3803e-12, 2.2809e-12,
    2.1862e-12, 2.0959e-12, 2.0097e-12, 1.9275e-12, 1.8490e-12, 1.7741e-12,
    1.7025e-12, 1.6341e-12, 1.5688e-12, 1.5064e-12, 1.4467e-12, 1.3896e-12,
    1.3350e-12, 1.2827e-12, 1.2327e-12, 1.1849e-12, 1.1391e-12, 1.0952e-12,
    1.0532e-12, 1.0130e-12, 9.7440e-13, 9.3745e-13, 9.0203e-13, 8.6807e-13,
    8.3551e-13, 8.0428e-13, 7.7433e-13, 7.4559e-13, 7.1801e-13, 6.9154e-13,
    6.6613e-13, 6.4174e-13, 6.1832e-13, 5.9582e-13, 5.7421e-13, 5.5345e-13,
    5.3351e-13, 5.1434e-13, 4.9592e-13, 4.7821e-13, 4.6118e-13, 4.4481e-13,
    4.2906e-13, 4.1392e-13, 3.9935e-13, 3.8533e-13, 3.7185e-13, 3.5887e-13,
    3.4638e-13, 3.3436e-13, 3.2278e-13, 3.1163e-13, 3.0090e-13, 2.9057e-13,
    2.8061e-13, 2.7102e-13, 2.6178e-13, 2.5288e-13, 2.4430e-13, 2.3604e-13,
    2.2807e-13, 2.2039e-13, 2.1298e-13, 2.0585e-13, 1.9896e-13, 1.9232e-13,
    1.8592e-13, 1.7975e-13, 1.7379e-13, 1.6805e-13, 1.6250e-13, 1.5715e-13,
    1.5199e-13, 1.4701e-13, 1.4220e-13, 1.3756e-13, 1.3308e-13, 1.2875e-13,
    1.2458e-13, 1.2054e-13, 1.1665e-13, 1.1289e-13, 1.0925e-13, 1.0575e-13,
    1.0236e-13, 9.9082e-14, 9.5918e-14, 9.2861e-14, 8.9907e-14, 8.7052e-14,
    8.4293e-14, 8.1627e-14, 7.9049e-14, 7.6558e-14, 7.4149e-14, 7.1820e-14,
    6.9568e-14, 6.7391e-14, 6.5286e-14, 6.3250e-14, 6.1280e-14, 5.9376e-14,
    5.7533e-14, 5.5751e-14, 5.4027e-14, 5.2359e-14, 5.0745e-14, 4.9183e-14,
    4.7672e-14, 4.6210e-14, 4.4794e-14, 4.3425e-14, 4.2099e-14, 4.0816e-14,
    3.9573e-14, 3.8371e-14, 3.7207e-14, 3.6079e-14, 3.4988e-14, 3.3931e-14,
    3.2908e-14, 3.1917e-14, 3.0957e-14, 3.0028e-14, 2.9127e-14, 2.8255e-14,
    2.7410e-14, 2.6592e-14, 2.5799e-14, 2.5031e-14, 2.4287e-14, 2.3566e-14,
    2.2867e-14, 2.2190e-14, 2.1534e-14, 2.0898e-14, 2.0282e-14, 1.9685e-14,
    1.9106e-14, 1.8545e-14, 1.8001e-14, 1.7474e-14, 1.6963e-14, 1.6467e-14,
    1.5987e-14, 1.5521e-14, 1.5069e-14, 1.4631e-14, 1.4206e-14, 1.3794e-14,
    1.3395e-14, 1.3008e-14, 1.2632e-14, 1.2267e-14, 1.1914e-14, 1.1571e-14,
    1.1238e-14, 1.0916e-14, 1.0603e-14, 1.0299e-14, 1.0005e-14, 9.7186e-15,
    9.4412e-15, 9.1721e-15, 8.9109e-15, 8.6575e-15, 8.4116e-15, 8.1729e-15,
    7.9412e-15, 7.7164e-15, 7.4982e-15, 7.2864e-15, 7.0808e-15, 6.8812e-15,
    6.6875e-15, 6.4994e-15, 6.3168e-15, 6.1396e-15, 5.9675e-15, 5.8004e-15,
    5.6382e-15, 5.4806e-15, 5.3277e-15, 5.1791e-15, 5.0349e-15, 4.8948e-15,
    4.7588e-15, 4.6267e-15, 4.4984e-15, 4.3737e-15, 4.2527e-15, 4.1351e-15,
    4.0209e-15, 3.9100e-15, 3.8022e-15, 3.6975e-15, 3.5958e-15, 3.4970e-15,
    3.4010e-15, 3.3078e-15, 3.2171e-15, 3.1291e-15, 3.0435e-15, 2.9604e-15,
    2.8796e-15, 2.8011e-15, 2.7248e-15, 2.6507e-15, 2.5787e-15, 2.5086e-15,
    2.4406e-15, 2.3744e-15, 2.3101e-15, 2.2477e-15, 2.1869e-15, 2.1279e-15,
    2.0705e-15, 2.0147e-15, 1.9604e-15, 1.9077e-15, 1.8564e-15, 1.8066e-15,
    1.7581e-15, 1.7110e-15, 1.6652e-15, 1.6207e-15, 1.5774e-15, 1.5352e-15,
    1.4943e-15, 1.4545e-15
    ])
    
    # Create the interpolation function
    rho_interp = interp1d(h_data, rho_data, bounds_error=False, fill_value="extrapolate")
    
    # Evaluate the function at a given altitude z
    rho = rho_interp(z)
    
    return rho
    
def g(z):
    # Constants of Neptune
    G = 6.67430e-11  # [m^3/(kg*s^2)], universal gravitational constant
    M_Nep = 1.02409e26  # [kg], Mass of Neptune
    R_Nep = 24622000  # [m], Mean Radius of Neptune
    return G * M_Nep / (R_Nep + z)**2  # [m/s^2]

def simulate_entry(v0, gamma0, B0, z0, dt, rc, c_D, A, m_Vehicle, LtoD, rn, R, density_FL, density_RL, plot, Q_aero):
    B = B0
    v = [v0]
    a = [v0 * dt]
    gamma = [gamma0]
    Q = [0.0]
    Q_dot = []
    Q_dot_conv = []
    Q_dot_rad = []
    z = [z0]
    t = [0.0]
    beta = []
    t_turn_off_magnet = 0.0
    
    while z[-1] <= z0:
    #for i in range(50):
        # Correction of equations of motion: https://en.wikipedia.org/wiki/Planar_reentry_equations
            
        rho_val = rho(z[-1])
        g_val = g(z[-1])
        beta_val = m_Vehicle / estimate_cdA(rho_val, v[-1], B, rc, c_D, A)
        beta.append(beta_val)
        
        # Acceleration
        new_a = -(rho_val * v[-1]**2) / (2 * beta_val) + g_val * np.sin(gamma[-1])
        a.append(new_a)
        
        # Velocity
        new_v = v[-1] + new_a * dt
        v.append(new_v)
        # Check if disable magnet
        if new_v < 23.56*1000 and B > 0:
            B = 0
            t_turn_off_magnet = t[-1]
        
        # Trajectory angle
        new_gamma = gamma[-1] - (rho_val * v[-1]) / (2 * beta_val) * LtoD \
                    + (g_val / v[-1]) * np.cos(gamma[-1]) \
                    - (v[-1] * np.cos(gamma[-1])) / (R + z[-1])
        gamma.append(new_gamma)        
        
        # Altitude
        z_new = z[-1] - v[-1] * np.sin(gamma[-1]) * dt
        z.append(z_new)
        
        # Convective heat flux
        K2 = 74.954*((v[-1]/45000)**2-0.1237*(v[-1])/45000)*rn**(-0.0190)*(np.log10(rho_val)/-5)**(-0.0309)
        Q_dot_conv_new = K2*np.sqrt(rho_val*v[-1]**2/rn)*(0.5453*v[-1]**2-2803*v[-1])/10**6
        Q_dot_conv_new = max(0.0, Q_dot_conv_new)
        Q_dot_conv.append(Q_dot_conv_new)
        
        # Radiative heat flux
        Q_dot_rad_ad = ((1-0.8)/0.8)*Q_dot_conv_new
        BigGamma = 4*Q_dot_rad_ad/(rho_val*v[-1]**3)
        if BigGamma > 0:
            Q_dot_rad_new = Q_dot_rad_ad / (1 + 3 * BigGamma**0.7)
        else:
            Q_dot_rad_new = 0
        Q_dot_rad.append(Q_dot_rad_new)
        
        #Total heat flux
        Q_dot_new = Q_dot_conv[-1]+Q_dot_rad[-1]
        Q_dot.append(Q_dot_new)
        
        # Heat
        Q_new = Q[-1] + Q_dot_new * dt
        Q.append(Q_new)
        
        # Time step
        t_new = t[-1] + dt
        t.append(t_new)
        
        if z[-1] <=0 or v[-1] < 0:
            v[-1] = 0
            z[-1] = 0
            break

    #print (Q[-1])
    mass_TPS = 0
    mass_Magnetoshell = 0
    
    if (v[-1] < 23.56*1000 and z[-1] >= z0 and v[-1] > 0):
        if B0 > 0:
            mass_Magnetoshell, N, I, volume, magnetPASS = magnet_dipole(B0, rc, 'Al', t_turn_off_magnet)
            
            #beta_low_avg = 0
            #max_beta = max(beta)
            #max_index = beta.index(max_beta)
            #beta_aero = m_Vehicle/(c_D*A)
            #beta_truncated = beta[:max_index]
            #for beta_low_avg_i in beta_truncated:
            #    beta_low_avg = beta_low_avg_i * dt 
            #u_b = 10
            #v_b = 1
            #print("Term 1: " + str(u_b*beta_low_avg/beta_aero/(u_b+v_b)))
            #print("Term 2: " + str(v_b*(1.0-t_turn_off_magnet/t[-1])/(u_b+v_b)))
            
            #beta_TPS = beta_aero * (u_b*beta_low_avg/beta_aero+v_b*(1.0-t_turn_off_magnet/t[-1]))/(u_b+v_b)
            #beta_at_q_dot_max = beta[Q_dot.index(max(Q_dot))]
            
            
            y1_beta = m_Vehicle / (c_D*A)
            y2_beta = 0.0
            #x1_beta = rho(z_lowest)
            #x2_beta = rho(z0)
            x1_beta = Q_aero
            x2_beta = 0
            
            m_beta, t_beta = np.polyfit([x1_beta, x2_beta], [y1_beta, y2_beta], 1)
            
            beta_TPS = m_beta*Q[-1]+t_beta
            #beta_TPS = m_beta*max(Q_dot)+t_beta
            #beta_TPS = sum(beta) / len(beta)
            
            mass_TPS = required_TPS_mass(gamma[0], beta_TPS, density_FL, density_RL, np.sqrt(4*A/np.pi))
        else:
            mass_TPS = required_TPS_mass(gamma[0], beta[-1], density_FL, density_RL, np.sqrt(4*A/np.pi))
            
    mass_ACS = mass_Magnetoshell+mass_TPS
    
    output = [mass_ACS, mass_TPS, mass_Magnetoshell, Q[-1]]
    
    if (plot == False):
        return output
        #return Q[-1] #for calculating Q values
    
    print(f"\nTotal heat load: {Q[-1]:.2f} J/m²")
    
    print(f"\nRequired TPS mass: {mass_TPS:.2f} kg")
    # Plotting
    # Set global font size
    plt.rcParams.update({'font.size': 14})  # Adjust this value as needed
    
    plt.figure(figsize=(12, 12))
    
    # Altitude vs Time
    plt.subplot(5, 1, 1)
    plt.plot(t, z, label='Altitude (z)', color='blue')
    plt.ylabel('Altitude (m)', fontsize=14)
    plt.grid()
    plt.legend(fontsize=12)
    
    # Velocity vs Time
    plt.subplot(5, 1, 2)
    plt.plot(t, v, label='Velocity (v)', color='green')
    plt.ylabel('Velocity (m/s)', fontsize=14)
    plt.grid()
    plt.legend(fontsize=12)
    
    # Flight Path Angle vs Time
    plt.subplot(5, 1, 3)
    plt.plot(t, np.degrees(gamma), label='Flight Path Angle (γ)', color='purple')
    plt.ylabel('Angle (deg)', fontsize=14)
    plt.grid()
    plt.legend(fontsize=12)
    
    # Heat Flux vs Time
    plt.subplot(5, 1, 4)
    plt.plot(t[1:], Q_dot_conv, label='Convective $\dot{q}_{conv}$', color='orange', linestyle='--')
    plt.plot(t[1:], Q_dot_rad, label='Radiative $\dot{q}_{rad}$', color='teal', linestyle='--')
    plt.plot(t[1:], Q_dot, label='Total $\dot{q}$', color='red')
    plt.ylabel('Heat Flux (W/m²)', fontsize=14)
    plt.grid()
    plt.legend(fontsize=12)
    
    # Ballistic Coefficient vs Time
    plt.subplot(5, 1, 5)
    plt.plot(t[1:], beta, label='Ballistic Coefficient ($\\beta$)', color='darkblue')
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Beta (kg/m²)', fontsize=14)
    plt.grid()
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return output
    

def convert_beta_to_Q (v, gamma, beta, LtoD, r_N):
    z0 = 1000000.0
    dt = 1.0
    c_D = 0.67
    A = np.pi*2.5**2
    m_Vehicle = c_D*beta*np.pi*2.5**2
    R = 6378137.0
    #make sure simulate_entry returns Q
    return simulate_entry(v, gamma, 0, z0, dt, 0, c_D, A, m_Vehicle, LtoD, r_N, R)
    
def create_coeffs(v):
    
    thickness_IL = [
    [1.347783304, 1.241379514, 1.108374592, 0.948768539],
    [1.387685001, 1.272413799, 1.134975355, 0.970936149],
    [1.445320418, 1.316749019, 1.174877053, 0.993103759],
    [1.520689923, 1.383251479, 1.228078948, 1.03743861],
    [1.649261323, 1.480788225, 1.299014931, 1.086206982],
    [1.866502424, 1.644827801, 1.409852242, 1.143842399]]
    
    thickness_RL = [
    [1.00072949, 1.047410761, 1.105762167, 1.17870179],
    [1.047410761, 1.099927221, 1.158278628, 1.234135602],
    [1.108679762, 1.167031412, 1.228300656, 1.309992819],
    [1.202042304, 1.266229143, 1.339168523, 1.423778281],
    [1.362509279, 1.435448659, 1.522976012, 1.61342096],
    [1.79139326, 1.884755802, 1.986870885, 2.100656589]]
    
    gammas_deg = [36, 32, 28, 24, 20, 16]
    betas = [350, 300, 250, 200]
    """
    Q_matrix = []
    
    for gamma in gammas_deg:
        Q_in = []
        for beta in betas:
            Q_in.append(convert_beta_to_Q(v, gamma*np.pi/180, beta, 0.24, 1.125))
            
        Q_matrix.append(Q_in)
    """
    
    a, b, c = np.polyfit(betas, thickness_IL[5], 2)
    print(a)
    print(b)
    print(c)
 
# For creating heat shield thickness data
#create_coeffs(24.73*1000)
    

# Extract the 'Speed' and 'Flight Path' columns: R = 6378137.0
#simulate_entry(v, gamma, 0, z0, dt, 0, c_D, A, m_Vehicle, LtoD, r_N, R)

    
    
# Define the row limit
row_limit = 3  # Change this value as needed

# Read the Excel file with the row limit
#df = pd.read_excel('neptune_results.xlsx', nrows=row_limit)  # Replace with your actual file path


# Extract the 'Speed' and 'Flight Path' columns:
#speed = df['Speed (km/s)'].to_numpy()
#flight_path = df['Flight Path (deg)'].to_numpy()

# Display the arrays
#print("Speed Array:", speed)
#print("Flight Path Array:", flight_path)

#for i in range(row_limit):
    #simulate_entry(speed[i]*1000, -flight_path[i]*np.pi/180, 1000000.0, 1.0, 0.67, np.pi*5.0**2, 2000.0, 0.24, 1.3, 6378137.0, 1.0)
    

#For conversion from beta to Q   
#simulate_entry(24.73*1000, 28*np.pi/180, 0.0, 1000000.0, 1.0, 5.0, 0.67, np.pi*2.5**2, 0.67*200*np.pi*2.5**2, 0.24, 1.125, 6378137.0)

#v0 = 23.8791085098175*1000 #Direct traj
v0 = 25.3510595492*1000 #Jupiter Fly by
gamma0 = 24.88*np.pi/180
z0 = 1000000.0
B0 = 0.000276
#B0 = 0.0
dt = 1.0
rc = 2.5
c_D = 0.67
A = np.pi*2.0**2
#beta = 199
#m_Vehicle = c_D*beta*np.pi*2.5**2
m_Vehicle = 2200
LtoD = 0.24
r_N = 1.125
R = 6378137.0
density_FL = 0.83*1000
density_RL = 1.1*1000
beta_pure_aero = m_Vehicle/(c_D*A)

gamma_deg = np.linspace(20, 29, 60)
gamma_in = np.deg2rad(gamma_deg)
B_in = np.linspace(0, 0.002, 30)
Q_aeros = []



for i in range(len(gamma_in)):
    out = simulate_entry(v0, gamma_in[i], 0.0, z0, dt, rc, c_D, A, m_Vehicle, LtoD, r_N, R, density_FL, density_RL, False, 0.0)
    Q_aeros.append(out[3])

# Create meshgrid
GAMMA, B = np.meshgrid(gamma_in, B_in)

# Compute mass values
mass = np.zeros_like(GAMMA)
for i in range(B.shape[0]):
    for j in range(GAMMA.shape[1]):
        out = simulate_entry(v0, GAMMA[i, j], B[i, j], z0, dt, rc, c_D, A, m_Vehicle, LtoD, r_N, R, density_FL, density_RL, False, Q_aeros[j])
        mass[i, j] = out[0]

#print(mass)

# Mask values that are exactly zero
mass_masked = np.ma.masked_where(mass == 0, mass)

# Use a modified colormap where masked (bad) values appear white
cmap = cm.get_cmap('viridis').copy()
cmap.set_bad(color='white')

# Plot heatmap
plt.figure(figsize=(8, 6))

# Plot the heatmap
heatmap = plt.imshow(
    mass_masked,
    extent=[gamma_deg[0], gamma_deg[-1], B_in[0], B_in[-1]],
    origin='lower',
    aspect='auto',
    cmap=cmap
)

# Set larger font sizes
plt.xlabel('Gamma (deg)', fontsize=16)
plt.ylabel('B (T)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Colorbar with larger font
cbar = plt.colorbar(heatmap)
cbar.set_label("Mass (kg)", fontsize=16)
cbar.ax.tick_params(labelsize=14)

# Optional: set a larger title
# plt.title("Mass Heatmap", fontsize=18)

plt.tight_layout()
plt.show()

# 1. Find the lowest overall mass
min_mass = np.min(mass_masked)
min_index = np.unravel_index(np.argmin(mass_masked), mass.shape)
min_gamma_deg = np.rad2deg(GAMMA[min_index])
min_B = B[min_index]

print(f"Lowest mass: {min_mass:.2f} kg at gamma = {min_gamma_deg:.2f} deg, B = {min_B:.6f} T")

# 2. Find the lowest mass where B_in = 0
mass_B_zero_raw = mass[0, :]  # first row corresponds to B = 0

# Filter out zero values
valid_mass_B_zero = mass_B_zero_raw[mass_B_zero_raw > 0]

if valid_mass_B_zero.size > 0:
    min_mass_B_zero = np.min(valid_mass_B_zero)
    min_index_B_zero = np.where(mass_B_zero_raw == min_mass_B_zero)[0][0]
    min_gamma_deg_B_zero = gamma_deg[min_index_B_zero]

    print(f"Lowest (nonzero) mass at B = 0: {min_mass_B_zero:.2f} kg at gamma = {min_gamma_deg_B_zero:.2f} deg")
else:
    print("No valid (nonzero) mass values found at B = 0.")
"""

out = simulate_entry(v0, gamma0, B0, z0, dt, rc, c_D, A, m_Vehicle, LtoD, r_N, R, density_FL, density_RL, True, 9240496.91)
print("Total Mass: " + str(out[0]))
print("TPS Mass: " + str(out[1]))
print("Magnetoshell Mass: " + str(out[2]))
"""