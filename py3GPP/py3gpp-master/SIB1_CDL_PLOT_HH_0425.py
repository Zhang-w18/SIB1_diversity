#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot 4x4 dual-pol UPA beamforming radiation patterns
using physical steering angles instead of DFT beam indices.

Goal:
  BS beam coverage:
    Horizontal: 120 deg  -> azimuth in [-60, +60]
    Vertical:   24 deg   -> elevation in [-12, +12]

Compare:
  - Array factor only
  - Array factor + BS element pattern

Notes:
  theta: zenith angle, 0 deg = upward z-axis, 90 deg = horizontal plane
  phi:   azimuth angle, 0 deg = broadside reference
  elevation = 90 - theta
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# Config
# =========================================================
M = 6          # vertical elements
N = 4          # horizontal elements
P = 2          # dual polarization, pattern shape only uses one pol spatial weight
dV = 0.8       # lambda
dH = 0.5       # lambda

OUT_DIR = "antenna_pattern_figures_angle"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Target sector coverage
# ---------------------------------------------------------
AZ_COVER_DEG = 120.0    # total horizontal sector coverage
EL_COVER_DEG = 24.0     # total vertical sector coverage

AZ_MIN = -AZ_COVER_DEG / 2.0
AZ_MAX =  AZ_COVER_DEG / 2.0
EL_MIN = -EL_COVER_DEG / 2.0
EL_MAX =  EL_COVER_DEG / 2.0

# Example scan beams:
# 8 horizontal beams across 120°, all at elevation = 0°
H_BEAM_AZ_LIST = np.linspace(AZ_MIN, AZ_MAX, 8)
H_BEAM_EL = 0.0

# 2 vertical beams across 24°, at azimuth = 0°
V_BEAM_EL_LIST = np.linspace(EL_MIN, EL_MAX, 2)
V_BEAM_AZ = 0.0

# Optional 2D beam set: 4x2 beams over the target coverage
AZ_2D_LIST = np.linspace(AZ_MIN, AZ_MAX, 4)
EL_2D_LIST = np.linspace(EL_MIN, EL_MAX, 2)

# ---------------------------------------------------------
# 3GPP BS element pattern (Table 7.3-1 style)
# ---------------------------------------------------------
# For shape comparison, keep max gain = 0 dBi and normalize peak to 0 dB
G_E_MAX_DBI = 0.0
THETA_3DB = 65.0
PHI_3DB = 65.0
SLA_V = 30.0
A_M = 30.0

NORMALIZE_TO_PEAK = True
FLOOR_DB = -45.0


# =========================================================
# Utilities
# =========================================================
def db10(x, floor=1e-15):
    return 10.0 * np.log10(np.maximum(np.asarray(x), floor))


def element_pattern_db(theta_deg, phi_deg,
                       theta_3db=65.0, phi_3db=65.0,
                       sla_v=30.0, a_m=30.0,
                       g_e_max_dbi=0.0):
    """
    3GPP-style single BS element pattern.

    theta_deg:
      zenith angle, broadside usually theta = 90 deg

    phi_deg:
      azimuth angle, broadside usually phi = 0 deg

    Return:
      element gain in dB
    """
    theta_deg = np.asarray(theta_deg, dtype=float)
    phi_deg = np.asarray(phi_deg, dtype=float)

    phi_wrapped = (phi_deg + 180.0) % 360.0 - 180.0

    A_v = -np.minimum(12.0 * ((theta_deg - 90.0) / theta_3db) ** 2, sla_v)
    A_h = -np.minimum(12.0 * (phi_wrapped / phi_3db) ** 2, a_m)
    A = -np.minimum(-(A_v + A_h), a_m)

    return g_e_max_dbi + A


def steering_vector(theta_deg, phi_deg, M=4, N=4, dV=0.5, dH=0.5):
    """
    UPA steering vector.

    Coordinate convention:
      vertical axis: z-axis
      horizontal axis: y-axis
      theta: zenith angle
      phi: azimuth angle

    Spatial frequency approximation:
      u_v = cos(theta)
      u_h = sin(theta) * sin(phi)

    Broadside:
      theta = 90 deg, phi = 0 deg => u_v = 0, u_h = 0

    Returns:
      a: shape [num_angles, M*N]
    """
    theta = np.deg2rad(np.asarray(theta_deg, dtype=float).ravel())
    phi = np.deg2rad(np.asarray(phi_deg, dtype=float).ravel())

    u_v = np.cos(theta)
    u_h = np.sin(theta) * np.sin(phi)

    mv = np.arange(M)
    mh = np.arange(N)

    a_v = np.exp(1j * 2.0 * np.pi * dV * np.outer(u_v, mv))
    a_h = np.exp(1j * 2.0 * np.pi * dH * np.outer(u_h, mh))

    # row-wise kron, spatial index = mv * N + mh
    a = np.einsum("iv,ih->ivh", a_v, a_h).reshape(len(theta), M * N)
    return a


def build_spatial_weight_from_angle(az_deg, el_deg, M=4, N=4, dV=0.5, dH=0.5):
    """
    Build one spatial beamforming vector from target azimuth/elevation.

    az_deg: azimuth angle around broadside, phi
    el_deg: elevation relative to horizon

    Since the steering_vector uses theta (zenith),
    convert by theta = 90 - elevation.
    """
    theta0 = 90.0 - el_deg
    phi0 = az_deg

    a0 = steering_vector(np.array([theta0]), np.array([phi0]), M=M, N=N, dV=dV, dH=dH)[0]
    w = a0 / np.linalg.norm(a0)
    return w.astype(np.complex128)


def array_factor_power(theta_deg, phi_deg, az_steer_deg, el_steer_deg):
    """
    Array factor power |a(theta,phi) @ conj(w)|^2
    """
    theta_arr = np.asarray(theta_deg, dtype=float)
    phi_arr = np.asarray(phi_deg, dtype=float)

    theta_flat = theta_arr.ravel()
    phi_flat = phi_arr.ravel()

    a = steering_vector(theta_flat, phi_flat, M=M, N=N, dV=dV, dH=dH)
    w = build_spatial_weight_from_angle(az_steer_deg, el_steer_deg, M=M, N=N, dV=dV, dH=dH)

    af = np.abs(a @ np.conj(w)) ** 2
    return af.reshape(theta_arr.shape)


def total_pattern_db(theta_deg, phi_deg, az_steer_deg, el_steer_deg,
                     with_element=True, normalize=True):
    """
    Total beam pattern in dB
    """
    af_pwr = array_factor_power(theta_deg, phi_deg, az_steer_deg, el_steer_deg)
    af_db = db10(af_pwr)

    if with_element:
        elem_db = element_pattern_db(
            theta_deg, phi_deg,
            theta_3db=THETA_3DB,
            phi_3db=PHI_3DB,
            sla_v=SLA_V,
            a_m=A_M,
            g_e_max_dbi=G_E_MAX_DBI
        )
        pat_db = af_db + elem_db
    else:
        pat_db = af_db

    if normalize:
        pat_db = pat_db - np.max(pat_db)

    return np.maximum(pat_db, FLOOR_DB)


# =========================================================
# Plot helpers
# =========================================================
def plot_horizontal_cut(az_steer_deg=0.0, el_steer_deg=0.0, theta_cut=None, fname=None):
    """
    Horizontal cut at fixed elevation / theta
    """
    if theta_cut is None:
        theta_cut = 90.0 - el_steer_deg

    phi = np.linspace(-90.0, 90.0, 1801)
    theta = np.full_like(phi, theta_cut)

    pat_no_elem = total_pattern_db(theta, phi, az_steer_deg, el_steer_deg,
                                   with_element=False, normalize=NORMALIZE_TO_PEAK)
    pat_with_elem = total_pattern_db(theta, phi, az_steer_deg, el_steer_deg,
                                     with_element=True, normalize=NORMALIZE_TO_PEAK)

    plt.figure(figsize=(7.4, 4.8))
    plt.plot(phi, pat_no_elem, label="Array factor only")
    plt.plot(phi, pat_with_elem, label="With BS element pattern")
    plt.grid(True, which="both")
    plt.xlabel("Azimuth angle φ (deg)")
    plt.ylabel("Normalized gain (dB)" if NORMALIZE_TO_PEAK else "Gain (dB)")
    plt.title(f"Horizontal Cut | steer(az={az_steer_deg:.1f}°, el={el_steer_deg:.1f}°)")
    plt.ylim(FLOOR_DB, 3)
    plt.xlim(-90, 90)

    # mark target sector
    plt.axvline(AZ_MIN, color="gray", linestyle="--", linewidth=1)
    plt.axvline(AZ_MAX, color="gray", linestyle="--", linewidth=1)

    plt.legend(loc="best")
    plt.tight_layout()

    if fname is None:
        fname = f"horizontal_cut_az{az_steer_deg:+.1f}_el{el_steer_deg:+.1f}.png".replace(" ", "")

    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_vertical_cut(az_steer_deg=0.0, el_steer_deg=0.0, phi_cut=None, fname=None):
    """
    Vertical cut at fixed azimuth / phi
    """
    if phi_cut is None:
        phi_cut = az_steer_deg

    theta = np.linspace(0.0, 180.0, 1801)
    phi = np.full_like(theta, phi_cut)
    elev = 90.0 - theta

    pat_no_elem = total_pattern_db(theta, phi, az_steer_deg, el_steer_deg,
                                   with_element=False, normalize=NORMALIZE_TO_PEAK)
    pat_with_elem = total_pattern_db(theta, phi, az_steer_deg, el_steer_deg,
                                     with_element=True, normalize=NORMALIZE_TO_PEAK)

    plt.figure(figsize=(7.4, 4.8))
    plt.plot(elev, pat_no_elem, label="Array factor only")
    plt.plot(elev, pat_with_elem, label="With BS element pattern")
    plt.grid(True, which="both")
    plt.xlabel("Elevation angle relative to horizon (deg)")
    plt.ylabel("Normalized gain (dB)" if NORMALIZE_TO_PEAK else "Gain (dB)")
    plt.title(f"Vertical Cut | steer(az={az_steer_deg:.1f}°, el={el_steer_deg:.1f}°)")
    plt.ylim(FLOOR_DB, 3)
    plt.xlim(-90, 90)

    # mark target sector
    plt.axvline(EL_MIN, color="gray", linestyle="--", linewidth=1)
    plt.axvline(EL_MAX, color="gray", linestyle="--", linewidth=1)

    plt.legend(loc="best")
    plt.tight_layout()

    if fname is None:
        fname = f"vertical_cut_az{az_steer_deg:+.1f}_el{el_steer_deg:+.1f}.png".replace(" ", "")

    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_all_horizontal_beams(with_element=True, theta_cut=90.0, fname=None):
    """
    Compare all horizontal steering beams over azimuth
    """
    phi = np.linspace(-90.0, 90.0, 1801)
    theta = np.full_like(phi, theta_cut)

    plt.figure(figsize=(7.6, 5.0))

    for az_deg in H_BEAM_AZ_LIST:
        pat = total_pattern_db(theta, phi, az_deg, H_BEAM_EL,
                               with_element=with_element,
                               normalize=NORMALIZE_TO_PEAK)
        plt.plot(phi, pat, label=f"az={az_deg:.1f}°")

    plt.grid(True, which="both")
    plt.xlabel("Azimuth angle φ (deg)")
    plt.ylabel("Normalized gain (dB)" if NORMALIZE_TO_PEAK else "Gain (dB)")
    suffix = "with BS element pattern" if with_element else "array factor only"
    plt.title(f"Horizontal Beam Sweep ({suffix})")
    plt.ylim(FLOOR_DB, 3)
    plt.xlim(-90, 90)
    plt.axvline(AZ_MIN, color="gray", linestyle="--", linewidth=1)
    plt.axvline(AZ_MAX, color="gray", linestyle="--", linewidth=1)
    plt.legend(loc="best", ncol=2, fontsize=9)
    plt.tight_layout()

    if fname is None:
        fname = f"horizontal_sweep_{'with_elem' if with_element else 'no_elem'}.png"

    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_all_vertical_beams(with_element=True, phi_cut=0.0, fname=None):
    """
    Compare vertical steering beams over elevation
    """
    theta = np.linspace(0.0, 180.0, 1801)
    phi = np.full_like(theta, phi_cut)
    elev = 90.0 - theta

    plt.figure(figsize=(7.6, 5.0))

    for el_deg in V_BEAM_EL_LIST:
        pat = total_pattern_db(theta, phi, V_BEAM_AZ, el_deg,
                               with_element=with_element,
                               normalize=NORMALIZE_TO_PEAK)
        plt.plot(elev, pat, label=f"el={el_deg:.1f}°")

    plt.grid(True, which="both")
    plt.xlabel("Elevation angle relative to horizon (deg)")
    plt.ylabel("Normalized gain (dB)" if NORMALIZE_TO_PEAK else "Gain (dB)")
    suffix = "with BS element pattern" if with_element else "array factor only"
    plt.title(f"Vertical Beam Sweep ({suffix})")
    plt.ylim(FLOOR_DB, 3)
    plt.xlim(-90, 90)
    plt.axvline(EL_MIN, color="gray", linestyle="--", linewidth=1)
    plt.axvline(EL_MAX, color="gray", linestyle="--", linewidth=1)
    plt.legend(loc="best")
    plt.tight_layout()

    if fname is None:
        fname = f"vertical_sweep_{'with_elem' if with_element else 'no_elem'}.png"

    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_2d_pattern(az_steer_deg=0.0, el_steer_deg=0.0, with_element=True, fname=None):
    """
    2D heatmap over azimuth/elevation
    """
    az = np.linspace(-90.0, 90.0, 721)
    el = np.linspace(-40.0, 40.0, 401)
    AZ, EL = np.meshgrid(az, el)

    THETA = 90.0 - EL
    PHI = AZ

    PAT = total_pattern_db(THETA, PHI, az_steer_deg, el_steer_deg,
                           with_element=with_element,
                           normalize=NORMALIZE_TO_PEAK)

    plt.figure(figsize=(8.0, 5.4))
    im = plt.imshow(
        PAT,
        extent=[az.min(), az.max(), el.min(), el.max()],
        origin="lower",
        aspect="auto",
        vmin=FLOOR_DB,
        vmax=0.0,
        cmap="jet"
    )
    plt.colorbar(im, label="Normalized gain (dB)")
    plt.xlabel("Azimuth angle φ (deg)")
    plt.ylabel("Elevation angle (deg)")
    suffix = "with BS element pattern" if with_element else "array factor only"
    plt.title(f"2D Pattern | steer(az={az_steer_deg:.1f}°, el={el_steer_deg:.1f}°), {suffix}")

    # sector boundary
    plt.axvline(AZ_MIN, color="white", linestyle="--", linewidth=1)
    plt.axvline(AZ_MAX, color="white", linestyle="--", linewidth=1)
    plt.axhline(EL_MIN, color="white", linestyle="--", linewidth=1)
    plt.axhline(EL_MAX, color="white", linestyle="--", linewidth=1)

    plt.tight_layout()

    if fname is None:
        fname = f"pattern2d_az{az_steer_deg:+.1f}_el{el_steer_deg:+.1f}_{'with_elem' if with_element else 'no_elem'}.png".replace(" ", "")

    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def main():
    saved = []

    # -----------------------------------------------------
    # Main demonstration: one central beam
    # -----------------------------------------------------
    saved.append(plot_horizontal_cut(az_steer_deg=0.0, el_steer_deg=0.0))
    saved.append(plot_vertical_cut(az_steer_deg=0.0, el_steer_deg=0.0))
    saved.append(plot_2d_pattern(az_steer_deg=0.0, el_steer_deg=0.0, with_element=False))
    saved.append(plot_2d_pattern(az_steer_deg=0.0, el_steer_deg=0.0, with_element=True))

    # -----------------------------------------------------
    # Edge beam demonstration
    # -----------------------------------------------------
    saved.append(plot_horizontal_cut(az_steer_deg=60.0, el_steer_deg=0.0,
                                     fname="horizontal_cut_edge_beam.png"))
    saved.append(plot_vertical_cut(az_steer_deg=0.0, el_steer_deg=12.0,
                                   fname="vertical_cut_edge_beam.png"))

    # -----------------------------------------------------
    # Beam sweep figures
    # -----------------------------------------------------
    saved.append(plot_all_horizontal_beams(with_element=False))
    saved.append(plot_all_horizontal_beams(with_element=True))
    saved.append(plot_all_vertical_beams(with_element=False))
    saved.append(plot_all_vertical_beams(with_element=True))

    print("Saved figures:")
    for p in saved:
        print("  ", p)


if __name__ == "__main__":
    main()