#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot 4x4 dual-pol UPA DFT beam radiation patterns.

Target antenna setting:
  (M, N, P, Mg, Ng; Mp, Np) = (4, 4, 2, 1, 1; 4, 4)
  dV = dH = 0.5 lambda

Figures:
  1) Horizontal cut, theta = 90 deg
  2) Vertical cut, phi = 0 deg
  3) Horizontal cut, all 4 horizontal beams
  4) Vertical cut, selected vertical beams

Compare:
  - Array factor only
  - Element pattern + array factor

Notes:
  theta: zenith angle, 0 deg = upward z-axis, 90 deg = horizontal plane
  phi: azimuth angle, 0 deg = broadside reference
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# Config
# =========================================================
M = 4          # vertical elements
N = 4          # horizontal elements
P = 2          # dual polarization, not explicitly used in pattern shape
dV = 0.5       # lambda
dH = 0.5       # lambda

OUT_DIR = "antenna_pattern_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# 3GPP-like element pattern parameters.
# For beam-shape comparison, use 0 dBi max gain and normalize final pattern to 0 dB.
# If you want absolute nominal gain, set G_E_MAX_DBI = 8.0 and NORMALIZE_TO_PEAK = False.
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
    3GPP-style single element pattern.

    theta_deg:
      zenith angle, broadside usually theta = 90 deg

    phi_deg:
      azimuth angle, broadside usually phi = 0 deg

    Return:
      element gain in dB.
    """
    theta_deg = np.asarray(theta_deg, dtype=float)
    phi_deg = np.asarray(phi_deg, dtype=float)

    # Wrap phi to [-180, 180]
    phi_wrapped = (phi_deg + 180.0) % 360.0 - 180.0

    # Vertical attenuation
    A_v = -np.minimum(12.0 * ((theta_deg - 90.0) / theta_3db) ** 2, sla_v)

    # Horizontal attenuation
    A_h = -np.minimum(12.0 * (phi_wrapped / phi_3db) ** 2, a_m)

    # Combined element attenuation
    A = -np.minimum(-(A_v + A_h), a_m)

    return g_e_max_dbi + A


from channel_functions import get_dft_codebook


def dft_weight_from_your_codebook(qv, qh, M=4, N=4, P=2, pol_idx=0):
    """
    严格复用链路仿真里的 get_dft_codebook()。

    对齐你的主脚本：
      C = get_dft_codebook(M, N, P, dft_offset=0)
      beam_idx = qv * N + qh
      pol0_row = beam_idx
      pol1_row = beam_idx + M*N
    """
    qv = int(qv) % M
    qh = int(qh) % N

    C = get_dft_codebook(M, N, P, dft_offset=0)

    num_spatial = M * N
    beam_idx = qv * N + qh

    if pol_idx == 0:
        row = beam_idx
        w_full = C[row, :]
        w_spatial = w_full[:num_spatial]
    elif pol_idx == 1:
        row = beam_idx + num_spatial
        w_full = C[row, :]
        w_spatial = w_full[num_spatial:]
    else:
        raise ValueError("pol_idx must be 0 or 1")

    w_spatial = np.asarray(w_spatial, dtype=np.complex128)
    w_spatial = w_spatial / np.linalg.norm(w_spatial)

    return w_spatial


def steering_vector(theta_deg, phi_deg, M=4, N=4, dV=0.5, dH=0.5):
    """
    UPA steering vector.

    Coordinate convention:
      vertical axis: z-axis
      horizontal axis: y-axis
      theta: zenith angle
      phi: azimuth angle

    Directional spatial frequency approximation:
      u_v = cos(theta)
      u_h = sin(theta) * sin(phi)

    Broadside:
      theta = 90 deg, phi = 0 deg => u_v = 0, u_h = 0
    """
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)

    u_v = np.cos(theta)
    u_h = np.sin(theta) * np.sin(phi)

    mv = np.arange(M)
    mh = np.arange(N)

    a_v = np.exp(1j * 2.0 * np.pi * dV * np.outer(np.asarray(u_v).ravel(), mv))
    a_h = np.exp(1j * 2.0 * np.pi * dH * np.outer(np.asarray(u_h).ravel(), mh))

    # row-wise kron: [mv0,mh0], [mv0,mh1], ...
    a = np.einsum("iv,ih->ivh", a_v, a_h).reshape(len(np.asarray(u_v).ravel()), M * N)

    return a


def array_factor_power(theta_deg, phi_deg, qv, qh):
    """
    Array factor power |a(theta,phi) @ conj(w)|^2.
    """
    theta_arr = np.asarray(theta_deg, dtype=float)
    phi_arr = np.asarray(phi_deg, dtype=float)

    theta_flat = theta_arr.ravel()
    phi_flat = phi_arr.ravel()

    a = steering_vector(theta_flat, phi_flat, M=M, N=N, dV=dV, dH=dH)
    w = dft_weight_from_your_codebook(qv, qh, M=M, N=N, P=2, pol_idx=0)

    # normalized beamforming vector, array response power
    af = np.abs(a @ np.conj(w)) ** 2

    return af.reshape(theta_arr.shape)


def total_pattern_db(theta_deg, phi_deg, qv, qh, with_element=True,
                     normalize=True):
    """
    Total beam pattern in dB.
    """
    af_pwr = array_factor_power(theta_deg, phi_deg, qv, qh)
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


def plot_horizontal_cut(qv=0, qh=0, theta_cut=90.0, fname=None):
    phi = np.linspace(-90.0, 90.0, 1801)
    theta = np.full_like(phi, theta_cut)

    pat_no_elem = total_pattern_db(theta, phi, qv, qh, with_element=False,
                                   normalize=NORMALIZE_TO_PEAK)
    pat_with_elem = total_pattern_db(theta, phi, qv, qh, with_element=True,
                                     normalize=NORMALIZE_TO_PEAK)

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(phi, pat_no_elem, label="Array factor only")
    plt.plot(phi, pat_with_elem, label="With element pattern")
    plt.grid(True, which="both")
    plt.xlabel("Azimuth angle phi (deg)")
    plt.ylabel("Normalized gain (dB)" if NORMALIZE_TO_PEAK else "Gain (dB)")
    plt.title(f"Horizontal Cut, theta={theta_cut:.0f}°, Beam(qv={qv}, qh={qh})")
    plt.ylim(FLOOR_DB, 3)
    plt.xlim(-90, 90)
    plt.legend(loc="best")
    plt.tight_layout()

    if fname is None:
        fname = f"horizontal_cut_qv{qv}_qh{qh}.png"

    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_vertical_cut(qv=0, qh=0, phi_cut=0.0, fname=None):
    theta = np.linspace(0.0, 180.0, 1801)
    phi = np.full_like(theta, phi_cut)

    pat_no_elem = total_pattern_db(theta, phi, qv, qh, with_element=False,
                                   normalize=NORMALIZE_TO_PEAK)
    pat_with_elem = total_pattern_db(theta, phi, qv, qh, with_element=True,
                                     normalize=NORMALIZE_TO_PEAK)

    # For vertical cut, x-axis is elevation offset from horizon:
    # elevation = 90 - theta
    elev = 90.0 - theta

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(elev, pat_no_elem, label="Array factor only")
    plt.plot(elev, pat_with_elem, label="With element pattern")
    plt.grid(True, which="both")
    plt.xlabel("Elevation angle relative to horizon (deg)")
    plt.ylabel("Normalized gain (dB)" if NORMALIZE_TO_PEAK else "Gain (dB)")
    plt.title(f"Vertical Cut, phi={phi_cut:.0f}°, Beam(qv={qv}, qh={qh})")
    plt.ylim(FLOOR_DB, 3)
    plt.xlim(-90, 90)
    plt.legend(loc="best")
    plt.tight_layout()

    if fname is None:
        fname = f"vertical_cut_qv{qv}_qh{qh}.png"

    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_all_horizontal_beams(qv=0, with_element=True, theta_cut=90.0):
    phi = np.linspace(-90.0, 90.0, 1801)
    theta = np.full_like(phi, theta_cut)

    plt.figure(figsize=(7.2, 4.8))

    for qh in range(N):
        pat = total_pattern_db(theta, phi, qv, qh, with_element=with_element,
                               normalize=NORMALIZE_TO_PEAK)
        plt.plot(phi, pat, label=f"qh={qh}")

    plt.grid(True, which="both")
    plt.xlabel("Azimuth angle phi (deg)")
    plt.ylabel("Normalized gain (dB)" if NORMALIZE_TO_PEAK else "Gain (dB)")
    suffix = "with element pattern" if with_element else "array factor only"
    plt.title(f"Horizontal Cut, qv={qv}, {suffix}")
    plt.ylim(FLOOR_DB, 3)
    plt.xlim(-90, 90)
    plt.legend(loc="best", ncol=2)
    plt.tight_layout()

    fname = f"horizontal_all_qv{qv}_{'with_elem' if with_element else 'no_elem'}.png"
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_all_vertical_beams(qh=0, with_element=True, phi_cut=0.0):
    theta = np.linspace(0.0, 180.0, 1801)
    phi = np.full_like(theta, phi_cut)
    elev = 90.0 - theta

    plt.figure(figsize=(7.2, 4.8))

    # Your target sector uses qv = 0 and 1
    for qv in [0, 1]:
        pat = total_pattern_db(theta, phi, qv, qh, with_element=with_element,
                               normalize=NORMALIZE_TO_PEAK)
        plt.plot(elev, pat, label=f"qv={qv}")

    plt.grid(True, which="both")
    plt.xlabel("Elevation angle relative to horizon (deg)")
    plt.ylabel("Normalized gain (dB)" if NORMALIZE_TO_PEAK else "Gain (dB)")
    suffix = "with element pattern" if with_element else "array factor only"
    plt.title(f"Vertical Cut, qh={qh}, {suffix}")
    plt.ylim(FLOOR_DB, 3)
    plt.xlim(-90, 90)
    plt.legend(loc="best")
    plt.tight_layout()

    fname = f"vertical_all_qh{qh}_{'with_elem' if with_element else 'no_elem'}.png"
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def main():
    saved = []

    # 推荐主展示图：beam(qv=0,qh=0) 的水平/垂直 cut
    saved.append(plot_horizontal_cut(qv=0, qh=0))
    saved.append(plot_vertical_cut(qv=0, qh=0))

    # 4 个水平 DFT beams 对比
    saved.append(plot_all_horizontal_beams(qv=0, with_element=False))
    saved.append(plot_all_horizontal_beams(qv=0, with_element=True))

    # 2 个垂直 DFT beams 对比
    saved.append(plot_all_vertical_beams(qh=0, with_element=False))
    saved.append(plot_all_vertical_beams(qh=0, with_element=True))

    print("Saved figures:")
    for p in saved:
        print("  ", p)


if __name__ == "__main__":
    main()