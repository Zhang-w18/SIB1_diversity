#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated SIB1 PDSCH BLER comparison script
============================================
Purpose:
  Compare two fine-beam realization families under one common link-level framework:

  1) Array-based fine beam
     - Coarse beam: reduced horizontal aperture, e.g. Mh_active=4
     - Fine beam: full horizontal aperture, e.g. Mh_active=8
     - Optional FDM refinement: different RB bundles use different full-aperture fine beams

  2) Precoding-based fine beam / beam-state variation
     - Use the same spatial DFT beam on two dual-pol branches, or two neighboring spatial beams
     - Apply PRG-level cyclic baseband precoding phase
     - Different PRGs experience different effective vector beam states

Notes:
  - This script keeps the SIB1 grid generation, CDL MIMO channel, AWGN, DMRS CE,
    equalization, and LDPC decoding in one common path.
  - It is designed to be directly adapted from the two scripts you pasted.
  - Default channel is CDL because polarization / array / beam-state effects are more meaningful there.
  - A lightweight TDL option is included only as a sanity check. For strong beam-direction claims,
    CDL is preferred.
"""

import csv
import math
import os
import concurrent.futures
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from scipy.signal import fftconvolve

from py3gpp import (
    nrPRBS, nrCRCEncode, nrCodeBlockSegmentLDPC, nrLDPCEncode,
    nrRateMatchLDPC, nrOFDMModulate, nrOFDMDemodulate,
    nrExtractResources, nrSetResources, nrRateRecoverLDPC,
    nrLDPCDecode, nrCodeBlockDesegmentLDPC, nrCRCDecode,
    nrDLSCHInfo, nrSymbolModulate,
)

from channel_initialize import channel_initialize
from channel_functions import channel_response_generate, channel_time_interpolation
from myChannelEstimation4 import myChannelEstimate_std


# =============================================================================
# Global switches
# =============================================================================
PRINT_DIAG = False


# =============================================================================
# SIB1 PDSCH grid / coding / modulation utilities
# =============================================================================
def _normalize_dmrs_loc(dmrs_loc, nsym_slot):
    if np.isscalar(dmrs_loc):
        locs = np.array([int(dmrs_loc)], dtype=int)
    else:
        locs = np.array(sorted(set(int(x) for x in dmrs_loc)), dtype=int)
    assert locs.size > 0
    assert np.all((0 <= locs) & (locs < nsym_slot))
    return locs


def nrSIB1DMRScinit(n_slot=0, l=2, ncellid=0):
    return (2 ** 17 * (14 * n_slot + l + 1) * (2 * ncellid + 1) + 2 * ncellid) % (2 ** 31)


def _pdsch_scrambling_bits_sib1(n_id_cell: int, E: int, rnti: int = 0xFFFF, q=0) -> np.ndarray:
    c_init = (int(rnti) << 15) + (int(q) << 14) + int(n_id_cell)
    return nrPRBS(c_init, E).astype(np.uint8)


def nrSIB1DMRSIndices_param(nrb_sib: int, nsym_slot: int, dmrs_loc, k_step: int = 2, style: str = "matlab"):
    Nsc = int(nrb_sib) * 12
    ks = np.arange(0, Nsc, k_step, dtype=int)
    locs = _normalize_dmrs_loc(dmrs_loc, nsym_slot)
    k_all = np.tile(ks, len(locs))
    l_all = np.repeat(locs, len(ks))
    if style == "python":
        return (tuple(k_all.tolist()), tuple(l_all.tolist()))
    if style == "matlab":
        return (k_all + l_all * Nsc).astype(int)
    raise ValueError("Unknown style")


def nrSIB1Indices_param(nrb_sib: int, nsym_slot: int, dmrs_idx: np.ndarray, style: str = "matlab"):
    Nsc = int(nrb_sib) * 12
    total_re = Nsc * nsym_slot
    all_idx = np.arange(total_re, dtype=int)
    dmrs_idx = np.asarray(dmrs_idx, dtype=int)
    dmrs_idx = dmrs_idx[(dmrs_idx >= 0) & (dmrs_idx < total_re)]
    dmrs_idx = np.unique(dmrs_idx)
    sib1_data_idx = np.setdiff1d(all_idx, dmrs_idx, assume_unique=True)
    if style == "matlab":
        return sib1_data_idx
    if style == "python":
        return (tuple((sib1_data_idx % Nsc).tolist()), tuple((sib1_data_idx // Nsc).tolist()))
    raise ValueError("Unknown style")


def nrSIB_param(
    in_bits: np.ndarray,
    n_id_cell: int,
    E: int,
    nLayers: int = 1,
    rv: int = 0,
    rnti: int = 0xFFFF,
    BGN: int = 2,
    CRC: str = "16",
):
    if in_bits.ndim == 2 and in_bits.shape[1] == 1:
        in_bits = in_bits[:, 0]
    in_bits = in_bits.astype(int).copy()
    A = int(in_bits.size)

    tb_in = nrCRCEncode(in_bits, CRC)
    cbs_in = nrCodeBlockSegmentLDPC(tb_in, BGN)
    enc = nrLDPCEncode(cbs_in, BGN)
    cw_bits_no_scram = nrRateMatchLDPC(enc, E, rv, 'QPSK', nLayers)
    scr = _pdsch_scrambling_bits_sib1(n_id_cell, E, rnti)
    cw_bits = (cw_bits_no_scram ^ scr).astype(np.int8)
    meta = {
        "A": A,
        "E": E,
        "BGN": BGN,
        "CRC": CRC,
        "rv": rv,
        "nLayers": nLayers,
        "cw_bits_no_scram": cw_bits_no_scram,
        "scr": scr,
        "cw_bits": cw_bits,
    }
    return cw_bits, meta


def nrSIBDMRS_param_multi(ncellid, nrb_sib, nsym_sib, dmrs_loc, k_step=2, n_slot=0):
    dmrs_cols = _normalize_dmrs_loc(dmrs_loc, nsym_sib)
    Nsc = nrb_sib * 12
    ks = np.arange(0, Nsc, k_step, dtype=int)
    dmrs_all = []
    for l_dmrs in dmrs_cols:
        cinit = nrSIB1DMRScinit(n_slot, int(l_dmrs), ncellid)
        c = nrPRBS(cinit, 2 * len(ks))
        s = nrSymbolModulate(c, "QPSK")
        dmrs_all.append(np.asarray(s, dtype=complex))
    return np.concatenate(dmrs_all, axis=0)


def build_SIB1pdsch_grid_only(
    ncellid,
    trblk_bits,
    nrb_sib=48,
    nsym_sib=12,
    dmrs_loc=(0, 6, 9),
    k_step=2,
    cbs_info=None,
):
    if cbs_info is None:
        cbs_info = nrDLSCHInfo(len(trblk_bits), 0.5)

    grid = np.zeros((nrb_sib * 12, nsym_sib), dtype=complex)
    dmrs_cols = _normalize_dmrs_loc(dmrs_loc, nsym_sib)
    dmrs_idx = nrSIB1DMRSIndices_param(nrb_sib, nsym_sib, dmrs_cols, k_step, "matlab")
    sib1_idx = nrSIB1Indices_param(nrb_sib, nsym_sib, dmrs_idx, "matlab")
    E = len(sib1_idx) * 2

    sib1_bits, meta = nrSIB_param(
        trblk_bits,
        ncellid,
        E,
        1,
        0,
        0xFFFF,
        cbs_info["BGN"],
        cbs_info["CRC"],
    )
    sib1_symb = nrSymbolModulate(sib1_bits, "QPSK")
    dmrs_symb = nrSIBDMRS_param_multi(ncellid, nrb_sib, nsym_sib, dmrs_cols, k_step, 0)
    nrSetResources(sib1_idx, grid, sib1_symb)
    nrSetResources(dmrs_idx, grid, dmrs_symb)

    return grid, meta, {
        "sib1_idx": sib1_idx,
        "dmrs_idx": dmrs_idx,
        "dmrs_cols": dmrs_cols,
        "E": E,
    }


# =============================================================================
# Beam and precoder utilities
# =============================================================================
def iter_prg_ranges(nrb_sib, prg_size_rb):
    for rb0 in range(0, nrb_sib, prg_size_rb):
        yield rb0, min(nrb_sib, rb0 + prg_size_rb)


def build_subarray_dft_vector_64tx(Mv=4, Mh_total=8, Mh_active=8, qv=0, qh_equiv=0.0):
    """
    Array/subarray-based beam.
    Mh_active controls aperture:
      - smaller Mh_active -> wider/coarser beam
      - larger Mh_active  -> narrower/finer beam
    """
    vv = np.exp(1j * 2.0 * np.pi * qv * np.arange(Mv, dtype=float) / Mv) / np.sqrt(Mv)
    vh = np.zeros(Mh_total, dtype=np.complex128)
    m_active = np.arange(Mh_active, dtype=float)
    vh[:Mh_active] = np.exp(1j * 2.0 * np.pi * m_active * qh_equiv / Mh_total) / np.sqrt(Mh_active)
    return np.kron(vv, vh).astype(np.complex128)


def build_upa_dft_vector(Mv=4, Mh=8, qv=0, qh=0):
    """
    Full-aperture 4x8 UPA DFT beam.
    Used by the precoding-based beam-state scheme.
    """
    qv, qh = int(qv) % Mv, int(qh) % Mh
    mv, mh = np.arange(Mv, dtype=float), np.arange(Mh, dtype=float)
    vv = np.exp(1j * 2.0 * np.pi * qv * mv / Mv) / np.sqrt(Mv)
    vh = np.exp(1j * 2.0 * np.pi * qh * mh / Mh) / np.sqrt(Mh)
    return np.kron(vv, vh).astype(np.complex128)


def map_spatial_beam_to_dualpol_64(v_spatial, pol_idx=0):
    v = np.asarray(v_spatial, dtype=np.complex128).ravel()
    if pol_idx == 0:
        w = np.concatenate([v, np.zeros_like(v)])
    elif pol_idx == 1:
        w = np.concatenate([np.zeros_like(v), v])
    else:
        raise ValueError("pol_idx must be 0 or 1")
    return w / np.linalg.norm(w)


def build_wbf_64x2_subarray(Mv=4, Mh_total=8, Mh_active=8, qv=0, qh_equiv=0.0):
    """64Tx x 2 dual-pol Wbf for array/subarray-based beam."""
    v_sp = build_subarray_dft_vector_64tx(Mv, Mh_total, Mh_active, qv, qh_equiv)
    return np.column_stack([
        map_spatial_beam_to_dualpol_64(v_sp, pol_idx=0),
        map_spatial_beam_to_dualpol_64(v_sp, pol_idx=1),
    ])


def build_wbf_64x2_upa(qv1=0, qh1=0, qv2=None, qh2=None):
    """
    64Tx x 2 dual-pol Wbf for precoding-based beam-state variation.

    Default qv2/qh2 = qv1/qh1:
      - same spatial beam on pol0 and pol1
      - cyclic phase changes dual-pol vector beam state

    If qh2 != qh1:
      - column 0 and column 1 correspond to two different spatial beam bases
      - cyclic phase also changes spatial composite pattern
    """
    if qv2 is None:
        qv2 = qv1
    if qh2 is None:
        qh2 = qh1

    v1 = build_upa_dft_vector(Mv=4, Mh=8, qv=qv1, qh=qh1)
    v2 = build_upa_dft_vector(Mv=4, Mh=8, qv=qv2, qh=qh2)
    w1 = map_spatial_beam_to_dualpol_64(v1, pol_idx=0)
    w2 = map_spatial_beam_to_dualpol_64(v2, pol_idx=1)
    return np.column_stack([w1, w2])


def bb_precoder_traditional():
    return np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)


def bb_precoder_cyclic(prg_idx, phase_table=(0, np.pi / 2, np.pi, 3 * np.pi / 2)):
    phi = phase_table[prg_idx % len(phase_table)]
    return np.array([1.0, np.exp(1j * phi)], dtype=np.complex128) / np.sqrt(2.0)


def get_bb_precoder(mode: str, prg_idx: int, phase_table):
    if mode == "traditional":
        return bb_precoder_traditional()
    if mode == "cyclic":
        return bb_precoder_cyclic(prg_idx, phase_table)
    raise ValueError(f"Unsupported PRG precoding mode: {mode}")


@dataclass
class SchemeConfig:
    label: str
    family: str
    scheme: str
    prg_size_rb: int
    prg_precoding_mode: str = "traditional"
    ce_mode: str = "fullband"
    # Optional for precoding-based scheme
    precoding_basis: str = "polarization"  # "polarization" or "spatial_adjacent"


def make_wbf_list_for_scheme(cfg: SchemeConfig, qv: int, qh_a: int, qh_b: int):
    """
    Return a list of Wbf matrices. build_precoded_tx_grids_unified() maps the list
    to the frequency axis. If len(Wbf_list)==1, all PRGs use the same Wbf but may
    still use different baseband precoders.
    """
    if cfg.scheme == "array_coarse_baseline":
        # Coarse beam: reduced horizontal aperture, pointing around the middle of qh_a and qh_b.
        return [build_wbf_64x2_subarray(4, 8, Mh_active=4, qv=qv, qh_equiv=(qh_a + qh_b) / 2.0)]

    if cfg.scheme == "array_fdm_fine":
        # FDM fine-beam refinement: different frequency parts use different full-aperture fine beams.
        return [
            build_wbf_64x2_subarray(4, 8, Mh_active=8, qv=qv, qh_equiv=qh_a),
            build_wbf_64x2_subarray(4, 8, Mh_active=8, qv=qv, qh_equiv=qh_b),
        ]

    if cfg.scheme == "array_refined_fine":
        # After refinement: full band uses the selected full-aperture fine beam.
        return [build_wbf_64x2_subarray(4, 8, Mh_active=8, qv=qv, qh_equiv=qh_a)]

    if cfg.scheme == "precoding_cyclic":
        if cfg.precoding_basis == "polarization":
            # Same spatial DFT beam on pol0/pol1; cyclic phase changes vector/polarization beam state.
            return [build_wbf_64x2_upa(qv1=qv, qh1=qh_a, qv2=qv, qh2=qh_a)]
        if cfg.precoding_basis == "spatial_adjacent":
            # Two neighboring spatial beam bases on two pol branches; cyclic phase changes composite pattern.
            return [build_wbf_64x2_upa(qv1=qv, qh1=qh_a, qv2=qv, qh2=qh_b)]
        raise ValueError(f"Unsupported precoding_basis: {cfg.precoding_basis}")

    raise ValueError(f"Unsupported scheme: {cfg.scheme}")


def build_precoded_tx_grids_unified(
    base_grid,
    nrb_sib,
    prg_size_rb=4,
    prg_precoding_mode="traditional",
    Wbf_list=None,
    phase_table=(0, np.pi / 2, np.pi, 3 * np.pi / 2),
):
    if Wbf_list is None or len(Wbf_list) == 0:
        Wbf_list = [build_wbf_64x2_subarray()]

    num_wbf = len(Wbf_list)
    K, N = base_grid.shape
    tx_grids = np.zeros((Wbf_list[0].shape[0], K, N), dtype=np.complex128)

    for g, (rb0, rb1) in enumerate(iter_prg_ranges(nrb_sib, prg_size_rb)):
        # Frequency-domain Wbf selection for array_fdm_fine; constant Wbf for other schemes.
        mid_rb = (rb0 + rb1) / 2.0
        wbf_idx = int((mid_rb / nrb_sib) * num_wbf)
        wbf_idx = min(wbf_idx, num_wbf - 1)
        Wbf_current = Wbf_list[wbf_idx]

        wbb = get_bb_precoder(prg_precoding_mode, g, phase_table)
        w_eff = Wbf_current @ wbb
        w_eff = w_eff / np.linalg.norm(w_eff)
        tx_grids[:, rb0 * 12:rb1 * 12, :] = w_eff[:, None, None] * base_grid[None, rb0 * 12:rb1 * 12, :]

    return tx_grids


# =============================================================================
# Channel and OFDM utilities
# =============================================================================
def nrOFDMModulate_multi_tx(tx_grids, scs, SampleRate):
    Nt = tx_grids.shape[0]
    waves = []
    max_len = 0
    for t in range(Nt):
        w, _ = nrOFDMModulate(carrier=None, grid=tx_grids[t], scs=scs, SampleRate=SampleRate)
        waves.append(w)
        max_len = max(max_len, len(w))

    tx_waves = np.zeros((max_len, Nt), dtype=np.complex128)
    for t, w in enumerate(waves):
        tx_waves[:len(w), t] = w
    return tx_waves


def cdl_impulse_response_64t4r_full(
    fs_hz,
    ds_ns=300,
    fc_GHz=7,
    seed=None,
    initial_time=0.0,
    ue_speed=3,
    cdl_type="CDLD",
):
    if seed is not None:
        np.random.seed(seed)
    ds = ds_ns * 1e-9
    tx_ant = [1, 1, 4, 8, 2]
    rx_ant = [1, 1, 1, 2, 2]
    channel, _, _ = channel_initialize(
        cdl_type=cdl_type,
        delay_spread=ds,
        Kf=[],
        tx_antenna=tx_ant,
        rx_antenna=rx_ant,
        ue_speed=ue_speed,
        fc_GHz=fc_GHz,
        fs=fs_hz,
        num_tti=0,
        tti_length=1e-3,
        angle_gap=0.0,
        power_gap=0.0,
        beamforming='no',
        chan_method='online',
    )
    h_paths, delays = channel_response_generate(channel, initial_time=initial_time)
    h00 = channel_time_interpolation(delays, h_paths[:, 0, 0], ts=1.0 / fs_hz, num_ext=4)
    H_mimo = np.zeros((4, 64, len(h00)), dtype=np.complex128)
    for rx in range(4):
        for tx in range(64):
            H_mimo[rx, tx, :] = channel_time_interpolation(delays, h_paths[:, rx, tx], ts=1.0 / fs_hz, num_ext=4)
    return H_mimo


def apply_mimo_fir(tx_waves, H_mimo):
    Ns, Nt = tx_waves.shape
    y = np.zeros((Ns, H_mimo.shape[0]), dtype=np.complex128)
    for rx in range(H_mimo.shape[0]):
        acc = np.zeros(Ns, dtype=np.complex128)
        for tx in range(Nt):
            acc += fftconvolve(tx_waves[:, tx], H_mimo[rx, tx, :], mode='same')
        y[:, rx] = acc
    return y


def apply_cdl_mimo_tx(
    tx_waves,
    fs_hz,
    seed=None,
    ds_ns=300,
    fc_hz=7e9,
    speed_kmh=3,
    initial_time=0.0,
    cdl_type="CDLD",
):
    H_mimo = cdl_impulse_response_64t4r_full(
        fs_hz,
        ds_ns,
        fc_hz / 1e9,
        seed,
        initial_time,
        speed_kmh,
        cdl_type=cdl_type,
    )
    return apply_mimo_fir(tx_waves, H_mimo), H_mimo


def get_simple_tdl_profile(profile="TDL-C"):
    """
    Lightweight sanity-check TDL profile, not a full 3GPP-complete implementation.
    Use CDL for main beam/polarization conclusions.
    """
    profile = profile.upper()
    if profile == "TDL-C":
        delays_ns = np.array([0, 30, 70, 90, 110, 190, 410, 730, 1090, 1730], dtype=float)
        powers_db = np.array([-4.4, -1.2, -3.5, -5.2, -2.5, 0.0, -2.2, -7.4, -7.1, -10.7], dtype=float)
    elif profile == "TDL-A":
        delays_ns = np.array([0, 30, 70, 90, 110, 190, 410], dtype=float)
        powers_db = np.array([0.0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8], dtype=float)
    else:
        raise ValueError(f"Unsupported lightweight TDL profile: {profile}")
    p = 10 ** (powers_db / 10.0)
    return delays_ns, p / np.sum(p)


def build_tx_steering_64(qv=0, qh=0.0):
    # Directional TDL helper: same spatial beam replicated on both polarizations.
    v_sp = build_upa_dft_vector(Mv=4, Mh=8, qv=qv, qh=int(round(qh)) % 8)
    a = np.concatenate([v_sp, v_sp])
    return a / np.linalg.norm(a)


def tdl_impulse_response_64t4r_full(
    fs_hz,
    ds_ns=300,
    seed=None,
    profile="TDL-C",
    directional=True,
    main_qv=0,
    main_qh=0,
    n_rx=4,
    n_tx=64,
):
    if seed is not None:
        np.random.seed(seed)

    delays_ns, powers_lin = get_simple_tdl_profile(profile)
    if np.max(delays_ns) > 0:
        delays_ns = delays_ns / np.sqrt(np.mean(delays_ns ** 2)) * ds_ns
    delay_samps = np.round(delays_ns * 1e-9 * fs_hz).astype(int)
    L = int(np.max(delay_samps)) + 1 + 4
    H = np.zeros((n_rx, n_tx, L), dtype=np.complex128)

    if directional:
        a_tx = build_tx_steering_64(qv=main_qv, qh=main_qh)
        for rx in range(n_rx):
            g = (np.random.randn(len(powers_lin)) + 1j * np.random.randn(len(powers_lin))) / np.sqrt(2.0)
            g = g * np.sqrt(powers_lin)
            for p, d in enumerate(delay_samps):
                H[rx, :, d] += g[p] * np.conj(a_tx[:n_tx])
    else:
        for rx in range(n_rx):
            for tx in range(n_tx):
                g = (np.random.randn(len(powers_lin)) + 1j * np.random.randn(len(powers_lin))) / np.sqrt(2.0)
                g = g * np.sqrt(powers_lin)
                for p, d in enumerate(delay_samps):
                    H[rx, tx, d] += g[p]
    return H


def apply_tdl_mimo_tx(
    tx_waves,
    fs_hz,
    seed=None,
    ds_ns=300,
    profile="TDL-C",
    directional=True,
    main_qv=0,
    main_qh=0,
):
    H_mimo = tdl_impulse_response_64t4r_full(
        fs_hz=fs_hz,
        ds_ns=ds_ns,
        seed=seed,
        profile=profile,
        directional=directional,
        main_qv=main_qv,
        main_qh=main_qh,
        n_rx=4,
        n_tx=tx_waves.shape[1],
    )
    return apply_mimo_fir(tx_waves, H_mimo), H_mimo


def ofdm_demodulate_multi_rx(waveform, nrb, scs, initialNSlot, SampleRate, nsym_keep, CyclicPrefixFraction=0.5):
    x = np.asarray(waveform)
    if x.ndim == 1:
        g = nrOFDMDemodulate(
            waveform=x,
            nrb=nrb,
            scs=scs,
            initialNSlot=initialNSlot,
            SampleRate=SampleRate,
            CyclicPrefixFraction=CyclicPrefixFraction,
        )[:, :nsym_keep]
        return g[None, :, :]

    nRx = x.shape[1]
    grids = []
    for r in range(nRx):
        g = nrOFDMDemodulate(
            waveform=x[:, r],
            nrb=nrb,
            scs=scs,
            initialNSlot=initialNSlot,
            SampleRate=SampleRate,
            CyclicPrefixFraction=CyclicPrefixFraction,
        )[:, :nsym_keep]
        grids.append(g)
    return np.stack(grids, axis=0)


def add_awgn_on_gri_per_symbol(rxGrid_clean, tx_grid_ref, snr_db_re, active_idx):
    K, N = rxGrid_clean.shape
    snr_lin = 10 ** (snr_db_re / 10.0)
    flat = tx_grid_ref.ravel(order="F")
    p_sig_global = np.mean(np.abs(flat[active_idx]) ** 2)
    nvar_cols = np.zeros(N, dtype=float)
    for l in range(N):
        in_col = (active_idx >= l * K) & (active_idx < (l + 1) * K)
        p_sig_l = np.mean(np.abs(flat[active_idx[in_col]]) ** 2) if np.any(in_col) else p_sig_global
        nvar_cols[l] = p_sig_l / snr_lin
    noise = (np.random.randn(K, N) + 1j * np.random.randn(K, N)) * np.sqrt(nvar_cols[None, :] / 2.0)
    return rxGrid_clean + noise, nvar_cols


def add_awgn_on_grid_multi_rx(rxGrids, txGrid_ref, snr_db, active_idx):
    rxGrids = np.asarray(rxGrids)
    out = np.zeros_like(rxGrids, dtype=complex)
    nVar_list = []
    for r in range(rxGrids.shape[0]):
        g_noisy, nVar_r = add_awgn_on_gri_per_symbol(rxGrids[r], txGrid_ref, snr_db, active_idx)
        out[r] = g_noisy
        nVar_list.append(nVar_r)
    return out, nVar_list


# =============================================================================
# Channel estimation / equalization / demodulation
# =============================================================================
def myChannelEstimate_prg_local_ls(rxGrid, refGrid, nrb_sib, prg_size_rb, dmrs_l):
    K, N = rxGrid.shape
    H = np.zeros_like(rxGrid, dtype=np.complex128)
    dmrs_cols = _normalize_dmrs_loc(dmrs_l, N)
    has_dmrs_col = np.zeros(N, dtype=bool)

    for col in dmrs_cols:
        for rb0, rb1 in iter_prg_ranges(nrb_sib, prg_size_rb):
            sc0, sc1 = rb0 * 12, rb1 * 12
            ref_blk = refGrid[sc0:sc1, col]
            pos = np.where(np.abs(ref_blk) > 1e-12)[0]
            if len(pos) == 0:
                continue
            y = rxGrid[sc0:sc1, col][pos]
            hls = y / ref_blk[pos]
            x_all = np.arange(sc1 - sc0)
            if len(pos) == 1:
                H_blk = np.full(sc1 - sc0, hls[0], dtype=np.complex128)
            else:
                H_blk = np.interp(x_all, pos, np.real(hls)) + 1j * np.interp(x_all, pos, np.imag(hls))
            H[sc0:sc1, col] = H_blk
            has_dmrs_col[col] = True

    cols_with, cols_wo = np.where(has_dmrs_col)[0], np.where(~has_dmrs_col)[0]
    if cols_with.size >= 2:
        for k in range(K):
            y = H[k, cols_with]
            H[k, cols_wo] = np.interp(cols_wo, cols_with, np.real(y)) + 1j * np.interp(cols_wo, cols_with, np.imag(y))
    elif cols_with.size == 1:
        H[:, cols_wo] = H[:, cols_with[0:1]]
    return H


def nrEqualizeZF_persym(rxSym, hest, nVar_cols, re_idx, K):
    nVar_re = nVar_cols[(re_idx // K).astype(np.int64)]
    h_abs2 = np.abs(hest) ** 2
    hest_safe = np.where(np.abs(hest) < 1e-12, 1e-12, hest)
    return rxSym / hest_safe, nVar_re / np.where(h_abs2 < 1e-12, 1e-12, h_abs2)


def nrSymbolDemodulate_qpsk_persym(input_sym, nVar_input, re_idx, K):
    input_sym = np.asarray(input_sym, dtype=np.complex128).ravel()
    nVar_arr = np.asarray(nVar_input, dtype=float)
    N0_re = nVar_arr if nVar_arr.size == len(re_idx) else nVar_arr[re_idx // K]
    xr, xi = np.real(input_sym), np.imag(input_sym)
    llr = np.empty(2 * len(input_sym), dtype=np.float64)
    llr[0::2], llr[1::2] = xr / np.maximum(N0_re, 1e-12), xi / np.maximum(N0_re, 1e-12)
    return llr


# =============================================================================
# One transmission / one trial
# =============================================================================
def _rx_round_get_llr(
    snr_db,
    ncellid,
    sib_grid,
    sib_idx,
    dmrs_idx,
    E,
    fs=61.44e6,
    scs_khz=30,
    nrb_sib=48,
    nsym_sib=12,
    meta=None,
    meta2=None,
    chan_seed=None,
    chan_initial_time=0.0,
    prg_size_rb=4,
    prg_precoding_mode="traditional",
    Wbf_list=None,
    phase_table=(0, np.pi / 2, np.pi, 3 * np.pi / 2),
    ce_mode="prg",
    channel_model="CDL",
    cdl_type="CDLD",
    tdl_profile="TDL-C",
    tdl_directional=True,
    main_qv=0,
    main_qh=0,
):
    tx_grids = build_precoded_tx_grids_unified(
        sib_grid,
        nrb_sib,
        prg_size_rb,
        prg_precoding_mode,
        Wbf_list,
        phase_table,
    )
    tx_waves = nrOFDMModulate_multi_tx(tx_grids, scs_khz, fs)

    if channel_model.upper() == "CDL":
        ch_out, _ = apply_cdl_mimo_tx(
            tx_waves,
            fs,
            seed=chan_seed,
            ds_ns=300,
            fc_hz=7e9,
            speed_kmh=3,
            initial_time=chan_initial_time,
            cdl_type=cdl_type,
        )
    elif channel_model.upper() == "TDL":
        ch_out, _ = apply_tdl_mimo_tx(
            tx_waves,
            fs,
            seed=chan_seed,
            ds_ns=300,
            profile=tdl_profile,
            directional=tdl_directional,
            main_qv=main_qv,
            main_qh=main_qh,
        )
    else:
        raise ValueError(f"Unsupported channel_model: {channel_model}")

    rxGrids_clean = ofdm_demodulate_multi_rx(ch_out, nrb_sib, scs_khz, 0, fs, nsym_sib, 0.5)

    active_idx = np.sort(np.concatenate([sib_idx, dmrs_idx]))
    rxGrids, nVar_list = add_awgn_on_grid_multi_rx(rxGrids_clean, sib_grid, snr_db, active_idx)

    dmrs_cols = meta2["dmrs_cols"]
    dmrs_symb = nrSIBDMRS_param_multi(ncellid, nrb_sib, nsym_sib, dmrs_cols, 2, 0)
    refGridH = np.zeros_like(rxGrids[0], dtype=complex)
    nrSetResources(dmrs_idx, refGridH, dmrs_symb)

    llr_sum_rx = np.zeros(E, dtype=np.float64)
    for r in range(rxGrids.shape[0]):
        if ce_mode == "fullband":
            # In this integrated script, "fullband" is interpreted as bundle-wise fullband CE.
            # If prg_size_rb == nrb_sib, this becomes true full-band CE.
            H_r = np.zeros_like(rxGrids[r], dtype=np.complex128)
            for rb0, rb1 in iter_prg_ranges(nrb_sib, prg_size_rb):
                H_bundle = myChannelEstimate_std(
                    rxGrid=rxGrids[r],
                    refGrid=refGridH,
                    EST_TFDOMAIN=False,
                    EST_PBCH=False,
                    EST_PDSCH_FULLBAND=True,
                    dmrs_l=dmrs_cols,
                    start_rb=rb0,
                    end_rb=rb1,
                    td_win_len_ratio=0.08,
                    td_window_type="fixed",
                )
                sc0, sc1 = rb0 * 12, rb1 * 12
                H_r[sc0:sc1, :] = H_bundle[sc0:sc1, :]
        elif ce_mode == "prg":
            H_r = myChannelEstimate_prg_local_ls(rxGrids[r], refGridH, nrb_sib, prg_size_rb, dmrs_cols)
        else:
            raise ValueError(f"Unsupported ce_mode: {ce_mode}")

        sib_eq_r, nVar_post_r = nrEqualizeZF_persym(
            nrExtractResources(sib_idx, rxGrids[r]),
            nrExtractResources(sib_idx, H_r),
            nVar_list[r],
            sib_idx,
            nrb_sib * 12,
        )
        llr_sum_rx += nrSymbolDemodulate_qpsk_persym(sib_eq_r, nVar_post_r, sib_idx, nrb_sib * 12)

    return llr_sum_rx * (1 - 2 * meta["scr"].astype(int))


def one_shot_sib_min_combined(
    snr_db,
    ncellid,
    scheme_cfg: SchemeConfig,
    qv,
    qh_a,
    qh_b,
    fs=61.44e6,
    scs_khz=30,
    nrb_sib=96,
    nsym_sib=12,
    rng_seed=None,
    N_comb=1,
    payload=1200,
    dmrs_step=2,
    dmrs_loc=(0, 6, 9),
    phase_table=(0, np.pi / 2, np.pi, 3 * np.pi / 2),
    chan_seed_base=1234,
    chan_initial_time0=0.0,
    chan_round_spacing_s=0.0,
    channel_evolution_mode="static_same",
    channel_model="CDL",
    cdl_type="CDLD",
    tdl_profile="TDL-C",
    tdl_directional=True,
):
    if rng_seed is not None:
        np.random.seed(rng_seed)

    dmrs_idx_tmp = nrSIB1DMRSIndices_param(nrb_sib, nsym_sib, dmrs_loc, dmrs_step, "matlab")
    sib_idx_tmp = nrSIB1Indices_param(nrb_sib, nsym_sib, dmrs_idx_tmp, "matlab")
    rate = (payload + 16) / (len(sib_idx_tmp) * 2)
    cbs_info = nrDLSCHInfo(payload, rate)
    trblk_bits = np.random.randint(0, 2, payload).astype(int)

    sib_grid, meta, meta2 = build_SIB1pdsch_grid_only(
        ncellid,
        trblk_bits,
        nrb_sib,
        nsym_sib,
        dmrs_loc,
        dmrs_step,
        cbs_info,
    )

    Wbf_list = make_wbf_list_for_scheme(scheme_cfg, qv, qh_a, qh_b)
    llr_sum = None

    for r in range(N_comb):
        if channel_evolution_mode == "static_same":
            c_seed = chan_seed_base
        elif channel_evolution_mode == "independent":
            c_seed = chan_seed_base + r
        else:
            c_seed = chan_seed_base
        t_init = chan_initial_time0 + r * chan_round_spacing_s

        llr_r = _rx_round_get_llr(
            snr_db,
            ncellid,
            sib_grid,
            meta2["sib1_idx"],
            meta2["dmrs_idx"],
            meta2["E"],
            fs,
            scs_khz,
            nrb_sib,
            nsym_sib,
            meta,
            meta2,
            c_seed,
            t_init,
            scheme_cfg.prg_size_rb,
            scheme_cfg.prg_precoding_mode,
            Wbf_list,
            phase_table,
            ce_mode=scheme_cfg.ce_mode,
            channel_model=channel_model,
            cdl_type=cdl_type,
            tdl_profile=tdl_profile,
            tdl_directional=tdl_directional,
            main_qv=qv,
            main_qh=qh_a,
        )
        llr_sum = llr_r.astype(np.float64, copy=True) if llr_sum is None else llr_sum + llr_r

    dec_bits, _ = nrLDPCDecode(
        nrRateRecoverLDPC(llr_sum, payload, rate, 0, "QPSK", 1),
        cbs_info["BGN"],
        25,
        blklen=payload,
    )
    blk, _ = nrCodeBlockDesegmentLDPC(dec_bits, cbs_info["BGN"], payload + cbs_info["L"])
    _, tb_err = nrCRCDecode(blk, cbs_info["CRC"])
    return (tb_err == 0), {}


def run_single_trial_global(args):
    snr_db, target_mapping, trial_seed, static_cfg, scheme_dict = args
    scheme_cfg = SchemeConfig(**scheme_dict)
    qv, qh_a, qh_b = target_mapping["qv"], target_mapping["qh_a"], target_mapping["qh_b"]

    ok, _ = one_shot_sib_min_combined(
        snr_db=snr_db,
        ncellid=static_cfg["ncellid"],
        scheme_cfg=scheme_cfg,
        qv=qv,
        qh_a=qh_a,
        qh_b=qh_b,
        fs=static_cfg["fs"],
        scs_khz=static_cfg["scs_khz"],
        nrb_sib=static_cfg["nrb_sib"],
        nsym_sib=static_cfg["nsym_sib"],
        rng_seed=trial_seed,
        N_comb=static_cfg["N_comb"],
        payload=static_cfg["payload"],
        dmrs_step=2,
        dmrs_loc=(0, 6, 9),
        phase_table=tuple(static_cfg["phase_table"]),
        chan_seed_base=trial_seed,
        channel_model=static_cfg["channel_model"],
        cdl_type=static_cfg["cdl_type"],
        tdl_profile=static_cfg["tdl_profile"],
        tdl_directional=static_cfg["tdl_directional"],
    )
    return ok


# =============================================================================
# BLER sweep
# =============================================================================
def simulate_bler_compare(
    snr_db_list,
    scheme_cfg: SchemeConfig,
    target_mappings,
    n_trials_per_target=200,
    ncellid=208,
    fs=61.44e6,
    scs_khz=30,
    seed=2029,
    nrb_sib=96,
    nsym_sib=12,
    N_comb=1,
    payload=1200,
    max_workers=4,
    target_errors=50,
    snr_early_stop_th=None,
    channel_model="CDL",
    cdl_type="CDLD",
    tdl_profile="TDL-C",
    tdl_directional=True,
    phase_table=(0, np.pi / 2, np.pi, 3 * np.pi / 2),
):
    bler_avg_list = []
    rng = np.random.RandomState(seed)

    static_cfg = {
        "ncellid": ncellid,
        "fs": fs,
        "scs_khz": scs_khz,
        "nrb_sib": nrb_sib,
        "nsym_sib": nsym_sib,
        "N_comb": N_comb,
        "payload": payload,
        "channel_model": channel_model,
        "cdl_type": cdl_type,
        "tdl_profile": tdl_profile,
        "tdl_directional": tdl_directional,
        "phase_table": list(phase_table),
    }

    scheme_dict = asdict(scheme_cfg)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for snr_db in snr_db_list:
            target_blers = []

            for mapping in target_mappings:
                err = 0
                run = 0
                tasks = [
                    (snr_db, mapping, rng.randint(1 << 31), static_cfg, scheme_dict)
                    for _ in range(n_trials_per_target)
                ]
                futures = [executor.submit(run_single_trial_global, t) for t in tasks]

                desc = f"{scheme_cfg.label} | SNR={snr_db:>5.1f} | {mapping['name']}"
                with tqdm(total=n_trials_per_target, desc=desc, ncols=110, leave=False) as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        ok = future.result()
                        run += 1
                        if not ok:
                            err += 1
                        pbar.update(1)

                        if err >= target_errors:
                            pbar.set_postfix_str(f"Early Stop: {err} errs")
                            for f in futures:
                                f.cancel()
                            break

                target_bler = err / max(run, 1)
                target_blers.append(target_bler)

            avg_bler = float(np.mean(target_blers))
            bler_avg_list.append(avg_bler)
            print(
                f"[{scheme_cfg.label}] SNR={snr_db:>5.1f} dB | "
                f"Avg BLER={avg_bler:.4e} | per-target={['%.4f' % b for b in target_blers]}"
            )

            if snr_early_stop_th is not None and avg_bler <= snr_early_stop_th:
                print(f"  -> Avg BLER <= {snr_early_stop_th}. Padding remaining SNRs.")
                remaining = len(snr_db_list) - len(bler_avg_list)
                bler_avg_list.extend([0.0] * remaining)
                break

    return np.array(bler_avg_list)


# =============================================================================
# Main demo
# =============================================================================
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Common simulation settings
    # -------------------------------------------------------------------------
    snr_points = np.arange(-25, -10, 1.0).tolist()

    # Keep this small for first debugging. Increase after basic validation.
    n_trials_per_target = 200
    max_workers = 4

    # Use 96RB for both families to keep the comparison fair.
    nrb_sib = 48
    fs = 61.44e6

    # Channel choice:
    #   CDL is recommended for array / polarization / beam-state conclusions.
    #   TDL is only for sanity checks.
    channel_model = "CDL"  # "CDL" or "TDL"
    cdl_type = "CDLC"      # e.g. "CDLB", "CDLC", "CDLD"

    # Target mapping list.
    # Default is one beam pair around boresight for quick comparison.
    # For sector average, uncomment the 8-pair list below.
    target_mappings = [
        {"name": "Pair_qh0_qh1", "qv": 0, "qh_a": 0, "qh_b": 1},
    ]

    # Example sector-average mappings. More expensive.
    # target_mappings = [
    #     {"name": f"Pair_qh{q}_qh{(q + 1) % 8}", "qv": 0, "qh_a": q, "qh_b": (q + 1) % 8}
    #     for q in range(8)
    # ]

    # -------------------------------------------------------------------------
    # Scheme configurations
    # -------------------------------------------------------------------------
    scheme_configs = [
        # Array-based family: aperture/subarray-based coarse/fine beams.
        SchemeConfig(
            label="Array-based: 4-ant coarse beam, full 48RB",
            family="array_based",
            scheme="array_coarse_baseline",
            prg_size_rb=48,
            prg_precoding_mode="traditional",
            ce_mode="fullband",
        ),
        SchemeConfig(
            label="Array-based: FDM fine beams, 24RB+24RB",
            family="array_based",
            scheme="array_fdm_fine",
            prg_size_rb=24,
            prg_precoding_mode="traditional",
            ce_mode="fullband",
        ),
        SchemeConfig(
            label="Array-based: refined 8-ant fine beam, full 48RB",
            family="array_based",
            scheme="array_refined_fine",
            prg_size_rb=48,
            prg_precoding_mode="traditional",
            ce_mode="fullband",
        ),

        # Precoding-based family: PRG-level cyclic phase changes vector beam state.
        SchemeConfig(
            label="Precoding-based: cyclic dual-pol, PRG=12RB",
            family="precoding_based",
            scheme="precoding_cyclic",
            prg_size_rb=24,
            prg_precoding_mode="cyclic",
            ce_mode="fullband",
            precoding_basis="polarization",
        ),
        # Optional stronger spatial-composite variant.
        SchemeConfig(
            label="Precoding-based: cyclic adjacent-beam basis, PRG=12RB",
            family="precoding_based",
            scheme="precoding_cyclic",
            prg_size_rb=24,
            prg_precoding_mode="cyclic",
            ce_mode="fullband",
            precoding_basis="spatial_adjacent",
        ),
    ]

    curves: Dict[str, np.ndarray] = {}
    out_csv = "sib1_array_vs_precoding_fine_beam_bler.csv"
    out_png = "sib1_array_vs_precoding_fine_beam_bler.png"

    for cfg in scheme_configs:
        bler_curve = simulate_bler_compare(
            snr_db_list=snr_points,
            scheme_cfg=cfg,
            target_mappings=target_mappings,
            n_trials_per_target=n_trials_per_target,
            ncellid=208,
            fs=fs,
            scs_khz=30,
            seed=2029,
            nrb_sib=nrb_sib,
            nsym_sib=12,
            N_comb=1,
            payload=1200,
            max_workers=max_workers,
            target_errors=50,
            snr_early_stop_th=0.04,
            channel_model=channel_model,
            cdl_type=cdl_type,
            tdl_profile="TDL-C",
            tdl_directional=True,
        )
        curves[cfg.label] = bler_curve

    # -------------------------------------------------------------------------
    # Save CSV and plot
    # -------------------------------------------------------------------------
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Config",
            "Family",
            "Scheme",
            "PRG_Size_RB",
            "Precoding_Mode",
            "CE_Mode",
            "Precoding_Basis",
            "Channel_Model",
            "CDL_Type",
            "SNR(dB)",
            "BLER",
        ])
        for cfg in scheme_configs:
            for snr, bler_val in zip(snr_points, curves[cfg.label]):
                writer.writerow([
                    cfg.label,
                    cfg.family,
                    cfg.scheme,
                    cfg.prg_size_rb,
                    cfg.prg_precoding_mode,
                    cfg.ce_mode,
                    cfg.precoding_basis,
                    channel_model,
                    cdl_type,
                    snr,
                    bler_val,
                ])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 6))
    for label, bler_curve in curves.items():
        plt.semilogy(snr_points, np.maximum(bler_curve, 1e-4), marker='o', label=label)
    plt.grid(True, which="both")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Average BLER")
    plt.title(f"SIB1 BLER: Array-based vs Precoding-based Fine Beam ({channel_model}/{cdl_type})")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)

    print(f"Saved CSV: {out_csv}")
    print(f"Saved figure: {out_png}")
