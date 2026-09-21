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

# Standard TDL implementation used in the previous SIB1 PDSCH script.
# The import is optional so CDL-only runs still work even if GenTDLChannel is absent.
try:
    from GenTDLChannel import ChannelInfo
except ImportError:
    ChannelInfo = None


# =============================================================================
# Global switches
# =============================================================================
PRINT_DIAG = False


# =============================================================================
# Beam-grid / antenna configuration: 2Vx4H coarse SSB, 2Vx8H fine beam
# =============================================================================
# User-specified beam hierarchy:
#   8 coarse beams  = 2 vertical beams x 4 horizontal beams
#   16 fine beams   = 2 vertical beams x 8 horizontal beams
#
# To keep the array/fine-beam comparison physically consistent, the CDL channel
# is generated on the larger physical array required by the fine beam:
#   [Mg, Ng, M, N, P] = [1, 1, 6, 8, 2] -> 96 Tx ports.
#
# The coarse SSB is then realized as a reduced-horizontal-aperture beam using
# only Np=4 active horizontal elements within the 8-column physical aperture.
# The fine beam uses all Np=8 horizontal elements.
TX_VERTICAL_SIZE = 6
HORIZONTAL_ARRAY_SIZE = 8
NUM_POLARIZATIONS = 2
TX_PORTS = TX_VERTICAL_SIZE * HORIZONTAL_ARRAY_SIZE * NUM_POLARIZATIONS  # 96T

# Coarse/fine grid dimensions.
N_COARSE_V = 2
N_COARSE_H = 4
N_FINE_V = 2
N_FINE_H = 8
N_COARSE_SSB = N_COARSE_V * N_COARSE_H     # 8 = 2V x 4H
N_FINE_BEAMS = N_FINE_V * N_FINE_H         # 16 = 2V x 8H

# DFT-index spacing in the vertical/horizontal domains.
COARSE_QV_STEP = TX_VERTICAL_SIZE / N_COARSE_V       # 6 / 2 = 3.0
COARSE_QH_STEP = HORIZONTAL_ARRAY_SIZE / N_COARSE_H  # 8 / 4 = 2.0
FINE_QV_STEP = TX_VERTICAL_SIZE / N_FINE_V           # 6 / 2 = 3.0
FINE_QH_STEP = HORIZONTAL_ARRAY_SIZE / N_FINE_H      # 8 / 8 = 1.0

# Fine-pair association mode for one coarse SSB horizontal sector.
# "center_plus" means SSB(h) -> fine qh = qh_center, qh_center + FINE_QH_STEP.
# This makes SSB0 include qh=0, which is important for the current CDL center.
# Other useful modes:
#   "center_minus"       -> qh_center - FINE_QH_STEP, qh_center
#   "symmetric_halfstep" -> qh_center - 0.5*FINE_QH_STEP, qh_center + 0.5*FINE_QH_STEP
FINE_PAIR_MODE = "center_plus"

# Array-aperture choices.
COARSE_MH_ACTIVE = N_COARSE_H       # 4 horizontal active elements -> ~29 deg
FINE_MH_ACTIVE = N_FINE_H           # 8 horizontal active elements -> ~14 deg


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
# RE-level comb DMRS utilities: 01020102... pattern
# =============================================================================

def nrSIB1DMRSIndices_comb2_re_param(
    nrb_sib: int,
    nsym_slot: int,
    dmrs_loc,
    k_step: int = 2,
    style: str = "matlab",
):
    """
    Generate two interleaved DMRS index sets while keeping the union of DMRS REs
    identical to the legacy 1/2-density pattern.

    Legacy SIB1 DMRS in this script uses k = 0, 2, 4, ... when k_step=2.
    For fair comparison, the comb2 RE-level scheme splits exactly this legacy
    DMRS union into two states:
      k = 0, 4, 8, ... -> state-1 DMRS
      k = 2, 6, 10,... -> state-2 DMRS

    On DMRS symbols, the resulting pattern is effectively:
      k mod 4 = 0 : DMRS for state 1
      k mod 4 = 1 : data associated with state 1
      k mod 4 = 2 : DMRS for state 2
      k mod 4 = 3 : data associated with state 2

    This is the same "two-state comb" idea as 01020102..., but with a comb
    offset chosen so the total DMRS RE positions are identical to the legacy
    baseline. Each state still sees 1/4-density DMRS in frequency.
    """
    Nsc = int(nrb_sib) * 12
    locs = _normalize_dmrs_loc(dmrs_loc, nsym_slot)

    k_union = np.arange(0, Nsc, k_step, dtype=int)
    k_b1 = k_union[0::2]
    k_b2 = k_union[1::2]

    k1_all = np.tile(k_b1, len(locs))
    l1_all = np.repeat(locs, len(k_b1))
    k2_all = np.tile(k_b2, len(locs))
    l2_all = np.repeat(locs, len(k_b2))

    if style == "python":
        return (tuple(k1_all.tolist()), tuple(l1_all.tolist())), (tuple(k2_all.tolist()), tuple(l2_all.tolist()))
    if style == "matlab":
        return (k1_all + l1_all * Nsc).astype(int), (k2_all + l2_all * Nsc).astype(int)
    raise ValueError("Unknown style")


def nrSIB1DataIndices_comb2_re_param(nrb_sib: int, nsym_slot: int, dmrs_loc):
    """
    Return total data REs and the two per-state data RE subsets.

    The DMRS union is identical to legacy k_step=2 DMRS. It is merely split into
    state-1 and state-2 DMRS. Data REs are assigned by the same 4-subcarrier
    comb period:
      k mod 4 in {0, 1} -> state 1
      k mod 4 in {2, 3} -> state 2

    On DMRS symbols, k mod 4 = 0 and 2 are removed as DMRS, leaving k mod 4 =
    1 for state-1 data and k mod 4 = 3 for state-2 data. On non-DMRS symbols,
    all REs are data and are split by the same state assignment rule.

    The union of data_idx_b1 and data_idx_b2 is exactly sib1_idx_all, and the
    order of sib1_idx_all is preserved for LDPC/rate-recovery consistency.
    """
    Nsc = int(nrb_sib) * 12
    total_re = Nsc * int(nsym_slot)
    all_idx = np.arange(total_re, dtype=int)

    dmrs_idx_b1, dmrs_idx_b2 = nrSIB1DMRSIndices_comb2_re_param(nrb_sib, nsym_slot, dmrs_loc, 2, "matlab")
    dmrs_all = np.sort(np.concatenate([dmrs_idx_b1, dmrs_idx_b2]))
    sib1_idx_all = np.setdiff1d(all_idx, dmrs_all, assume_unique=False)

    k = sib1_idx_all % Nsc
    data_idx_b1 = sib1_idx_all[(k % 4 == 0) | (k % 4 == 1)]
    data_idx_b2 = sib1_idx_all[(k % 4 == 2) | (k % 4 == 3)]

    return sib1_idx_all, data_idx_b1, data_idx_b2, dmrs_idx_b1, dmrs_idx_b2

def nrSIBDMRS_param_for_indices(ncellid, nrb_sib, dmrs_idx, n_slot=0, cinit_offset=0):
    """
    Generate QPSK DMRS symbols for an arbitrary DMRS index set.

    The output order matches dmrs_idx order. For multi-symbol DMRS, PRBS is
    generated independently per DMRS symbol and concatenated in ascending symbol
    order, matching the index construction above.

    cinit_offset is a simulation hook to distinguish state-1/state-2 DMRS
    sequences if desired. Since the two DMRS sets are FDM-disjoint, using the
    same sequence family is also acceptable for a first-order link comparison.
    """
    dmrs_idx = np.asarray(dmrs_idx, dtype=int)
    if dmrs_idx.size == 0:
        return np.array([], dtype=complex)

    Nsc = int(nrb_sib) * 12
    out = []
    for l_dmrs in np.unique(dmrs_idx // Nsc):
        idx_l = dmrs_idx[(dmrs_idx // Nsc) == l_dmrs]
        M = len(idx_l)
        cinit = (nrSIB1DMRScinit(n_slot, int(l_dmrs), ncellid) + int(cinit_offset)) % (2 ** 31)
        c = nrPRBS(cinit, 2 * M)
        out.append(np.asarray(nrSymbolModulate(c, "QPSK"), dtype=complex))
    return np.concatenate(out, axis=0)


def build_SIB1pdsch_grid_comb2_re_dmrs(
    ncellid,
    trblk_bits,
    nrb_sib=48,
    nsym_sib=12,
    dmrs_loc=(0, 6, 9),
    cbs_info=None,
    dmrs2_cinit_offset=1009,
):
    """
    Build a SIB1 PDSCH grid using RE-level comb DMRS:
      01020102... on each DMRS symbol.

    The total DMRS overhead remains 1/2 on DMRS symbols, but each beam/precoder
    state has only 1/4-density DMRS. Data REs are split into two state-specific
    subsets so the receiver can equalize state-1 data with H1 and state-2 data
    with H2.
    """
    if cbs_info is None:
        cbs_info = nrDLSCHInfo(len(trblk_bits), 0.5)

    grid = np.zeros((nrb_sib * 12, nsym_sib), dtype=complex)
    dmrs_cols = _normalize_dmrs_loc(dmrs_loc, nsym_sib)

    sib1_idx, data_idx_b1, data_idx_b2, dmrs_idx_b1, dmrs_idx_b2 = nrSIB1DataIndices_comb2_re_param(
        nrb_sib, nsym_sib, dmrs_cols
    )
    dmrs_idx = np.sort(np.concatenate([dmrs_idx_b1, dmrs_idx_b2]))
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

    dmrs_symb_b1 = nrSIBDMRS_param_for_indices(ncellid, nrb_sib, dmrs_idx_b1, n_slot=0, cinit_offset=0)
    dmrs_symb_b2 = nrSIBDMRS_param_for_indices(ncellid, nrb_sib, dmrs_idx_b2, n_slot=0, cinit_offset=dmrs2_cinit_offset)

    nrSetResources(sib1_idx, grid, sib1_symb)
    nrSetResources(dmrs_idx_b1, grid, dmrs_symb_b1)
    nrSetResources(dmrs_idx_b2, grid, dmrs_symb_b2)

    return grid, meta, {
        "sib1_idx": sib1_idx,
        "data_idx_b1": data_idx_b1,
        "data_idx_b2": data_idx_b2,
        "dmrs_idx": dmrs_idx,
        "dmrs_idx_b1": dmrs_idx_b1,
        "dmrs_idx_b2": dmrs_idx_b2,
        "dmrs_cols": dmrs_cols,
        "E": E,
        "dmrs_pattern": "comb2_re",
        "dmrs2_cinit_offset": dmrs2_cinit_offset,
    }


# =============================================================================
# Beam and precoder utilities
# =============================================================================
def iter_prg_ranges(nrb_sib, prg_size_rb):
    for rb0 in range(0, nrb_sib, prg_size_rb):
        yield rb0, min(nrb_sib, rb0 + prg_size_rb)


def build_subarray_dft_vector_64tx(Mv=TX_VERTICAL_SIZE, Mh_total=HORIZONTAL_ARRAY_SIZE, Mh_active=FINE_MH_ACTIVE, qv=0, qh_equiv=0.0):
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



def build_upa_dft_vector(Mv=TX_VERTICAL_SIZE, Mh=HORIZONTAL_ARRAY_SIZE, qv=0, qh=0):
    """
    Full-aperture 6x4 UPA steering/DFT-like beam aligned with the PBCH CDL setup.

    qh is intentionally allowed to be fractional, e.g. qh=0.5, so that the
    precoding-based scheme can be geometrically aligned with the center between
    two array fine beams qh=0 and qh=1. Integer qh values reproduce the original
    DFT codebook points.
    """
    qv = float(qv) % Mv
    qh = float(qh) % Mh
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


def build_wbf_64x2_subarray(Mv=TX_VERTICAL_SIZE, Mh_total=HORIZONTAL_ARRAY_SIZE, Mh_active=FINE_MH_ACTIVE, qv=0, qh_equiv=0.0):
    """48Tx x 2 dual-pol Wbf for array/subarray-based beam."""
    v_sp = build_subarray_dft_vector_64tx(Mv, Mh_total, Mh_active, qv, qh_equiv)
    return np.column_stack([
        map_spatial_beam_to_dualpol_64(v_sp, pol_idx=0),
        map_spatial_beam_to_dualpol_64(v_sp, pol_idx=1),
    ])


def build_wbf_64x2_upa(qv1=0, qh1=0, qv2=None, qh2=None):
    """
    48Tx x 2 dual-pol Wbf for precoding-based beam-state variation.

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

    v1 = build_upa_dft_vector(Mv=TX_VERTICAL_SIZE, Mh=HORIZONTAL_ARRAY_SIZE, qv=qv1, qh=qh1)
    v2 = build_upa_dft_vector(Mv=TX_VERTICAL_SIZE, Mh=HORIZONTAL_ARRAY_SIZE, qv=qv2, qh=qh2)
    w1 = map_spatial_beam_to_dualpol_64(v1, pol_idx=0)
    w2 = map_spatial_beam_to_dualpol_64(v2, pol_idx=1)
    return np.column_stack([w1, w2])


def bb_precoder_traditional():
    return np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)



def bb_precoder_cyclic(prg_idx, phase_table=(0, np.pi / 2, np.pi, 3 * np.pi / 2)):
    phi = phase_table[int(prg_idx) % len(phase_table)]
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
    # DMRS layout: "legacy" keeps 010101..., "comb2_re" enables 01020102...
    dmrs_pattern: str = "legacy"



def circular_mid_qh(qh_a, qh_b, Mh=HORIZONTAL_ARRAY_SIZE):
    """
    Circular midpoint between two horizontal beam indices.

    Examples with Mh=8:
      qh_a=0, qh_b=1 -> 0.5
      qh_a=7, qh_b=0 -> 7.5 (equivalent to -0.5), not 3.5
    """
    qh_a = float(qh_a) % Mh
    qh_b = float(qh_b) % Mh
    diff = ((qh_b - qh_a + Mh / 2.0) % Mh) - Mh / 2.0
    return (qh_a + diff / 2.0) % Mh


def make_target_mappings_8ssb_16fine(run_all_ssb=False, center_ssb_idx=0):
    """
    Build target mappings for the configured 2Vx4H coarse SSB grid and 2Vx8H
    fine-beam grid.

    Coarse SSB index mapping:
      ssb_idx = v_idx * N_COARSE_H + h_idx
      qv_center = v_idx * COARSE_QV_STEP
      qh_center = h_idx * COARSE_QH_STEP

    Fine-beam association for each coarse SSB keeps the same vertical beam and
    selects two horizontal fine beams according to FINE_PAIR_MODE.  With the
    default center_plus mode, SSB00 maps to fine qh=(0, 1), so the current CDL
    boresight at qh≈0 is included in the array fine-beam set.
    """
    ssb_indices = range(N_COARSE_SSB) if run_all_ssb else [int(center_ssb_idx) % N_COARSE_SSB]
    mappings = []

    for ssb_idx in ssb_indices:
        v_idx = int(ssb_idx) // N_COARSE_H
        h_idx = int(ssb_idx) % N_COARSE_H

        qv_center = (v_idx * COARSE_QV_STEP) % TX_VERTICAL_SIZE
        qh_center = (h_idx * COARSE_QH_STEP) % HORIZONTAL_ARRAY_SIZE

        if FINE_PAIR_MODE == "center_plus":
            qh_a = qh_center
            qh_b = (qh_center + FINE_QH_STEP) % HORIZONTAL_ARRAY_SIZE
            fine_h_idx_a = (2 * h_idx) % N_FINE_H
            fine_h_idx_b = (2 * h_idx + 1) % N_FINE_H
        elif FINE_PAIR_MODE == "center_minus":
            qh_a = (qh_center - FINE_QH_STEP) % HORIZONTAL_ARRAY_SIZE
            qh_b = qh_center
            fine_h_idx_a = (2 * h_idx - 1) % N_FINE_H
            fine_h_idx_b = (2 * h_idx) % N_FINE_H
        elif FINE_PAIR_MODE == "symmetric_halfstep":
            qh_a = (qh_center - 0.5 * FINE_QH_STEP) % HORIZONTAL_ARRAY_SIZE
            qh_b = (qh_center + 0.5 * FINE_QH_STEP) % HORIZONTAL_ARRAY_SIZE
            fine_h_idx_a = None
            fine_h_idx_b = None
        else:
            raise ValueError(f"Unsupported FINE_PAIR_MODE: {FINE_PAIR_MODE}")

        fine_v_idx = v_idx  # vertical beam count is unchanged: 2 coarse V = 2 fine V
        fine_idx_a = None if fine_h_idx_a is None else fine_v_idx * N_FINE_H + fine_h_idx_a
        fine_idx_b = None if fine_h_idx_b is None else fine_v_idx * N_FINE_H + fine_h_idx_b

        mappings.append({
            "name": (
                f"SSB{ssb_idx:02d}_v{v_idx}_h{h_idx}_"
                f"center_qv{qv_center:g}_qh{qh_center:g}_"
                f"fine_qh{qh_a:g}_qh{qh_b:g}"
            ),
            "ssb_idx": ssb_idx,
            "v_idx": v_idx,
            "h_idx": h_idx,
            "qv": qv_center,
            "qh_center": qh_center,
            "qh_a": qh_a,
            "qh_b": qh_b,
            "fine_idx_left": fine_idx_a,
            "fine_idx_right": fine_idx_b,
            "fine_pair_mode": FINE_PAIR_MODE,
        })
    return mappings

def resolve_target_mapping(target_mapping, Mh=HORIZONTAL_ARRAY_SIZE):
    """
    Resolve one target mapping into a beam pair.

    Preferred input generated by make_target_mappings_8ssb_16fine():
      {"qh_center": ..., "qh_a": ..., "qh_b": ...}

    Backward-compatible input options:
      {"qh_a": 0, "qh_b": 1}
      {"qh_center": 0.0, "qh_delta": 0.5}

    Priority rule:
      1) If qh_a/qh_b are explicitly provided, use them directly.
         This is required for center_plus / center_minus fine-pair association,
         where the pair is not symmetric around qh_center.
      2) Else if qh_center/qh_delta are provided, synthesize a symmetric pair.
      3) Else raise an error.
    """
    qv = target_mapping.get("qv", 0)

    if "qh_a" in target_mapping and "qh_b" in target_mapping:
        qh_a = float(target_mapping["qh_a"]) % Mh
        qh_b = float(target_mapping["qh_b"]) % Mh
        if "qh_center" in target_mapping:
            qh_center = float(target_mapping["qh_center"]) % Mh
        else:
            qh_center = circular_mid_qh(qh_a, qh_b, Mh=Mh)
    elif "qh_center" in target_mapping:
        qh_center = float(target_mapping["qh_center"]) % Mh
        # Symmetric fallback only for old/manual mappings. In the new 2Vx4H/2Vx8H
        # grid, make_target_mappings_8ssb_16fine() should provide qh_a/qh_b.
        qh_delta = float(target_mapping.get("qh_delta", 0.5 * FINE_QH_STEP))
        qh_a = (qh_center - qh_delta) % Mh
        qh_b = (qh_center + qh_delta) % Mh
    else:
        raise ValueError("target_mapping must contain either qh_a/qh_b or qh_center")

    return qv, qh_a, qh_b, qh_center

def make_wbf_list_for_scheme(cfg: SchemeConfig, qv: int, qh_a: float, qh_b: float):
    """
    Return a list of Wbf matrices.

    Alignment rule:
      - qh_center is always the circular midpoint between qh_a and qh_b.
      - array coarse / centered reference schemes point to qh_center.
      - array comb pair uses qh_a and qh_b, whose pair center is qh_center.
      - precoding comb schemes use the same spatial beam centered at qh_center;
        state differences are introduced only by dual-pol baseband precoding.
    """
    qh_center = circular_mid_qh(qh_a, qh_b, Mh=HORIZONTAL_ARRAY_SIZE)

    if cfg.scheme == "array_coarse_baseline":
        # Coarse beam: reduced aperture, centered between the two fine beams.
        return [build_wbf_64x2_subarray(TX_VERTICAL_SIZE, HORIZONTAL_ARRAY_SIZE, Mh_active=COARSE_MH_ACTIVE, qv=qv, qh_equiv=qh_center)]

    if cfg.scheme == "array_centered_fine":
        # Full-aperture reference beam exactly at the same center as the coarse beam.
        return [build_wbf_64x2_subarray(TX_VERTICAL_SIZE, HORIZONTAL_ARRAY_SIZE, Mh_active=FINE_MH_ACTIVE, qv=qv, qh_equiv=qh_center)]

    if cfg.scheme == "array_fdm_fine":
        # FDM fine-beam refinement: two full-aperture fine beams centered around qh_center.
        return [
            build_wbf_64x2_subarray(TX_VERTICAL_SIZE, HORIZONTAL_ARRAY_SIZE, Mh_active=FINE_MH_ACTIVE, qv=qv, qh_equiv=qh_a),
            build_wbf_64x2_subarray(TX_VERTICAL_SIZE, HORIZONTAL_ARRAY_SIZE, Mh_active=FINE_MH_ACTIVE, qv=qv, qh_equiv=qh_b),
        ]

    if cfg.scheme == "array_refined_fine":
        # Backward-compatible original behavior: selected fine beam qh_a.
        return [build_wbf_64x2_subarray(TX_VERTICAL_SIZE, HORIZONTAL_ARRAY_SIZE, Mh_active=FINE_MH_ACTIVE, qv=qv, qh_equiv=qh_a)]

    if cfg.scheme == "array_comb_re_dmrs":
        # RE-level comb DMRS: state 1 and state 2 use the two fine beams whose
        # pair center is aligned to the coarse/reference center.
        return [
            build_wbf_64x2_subarray(TX_VERTICAL_SIZE, HORIZONTAL_ARRAY_SIZE, Mh_active=FINE_MH_ACTIVE, qv=qv, qh_equiv=qh_a),
            build_wbf_64x2_subarray(TX_VERTICAL_SIZE, HORIZONTAL_ARRAY_SIZE, Mh_active=FINE_MH_ACTIVE, qv=qv, qh_equiv=qh_b),
        ]

    if cfg.scheme == "array_comb_re_dmrs_same_center":
        # Control: both comb states use the same full-aperture centered beam.
        return [
            build_wbf_64x2_subarray(TX_VERTICAL_SIZE, HORIZONTAL_ARRAY_SIZE, Mh_active=FINE_MH_ACTIVE, qv=qv, qh_equiv=qh_center),
            build_wbf_64x2_subarray(TX_VERTICAL_SIZE, HORIZONTAL_ARRAY_SIZE, Mh_active=FINE_MH_ACTIVE, qv=qv, qh_equiv=qh_center),
        ]

    if cfg.scheme == "precoding_comb_re_dmrs":
        # RE-level comb DMRS: both states share the same centered spatial beam
        # basis; state 1/state 2 can differ by dual-pol baseband precoder.
        Wbf = build_wbf_64x2_upa(qv1=qv, qh1=qh_center, qv2=qv, qh2=qh_center)
        return [Wbf, Wbf]

    if cfg.scheme == "precoding_cyclic":
        if cfg.precoding_basis == "polarization":
            return [build_wbf_64x2_upa(qv1=qv, qh1=qh_center, qv2=qv, qh2=qh_center)]
        if cfg.precoding_basis == "spatial_adjacent":
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


def build_precoded_tx_grids_comb2_re_dmrs(
    base_grid,
    meta2,
    Wbf_list,
    prg_precoding_mode="traditional",
    phase_table=(0, np.pi / 2, np.pi, 3 * np.pi / 2),
):
    """
    RE-level beam/precoder mapping for comb2 DMRS.

    State-1 REs: data_idx_b1 + dmrs_idx_b1 -> Wbf_list[0] and precoder phase 0.
    State-2 REs: data_idx_b2 + dmrs_idx_b2 -> Wbf_list[1] and precoder phase 1
                 if prg_precoding_mode == "cyclic"; otherwise traditional.

    This deliberately does not use PRG/bundle selection. The granularity is RE.
    """
    if Wbf_list is None or len(Wbf_list) == 0:
        raise ValueError("comb2_re DMRS requires Wbf_list with at least one Wbf")
    if len(Wbf_list) == 1:
        Wbf_list = [Wbf_list[0], Wbf_list[0]]

    Wbf_b1, Wbf_b2 = Wbf_list[0], Wbf_list[1]
    K, N = base_grid.shape
    Nt = Wbf_b1.shape[0]
    tx_grids = np.zeros((Nt, K, N), dtype=np.complex128)

    # State 1 uses phase_table[0]. State 2 uses phase_table[1] in cyclic mode.
    wbb_b1 = get_bb_precoder("traditional", 0, phase_table)
    if prg_precoding_mode == "cyclic":
        wbb_b2 = get_bb_precoder("cyclic", 1, phase_table)
    else:
        wbb_b2 = get_bb_precoder("traditional", 0, phase_table)

    w_eff_b1 = Wbf_b1 @ wbb_b1
    w_eff_b2 = Wbf_b2 @ wbb_b2
    w_eff_b1 = w_eff_b1 / np.linalg.norm(w_eff_b1)
    w_eff_b2 = w_eff_b2 / np.linalg.norm(w_eff_b2)

    idx_b1 = np.sort(np.concatenate([meta2["data_idx_b1"], meta2["dmrs_idx_b1"]])).astype(int)
    idx_b2 = np.sort(np.concatenate([meta2["data_idx_b2"], meta2["dmrs_idx_b2"]])).astype(int)

    k1, l1 = idx_b1 % K, idx_b1 // K
    k2, l2 = idx_b2 % K, idx_b2 // K

    tx_grids[:, k1, l1] = w_eff_b1[:, None] * base_grid[k1, l1][None, :]
    tx_grids[:, k2, l2] = w_eff_b2[:, None] * base_grid[k2, l2][None, :]
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


def cdl_impulse_response_48t4r_full(
    fs_hz,
    ds_ns=300,
    fc_GHz=7,
    seed=None,
    initial_time=0.0,
    ue_speed=120,
    cdl_type="CDLC",
):
    """
    CDL channel generation aligned with the converged PBCH simulation script.

    PBCH reference configuration:
      - cdl_type='CDLC'
      - Tx antenna [1, 1, 6, 4, 2] => 48 Tx ports
      - Rx antenna [1, 1, 1, 2, 2] => 4 Rx ports
      - delay_spread = 300 ns
      - fc = 7 GHz
      - ue_speed = 120 km/h
      - angle_gap = 0.0, power_gap = 0.0
      - beamforming='no', chan_method='online'
    """
    if seed is not None:
        np.random.seed(seed)
    ds = ds_ns * 1e-9

    # Exactly follow the converged PBCH CDL setup.
    tx_ant = [1, 1, TX_VERTICAL_SIZE, HORIZONTAL_ARRAY_SIZE, NUM_POLARIZATIONS]  # [1, 1, 6, 8, 2] -> 96T
    rx_ant = [1, 1, 1, 2, 2]                                                     # -> 4R

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

    n_rx = 4
    n_tx = TX_PORTS
    H_mimo = np.zeros((n_rx, n_tx, len(h00)), dtype=np.complex128)
    for rx in range(n_rx):
        for tx in range(n_tx):
            H_mimo[rx, tx, :] = channel_time_interpolation(
                delays, h_paths[:, rx, tx], ts=1.0 / fs_hz, num_ext=4
            )
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
    speed_kmh=120,
    initial_time=0.0,
    cdl_type="CDLC",
):
    H_mimo = cdl_impulse_response_48t4r_full(
        fs_hz,
        ds_ns,
        fc_hz / 1e9,
        seed,
        initial_time,
        speed_kmh,
        cdl_type=cdl_type,
    )
    return apply_mimo_fir(tx_waves, H_mimo), H_mimo


def _tdl_profile_to_chantype(profile: str) -> str:
    """
    Convert user-facing profile names to the ChanType expected by GenTDLChannel.
      "TDL-C" -> "TDLC"
      "TDLC"  -> "TDLC"
    """
    p = str(profile).upper().replace("_", "-").replace(" ", "")
    if p.startswith("TDL-"):
        p = p.replace("TDL-", "TDL")
    valid = {"TDLA", "TDLB", "TDLC", "TDLD", "TDLE"}
    if p not in valid:
        raise ValueError(f"Unsupported TDL profile for GenTDLChannel: {profile}")
    return p


def standard_tdl_impulse_response_64t4r_full(
    fs_hz,
    ds_ns=300.0,
    fc_hz=7e9,
    speed_kmh=3.0,
    nfft=4096,
    seed=None,
    profile="TDL-C",
    n_rx=4,
    n_tx=TX_PORTS,
    normalize=False,
):
    """
    Standard TDL MIMO channel based on GenTDLChannel.ChannelInfo.

    This follows the older STD_SIB1_PDSCH script's TDL path:
      ch = ChannelInfo(..., ChanType='TDLC', DS=..., NFFT=...)
      ch.gen_Rayleigh(num_samples=1)
      H = ch.time_channel[:, :, :, 0]

    Notes:
      - DS is passed to ChannelInfo in ns, consistent with the previous script's
        actual call pattern apply_tdl_c(..., ds=300).
      - The generated channel is an i.i.d. MIMO TDL channel. It intentionally does
        not model CDL-like AoD/AoA, array manifold, or dual-pol XPR structure.
      - This is therefore useful for sanity checks and for reducing CDL-specific
        polarization/angle effects, but CDL remains preferred for beam-state claims.
    """
    if ChannelInfo is None:
        raise ImportError(
            "GenTDLChannel.ChannelInfo is not available. "
            "Please ensure GenTDLChannel.py is in PYTHONPATH before using channel_model='TDL'."
        )

    if seed is not None:
        np.random.seed(seed)

    chan_type = _tdl_profile_to_chantype(profile)

    ch = ChannelInfo(
        nTx=int(n_tx),
        nRx=int(n_rx),
        Speed=float(speed_kmh),
        Fc=float(fc_hz),
        Fs=float(fs_hz),
        ChanType=chan_type,
        DS=float(ds_ns),
        NFFT=int(nfft),
        TO=0,
        FO=0,
    )

    # Static one-shot Rayleigh TDL realization.
    ch.gen_Rayleigh(num_samples=1)

    H = np.asarray(ch.time_channel[:, :, :, 0], dtype=np.complex128)

    if normalize:
        # Optional average-link normalization: E_link[sum_l |h_l|^2] -> 1.
        # Keep False by default to match the user's previous script, where the
        # normalization lines were intentionally commented out.
        p = np.mean(np.sum(np.abs(H) ** 2, axis=2))
        if p > 0:
            H = H / np.sqrt(p)

    return H


def tdl_c_impulse_response(
    fs_hz,
    ds_ns=300.0,
    fc_hz=7e9,
    speed_kmh=3.0,
    nfft=4096,
    seed=None,
    normalize=False,
):
    """
    SISO helper equivalent to the previous apply_tdl_c()/tdl_c_impulse_response()
    path, kept for debugging and cross-checking.
    """
    H = standard_tdl_impulse_response_64t4r_full(
        fs_hz=fs_hz,
        ds_ns=ds_ns,
        fc_hz=fc_hz,
        speed_kmh=speed_kmh,
        nfft=nfft,
        seed=seed,
        profile="TDL-C",
        n_rx=1,
        n_tx=1,
        normalize=normalize,
    )
    return H[0, 0, :]


def apply_tdl_mimo_tx(
    tx_waves,
    fs_hz,
    seed=None,
    ds_ns=300.0,
    fc_hz=7e9,
    speed_kmh=3.0,
    nfft=4096,
    profile="TDL-C",
    directional=False,
    main_qv=0,
    main_qh=0,
    normalize=False,
):
    """
    Apply standard GenTDLChannel-based MIMO TDL.

    The 'directional', 'main_qv', and 'main_qh' arguments are accepted only for
    interface compatibility with the previous lightweight directional TDL path.
    They are intentionally not used here, because the standard TDL path is meant
    to remove CDL-style angle/array/polarization geometry from the sanity check.
    """
    H_mimo = standard_tdl_impulse_response_64t4r_full(
        fs_hz=fs_hz,
        ds_ns=ds_ns,
        fc_hz=fc_hz,
        speed_kmh=speed_kmh,
        nfft=nfft,
        seed=seed,
        profile=profile,
        n_rx=4,
        n_tx=tx_waves.shape[1],
        normalize=normalize,
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


def estimate_comb2_re_dmrs_reuse_existing_ce(
    rxGrid,
    refGridH_b1,
    refGridH_b2,
    meta2,
    nrb_sib,
    prg_size_rb=None,
    td_win_len_ratio=0.08,
):
    """
    Reuse the existing myChannelEstimate_std() twice:
      - refGridH_b1 only contains state-1 DMRS, so it estimates H1 with 1/4-density DMRS.
      - refGridH_b2 only contains state-2 DMRS, so it estimates H2 with 1/4-density DMRS.

    Then fill one effective channel grid H_eff:
      data_idx_b1 -> H1
      data_idx_b2 -> H2

    This keeps the legacy equalization path unchanged and preserves sib1_idx order.
    """
    if prg_size_rb is None or prg_size_rb <= 0:
        prg_size_rb = nrb_sib

    H1_all = np.zeros_like(rxGrid, dtype=np.complex128)
    H2_all = np.zeros_like(rxGrid, dtype=np.complex128)
    dmrs_cols = meta2["dmrs_cols"]

    # For this RE-level scheme, using prg_size_rb=nrb_sib gives true full-band CE.
    # Smaller values are still supported for debugging, but are not needed.
    for rb0, rb1 in iter_prg_ranges(nrb_sib, prg_size_rb):
        H1_bundle = myChannelEstimate_std(
            rxGrid=rxGrid,
            refGrid=refGridH_b1,
            EST_TFDOMAIN=False,
            EST_PBCH=False,
            EST_PDSCH_FULLBAND=True,
            dmrs_l=dmrs_cols,
            start_rb=rb0,
            end_rb=rb1,
            td_win_len_ratio=td_win_len_ratio,
            td_window_type="fixed",
        )
        H2_bundle = myChannelEstimate_std(
            rxGrid=rxGrid,
            refGrid=refGridH_b2,
            EST_TFDOMAIN=False,
            EST_PBCH=False,
            EST_PDSCH_FULLBAND=True,
            dmrs_l=dmrs_cols,
            start_rb=rb0,
            end_rb=rb1,
            td_win_len_ratio=td_win_len_ratio,
            td_window_type="fixed",
        )
        sc0, sc1 = rb0 * 12, rb1 * 12
        H1_all[sc0:sc1, :] = H1_bundle[sc0:sc1, :]
        H2_all[sc0:sc1, :] = H2_bundle[sc0:sc1, :]

    H_eff = np.zeros_like(rxGrid, dtype=np.complex128)
    K = rxGrid.shape[0]

    idx1 = np.asarray(meta2["data_idx_b1"], dtype=int)
    idx2 = np.asarray(meta2["data_idx_b2"], dtype=int)
    k1, l1 = idx1 % K, idx1 // K
    k2, l2 = idx2 % K, idx2 // K
    H_eff[k1, l1] = H1_all[k1, l1]
    H_eff[k2, l2] = H2_all[k2, l2]

    return H_eff


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
    cdl_type="CDLC",
    tdl_profile="TDL-C",
    tdl_directional=True,
    tdl_normalize=False,
    main_qv=0,
    main_qh=0,
):
    if meta2 is not None and meta2.get("dmrs_pattern") == "comb2_re":
        tx_grids = build_precoded_tx_grids_comb2_re_dmrs(
            sib_grid,
            meta2,
            Wbf_list,
            prg_precoding_mode=prg_precoding_mode,
            phase_table=phase_table,
        )
    else:
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
            speed_kmh=120,
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
            normalize=tdl_normalize,
        )
    else:
        raise ValueError(f"Unsupported channel_model: {channel_model}")

    rxGrids_clean = ofdm_demodulate_multi_rx(ch_out, nrb_sib, scs_khz, 0, fs, nsym_sib, 0.5)

    active_idx = np.sort(np.concatenate([sib_idx, dmrs_idx]))
    rxGrids, nVar_list = add_awgn_on_grid_multi_rx(rxGrids_clean, sib_grid, snr_db, active_idx)

    if meta2.get("dmrs_pattern") == "comb2_re":
        refGridH_b1 = np.zeros_like(rxGrids[0], dtype=complex)
        refGridH_b2 = np.zeros_like(rxGrids[0], dtype=complex)
        dmrs_symb_b1 = nrExtractResources(meta2["dmrs_idx_b1"], sib_grid)
        dmrs_symb_b2 = nrExtractResources(meta2["dmrs_idx_b2"], sib_grid)
        nrSetResources(meta2["dmrs_idx_b1"], refGridH_b1, dmrs_symb_b1)
        nrSetResources(meta2["dmrs_idx_b2"], refGridH_b2, dmrs_symb_b2)

        llr_sum_rx = np.zeros(E, dtype=np.float64)
        # For the pure RE-level comb scheme, do not couple CE to PRG/bundle.
        comb_ce_prg_size = nrb_sib
        for r in range(rxGrids.shape[0]):
            H_eff = estimate_comb2_re_dmrs_reuse_existing_ce(
                rxGrid=rxGrids[r],
                refGridH_b1=refGridH_b1,
                refGridH_b2=refGridH_b2,
                meta2=meta2,
                nrb_sib=nrb_sib,
                prg_size_rb=comb_ce_prg_size,
                td_win_len_ratio=0.08,
            )
            sib_eq_r, nVar_post_r = nrEqualizeZF_persym(
                nrExtractResources(sib_idx, rxGrids[r]),
                nrExtractResources(sib_idx, H_eff),
                nVar_list[r],
                sib_idx,
                nrb_sib * 12,
            )
            llr_sum_rx += nrSymbolDemodulate_qpsk_persym(
                sib_eq_r,
                nVar_post_r,
                sib_idx,
                nrb_sib * 12,
            )

        return llr_sum_rx * (1 - 2 * meta["scr"].astype(int))

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
    channel_evolution_mode="pbch_like",
    channel_model="CDL",
    cdl_type="CDLC",
    tdl_profile="TDL-C",
    tdl_directional=True,
    tdl_normalize=False,
):
    if rng_seed is not None:
        np.random.seed(rng_seed)

    if scheme_cfg.dmrs_pattern == "comb2_re":
        sib_idx_tmp, _, _, _, _ = nrSIB1DataIndices_comb2_re_param(nrb_sib, nsym_sib, dmrs_loc)
    else:
        dmrs_idx_tmp = nrSIB1DMRSIndices_param(nrb_sib, nsym_sib, dmrs_loc, dmrs_step, "matlab")
        sib_idx_tmp = nrSIB1Indices_param(nrb_sib, nsym_sib, dmrs_idx_tmp, "matlab")

    rate = (payload + 16) / (len(sib_idx_tmp) * 2)
    cbs_info = nrDLSCHInfo(payload, rate)
    trblk_bits = np.random.randint(0, 2, payload).astype(int)

    if scheme_cfg.dmrs_pattern == "comb2_re":
        sib_grid, meta, meta2 = build_SIB1pdsch_grid_comb2_re_dmrs(
            ncellid,
            trblk_bits,
            nrb_sib,
            nsym_sib,
            dmrs_loc,
            cbs_info,
        )
    else:
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
        # PBCH-converged behavior:
        #   c_seed = rng_seed + r
        #   initial_time = r * 0.005 s
        if channel_evolution_mode == "pbch_like":
            c_seed = chan_seed_base + r
            t_init = chan_initial_time0 + r * 0.005
        elif channel_evolution_mode == "static_same":
            c_seed = chan_seed_base
            t_init = chan_initial_time0 + r * chan_round_spacing_s
        elif channel_evolution_mode == "independent":
            c_seed = chan_seed_base + r
            t_init = chan_initial_time0 + r * chan_round_spacing_s
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
            tdl_normalize=tdl_normalize,
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
    qv, qh_a, qh_b, qh_center = resolve_target_mapping(target_mapping, Mh=HORIZONTAL_ARRAY_SIZE)

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
        tdl_normalize=static_cfg["tdl_normalize"],
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
    cdl_type="CDLC",
    tdl_profile="TDL-C",
    tdl_directional=False,
    tdl_normalize=False,
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
        "tdl_normalize": tdl_normalize,
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
    # snr_points = np.arange(-25, -10, 1.0).tolist()

    # Keep this small for first debugging. Increase after basic validation.
    n_trials_per_target = 100
    max_workers = 4

    # Use 48RB for both families to keep the comparison fair.
    nrb_sib = 48
    fs = 61.44e6

    # Channel choice:
    #   CDL is recommended for array / polarization / beam-state conclusions.
    #   TDL is only for sanity checks.
    channel_model = "CDL"  # "CDL" or "TDL"
    cdl_type = "CDLC"      # e.g. "CDLB", "CDLC", "CDLD"
    tdl_profile = "TDL-C"
    if channel_model.upper() == "TDL":
        snr_points = np.arange(-10, 11, 1.0).tolist()
    else:
        snr_points = np.arange(-25, -10, 1.0).tolist()
    tdl_directional = False
    tdl_normalize = False
    # Target mapping list for the requested 2Vx4H coarse grid / 2Vx8H fine grid.
    # Default: SSB00 has qv=0, qh=0.  With FINE_PAIR_MODE="center_plus", its
    # associated fine horizontal beams are qh=0 and qh=1.
    #
    # Set RUN_ALL_8_SSB_SECTORS=True to average over all 8 coarse SSB sectors.
    RUN_ALL_8_SSB_SECTORS = False
    CENTER_SSB_INDEX = 0
    target_mappings = make_target_mappings_8ssb_16fine(
        run_all_ssb=RUN_ALL_8_SSB_SECTORS,
        center_ssb_idx=CENTER_SSB_INDEX,
    )

    print(f"[Beam grid] coarse={N_COARSE_V}Vx{N_COARSE_H}H={N_COARSE_SSB}, "
          f"fine={N_FINE_V}Vx{N_FINE_H}H={N_FINE_BEAMS}, "
          f"COARSE_QV_STEP={COARSE_QV_STEP}, COARSE_QH_STEP={COARSE_QH_STEP}, "
          f"FINE_QV_STEP={FINE_QV_STEP}, FINE_QH_STEP={FINE_QH_STEP}, "
          f"FINE_PAIR_MODE={FINE_PAIR_MODE}")
    for m in target_mappings:
        qv_dbg, qh_a_dbg, qh_b_dbg, qh_c_dbg = resolve_target_mapping(m)
        print(f"[Target] {m['name']} | qv={qv_dbg}, qh_center={qh_c_dbg}, "
              f"fine_pair=({qh_a_dbg}, {qh_b_dbg}), "
              f"fine_idx=({m.get('fine_idx_left')}, {m.get('fine_idx_right')})")

    # -------------------------------------------------------------------------
    # Scheme configurations
    # -------------------------------------------------------------------------
    scheme_configs = [
        # 8 coarse SSB directions: this reference uses the selected SSB center qh_c.
        SchemeConfig(
            label="8-SSB reference: 4-ant coarse beam centered at SSB qh_c",
            family="array_based",
            scheme="array_coarse_baseline",
            prg_size_rb=48,
            prg_precoding_mode="traditional",
            ce_mode="fullband",
            dmrs_pattern="legacy",
        ),

        # Ideal full-aperture reference exactly at the SSB center qh_c. Not one of the shifted 16 fine directions; use as an upper-bound/reference.
        SchemeConfig(
            label="Ideal reference: 8-ant fine beam exactly centered at SSB qh_c",
            family="array_based",
            scheme="array_centered_fine",
            prg_size_rb=48,
            prg_precoding_mode="traditional",
            ce_mode="fullband",
            dmrs_pattern="legacy",
        ),

        # # Control: comb DMRS structure only, no state misalignment.
        # SchemeConfig(
        #     label="Comb-DMRS control: same centered fine beam/state1=state2",
        #     family="array_based",
        #     scheme="array_comb_re_dmrs_same_center",
        #     prg_size_rb=48,
        #     prg_precoding_mode="traditional",
        #     ce_mode="fullband",
        #     dmrs_pattern="comb2_re",
        # ),

        # 16 fine directions: two adjacent fine beams around the selected SSB center, qh_c ± 0.25.
        SchemeConfig(
            label="Comb-DMRS aligned: 16-dir array fine pair around SSB qh_c",
            family="array_based",
            scheme="array_comb_re_dmrs",
            prg_size_rb=48,
            prg_precoding_mode="traditional",
            ce_mode="fullband",
            dmrs_pattern="comb2_re",
        ),

        # Control: same centered spatial beam and same dual-pol phase on both RE states.
        # SchemeConfig(
        #     label="Comb-DMRS control: centered precoding state [1,1]/[1,1]",
        #     family="precoding_based",
        #     scheme="precoding_comb_re_dmrs",
        #     prg_size_rb=48,
        #     prg_precoding_mode="traditional",
        #     ce_mode="fullband",
        #     precoding_basis="polarization",
        #     dmrs_pattern="comb2_re",
        # ),

        # Same centered spatial beam, two dual-pol precoder states [1,1] and [1,j].
        SchemeConfig(
            label="Comb-DMRS aligned: centered precoding states [1,1]/[1,j]",
            family="precoding_based",
            scheme="precoding_comb_re_dmrs",
            prg_size_rb=48,
            prg_precoding_mode="cyclic",
            ce_mode="fullband",
            precoding_basis="polarization",
            dmrs_pattern="comb2_re",
        ),
    ]

    curves: Dict[str, np.ndarray] = {}
    out_csv = "sib1_8ssb_16fine_comb_re_dmrs_bler.csv"
    out_png = "sib1_8ssb_16fine_comb_re_dmrs_bler.png"

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
            tdl_directional=False,   # ignored by standard GenTDLChannel-based TDL
            tdl_normalize=False,     # keep False to match previous STD_SIB1_PDSCH script
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
            "DMRS_Pattern",
            "N_Coarse_SSB",
            "N_Fine_Beams",
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
                    cfg.dmrs_pattern,
                    N_COARSE_SSB,
                    N_FINE_BEAMS,
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
    plt.title(f"SIB1 BLER: 8-SSB Coarse / 16-Direction Fine Comb DMRS ({channel_model}/{tdl_profile if channel_model.upper()=='TDL' else cdl_type})")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)

    print(f"Saved CSV: {out_csv}")
    print(f"Saved figure: {out_png}")
