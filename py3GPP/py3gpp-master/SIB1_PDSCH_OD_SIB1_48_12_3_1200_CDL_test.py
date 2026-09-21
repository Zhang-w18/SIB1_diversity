# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm

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
# =========================================================
# 基本配置
# =========================================================
CDL_CHANNEL = True
PRINT_DIAG = False


# ... [前面的 _normalize_dmrs_loc 到 build_SIB1pdsch_grid_only 函数保持完全不变] ...
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
    elif style == "matlab":
        return (k_all + l_all * Nsc).astype(int)
    else:
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
    elif style == "python":
        return (tuple((sib1_data_idx % Nsc).tolist()), tuple((sib1_data_idx // Nsc).tolist()))
    else:
        raise ValueError("Unknown style")


def nrSIB_param(in_bits: np.ndarray, n_id_cell: int, E: int, nLayers: int = 1, rv: int = 0, rnti: int = 0xFFFF,
                BGN: int = 2, CRC: str = "16"):
    if in_bits.ndim == 2 and in_bits.shape[1] == 1: in_bits = in_bits[:, 0]
    in_bits = in_bits.astype(int).copy()
    A = int(in_bits.size)
    tb_in = nrCRCEncode(in_bits, CRC)
    cbs_in = nrCodeBlockSegmentLDPC(tb_in, BGN)
    enc = nrLDPCEncode(cbs_in, BGN)
    cw_bits_no_scram = nrRateMatchLDPC(enc, E, rv, 'QPSK', nLayers)
    scr = _pdsch_scrambling_bits_sib1(n_id_cell, E, rnti)
    cw_bits = (cw_bits_no_scram ^ scr).astype(np.int8)
    meta = {"A": A, "E": E, "BGN": BGN, "CRC": CRC, "rv": rv, "nLayers": nLayers, "cw_bits_no_scram": cw_bits_no_scram,
            "scr": scr, "cw_bits": cw_bits}
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


def build_SIB1pdsch_grid_only(ncellid, trblk_bits, nrb_sib=48, nsym_sib=12, dmrs_loc=(0, 6, 9), k_step=2,
                              cbs_info=None):
    if cbs_info is None: cbs_info = nrDLSCHInfo(len(trblk_bits), 0.5)
    grid = np.zeros((nrb_sib * 12, nsym_sib), dtype=complex)
    dmrs_cols = _normalize_dmrs_loc(dmrs_loc, nsym_sib)
    dmrs_idx = nrSIB1DMRSIndices_param(nrb_sib, nsym_sib, dmrs_cols, k_step, "matlab")
    sib1_idx = nrSIB1Indices_param(nrb_sib, nsym_sib, dmrs_idx, "matlab")
    E = len(sib1_idx) * 2
    sib1_bits, meta = nrSIB_param(trblk_bits, ncellid, E, 1, 0, 0xFFFF, cbs_info["BGN"], cbs_info["CRC"])
    sib1_symb = nrSymbolModulate(sib1_bits, "QPSK")
    dmrs_symb = nrSIBDMRS_param_multi(ncellid, nrb_sib, nsym_sib, dmrs_cols, k_step, 0)
    nrSetResources(sib1_idx, grid, sib1_symb)
    nrSetResources(dmrs_idx, grid, dmrs_symb)
    return grid, meta, {"sib1_idx": sib1_idx, "dmrs_idx": dmrs_idx, "dmrs_cols": dmrs_cols, "E": E}


# =========================================================
# PRG-level precoding cyclic & 波束映射
# =========================================================
def iter_prg_ranges(nrb_sib, prg_size_rb):
    for rb0 in range(0, nrb_sib, prg_size_rb):
        yield rb0, min(nrb_sib, rb0 + prg_size_rb)


def build_upa_dft_vector(Mv=4, Mh=8, qv=0, qh=0):
    qv, qh = int(qv) % Mv, int(qh) % Mh
    mv, mh = np.arange(Mv, dtype=float), np.arange(Mh, dtype=float)
    vv = np.exp(1j * 2.0 * np.pi * qv * mv / Mv) / np.sqrt(Mv)
    vh = np.exp(1j * 2.0 * np.pi * qh * mh / Mh) / np.sqrt(Mh)
    v2d = np.kron(vv, vh)
    return v2d.astype(np.complex128)


def map_spatial_beam_to_dualpol_basis(v_spatial, pol_idx=0):
    v_spatial = np.asarray(v_spatial, dtype=np.complex128).ravel()
    if pol_idx == 0:
        w64 = np.concatenate([v_spatial, np.zeros_like(v_spatial)])
    elif pol_idx == 1:
        w64 = np.concatenate([np.zeros_like(v_spatial), v_spatial])
    else:
        raise ValueError("pol_idx must be 0 or 1")
    return w64 / np.linalg.norm(w64)


def build_wbf_64x2_upa(qv1=0, qh1=0, qv2=None, qh2=None):
    """
    修改点 1：移除多余的 QR 分解，直接返回物理上严格正交的矩阵 W。
    默认 qh1 改回 0，由外层脚本控制方向。
    """
    if qv2 is None: qv2 = qv1
    if qh2 is None: qh2 = qh1

    v1 = build_upa_dft_vector(Mv=4, Mh=8, qv=qv1, qh=qh1)
    v2 = build_upa_dft_vector(Mv=4, Mh=8, qv=qv2, qh=qh2)

    w1 = map_spatial_beam_to_dualpol_basis(v1, pol_idx=0)
    w2 = map_spatial_beam_to_dualpol_basis(v2, pol_idx=1)

    W = np.column_stack([w1, w2])
    return W  # 坚决去掉了 QR 分解！


def bb_precoder_traditional():
    return np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)


def bb_precoder_cyclic(prg_idx, phase_table=(0, np.pi / 2, np.pi, 3 * np.pi / 2)):
    phi = phase_table[prg_idx % len(phase_table)]
    return np.array([1.0, np.exp(1j * phi)], dtype=np.complex128) / np.sqrt(2.0)


def build_precoded_tx_grids(base_grid, nrb_sib, prg_size_rb=4, mode="traditional", Wbf=None,
                            phase_table=(0, np.pi / 2, np.pi, 3 * np.pi / 2)):
    if Wbf is None: Wbf = build_wbf_64x2_upa(qv1=0, qh1=0)
    K, N = base_grid.shape
    tx_grids = np.zeros((Wbf.shape[0], K, N), dtype=np.complex128)

    for g, (rb0, rb1) in enumerate(iter_prg_ranges(nrb_sib, prg_size_rb)):
        if mode == "traditional":
            wbb = bb_precoder_traditional()
        elif mode == "cyclic":
            wbb = bb_precoder_cyclic(g, phase_table)

        w_eff = Wbf @ wbb
        w_eff = w_eff / np.linalg.norm(w_eff)
        tx_grids[:, rb0 * 12:rb1 * 12, :] = w_eff[:, None, None] * base_grid[None, rb0 * 12:rb1 * 12, :]
    return tx_grids


# ... [中间的 NR OFDM，CDL，和接收机多天线处理保持不变，直到 _rx_round_get_llr] ...
# (为节省篇幅，这里略去了 nrOFDMModulate_multi_tx 到 nrSymbolDemodulate_qpsk_persym，与你提供的完全一致)

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


def cdl_c_impulse_response_64t4r_full(fs_hz, ds_ns=300, fc_GHz=7, seed=None, initial_time=0.0, ue_speed=3):
    if seed is not None:
        np.random.seed(seed)
    ds = ds_ns * 1e-9
    tx_ant = [1, 1, 4, 8, 2]
    rx_ant = [1, 1, 1, 2, 2]
    channel, _, _ = channel_initialize(
        cdl_type='CDLB', delay_spread=ds, Kf=[], tx_antenna=tx_ant, rx_antenna=rx_ant,
        ue_speed=ue_speed, fc_GHz=fc_GHz, fs=fs_hz, num_tti=0, tti_length=1e-3,
        angle_gap=0.0, power_gap=0.0, beamforming='no', chan_method='online'
    )
    h_paths, delays = channel_response_generate(channel, initial_time=initial_time)
    h00 = channel_time_interpolation(delays, h_paths[:, 0, 0], ts=1.0 / fs_hz, num_ext=4)
    H_mimo = np.zeros((4, 64, len(h00)), dtype=np.complex128)
    for rx in range(4):
        for tx in range(64):
            H_mimo[rx, tx, :] = channel_time_interpolation(delays, h_paths[:, rx, tx], ts=1.0 / fs_hz, num_ext=4)
    return H_mimo


def apply_cdl_c_mimo_tx(tx_waves, fs_hz, seed=None, ds_ns=300, fc_hz=7e9, speed_kmh=3, initial_time=0.0):
    H_mimo = cdl_c_impulse_response_64t4r_full(fs_hz, ds_ns, fc_hz / 1e9, seed, initial_time, speed_kmh)
    Ns, Nt = tx_waves.shape
    y = np.zeros((Ns, H_mimo.shape[0]), dtype=np.complex128)
    for rx in range(H_mimo.shape[0]):
        acc = np.zeros(Ns, dtype=np.complex128)
        for tx in range(Nt):
            acc += np.convolve(tx_waves[:, tx], H_mimo[rx, tx, :], mode='same')
        y[:, rx] = acc
    return y, H_mimo


def ofdm_demodulate_multi_rx(waveform, nrb, scs, initialNSlot, SampleRate, nsym_keep, CyclicPrefixFraction=0.5):
    x = np.asarray(waveform)
    if x.ndim == 1:
        g = nrOFDMDemodulate(
            waveform=x,
            nrb=nrb,
            scs=scs,
            initialNSlot=initialNSlot,
            SampleRate=SampleRate,
            CyclicPrefixFraction=CyclicPrefixFraction
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
            CyclicPrefixFraction=CyclicPrefixFraction
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

import concurrent.futures
def add_awgn_on_grid_multi_rx(rxGrids, txGrid_ref, snr_db, active_idx):
    rxGrids = np.asarray(rxGrids)
    out = np.zeros_like(rxGrids, dtype=complex)
    nVar_list = []
    for r in range(rxGrids.shape[0]):
        g_noisy, nVar_r = add_awgn_on_gri_per_symbol(rxGrids[r], txGrid_ref, snr_db, active_idx)
        out[r] = g_noisy
        nVar_list.append(nVar_r)
    return out, nVar_list


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
            if len(pos) == 0: continue
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


def _rx_round_get_llr(
        snr_db, ncellid, sib_grid, sib_idx, dmrs_idx, E,
        fs=61.44e6, scs_khz=30, nrb_sib=48, nsym_sib=12,
        meta=None, meta2=None, cdl_seed=None, cdl_initial_time=0.0,
        prg_size_rb=4, prg_precoding_mode="traditional",
        Wbf=None, phase_table=(0, np.pi / 2, np.pi, 3 * np.pi / 2),
        ce_mode="prg",
):
    tx_grids = build_precoded_tx_grids(sib_grid, nrb_sib, prg_size_rb, prg_precoding_mode, Wbf, phase_table)
    tx_waves = nrOFDMModulate_multi_tx(tx_grids, scs_khz, fs)
    ch_out, _ = apply_cdl_c_mimo_tx(tx_waves, fs, cdl_seed, 300, 7e9, 3, cdl_initial_time)
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
            H_r = myChannelEstimate_std(
                rxGrid=rxGrids[r],
                refGrid=refGridH,
                EST_TFDOMAIN=False,
                EST_PBCH=False,
                EST_PDSCH_FULLBAND=True,
                dmrs_l=dmrs_cols,
                start_rb=0,
                end_rb=nrb_sib,  # 动态适配 nrb_sib
                td_win_len_ratio=0.08,
                td_window_type="fixed",
            )
        else:
            H_r = myChannelEstimate_prg_local_ls(rxGrids[r], refGridH, nrb_sib, prg_size_rb, dmrs_cols)

        sib_eq_r, nVar_post_r = nrEqualizeZF_persym(
            nrExtractResources(sib_idx, rxGrids[r]),
            nrExtractResources(sib_idx, H_r),
            nVar_list[r], sib_idx, nrb_sib * 12
        )
        llr_sum_rx += nrSymbolDemodulate_qpsk_persym(sib_eq_r, nVar_post_r, sib_idx, nrb_sib * 12)

    return llr_sum_rx * (1 - 2 * meta["scr"].astype(int))


# =========================================================
# 单次传输配置 (新增 qv, qh 接口)
# =========================================================
def one_shot_sib_min_combined(
        snr_db, ncellid, fs=61.44e6, scs_khz=30, nrb_sib=48, nsym_sib=12,
        rng_seed=None, N_comb=4, payload=1200, dmrs_step=2, dmrs_loc=(0, 6, 9),
        prg_size_rb=4, prg_precoding_mode="traditional", phase_table=(0, np.pi / 2, np.pi, 3 * np.pi / 2),
        cdl_seed_base=1234, cdl_initial_time0=0.0, cdl_round_spacing_s=0.0, channel_evolution_mode="static_same",
        qv=0, qh=0,  # <--- 修改点 2：开放波束方向接口
        ce_mode = "prg"
):
    if rng_seed is not None: np.random.seed(rng_seed)
    A, rate = payload, (payload + 16) / (len(nrSIB1Indices_param(nrb_sib, nsym_sib,
                                                                 nrSIB1DMRSIndices_param(nrb_sib, nsym_sib, dmrs_loc,
                                                                                         dmrs_step, "matlab"),
                                                                 "matlab")) * 2)
    cbs_info = nrDLSCHInfo(A, rate)
    trblk_bits = np.random.randint(0, 2, A).astype(int)

    sib_grid, meta, meta2 = build_SIB1pdsch_grid_only(ncellid, trblk_bits, nrb_sib, nsym_sib, dmrs_loc, dmrs_step,
                                                      cbs_info)

    # 修改点 3：根据传入的 qv, qh 动态生成预编码矩阵
    Wbf = build_wbf_64x2_upa(qv1=qv, qh1=qh)
    llr_sum = None

    for r in range(N_comb):
        c_seed = cdl_seed_base if channel_evolution_mode == "static_same" else (
            cdl_seed_base + r if channel_evolution_mode == "independent" else cdl_seed_base)
        t_init = cdl_initial_time0 + r * cdl_round_spacing_s

        llr_r = _rx_round_get_llr(
            snr_db, ncellid, sib_grid, meta2["sib1_idx"], meta2["dmrs_idx"], meta2["E"], fs, scs_khz, nrb_sib, nsym_sib,
            meta, meta2, c_seed, t_init, prg_size_rb, prg_precoding_mode, Wbf, phase_table,ce_mode=ce_mode
        )
        llr_sum = llr_r.astype(np.float64, copy=True) if llr_sum is None else llr_sum + llr_r

    dec_bits, _ = nrLDPCDecode(nrRateRecoverLDPC(llr_sum, A, rate, 0, "QPSK", 1), cbs_info["BGN"], 25, blklen=A)
    blk, _ = nrCodeBlockDesegmentLDPC(dec_bits, cbs_info["BGN"], A + cbs_info["L"])
    _, tb_err = nrCRCDecode(blk, cbs_info["CRC"])
    return (tb_err == 0), {}


def run_single_trial_global(args):
    """
    接收一个 tuple: (snr_db, qv, qh, trial_seed, static_cfg)
    """
    snr_db, qv, qh, trial_seed, cfg = args

    ok, _ = one_shot_sib_min_combined(
        snr_db=snr_db,
        ncellid=cfg['ncellid'],
        fs=cfg['fs'],
        scs_khz=cfg['scs_khz'],
        nrb_sib=cfg['nrb_sib'],
        nsym_sib=cfg['nsym_sib'],
        rng_seed=trial_seed,
        N_comb=cfg['N_comb'],
        payload=cfg['payload'],
        dmrs_step=2,
        dmrs_loc=(0, 6, 9),
        prg_size_rb=cfg['prg_size_rb'],
        prg_precoding_mode=cfg['prg_precoding_mode'],
        phase_table=(0, np.pi / 2),
        cdl_seed_base=trial_seed,
        qv=qv, qh=qh,
        ce_mode=cfg.get('ce_mode', 'prg')
    )
    return ok
# =========================================================
# 扫 BLER（支持多波束平均）
# =========================================================
def simulate_bler_sector_smart(
        snr_db_list, n_trials_per_beam=1000, ncellid=208,
        fs=30.72e6, scs_khz=30, seed=2029, nrb_sib=48, nsym_sib=12,
        N_comb=1, payload=1200, prg_size_rb=2, prg_precoding_mode="traditional",
        ssb_beams=[(0, 0), (0, 1), (0, 2), (0, 7)],
        max_workers=4,
        target_errors=50,
        snr_early_stop_th=0.04,
        ce_mode="prg"
):
    bler_avg_list = []
    rng = np.random.RandomState(seed)

    static_cfg = {
        'ncellid': ncellid, 'fs': fs, 'scs_khz': scs_khz,
        'nrb_sib': nrb_sib, 'nsym_sib': nsym_sib, 'N_comb': N_comb,
        'payload': payload, 'prg_size_rb': prg_size_rb,
        'prg_precoding_mode': prg_precoding_mode,
        'ce_mode': ce_mode,
    }

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for snr_db in snr_db_list:
            beam_blers = []

            for (qv, qh) in ssb_beams:
                beam_err = 0
                beam_run = 0

                # 为该波束生成所有 1000 个 trial 的参数
                tasks = [
                    (snr_db, qv, qh, rng.randint(1 << 31), static_cfg)
                    for _ in range(n_trials_per_beam)
                ]

                # 一次性丢给进程池，拿到 Future 句柄列表（此时任务已开始排队执行）
                futures = [executor.submit(run_single_trial_global, t) for t in tasks]

                # 配合进度条，as_completed 会在每一个 trial 完成时立刻 yield
                with tqdm(total=n_trials_per_beam, desc=f"SNR={snr_db:>5.1f}dB Beam({qv},{qh})", ncols=85,
                          leave=False) as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        ok = future.result()
                        beam_run += 1
                        if not ok:
                            beam_err += 1

                        # 核心：进度条现在是真正的“每跑完1个动1下”，顺滑无比
                        pbar.update(1)

                        # 波束级精确早停
                        if beam_err >= target_errors:
                            pbar.set_postfix_str(f"Early Stop: {beam_err} errs")
                            # 杀手锏：取消掉队列里还没有被进程领走的多余任务，瞬间释放算力！
                            for f in futures:
                                f.cancel()
                            break

                beam_bler = beam_err / max(beam_run, 1)
                beam_blers.append(beam_bler)

            sector_avg_bler = np.mean(beam_blers)
            bler_avg_list.append(sector_avg_bler)
            print(f"[{prg_precoding_mode.upper()} PRG={prg_size_rb}] SNR={snr_db:>5.1f} dB | "
                  f"Sector Avg BLER={sector_avg_bler:.4e} (Beams: {[round(b, 4) for b in beam_blers]})")

            # SNR 级瀑布区截断早停
            if sector_avg_bler <= snr_early_stop_th:
                print(f"  -> Sector Avg BLER ({sector_avg_bler:.4f}) <= {snr_early_stop_th}. Early stopping.")
                remaining_snrs = len(snr_db_list) - len(bler_avg_list)
                bler_avg_list.extend([0.0] * remaining_snrs)
                break

    return np.array(bler_avg_list)
# =========================================================
# demo
# =========================================================
if __name__ == "__main__":
    snr_points = np.arange(-25, -10, 1.0).tolist()#[-20.0, -17.0, -16.0]

    # 假设一个 120 度扇区覆盖，选用 4 个典型的方位角
    sector_beams = [(0, 0), (0, 1),(0, 2),(0, 3),(0, 4),(0, 5), (0, 6), (0, 7)]

    # 每个波束下的试验次数（总试验次数 = n_trials_per_beam * len(sector_beams)）
    # 比如这里总共就是 500 * 4 = 2000 次 trial
    n_trials = 5000

    curves = {}
    for label, prg_mode,ce_mode in [
        ("Traditional (Fullband CE)", "traditional", "fullband"),
        # ("Traditional PRG precoding", "traditional","prg"),
        # ("Cyclic PRG precoding", "cyclic","prg"),
    ]:
        bler_curve = simulate_bler_sector_smart(
            snr_db_list=snr_points,
            n_trials_per_beam=n_trials,
            ncellid=208,
            fs=61.44e6,
            scs_khz=30,
            seed=2029,
            nrb_sib=48,
            nsym_sib=12,
            N_comb=1,
            payload=1200,
            prg_size_rb=2,
            prg_precoding_mode=prg_mode,
            ssb_beams=sector_beams,
            ce_mode =ce_mode,
        )
        curves[label] = bler_curve

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import csv

    plt.figure(figsize=(8, 5.5))
    for label, bler_curve in curves.items():
        plt.semilogy(snr_points, np.maximum(bler_curve, 1e-4), marker='o', label=label)

    plt.grid(True, which="both")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Sector Average BLER")
    plt.title("SIB1 Average BLER vs SNR (Sector Sweep)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("prg_precoding_sector_bler.png", dpi=150)

    with open("prg_precoding_sector_bler.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Config", "SNR(dB)", "Sector_Avg_BLER"])
        for label, bler_curve in curves.items():
            for snr, bler_val in zip(snr_points, bler_curve):
                writer.writerow([label, snr, bler_val])

    print("扇区平均 BLER 曲线已保存到 prg_precoding_sector_bler.png")