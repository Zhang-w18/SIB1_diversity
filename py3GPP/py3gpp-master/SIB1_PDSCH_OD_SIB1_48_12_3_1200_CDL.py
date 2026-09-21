# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm

from py3gpp import (
    nrPRBS,
    nrCRCEncode,
    nrCodeBlockSegmentLDPC,
    nrLDPCEncode,
    nrRateMatchLDPC,
    nrOFDMModulate,
    nrOFDMDemodulate,
    nrExtractResources,
    nrSetResources,
    nrRateRecoverLDPC,
    nrLDPCDecode,
    nrCodeBlockDesegmentLDPC,
    nrCRCDecode,
    nrDLSCHInfo,
    nrSymbolModulate,
)

from channel_initialize import channel_initialize
from channel_functions import channel_response_generate, channel_time_interpolation


# =========================================================
# 基本配置
# =========================================================
CDL_CHANNEL = True
PRINT_DIAG = False


# =========================================================
# 基础工具
# =========================================================
def _normalize_dmrs_loc(dmrs_loc, nsym_slot):
    if np.isscalar(dmrs_loc):
        locs = np.array([int(dmrs_loc)], dtype=int)
    else:
        locs = np.array(sorted(set(int(x) for x in dmrs_loc)), dtype=int)
    assert locs.size > 0
    assert np.all((0 <= locs) & (locs < nsym_slot))
    return locs


def nrSIB1DMRScinit(n_slot=0, l=2, ncellid=0):
    return (2**17 * (14 * n_slot + l + 1) * (2 * ncellid + 1) + 2 * ncellid) % (2**31)


def _pdsch_scrambling_bits_sib1(n_id_cell: int, E: int, rnti: int = 0xFFFF, q=0) -> np.ndarray:
    c_init = (int(rnti) << 15) + (int(q) << 14) + int(n_id_cell)
    return nrPRBS(c_init, E).astype(np.uint8)


def nrSIB1DMRSIndices_param(
    nrb_sib: int,
    nsym_slot: int,
    dmrs_loc,
    k_step: int = 2,
    style: str = "matlab",
):
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


def nrSIB1Indices_param(
    nrb_sib: int,
    nsym_slot: int,
    dmrs_idx: np.ndarray,
    style: str = "matlab",
):
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
        k_idx = sib1_data_idx % Nsc
        l_idx = sib1_data_idx // Nsc
        return (tuple(k_idx.tolist()), tuple(l_idx.tolist()))
    else:
        raise ValueError("Unknown style")


# =========================================================
# 编码 / 调制
# =========================================================
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


def nrSIBDMRS_param_multi(
    ncellid,
    nrb_sib,
    nsym_sib,
    dmrs_loc,
    k_step=2,
    n_slot=0,
):
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

    dmrs_idx = nrSIB1DMRSIndices_param(
        nrb_sib=nrb_sib,
        nsym_slot=nsym_sib,
        dmrs_loc=dmrs_cols,
        k_step=k_step,
        style="matlab"
    )

    sib1_idx = nrSIB1Indices_param(
        nrb_sib=nrb_sib,
        nsym_slot=nsym_sib,
        dmrs_idx=dmrs_idx,
        style="matlab"
    )

    E = len(sib1_idx) * 2  # QPSK

    sib1_bits, meta = nrSIB_param(
        in_bits=trblk_bits,
        n_id_cell=ncellid,
        E=E,
        nLayers=1,
        rv=0,
        rnti=0xFFFF,
        BGN=cbs_info["BGN"],
        CRC=cbs_info["CRC"],
    )

    sib1_symb = nrSymbolModulate(sib1_bits, "QPSK")
    assert len(sib1_symb) == len(sib1_idx)

    dmrs_symb = nrSIBDMRS_param_multi(
        ncellid=ncellid,
        nrb_sib=nrb_sib,
        nsym_sib=nsym_sib,
        dmrs_loc=dmrs_cols,
        k_step=k_step,
        n_slot=0,
    )
    assert len(dmrs_symb) == len(dmrs_idx)

    nrSetResources(sib1_idx, grid, sib1_symb)
    nrSetResources(dmrs_idx, grid, dmrs_symb)

    meta2 = {
        "sib1_idx": sib1_idx,
        "dmrs_idx": dmrs_idx,
        "dmrs_cols": dmrs_cols,
        "E": E,
    }
    return grid, meta, meta2


# =========================================================
# PRG-level precoding cyclic
# =========================================================
def iter_prg_ranges(nrb_sib, prg_size_rb):
    for rb0 in range(0, nrb_sib, prg_size_rb):
        rb1 = min(nrb_sib, rb0 + prg_size_rb)
        yield rb0, rb1


import numpy as np

def build_upa_dft_vector(Mv=4, Mh=8, qv=0, qh=0):
    """
    2D UPA DFT beam:
      vertical: Mv
      horizontal: Mh
    输出长度 = Mv * Mh = 32
    """
    qv = int(qv) % Mv
    qh = int(qh) % Mh

    mv = np.arange(Mv, dtype=float)
    mh = np.arange(Mh, dtype=float)

    vv = np.exp(1j * 2.0 * np.pi * qv * mv / Mv) / np.sqrt(Mv)
    vh = np.exp(1j * 2.0 * np.pi * qh * mh / Mh) / np.sqrt(Mh)

    # 和你前面 channel_initialize 里“水平优先再垂直”的展开顺序一致时，
    # 一般用 kron(vv, vh) 或 kron(vh, vv) 要核对一下索引。
    # 按你前面 bs_loc 的写法，更可能是：
    v2d = np.kron(vv, vh)   # 长度 32

    return v2d.astype(np.complex128)


def map_spatial_beam_to_dualpol_basis(v_spatial, pol_idx=0):
    """
    pol_idx = 0 -> 只映射到第一极化
    pol_idx = 1 -> 只映射到第二极化
    """
    v_spatial = np.asarray(v_spatial, dtype=np.complex128).ravel()
    assert len(v_spatial) == 32

    if pol_idx == 0:
        w64 = np.concatenate([v_spatial, np.zeros_like(v_spatial)])
    elif pol_idx == 1:
        w64 = np.concatenate([np.zeros_like(v_spatial), v_spatial])
    else:
        raise ValueError("pol_idx must be 0 or 1")

    w64 = w64 / np.linalg.norm(w64)
    return w64


def build_wbf_64x2_upa(qv1=0, qh1=1, qv2=None, qh2=None):
    """
    推荐默认：
      两列共享同一个粗空间方向，但分别映射到两个正交极化
    """
    if qv2 is None:
        qv2 = qv1
    if qh2 is None:
        qh2 = qh1

    v1 = build_upa_dft_vector(Mv=4, Mh=8, qv=qv1, qh=qh1)
    v2 = build_upa_dft_vector(Mv=4, Mh=8, qv=qv2, qh=qh2)

    w1 = map_spatial_beam_to_dualpol_basis(v1, pol_idx=0)
    w2 = map_spatial_beam_to_dualpol_basis(v2, pol_idx=1)

    W = np.column_stack([w1, w2])
    return W



def bb_precoder_traditional():
    return np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)


def bb_precoder_cyclic(prg_idx, phase_table=(0, np.pi/2, np.pi, 3*np.pi/2)):
    phi = phase_table[prg_idx % len(phase_table)]
    return np.array([1.0, np.exp(1j * phi)], dtype=np.complex128) / np.sqrt(2.0)


def build_precoded_tx_grids(
    base_grid,
    nrb_sib,
    prg_size_rb=4,
    mode="traditional",
    Wbf=None,
    phase_table=(0, np.pi/2, np.pi, 3*np.pi/2),
):
    if Wbf is None:
        Wbf = build_wbf_64x2_upa(qv1=0, qh1=0)

    K, N = base_grid.shape
    Nt = Wbf.shape[0]
    tx_grids = np.zeros((Nt, K, N), dtype=np.complex128)

    for g, (rb0, rb1) in enumerate(iter_prg_ranges(nrb_sib, prg_size_rb)):
        if mode == "traditional":
            wbb = bb_precoder_traditional()
        elif mode == "cyclic":
            wbb = bb_precoder_cyclic(g, phase_table)
        else:
            raise ValueError("mode must be 'traditional' or 'cyclic'")

        w_eff = Wbf @ wbb
        w_eff = w_eff / np.linalg.norm(w_eff)

        sc0 = rb0 * 12
        sc1 = rb1 * 12
        tx_grids[:, sc0:sc1, :] = w_eff[:, None, None] * base_grid[None, sc0:sc1, :]

    return tx_grids


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


# =========================================================
# CDL: 32位置×双极化 => 64Tx, 4Rx
# =========================================================
def cdl_c_impulse_response_64t4r_full(fs_hz, ds_ns=300, fc_GHz=7, seed=None, initial_time=0.0, ue_speed=3):
    if seed is not None:
        np.random.seed(seed)

    ds = ds_ns * 1e-9

    tx_ant = [1, 1, 4, 8, 2]   # 32 positions × dual-pol => 64 physical Tx
    rx_ant = [1, 1, 1, 2, 2]   # 4Rx

    channel, _, _ = channel_initialize(
        cdl_type='CDLB',
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
        chan_method='online'
    )

    h_paths, delays = channel_response_generate(channel, initial_time=initial_time)

    h00 = channel_time_interpolation(delays, h_paths[:, 0, 0], ts=1.0 / fs_hz, num_ext=4)
    num_taps = len(h00)

    H_mimo = np.zeros((4, 64, num_taps), dtype=np.complex128)
    for rx in range(4):
        for tx in range(64):
            H_mimo[rx, tx, :] = channel_time_interpolation(
                delays, h_paths[:, rx, tx], ts=1.0 / fs_hz, num_ext=4
            )
    return H_mimo


def apply_cdl_c_mimo_tx(tx_waves, fs_hz, seed=None, ds_ns=300, fc_hz=7e9, speed_kmh=3, initial_time=0.0):
    H_mimo = cdl_c_impulse_response_64t4r_full(
        fs_hz=fs_hz,
        ds_ns=ds_ns,
        fc_GHz=fc_hz / 1e9,
        seed=seed,
        initial_time=initial_time,
        ue_speed=speed_kmh
    )

    Ns, Nt = tx_waves.shape
    nRx = H_mimo.shape[0]
    y = np.zeros((Ns, nRx), dtype=np.complex128)

    for rx in range(nRx):
        acc = np.zeros(Ns, dtype=np.complex128)
        for tx in range(Nt):
            acc += np.convolve(tx_waves[:, tx], H_mimo[rx, tx, :], mode='same')
        y[:, rx] = acc

    return y, H_mimo


# =========================================================
# 接收端：多Rx OFDM / 加噪 / CE / EQ / LLR
# =========================================================
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
        if np.any(in_col):
            p_sig_l = np.mean(np.abs(flat[active_idx[in_col]]) ** 2)
        else:
            p_sig_l = p_sig_global
        nvar_cols[l] = p_sig_l / snr_lin

    noise = (np.random.randn(K, N) + 1j * np.random.randn(K, N)) * np.sqrt(nvar_cols[None, :] / 2.0)
    return rxGrid_clean + noise, nvar_cols


def add_awgn_on_grid_multi_rx(rxGrids, txGrid_ref, snr_db, active_idx):
    rxGrids = np.asarray(rxGrids)
    nRx = rxGrids.shape[0]

    out = np.zeros_like(rxGrids, dtype=complex)
    nVar_list = []
    for r in range(nRx):
        g_noisy, nVar_r = add_awgn_on_gri_per_symbol(rxGrids[r], txGrid_ref, snr_db, active_idx)
        out[r] = g_noisy
        nVar_list.append(nVar_r)
    return out, nVar_list


def myChannelEstimate_prg_local_ls(
    rxGrid,
    refGrid,
    nrb_sib,
    prg_size_rb,
    dmrs_l,
):
    K, N = rxGrid.shape
    H = np.zeros_like(rxGrid, dtype=np.complex128)

    dmrs_cols = _normalize_dmrs_loc(dmrs_l, N)
    has_dmrs_col = np.zeros(N, dtype=bool)

    # 频域：每个 PRG 独立 LS + 线性插值
    for col in dmrs_cols:
        for rb0, rb1 in iter_prg_ranges(nrb_sib, prg_size_rb):
            sc0 = rb0 * 12
            sc1 = rb1 * 12

            ref_blk = refGrid[sc0:sc1, col]
            pos = np.where(np.abs(ref_blk) > 1e-12)[0]
            if len(pos) == 0:
                continue

            y = rxGrid[sc0:sc1, col][pos]
            x = ref_blk[pos]
            hls = y / x

            x_all = np.arange(sc1 - sc0)
            if len(pos) == 1:
                H_blk = np.full(sc1 - sc0, hls[0], dtype=np.complex128)
            else:
                H_blk = (
                    np.interp(x_all, pos, np.real(hls)) +
                    1j * np.interp(x_all, pos, np.imag(hls))
                )

            H[sc0:sc1, col] = H_blk
            has_dmrs_col[col] = True

    # 时域：不同 DMRS symbol 之间做线性插值
    cols_with = np.where(has_dmrs_col)[0]
    cols_wo = np.where(~has_dmrs_col)[0]

    if cols_with.size >= 2:
        for k in range(K):
            y = H[k, cols_with]
            H[k, cols_wo] = (
                np.interp(cols_wo, cols_with, np.real(y)) +
                1j * np.interp(cols_wo, cols_with, np.imag(y))
            )
    elif cols_with.size == 1:
        H[:, cols_wo] = H[:, cols_with[0:1]]

    return H


def nrEqualizeZF_persym(rxSym, hest, nVar_cols, re_idx, K):
    sym_ids = (re_idx // K).astype(np.int64)
    nVar_re = nVar_cols[sym_ids]

    h_abs2 = np.abs(hest) ** 2
    h_abs2 = np.where(h_abs2 < 1e-12, 1e-12, h_abs2)

    rxEq = rxSym / np.where(np.abs(hest) < 1e-12, 1e-12, hest)
    nVar_post = nVar_re / h_abs2
    return rxEq, nVar_post


def nrSymbolDemodulate_qpsk_persym(input_sym, nVar_input, re_idx, K):
    input_sym = np.asarray(input_sym, dtype=np.complex128).ravel()
    re_idx = np.asarray(re_idx, dtype=np.int64)
    nVar_arr = np.asarray(nVar_input, dtype=float)

    if nVar_arr.size == len(re_idx):
        N0_re = nVar_arr
    else:
        sym_ids = re_idx // K
        N0_re = nVar_arr[sym_ids]

    xr = np.real(input_sym)
    xi = np.imag(input_sym)

    llr0 = xr / np.maximum(N0_re, 1e-12)
    llr1 = xi / np.maximum(N0_re, 1e-12)

    llr = np.empty(2 * len(input_sym), dtype=np.float64)
    llr[0::2] = llr0
    llr[1::2] = llr1
    return llr


# =========================================================
# 单轮接收
# =========================================================
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
    cdl_seed=None,
    cdl_initial_time=0.0,
    prg_size_rb=4,
    prg_precoding_mode="traditional",   # "traditional" / "cyclic"
    Wbf=None,
    phase_table=(0, np.pi/2, np.pi, 3*np.pi/2),
):
    # 1) PRG-level precoded Tx grids
    tx_grids = build_precoded_tx_grids(
        base_grid=sib_grid,
        nrb_sib=nrb_sib,
        prg_size_rb=prg_size_rb,
        mode=prg_precoding_mode,
        Wbf=Wbf,
        phase_table=phase_table,
    )

    # 2) 多 Tx OFDM
    tx_waves = nrOFDMModulate_multi_tx(
        tx_grids,
        scs=scs_khz,
        SampleRate=fs
    )

    # 3) 过 CDL
    if CDL_CHANNEL:
        ch_out, H_mimo = apply_cdl_c_mimo_tx(
            tx_waves,
            fs_hz=fs,
            seed=cdl_seed,
            ds_ns=300,
            fc_hz=7e9,
            speed_kmh=3,
            initial_time=cdl_initial_time
        )
    else:
        raise NotImplementedError("This minimal version keeps only CDL path.")

    # 4) 多 Rx OFDM demod
    rxGrids_clean = ofdm_demodulate_multi_rx(
        waveform=ch_out,
        nrb=nrb_sib,
        scs=scs_khz,
        initialNSlot=0,
        SampleRate=fs,
        nsym_keep=nsym_sib,
        CyclicPrefixFraction=0.5
    )
    nRx = rxGrids_clean.shape[0]

    # 5) 加噪
    active_idx = np.sort(np.concatenate([sib_idx, dmrs_idx]))
    rxGrids, nVar_list = add_awgn_on_grid_multi_rx(
        rxGrids_clean,
        sib_grid,
        snr_db,
        active_idx
    )

    # 6) 构造 DMRS ref grid
    dmrs_cols = meta2["dmrs_cols"]
    dmrs_symb = nrSIBDMRS_param_multi(
        ncellid=ncellid,
        nrb_sib=nrb_sib,
        nsym_sib=nsym_sib,
        dmrs_loc=dmrs_cols,
        k_step=2,
        n_slot=0,
    )
    refGridH = np.zeros_like(rxGrids[0], dtype=complex)
    nrSetResources(dmrs_idx, refGridH, dmrs_symb)

    # 7) 4Rx 独立 CE / EQ / LLR，再合并
    llr_sum_rx = np.zeros(E, dtype=np.float64)

    for r in range(nRx):
        rxGrid_r = rxGrids[r]
        nVar_r = nVar_list[r]

        H_r = myChannelEstimate_prg_local_ls(
            rxGrid=rxGrid_r,
            refGrid=refGridH,
            nrb_sib=nrb_sib,
            prg_size_rb=prg_size_rb,
            dmrs_l=dmrs_cols,
        )

        y_sib_r = nrExtractResources(sib_idx, rxGrid_r)
        h_sib_r = nrExtractResources(sib_idx, H_r)

        sib_eq_r, nVar_post_r = nrEqualizeZF_persym(
            y_sib_r, h_sib_r, nVar_r, sib_idx, K=nrb_sib * 12
        )

        llr_r = nrSymbolDemodulate_qpsk_persym(
            sib_eq_r, nVar_post_r, sib_idx, K=nrb_sib * 12
        )
        llr_sum_rx += llr_r

    # 8) descramble
    scr_rx = meta["scr"].astype(int)
    llr_descrambled = llr_sum_rx * (1 - 2 * scr_rx)

    assert len(llr_descrambled) == E
    return llr_descrambled


# =========================================================
# 多轮 combining
# =========================================================
def one_shot_sib_min_combined(
    snr_db,
    ncellid,
    fs=61.44e6,
    scs_khz=30,
    nrb_sib=48,
    nsym_sib=12,
    rng_seed=None,
    N_comb=4,
    payload=1200,

    # DMRS / PDSCH
    dmrs_step=2,
    dmrs_loc=(0, 6, 9),

    # PRG-level cyclic
    prg_size_rb=4,
    prg_precoding_mode="traditional",   # "traditional" / "cyclic"
    phase_table=(0, np.pi/2, np.pi, 3*np.pi/2),

    # CDL
    cdl_seed_base=1234,
    cdl_initial_time0=0.0,
    cdl_round_spacing_s=0.0,
    channel_evolution_mode="static_same",   # static_same / time_evolving / independent
):
    if rng_seed is not None:
        np.random.seed(rng_seed)

    A = payload
    rv = 0
    modulation = "QPSK"
    nlayers = 1

    dmrs_idx_tmp = nrSIB1DMRSIndices_param(
        nrb_sib=nrb_sib,
        nsym_slot=nsym_sib,
        dmrs_loc=dmrs_loc,
        k_step=dmrs_step,
        style="matlab",
    )
    sib_idx_tmp = nrSIB1Indices_param(
        nrb_sib=nrb_sib,
        nsym_slot=nsym_sib,
        dmrs_idx=dmrs_idx_tmp,
        style="matlab",
    )

    E_tmp = len(sib_idx_tmp) * 2
    rate = (A + 16) / E_tmp
    cbs_info = nrDLSCHInfo(A, rate)

    trblk_bits = np.random.randint(0, 2, A).astype(int)

    sib_grid, meta, meta2 = build_SIB1pdsch_grid_only(
        ncellid=ncellid,
        trblk_bits=trblk_bits,
        nrb_sib=nrb_sib,
        nsym_sib=nsym_sib,
        dmrs_loc=dmrs_loc,
        k_step=dmrs_step,
        cbs_info=cbs_info
    )

    sib_idx = meta2["sib1_idx"]
    dmrs_idx = meta2["dmrs_idx"]
    E = meta2["E"]

    llr_sum = None
    Wbf = build_wbf_64x2_upa()
    for r in range(N_comb):
        if channel_evolution_mode == "static_same":
            cdl_seed_r = cdl_seed_base
            cdl_initial_time_r = cdl_initial_time0
        elif channel_evolution_mode == "time_evolving":
            cdl_seed_r = cdl_seed_base
            cdl_initial_time_r = cdl_initial_time0 + r * cdl_round_spacing_s
        elif channel_evolution_mode == "independent":
            cdl_seed_r = cdl_seed_base + r if cdl_seed_base is not None else None
            cdl_initial_time_r = cdl_initial_time0 + r * cdl_round_spacing_s
        else:
            raise ValueError("unknown channel_evolution_mode")

        llr_r = _rx_round_get_llr(
            snr_db=snr_db,
            ncellid=ncellid,
            sib_grid=sib_grid,
            sib_idx=sib_idx,
            dmrs_idx=dmrs_idx,
            E=E,
            fs=fs,
            scs_khz=scs_khz,
            nrb_sib=nrb_sib,
            nsym_sib=nsym_sib,
            meta=meta,
            meta2=meta2,
            cdl_seed=cdl_seed_r,
            cdl_initial_time=cdl_initial_time_r,
            prg_size_rb=prg_size_rb,
            prg_precoding_mode=prg_precoding_mode,
            Wbf=Wbf,
            phase_table=phase_table,
        )

        if llr_sum is None:
            llr_sum = llr_r.astype(np.float64, copy=True)
        else:
            llr_sum += llr_r

    raterec = nrRateRecoverLDPC(llr_sum, A, rate, rv, modulation, nlayers)
    dec_bits, iters = nrLDPCDecode(raterec, cbs_info["BGN"], 25, blklen=A)
    blk, blk_err = nrCodeBlockDesegmentLDPC(dec_bits, cbs_info["BGN"], A + cbs_info["L"])
    out, tb_err = nrCRCDecode(blk, cbs_info["CRC"])

    crc_ok = (tb_err == 0)

    info = {
        "A": A,
        "E": E,
        "rate": rate,
        "N_comb": N_comb,
        "prg_size_rb": prg_size_rb,
        "prg_precoding_mode": prg_precoding_mode,
        "channel_evolution_mode": channel_evolution_mode,
    }
    return crc_ok, info


# =========================================================
# 扫 BLER
# =========================================================
def simulate_bler_config(
    snr_db_list,
    n_trials=3000,
    ncellid=208,
    fs=61.44e6,
    scs_khz=30,
    seed=2029,
    nrb_sib=48,
    nsym_sib=12,
    N_comb=4,
    payload=1200,
    prg_size_rb=4,
    prg_precoding_mode="traditional",
):
    bler = []
    rng = np.random.RandomState(seed)

    for snr_db in snr_db_list:
        n_err = 0
        n_run = 0

        with tqdm(total=n_trials, desc=f"SNR={snr_db:>5.1f} dB", ncols=72, leave=False) as pbar:
            for _ in range(n_trials):
                ok, info = one_shot_sib_min_combined(
                    snr_db=snr_db,
                    ncellid=ncellid,
                    fs=fs,
                    scs_khz=scs_khz,
                    nrb_sib=nrb_sib,
                    nsym_sib=nsym_sib,
                    rng_seed=rng.randint(1 << 31),
                    N_comb=N_comb,
                    payload=payload,
                    dmrs_step=2,
                    dmrs_loc=(0, 6, 9),
                    prg_size_rb=prg_size_rb,
                    prg_precoding_mode=prg_precoding_mode,
                    phase_table=(0, np.pi/2),
                    cdl_seed_base=rng.randint(1 << 31), #1234
                    cdl_initial_time0=0.0,
                    cdl_round_spacing_s=0.0,
                    channel_evolution_mode="static_same",
                )

                n_run += 1
                if not ok:
                    n_err += 1

                pbar.update(1)

                # 简单早停
                if n_err >= 50:
                    break

        bler_val = n_err / max(n_run, 1)
        bler.append(bler_val)
        print(f"[CFG nrb={nrb_sib}, nsym={nsym_sib}, N={N_comb}, PRG={prg_size_rb}, mode={prg_precoding_mode}] "
              f"SNR={snr_db:>5.1f} dB  BLER={bler_val:.4f}")

    return np.array(bler)




# =========================================================
# demo
# =========================================================
if __name__ == "__main__":
    snr_points = [-20] #,-17,-16

    curves = {}
    for label, prg_mode in [
        ("Cyclic PRG precoding", "cyclic"),
        ("Traditional PRG precoding", "traditional"),
    ]:
        bler_curve = simulate_bler_config(
            snr_db_list=snr_points,
            n_trials=3000,
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
    plt.ylabel("BLER")
    plt.title("SIB1/Paging-like PDSCH BLER vs SNR (PRG-level Precoding)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("prg_precoding_bler.png", dpi=150)

    with open("prg_precoding_bler.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Config", "SNR(dB)", "BLER"])
        for label, bler_curve in curves.items():
            for snr, bler_val in zip(snr_points, bler_curve):
                writer.writerow([label, snr, bler_val])

    print("BLER 曲线已保存到 prg_precoding_bler.png")
    print("数值结果已保存到 prg_precoding_bler.csv")