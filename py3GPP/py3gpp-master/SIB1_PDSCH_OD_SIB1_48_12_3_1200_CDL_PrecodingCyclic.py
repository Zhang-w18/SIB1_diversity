# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 14:23:13 2025

 File name: STD_SIB1_PDSCH
 Author: m00829866	Version: 0.1	Date: 2025-11-11
 Description:
     1. standard SIB1 PDSCH Link Level Simulation

 修改记录：
 date name line xxx
"""
import numpy as np
from py3gpp import *

from nrPBCHIndicesVariable import nrPBCHIndices_param,nrPBCHDMRSIndices_param,nrPBCHDMRS_param,nrBCH_param,nrSymbolDemodulate_param,nrSymbolDemodulate_param_persym
from channel_initialize import channel_initialize,channel_settings_initialize,channel_cdl_initialize,rotate_angles_max_direction,scaling_angles,calculate_delay_spread,calculate_angular_spread
from channel_functions import channel_fading_signal, channel_response_generate, calculate_antenna_pattern, get_dft_codebook, channel_virtualization, channel_time_interpolation, channel_frequency_response
from nrPBCHIndicesVariable import nrPBCHIndices_param,nrPBCHDMRSIndices_param,nrPBCHDMRS_param,nrBCH_param,nrSymbolDemodulate_param,nrSymbolDemodulate_param_persym,nrEqualizeMMSE_persym
from tqdm import tqdm
from GenTDLChannel import ChannelInfo
# from myChannelEstimation_0311 import DMRSFilterGenerate_v2_explicit,myChannelEstimate,myChannelEstimate_std,myChannelEstimate_bundle_MMSE_robust
from myChannelEstimation4 import myChannelEstimate_std
from tqdm import tqdm
NO_CHANNEL = False          # 先用纯 AWGN 验证链路 ——> True
CDL_CHANNEL = True
USE_KNOWN_NVAR = False      # 用 SNR 推噪声方差 ——> True
PRINT_DIAG = False #True          # 打印关键诊断信息
POWER_NORM_PER_RE = False
USE_SCL = True
ONLY_TDLC = False
USE_RX_SNR = False
def nrSIB1DMRScinit(n_slot=0,l=2,ncellid=0):
    return (2**17 * (14*n_slot + l + 1) * (2*ncellid + 1) + 2 * ncellid) % (2**31)
def _pdsch_scrambling_bits_sib1(n_id_cell: int, E: int, rnti: int = 0xFFFF,q=0) -> np.ndarray:
    # TS 38.211 7.3.1.1: c_init = RNTI * 2^15 + nID；SIB1 用 SI-RNTI，nID = N_ID_cell
    c_init = (int(rnti) << 15) + (int(q) << 14) + int(n_id_cell)
    return nrPRBS(c_init, E).astype(np.uint8)

import numpy as np

def _normalize_dmrs_loc(dmrs_loc, nsym_slot):
    """
    支持:
      - int: 2
      - list/tuple/ndarray: [0, 10, 11]
    返回:
      np.ndarray, 已排序去重, dtype=int
    """
    if np.isscalar(dmrs_loc):
        locs = np.array([int(dmrs_loc)], dtype=int)
    else:
        locs = np.array(sorted(set(int(x) for x in dmrs_loc)), dtype=int)

    assert locs.size > 0, "dmrs_loc must not be empty"
    assert np.all((0 <= locs) & (locs < nsym_slot)), "dmrs_loc out of range"
    return locs

def nrSIB1DMRSIndices_param(
        nrb_sib: int,
        nsym_slot: int,
        dmrs_loc,
        k_step: int = 2,
        style: str = "matlab",
):
    """
    生成 SIB1 的 PDSCH DM-RS 索引。
    支持单个或多个 DMRS 符号，例如:
      dmrs_loc = 2
      dmrs_loc = [0, 10, 11]

    参数:
        nrb_sib   : SIB1/PDSCH 占用 PRB 数
        nsym_slot : 总符号数
        dmrs_loc  : int 或 list[int]
        k_step    : comb 间隔
        style     : "matlab" -> flat index; "python" -> (k_idx, l_idx)
    """
    assert nrb_sib >= 1, "nrb_sib must be >= 1"
    assert nsym_slot >= 1, "nsym_slot must be >= 1"
    assert k_step >= 1, "k_step must be >= 1"

    Nsc = int(nrb_sib) * 12
    ks = np.arange(0, Nsc, k_step, dtype=int)
    locs = _normalize_dmrs_loc(dmrs_loc, nsym_slot)

    if ks.size == 0:
        if style == "python":
            return (tuple(), tuple())
        return np.array([], dtype=int)

    # 按 Fortran/列优先展平顺序组织：
    # 每个 symbol 内 k 从小到大，symbol 按 l 从小到大拼接
    k_all = np.tile(ks, len(locs))
    l_all = np.repeat(locs, len(ks))

    if style == "python":
        return (tuple(k_all.tolist()), tuple(l_all.tolist()))

    if style == "matlab":
        flat = (k_all + l_all * Nsc).astype(int)
        return flat

    raise ValueError("Unknown style")
def nrSIB1Indices_param(
        nrb_sib: int,
        nsym_slot: int,
        dmrs_idx: np.ndarray,
        style: str = "matlab",
):
    """
    生成 SIB1 的 PDSCH 数据 RE 索引（展平形式），去除 DMRS 对应 RE。
    支持单/多 DMRS 符号，因为 dmrs_idx 本身可以跨多个 symbol。
    """
    assert nrb_sib >= 1 and nsym_slot >= 1
    Nsc = int(nrb_sib) * 12
    total_re = Nsc * nsym_slot

    all_idx = np.arange(total_re, dtype=int)

    if dmrs_idx is not None and len(dmrs_idx) > 0:
        dmrs_idx = np.asarray(dmrs_idx, dtype=int)
        dmrs_idx = dmrs_idx[(dmrs_idx >= 0) & (dmrs_idx < total_re)]
        dmrs_idx = np.unique(dmrs_idx)   # 去重 + 排序
        sib1_data_idx = np.setdiff1d(all_idx, dmrs_idx, assume_unique=True)
    else:
        sib1_data_idx = all_idx

    if style == "matlab":
        return sib1_data_idx
    elif style == "python":
        k_idx = sib1_data_idx % Nsc
        l_idx = sib1_data_idx // Nsc
        return (tuple(k_idx.tolist()), tuple(l_idx.tolist()))
    else:
        raise ValueError("Unknown style")

def nrSIB_param(
    in_bits: np.ndarray,     # SIB1 净载荷 A，比特向量 shape=(A,) 或 (A,1)
    nrb_sib: int,
    nsym_slot: int,          # normal CP 通常 14
    dmrs_idx: np.ndarray,    # 展平DMRS索引 (k + l*Nsc)
    sib1_idx: np.ndarray,    # 展平数据RE索引（你前面函数的输出）
    n_id_cell: int,
    nLayers: int = 1,
    rv: int = 0,
    rnti: int = 0xFFFF,      # SIB1 固定 SI-RNTI
    E: int = 128,
    BGN: int = 2,
    CRC = "16"
):
    """
    返回：
      cw_bits : 速率匹配 + 扰码后的码字比特，长度 E
      E       : 发比特数
      meta    : 一些中间参数（BGN、Zc、TBS等）
    """
    # --- 0) 规范化输入 ---
    if in_bits.ndim == 2 and in_bits.shape[1] == 1:
        in_bits = in_bits[:,0]
    in_bits = in_bits.astype(int).copy()
    A = int(in_bits.size)

    # --- 1) 计算发比特数 E ---
    Nsc = int(nrb_sib) * 12
    assert (dmrs_idx is not None) and (sib1_idx is not None)
    assert E > 0, "E must be positive"

    # # --- 2) 从 cbs_info 读取（如有），覆盖 BGN/CRC（可选） ---
    # if cbs_info is not None:
    #     if 'BGN' in cbs_info: BGN = int(cbs_info['BGN'])
    #     if 'CRC' in cbs_info and CRC is None:
    #         CRC = cbs_info['CRC']  # 不建议覆盖 24A，除非你刻意用 16 做实验

    # --- 3) TB CRC 16 ---
    tb_in = nrCRCEncode(in_bits, CRC)            # shape=(A+16,1)

    # --- 4) CB 分段 + (必要时) 24B CB CRC ---
    cbs_in = nrCodeBlockSegmentLDPC(tb_in, BGN)    # shape=(K, C)
    ## TODO: check spec for  Block Segment
    # --- 5) LDPC 编码 ---
    enc = nrLDPCEncode(cbs_in, BGN)                # shape=(N', C)  (已打孔2Zc)
    ## TODO: check LDPC in sionna
    # --- 6) 速率匹配到 E ---
    cw_bits_no_scram = nrRateMatchLDPC(enc, E, rv, 'QPSK', nLayers)  # shape=(E,)

    # --- 7) SI-RNTI 扰码（数据扰码） ---
    scr = _pdsch_scrambling_bits_sib1(n_id_cell, E, rnti)
    cw_bits = (cw_bits_no_scram ^ scr).astype(np.int8)

    meta = {
        'A': A,
        'E': E,
        'BGN': BGN,
        'Qm': 'QPSK',
        'nLayers': nLayers,
        'rv': rv,
        'Nsc': Nsc,
        'Nre_data': len(sib1_idx),
        'code_rate_est': (A+16)/E if E>0 else 0.0,
        'tb_in':tb_in,
        'cbs_in':cbs_in,
        'enc':enc,
        'cw_bits_no_scram':cw_bits_no_scram,
        'scr':scr,
        'cw_bits':cw_bits
    }
    return cw_bits, meta

def nrSIBDMRS_param(ncellid,len,n_slot=0,l_dmrs=2):
    # 每PRB DMRS个数：Type1→6, Type2→4（与库保持一致）
    # n_per_prb = 2
    M = len            # 本符号上需要的DMRS个数
    cinit = nrSIB1DMRScinit(n_slot,l_dmrs,ncellid)
    c = nrPRBS(cinit, 2*M)
    return nrSymbolModulate(c, "QPSK")  # 复数长度 = M

def add_awgn_on_gri_per_symbol(rxGrid_clean, ssb_grid,  snr_db_re, active_idx):
    """
    逐 OFDM 符号统计信号功率并加噪：
      - snr_db_re: 以“每 RE 的发射口径 Es/N0 (dB)”定义
      - active_idx: 列优先的一维索引（如 PBCH∪DMRS），用于该符号内 p_sig 估计
    返回:
      rxGrid_noisy: 加噪后的频域网格（与 rxGrid_clean 同形状）
      nvar_cols:    每个符号列对应的复符号噪声功率 (长度 = Ns)
    """
    K, N = rxGrid_clean.shape
    snr_lin = 10**(snr_db_re/10.0)
    # 以“每个 RE 的平均信号功率”为口径配噪
    if USE_RX_SNR:
        flat = rxGrid_clean.ravel(order="F")
    else:
        flat = ssb_grid.ravel(order="F")
    # 预计算一个“全局”发射功率（当某列无 active RE 时做回退）
    p_sig_global = np.mean(np.abs(flat[active_idx]) ** 2)
    nvar_cols = np.zeros(N, dtype=float)
    for l in range(N):
        # 该符号列内的 active RE
        in_col = (active_idx >= l*K) & (active_idx < (l+1)*K)
        if np.any(in_col):
            p_sig_l = np.mean(np.abs(flat[active_idx[in_col]])**2)
        else:
            p_sig_l = p_sig_global
        nvar_cols[l] = p_sig_l / snr_lin
    noise = (np.random.randn(K, N) + 1j * np.random.randn(K, N)) * np.sqrt(nvar_cols[None, :] / 2.0)
    # p_sig = np.mean(np.abs(flat[active_idx]) ** 2)
    # nvar_complex = np.mean(nvar_cols)                        # 复符号噪声功率
    # noise = (np.random.randn(*ssb_grid.shape) + 1j*np.random.randn(*ssb_grid.shape)) \
    #         * np.sqrt(nvar_complex/2.0)
    return rxGrid_clean + noise, nvar_cols
# ---------------------------
# 发端：只构造 PBCH + PBCH DMRS
# ---------------------------
def build_SIB1pdsch_grid_only(
        ncellid,
        trblk_bits,
        nrb_sib=15,
        nsym_sib=12,
        dmrs_loc=0,
        k_step=2,
        cbs_info=None
):
    if cbs_info is None:
        cbs_info = {
            'CRC': '16', 'L': 16, 'BGN': 2, 'C': 1,
            'Lcb': 0, 'F': 8, 'Zc': 128, 'K': 1280, 'N': 6400
        }

    grid = np.zeros((nrb_sib * 12, nsym_sib), dtype=complex)

    dmrs_cols = _normalize_dmrs_loc(dmrs_loc, nsym_sib)

    # 统一索引
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

    # 编码 -> 速配(E) -> QPSK
    E = len(sib1_idx) * 2  # QPSK

    sib1_bits, meta = nrSIB_param(
        in_bits=trblk_bits,
        nrb_sib=nrb_sib,
        nsym_slot=nsym_sib,
        dmrs_idx=dmrs_idx,
        sib1_idx=sib1_idx,
        n_id_cell=ncellid,
        nLayers=1,
        rv=0,
        rnti=0xFFFF,   # SI-RNTI
        E=E,
        BGN=cbs_info['BGN'],
        CRC=cbs_info['CRC']
    )

    sib1_symb = nrSymbolModulate(sib1_bits, 'QPSK')
    assert len(sib1_symb) == len(sib1_idx)

    # DMRS
    # 最小改法：沿用你原接口
    dmrs_symb = nrSIBDMRS_param(ncellid, len(dmrs_idx))
    assert len(dmrs_symb) == len(dmrs_idx)

    nrSetResources(sib1_idx, grid, sib1_symb)
    nrSetResources(dmrs_idx, grid, dmrs_symb)

    meta2 = dict(
        sib1_idx=sib1_idx,
        dmrs_idx=dmrs_idx,
        dmrs_cols=dmrs_cols,
        E=E
    )
    return grid, meta, meta2, sib1_symb

def tdl_c_impulse_response(fs_hz, ds=300e-9, fc_hz=3.5e9, speed_kmh=3.0, nfft=4096, seed=None, normalize=True):
    """
    用 ChannelInfo 生成 3GPP TR 38.901 TDL-C 的时域冲激响应（静态一次抽样）。
    参数:
      fs_hz     : 采样率 (Hz)
      ds        : Delay Spread (秒)，如 300e-9
      fc_hz     : 载频 (Hz)，默认 3.5e9
      speed_kmh : 速度 (km/h)，这里只做参数占位（静态抽样）
      nfft      : NFFT（ChannelInfo 需要）
      seed      : 随机种子
      normalize : 是否归一化到单位功率
    返回:
      h : 复数一维 ndarray，长度 = DS_tap
    """
    if seed is not None:
        np.random.seed(seed)

    # ChannelInfo 的 DS 单位是 ns，这里转换
    DS_ns = ds

    ch = ChannelInfo(
        nTx=1, nRx=1,
        Speed=speed_kmh,      # km/h
        Fc=fc_hz,             # Hz
        Fs=fs_hz,             # Hz
        ChanType='TDLC',
        DS=DS_ns,             # ns
        NFFT=nfft,
        TO=0, FO=0
    )

    # 生成一次静态 Rayleigh 多径（不随时间变化）
    ch.gen_Rayleigh(num_samples=1)  # time_channel shape = [nRx, nTx, DS_tap, 1]

    # 取 1x1 MIMO 的时域冲激响应
    h = ch.time_channel[0, 0, :, 0].astype(np.complex128, copy=True)

    # if normalize:
    #     p = np.sum(np.abs(h)**2)
    #     if p > 0:
    #         h /= np.sqrt(p)

    return h

# def apply_tdl_c(x, fs_hz, seed=None):
#     h = tdl_c_impulse_response(fs_hz, seed)
#     # 用“same”保持长度和对齐，不引入整体时移（避免起点错位）
#     y = np.convolve(x, h, mode='same')
#     #print('h=',h)
#     return y, h
def apply_tdl_c(x, fs_hz, seed=None, ds=300, fc_hz=7e9, speed_kmh=3, nfft=4096, normalize=True):
    """
    与你原先接口保持一致：返回 (y, h)，其中 y = x * h （same 模式卷积）
    """
    h = tdl_c_impulse_response(
        fs_hz=fs_hz, ds=ds, fc_hz=fc_hz,
        speed_kmh=speed_kmh, nfft=nfft,
        seed=seed, normalize=normalize
    )
    y = np.convolve(x, h, mode='same')
    return y, h

# --------------------------
# CDLC Channel
# --------------------------
def cdl_c_impulse_response(fs_hz, ds=300, fc_GHz=3.5, seed=None):
    """
    生成 3GPP TR 38.901 CDL-C 的离散基带冲激响应（SISO，静态，单位功率）。
    参数:
        fs_hz : 采样率 (Hz)
        ds    : Delay Spread (秒)，默认 300 ns
        fc_GHz: 载频 (GHz)，默认 3.5 GHz
        seed  : 随机种子，用于相位和子径打乱
    返回:
        h : 复数组 (长度 = ceil(max_delay*fs) + 2*num_ext + 1)
    """
    if seed is not None:
        np.random.seed(seed)

    # SISO + 无波束赋形 + 在线（即时生成）+ 静态（无多普勒）
    tx_ant = [1, 1, 1, 1, 1]  # [Mg, Ng, M, N, P]
    rx_ant = [1, 1, 1, 1, 1]
    ue_speed = 0.0
    num_tti = 0          # online 模式，不预生成
    tti_len = 1e-3       # 任意给一个，用不到
    angle_gap = 0.0
    power_gap = 0.0

    channel, _, _ = channel_initialize(
        cdl_type='CDLC',
        delay_spread=ds,
        Kf=[],                       # NLOS，无 Kf
        tx_antenna=tx_ant,
        rx_antenna=rx_ant,
        ue_speed=ue_speed,
        fc_GHz=fc_GHz,
        fs=fs_hz,
        num_tti=num_tti,
        tti_length=tti_len,
        angle_gap=angle_gap,
        power_gap=power_gap,
        beamforming='no',
        chan_method='online'
    )

    # 生成元素级通道（SISO 就是 1×1），取当前时刻 initial_time=0
    h_paths, delays = channel_response_generate(channel, initial_time=0.0)  # h_paths: [cluster, Nrx, Ntx]
    # 压成每径系数向量（SISO → 标量），并按采样内插成均匀抽头 h(t)
    # h_paths[:, 0, 0] 对应每个 cluster 的复系数
    h = channel_time_interpolation(delays, h_paths[:, 0, 0], ts=1.0/fs_hz, num_ext=4)

    # 单位功率归一化：与 TDL 的习惯保持一致
    p = np.sum(np.abs(h)**2)
    if p > 0:
        h = h / np.sqrt(p)

    return h

# def apply_cdl_c(x, fs_hz, ds=300e-9, fc_GHz=7, seed=None):
#     """
#     用 CDL-C 冲激响应卷积输入波形，保持与 TDL 版本相同的接口与对齐策略。
#     返回:
#         y, h
#     """
#     h = cdl_c_impulse_response(fs_hz, ds=ds, fc_GHz=fc_GHz, seed=seed)
#     y = np.convolve(x, h, mode='same')  # 与你现在的 TDL 实现保持一致
#
#     return y, h


def cdl_c_impulse_response_64t4r(fs_hz, ds=300, fc_GHz=7, seed=None):
    """
    生成 64T4R CDL-C 的等效基带冲激响应 (已包含 64T 全 1 波束赋形)。
    返回:
        H_eff : 形状为 (4, num_taps)，表示 4 根接收天线的等效信道冲激响应
    """
    if seed is not None:
        np.random.seed(seed)
    ds = ds * 1e-9
    # ==========================================
    # 物理天一阵列配置
    # ==========================================
    tx_ant = [1, 1, 8, 8, 1]  # 64T: 1面板, 8行, 8列, 单极化
    rx_ant = [1, 1, 1, 2, 2]  # 4R:  1面板, 1行, 2列, 双极化

    # 初始化 CDL-C
    channel, _, _ = channel_initialize(
        cdl_type='CDLB', delay_spread=ds, Kf=[], tx_antenna=tx_ant, rx_antenna=rx_ant,
        ue_speed=3, fc_GHz=fc_GHz, fs=fs_hz, num_tti=0, tti_length=1e-3,
        angle_gap=0.0, power_gap=0.0, beamforming='no', chan_method='online'
    )

    h_paths, delays = channel_response_generate(channel, initial_time=0.0)

    h00 = channel_time_interpolation(delays, h_paths[:, 0, 0], ts=1.0 / fs_hz, num_ext=4)
    num_taps = len(h00)

    H_mimo = np.zeros((4, 64, num_taps), dtype=complex)
    for rx in range(4):
        for tx in range(64):
            H_mimo[rx, tx, :] = channel_time_interpolation(
                delays, h_paths[:, rx, tx], ts=1.0 / fs_hz, num_ext=4
            )

    B = np.ones(64, dtype=complex) / np.sqrt(64)

    H_eff = np.zeros((4, num_taps), dtype=complex)
    for rx in range(4):
        H_eff[rx, :] = np.sum(H_mimo[rx, :, :] * B[:, None], axis=0)

    return H_eff


def apply_cdl_c(x, fs_hz, seed=None, ds=300, fc_hz=7e9, speed_kmh=3, nfft=2048, normalize=False, nRx=4):
    """
    与你原先接口保持一致，但处理 1T4R。
    返回:
      y: 形状为 (len(x), nRx)，表示 4 根天线的时域接收信号
      h: 形状为 (nRx, DS_tap)，表示 4 根天线的信道冲激响应
    """
    h = cdl_c_impulse_response_64t4r(
        fs_hz=fs_hz, ds=ds, fc_GHz=fc_hz / 1e9, seed=seed
    )
    # h/=np.sqrt(40)
    # print(np.sum(np.square(np.abs(h))))
    # 针对每根接收天线分别进行一维卷积
    # 初始化二维输出数组
    y = np.zeros((len(x), nRx), dtype=np.complex128)
    for i in range(nRx):
        y[:, i] = np.convolve(x, h[i, :], mode='same')

    return y, h
def cdl_c_impulse_response_64t4r_full(fs_hz, ds_ns=300, fc_GHz=7, seed=None, initial_time=0.0, ue_speed=3):
    """
    生成 64T4R CDL 的全 MIMO 基带冲激响应。
    返回:
        H_mimo : shape=(4, 64, num_taps)
    """
    if seed is not None:
        np.random.seed(seed)

    ds = ds_ns * 1e-9

    tx_ant = [1, 1, 8, 8, 1]   # 64T
    rx_ant = [1, 1, 1, 2, 2]   # 4R

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
def effective_impulse_from_precoder(H_mimo, w_tx):
    """
    H_mimo: shape=(4, 64, L)
    w_tx  : shape=(64,)
    返回:
        h_eff: shape=(4, L)
    """
    w = np.asarray(w_tx, dtype=np.complex128).ravel()
    assert len(w) == 64
    norm = np.linalg.norm(w)
    assert norm > 0
    w = w / norm

    # 对 64 根发射天线加权求和
    h_eff = np.einsum('rtl,t->rl', H_mimo, w)
    return h_eff
def apply_cdl_c_precoded(x, fs_hz, w_tx, seed=None, ds_ns=300, fc_hz=7e9, speed_kmh=3, nRx=4, initial_time=0.0):
    """
    输入:
      x    : shape=(Ns,) 单层时域波形
      w_tx : shape=(64,) 本轮传输采用的发射 precoder
    返回:
      y     : shape=(Ns, 4)
      h_eff : shape=(4, L)
      H_mimo: shape=(4, 64, L)
    """
    H_mimo = cdl_c_impulse_response_64t4r_full(
        fs_hz=fs_hz,
        ds_ns=ds_ns,
        fc_GHz=fc_hz / 1e9,
        seed=seed,
        initial_time=initial_time,
        ue_speed=speed_kmh
    )

    h_eff = effective_impulse_from_precoder(H_mimo, w_tx)

    x = np.asarray(x, dtype=np.complex128).ravel()
    y = np.zeros((len(x), nRx), dtype=np.complex128)

    for rx in range(nRx):
        y[:, rx] = np.convolve(x, h_eff[rx, :], mode='same')

    return y, h_eff, H_mimo
def precoder_traditional_ones(Nt=64):
    return np.ones(Nt, dtype=np.complex128) / np.sqrt(Nt)
def precoder_cyclic_dft(q, Nt=64):
    """
    q: cyclic index, 0~Nt-1
    """
    m = np.arange(Nt, dtype=np.float64)
    w = np.exp(1j * 2.0 * np.pi * q * m / Nt) / np.sqrt(Nt)
    return w.astype(np.complex128)

def simulate_bler_config(
    snr_db_list,
    n_trials=200,
    ncellid=210,
    fs=30.72e6,
    scs_khz=15,
    seed=2025,
    nrb_sib=15,
    nsym_sib=12,
    use_combining=False,
    N_comb=4,
    payload = 1256
):
    """对单个配置 (nrb_ssb, nsym_ssb, nrb_ss) 扫 SNR 生成 BLER 曲线"""

    # 早停配置
    p_goal = 2*1e-3
    min_trials = 500
    bler = []
    rng = np.random.RandomState(seed)
    for snr_db in snr_db_list:
        n_err = 0
        round_total = 0
        with tqdm(total=n_trials, desc=f"SNR={snr_db:>5.1f} dB", ncols=70, leave=False) as pbar:
            for _ in range(n_trials):
                if use_combining:
                    ok, info = one_shot_sib_min_combined(
                        snr_db, ncellid=ncellid, fs=fs, scs_khz=scs_khz,
                        nrb_sib=nrb_sib, nsym_sib=nsym_sib, rng_seed=rng.randint(1 << 31),
                        N_comb=N_comb,payload = payload_cfg
                    )
                else:
                    # ok, info = one_shot_pbch_min(
                    #     snr_db, ncellid=ncellid, fs=fs, scs_khz=scs_khz,
                    #     nrb_ssb=nrb_ssb, nsym_ssb=nsym_ssb, nrb_ss=nrb_ss,
                    #     ibar_tx=None, rng_seed=rng.randint(1 << 31)
                    # )
                    round_total = round_total
                round_total += 1
                if not ok:
                    n_err += 1
                if n_err >= 50:
                    # 早停：出现 10 次错误就提前结束，提高速度
                    break
                if round_total >= min_trials and n_err <= 2 and (3.0 / round_total <= p_goal):
                    # 成功率高且试验够多 → 早停
                    break
                if round_total >= 3000 and n_err <= 1:
                    break
                pbar.update(1)
        bler.append(n_err / round_total)
        print(f"[CFG nrb={nrb_sib}, nsym={nsym_sib}, comb={use_combining}, N={N_comb}] "
              f"SNR={snr_db:>5.1f} dB  BLER={bler[-1]:.4f}")
    return np.array(bler)

def one_shot_sib_min_combined(
    snr_db, ncellid,
    fs=30.72e6, scs_khz=15,
    nrb_sib=48, nsym_sib=12,
    rng_seed=None,
    N_comb=4,
    payload=1256,

    # ---------- 传输模式 ----------
    mode="cyclic",          # "traditional" or "cyclic"
    cyclic_q_list=(0, 16, 32, 48),

    # ---------- DMRS / PDSCH 配置 ----------
    dmrs_step=2,
    dmrs_loc=(0, 6, 9),

    # ---------- CDL 控制 ----------
    cdl_seed_base=1234,
    cdl_initial_time0=0.0,
    cdl_round_spacing_s=0.0,
    channel_evolution_mode="static_same",
    # 取值:
    #   "static_same"   : 每轮同一个 seed、同一个 initial_time
    #   "time_evolving" : seed固定，initial_time 随轮次增加
    #   "independent"   : 每轮 seed 不同，initial_time 也可不同
):
    """
    返回:
      ok, info
    """

    if rng_seed is not None:
        np.random.seed(rng_seed)

    A = payload
    rv = 0
    modulation = 'QPSK'
    nlayers = 1

    # -----------------------------
    # 1) 先按当前配置自动算 E / rate
    #    避免 rate 写死成 48RB, 3DMRS
    # -----------------------------
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

    E_tmp = len(sib_idx_tmp) * 2   # QPSK
    rate = (A + 16) / E_tmp

    cbs_info = nrDLSCHInfo(A, rate)

    # -----------------------------
    # 2) 建 grid
    # -----------------------------
    sib1_pdsch_bits = np.random.randint(0, 2, A).astype(int)

    sib_grid, meta, meta2, sib1_symb_in = build_SIB1pdsch_grid_only(
        ncellid=ncellid,
        trblk_bits=sib1_pdsch_bits,
        nrb_sib=nrb_sib,
        nsym_sib=nsym_sib,
        dmrs_loc=dmrs_loc,
        k_step=dmrs_step,
        cbs_info=cbs_info
    )

    sib_idx = meta2['sib1_idx']
    dmrs_idx = meta2['dmrs_idx']
    dmrs_cols = meta2['dmrs_cols']
    E = meta2['E']

    # -----------------------------
    # 3) 多轮接收 + LLR combining
    # -----------------------------
    llr_sum = None

    for r in range(N_comb):
        # ---- 3.1 选择发端 precoder ----
        if mode == "traditional":
            w_r = precoder_traditional_ones(64)
        elif mode == "cyclic":
            q = cyclic_q_list[r % len(cyclic_q_list)]
            w_r = precoder_cyclic_dft(q, 64)
        else:
            raise ValueError("mode must be 'traditional' or 'cyclic'")

        # ---- 3.2 选择 CDL seed / initial_time ----
        if channel_evolution_mode == "static_same":
            cdl_seed_r = cdl_seed_base
            cdl_initial_time_r = cdl_initial_time0

        elif channel_evolution_mode == "time_evolving":
            # 同一个几何散射场景，但时间推进
            cdl_seed_r = cdl_seed_base
            cdl_initial_time_r = cdl_initial_time0 + r * cdl_round_spacing_s

        elif channel_evolution_mode == "independent":
            # 每轮独立信道 realization
            cdl_seed_r = None if cdl_seed_base is None else (cdl_seed_base + r)
            cdl_initial_time_r = cdl_initial_time0 + r * cdl_round_spacing_s

        else:
            raise ValueError("channel_evolution_mode must be "
                             "'static_same', 'time_evolving', or 'independent'")

        llr_r = _rx_round_get_llr(
            snr_db, ncellid,
            sib_grid, sib_idx, dmrs_idx, E,
            fs=fs,
            scs_khz=scs_khz,
            nrb_sib=nrb_sib,
            nsym_sib=nsym_sib,
            meta=meta,
            meta2=meta2,
            tx_precoder=w_r,
            cdl_seed=cdl_seed_r,
            cdl_initial_time=cdl_initial_time_r,
        )

        if llr_sum is None:
            llr_sum = llr_r.astype(np.float64, copy=True)
        else:
            llr_sum += llr_r

    # -----------------------------
    # 4) LDPC 恢复 / 解码
    # -----------------------------
    raterec = nrRateRecoverLDPC(llr_sum, A, rate, rv, modulation, nlayers)

    dec_bits, iters = nrLDPCDecode(
        raterec, cbs_info['BGN'], 25, blklen=A
    )
    blk, blk_err = nrCodeBlockDesegmentLDPC(
        dec_bits, cbs_info['BGN'], A + cbs_info['L']
    )
    out, tb_err = nrCRCDecode(blk, cbs_info['CRC'])

    crc_ok = (tb_err == 0)

    info = dict(
        N_comb=N_comb,
        E=E,
        A=A,
        rate=rate,
        mode=mode,
        dmrs_loc=list(dmrs_cols),
        cdl_seed_base=cdl_seed_base,
        cdl_initial_time0=cdl_initial_time0,
        cdl_round_spacing_s=cdl_round_spacing_s,
        channel_evolution_mode=channel_evolution_mode,
    )

    return crc_ok, info
def add_awgn_on_grid_multi_rx(rxGrids, txGrid_ref, snr_db, active_idx):
    """
    rxGrids: shape=(nRx, K, N)
    返回:
      rxGrids_noisy: shape=(nRx, K, N)
      nVar_list: 长度 nRx 的列表
    """
    rxGrids = np.asarray(rxGrids)
    nRx = rxGrids.shape[0]

    out = np.zeros_like(rxGrids, dtype=complex)
    nVar_list = []
    for r in range(nRx):
        g_noisy, nVar_r = add_awgn_on_gri_per_symbol(rxGrids[r], txGrid_ref, snr_db, active_idx)
        out[r] = g_noisy
        nVar_list.append(nVar_r)
    return out, nVar_list
def ofdm_demodulate_multi_rx(waveform, nrb, scs, initialNSlot, SampleRate, nsym_keep, CyclicPrefixFraction=0.5):
    """
    waveform:
      - shape=(Ns,)      -> 返回 (1, K, N)
      - shape=(Ns, nRx)  -> 返回 (nRx, K, N)
    """
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
    return np.stack(grids, axis=0)   # (nRx, K, N)
def _rx_round_get_llr(
    snr_db, ncellid,
    sib_grid,
    sib_idx, dmrs_idx, E,
    fs=61.44e6, scs_khz=30, nrb_sib=48, nsym_sib=12,
    meta=None, meta2=None,
    tx_precoder=None,
    cdl_seed=None,
    cdl_initial_time=0.0,
):
    # -----------------------------
    # 1) 发端 OFDM
    # -----------------------------
    tx_wave, _ = nrOFDMModulate(
        carrier=None, grid=sib_grid, scs=scs_khz, SampleRate=fs
    )

    # -----------------------------
    # 2) 过信道
    # -----------------------------
    if NO_CHANNEL:
        ch_out = tx_wave
        h_eff_cdl = None
        H_mimo = None
    else:
        if CDL_CHANNEL:
            if tx_precoder is None:
                tx_precoder = precoder_traditional_ones(64)

            ch_out, h_eff_cdl, H_mimo = apply_cdl_c_precoded(
                tx_wave,
                fs_hz=fs,
                w_tx=tx_precoder,
                seed=cdl_seed,
                ds_ns=300,
                fc_hz=7e9,
                speed_kmh=3,
                nRx=4,
                initial_time=cdl_initial_time
            )
        else:
            ch_out, H_tdl_timedomain = apply_tdl_c(tx_wave, fs, ds=300)

    # -----------------------------
    # 3) 多 Rx OFDM 解调
    # -----------------------------
    rxGrids_clean = ofdm_demodulate_multi_rx(
        waveform=ch_out,
        nrb=nrb_sib,
        scs=scs_khz,
        initialNSlot=0,
        SampleRate=fs,
        nsym_keep=nsym_sib,
        CyclicPrefixFraction=0.5
    )   # shape=(nRx, K, N)

    nRx = rxGrids_clean.shape[0]

    # -----------------------------
    # 4) 每个 Rx 单独加噪
    # -----------------------------
    active_idx = np.sort(np.concatenate([sib_idx, dmrs_idx]))
    rxGrids, nVar_list = add_awgn_on_grid_multi_rx(
        rxGrids_clean, sib_grid, snr_db, active_idx
    )

    # -----------------------------
    # 5) 构造参考 DMRS 网格
    # -----------------------------
    dmrs_symb = nrSIBDMRS_param(ncellid, len(dmrs_idx))
    refGridH = np.zeros_like(rxGrids[0], dtype=complex)
    nrSetResources(dmrs_idx, refGridH, dmrs_symb)

    # -----------------------------
    # 6) 单天线无信道直通
    # -----------------------------
    if NO_CHANNEL:
        # 这里 NO_CHANNEL 下一般就是 nRx=1
        rxGrid = rxGrids[0]
        nVar = nVar_list[0]
        y_sib = nrExtractResources(sib_idx, rxGrid)
        llr = nrSymbolDemodulate(y_sib, 'QPSK', np.mean(nVar), 'soft')
        scr_rx = meta['scr'].astype(int)
        llr_descrambled = llr * (1 - 2 * scr_rx)
        assert len(llr_descrambled) == E
        return llr_descrambled

    # -----------------------------
    # 7) 理想 CE / 非理想 CE：每个 Rx 单独做，再 LLR 合并
    # -----------------------------
    scr_rx = meta['scr'].astype(int)
    llr_sum_rx = np.zeros(E, dtype=np.float64)

    if ONLY_TDLC:
        # txGrid 只需要算一次
        txGrid_id = nrOFDMDemodulate(
            waveform=tx_wave,
            nrb=nrb_sib,
            scs=scs_khz,
            initialNSlot=0,
            SampleRate=fs
        )[:, :nsym_sib]

        mask = (np.abs(txGrid_id) > 0)

        for r in range(nRx):
            ch_only_r = rxGrids_clean[r]

            H_true_r = np.ones_like(ch_only_r, dtype=complex)
            H_true_r[mask] = ch_only_r[mask] / txGrid_id[mask]

            nVar_r = nVar_list[r]
            y_sib_r = nrExtractResources(sib_idx, rxGrids[r])
            h_sib_r = H_true_r.ravel(order="F")[sib_idx]

            sib_eq_r, csi_r = nrEqualizeMMSE_persym(
                y_sib_r, h_sib_r, nVar_r, sib_idx, K=nrb_sib * 12
            )

            llr_r = nrSymbolDemodulate_param_persym(
                sib_eq_r, 'QPSK', nVar_r, sib_idx, nrb_sib * 12, csi_r, 'soft'
            )
            llr_sum_rx += np.real(llr_r)

    else:
        dmrs_cols = meta2['dmrs_cols']

        for r in range(nRx):
            rxGrid_r = rxGrids[r]
            nVar_r = nVar_list[r]

            H_r = myChannelEstimate_std(
                rxGrid=rxGrid_r,
                refGrid=refGridH,
                EST_TFDOMAIN=False,
                EST_PBCH=False,
                EST_PDSCH_FULLBAND=True,
                dmrs_l=dmrs_cols,
                start_rb=0,
                end_rb=nrb_sib,
                td_win_len_ratio=0.08,
                td_window_type="fixed",
            )

            y_sib_r = nrExtractResources(sib_idx, rxGrid_r)
            h_sib_r = nrExtractResources(sib_idx, H_r)

            sib_eq_r, csi_r = nrEqualizeMMSE_persym(
                y_sib_r, h_sib_r, nVar_r, sib_idx, K=nrb_sib * 12
            )

            llr_r = nrSymbolDemodulate_param_persym(
                sib_eq_r, 'QPSK', nVar_r, sib_idx, nrb_sib * 12, csi_r, 'soft'
            )
            llr_sum_rx += np.real(llr_r)

    # -----------------------------
    # 8) descramble
    # -----------------------------
    llr_descrambled = llr_sum_rx * (1 - 2 * scr_rx)

    assert len(llr_descrambled) == E
    return llr_descrambled
if __name__ == "__main__":
    # 建议用稍“温和”的 SNR 段，便于拉开曲线
    # print_nmse_compare_db(snr_db=-2.0, ncellid=208, fs=30.72e6, scs_khz=30, trials_each=300)
    snr_points =  [-26.5,-26,-25.5]  # 也可以换回你之前的段, -13,   `-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,0,1,2,3,4,5,6,8,10 -3,-2,-1,-0.5,0,0.5,1,1.5,2

    # 要对比的 7 组配置（标签, nrb_ssb, nsym_ssb, nrb_ss）
    # 注意：nrb_ss 不要大于 nrb_ssb；这里统一用 12 个 RB 的中心带宽，
    # 在 nrb_ssb = 12 的场景会让 PSS/SSS 覆盖整个带宽（合法，只是 PBCH 在对应符号上被挖空）
    configs = [
        # ("12RB × 4sym 1 comb", 12, 4, 1),
        # ("20RB × 4sym 1 comb", 20, 4,  1),
        # ("20RB × 4sym 2 comb" , 20,  4,  2),
        # ("12RB × 6sym 1 comb" , 12,  6,  1),

        # ("24RB × 4sym 1 comb", 24, 4, 1),

        # ("96RB × 12sym 1 comb 200 bit payload", 96, 12, 1,200),
        # ("96RB × 12sym 1 comb 1200 bit payload", 96, 12, 1, 1200),
        # ("48RB × 12sym 1 comb 200 bit payload", 48, 12, 1, 200),
        ("48RB × 12sym 1 comb 1200 bit payload", 48, 12, 4, 1200),

    ]

    curves = {}
    for label, nrb_cfg, nsym_cfg, comb_cfg, payload_cfg in configs:
        bler_curve = simulate_bler_config(
            snr_points,
            n_trials=3000,          # 你可以调大/调小
            ncellid=208,
            fs=61.44e6,
            scs_khz=30, # TODO fit 30 kHz
            seed=2029,
            nrb_sib=nrb_cfg,
            nsym_sib=nsym_cfg,
            use_combining=True,
            N_comb=comb_cfg,
            payload = payload_cfg
        )
        curves[label] = bler_curve

    # 画图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5.5))
    bler = []
    for label, bler in curves.items():
        plt.semilogy(snr_points, np.maximum(bler, 1e-3), marker='o', label=label)

    plt.grid(True, which='both')
    plt.xlabel("SNR (dB)")
    plt.ylabel("BLER")
    plt.title(f"PBCH BLER vs SNR  (NO_CHANNEL={NO_CHANNEL}, KNOWN_nVar={USE_KNOWN_NVAR})")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("odsib_test_CE_rx_SNR_CE.png", dpi=150)
    print("多配置 BLER 曲线已保存到 odsib2_test_CE_rx_SNR_CE.png")
    # === 保存数值结果 ===
    import csv
    with open("odsib2_test_CE_rx_SNR_CE.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Config", "SNR(dB)", "BLER"])
        for label, bler_curve in curves.items():
            for snr, bler_val in zip(snr_points, bler_curve):
                writer.writerow([label, snr, bler_val])
    print("数值结果已保存到 odsib2_test_CE_rx_SNR_CE.csv")