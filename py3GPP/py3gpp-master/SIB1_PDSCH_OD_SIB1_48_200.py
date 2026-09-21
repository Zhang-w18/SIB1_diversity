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
from myChannelEstimation import DMRSFilterGenerate_v2_explicit,myChannelEstimate,myChannelEstimate_std
from tqdm import tqdm
NO_CHANNEL = False          # 先用纯 AWGN 验证链路 ——> True
CDL_CHANNEL = False
USE_KNOWN_NVAR = False      # 用 SNR 推噪声方差 ——> True
PRINT_DIAG = False #True          # 打印关键诊断信息
POWER_NORM_PER_RE = False
USE_SCL = True
ONLY_TDLC = False
USE_RX_SNR = True
def nrSIB1DMRScinit(n_slot=0,l=2,ncellid=0):
    return (2**17 * (14*n_slot + l + 1) * (2*ncellid + 1) + 2 * ncellid) % (2**31)
def _pdsch_scrambling_bits_sib1(n_id_cell: int, E: int, rnti: int = 0xFFFF,q=0) -> np.ndarray:
    # TS 38.211 7.3.1.1: c_init = RNTI * 2^15 + nID；SIB1 用 SI-RNTI，nID = N_ID_cell
    c_init = (int(rnti) << 15) + (int(q) << 14) + int(n_id_cell)
    return nrPRBS(c_init, E).astype(np.uint8)
def nrSIB1DMRSIndices_param(
        nrb_sib: int,
        nsym_slot: int,
        dmrs_loc: int,
        k_step: int = 2,
        style: str = "matlab",
):
    """
    生成 SIB1 的 PDSCH DM-RS 索引（仅在一个符号上，频域从0开始每 k_step 个RE放1个）。

    参数:
        nrb_sib   : SIB1 所在带宽的 PRB 数 (>=1)
        nsym_slot : 每个slot的OFDM符号数（normal CP = 14）
        dmrs_loc  : DMRS 所在符号索引 l (0..nsym_slot-1)，SIB1常用 l=2
        k_step    : 频域 comb 间隔（SIB1一般为2）
        style     : "matlab" 返回一维展平索引；"python" 返回 (k_idx_tuple, l_idx_tuple)

    返回:
        - style == "matlab": 一维展平索引 (flat = k + l * Nsc)
        - style == "python": (k_idx_tuple, l_idx_tuple)
    """
    assert nrb_sib >= 1, "nrb_sib must be >= 1"
    assert nsym_slot >= 1, "nsym_slot must be >= 1"
    assert 0 <= dmrs_loc < nsym_slot, "dmrs_loc out of range"
    assert k_step >= 1, "k_step must be >= 1"

    Nsc = int(nrb_sib) * 12  # 每PRB 12个RE
    # 频域从第0个RE开始，步长为 k_step
    ks = np.arange(0, Nsc, k_step, dtype=int)

    if ks.size == 0:
        # 没有可放置的 RE（理论上不会发生，除非Nsc<1或k_step过大）
        if style == "python":
            return (tuple(), tuple())
        else:
            return np.array([], dtype=int)

    # 仅在 l=dmrs_loc 这个符号放DMRS
    ls = np.full_like(ks, dmrs_loc, dtype=int)

    if style == "python":
        return (tuple(ks.tolist()), tuple(ls.tolist()))

    if style == "matlab":
        return (ks + ls * Nsc).astype(int)

    raise ValueError("Unknown style")
def nrSIB1Indices_param(
        nrb_sib: int,
        nsym_slot: int,
        dmrs_idx: np.ndarray,
        style: str = "matlab",
):
    """
    生成 SIB1 的 PDSCH 数据 RE 索引（展平形式），并去除 DMRS 对应的 RE。

    参数：
        nrb_sib   : PDSCH 占用的 RB 数（>=1）
        nsym_slot : 每slot符号数（normal CP 通常14）
        dmrs_idx  : 已知 DMRS 的展平索引 (k + l*Nsc)
        style     : "matlab" 返回展平索引；"python" 返回 (k_idx_tuple, l_idx_tuple)

    返回：
        - style=="matlab": 一维展平索引 (flat = k + l*Nsc)
        - style=="python": (k_idx_tuple, l_idx_tuple)
    """
    assert nrb_sib >= 1 and nsym_slot >= 1
    Nsc = int(nrb_sib) * 12  # 每个符号的子载波数

    # 1) 整个SIB1 PDSCH区域（所有RE）
    all_idx = np.arange(Nsc * nsym_slot, dtype=int)

    # 2) 去除 DMRS 对应的 RE
    if dmrs_idx is not None and len(dmrs_idx) > 0:
        dmrs_idx = np.asarray(dmrs_idx, dtype=int)
        dmrs_idx = dmrs_idx[(dmrs_idx >= 0) & (dmrs_idx < Nsc * nsym_slot)]
        sib1_data_idx = np.setdiff1d(all_idx, dmrs_idx, assume_unique=False)
    else:
        sib1_data_idx = all_idx

    # 3) 根据 style 输出
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
def build_SIB1pdsch_grid_only(ncellid, trblk_bits,nrb_sib=15, nsym_sib=12, dmrs_loc=0,k_step=2,cbs_info={'CRC': '16', 'L': 16, 'BGN': 2, 'C': 1, 'Lcb': 0, 'F': 8, 'Zc': 128, 'K': 1280, 'N': 6400}):
    grid = np.zeros((nrb_sib*12, nsym_sib), dtype=complex)
    # 统一索引
    dmrs_idx = nrSIB1DMRSIndices_param(nrb_sib, nsym_sib,dmrs_loc,k_step)
    sib1_idx = nrSIB1Indices_param(nrb_sib, nsym_sib,dmrs_idx)
    # 编码 -> 速配(E) -> QPSK
    E = len(sib1_idx)*2 #QPSK 承载的资源bit数
    sib1_bits,meta = nrSIB_param(
        in_bits=trblk_bits,
        nrb_sib=nrb_sib,
        nsym_slot=nsym_sib,
        dmrs_idx=dmrs_idx,
        sib1_idx=sib1_idx,
        n_id_cell=ncellid,
        nLayers=1,
        rv=0,  # SIB1 初发一般 RV=0
        rnti=0xFFFF,  # SI-RNTI
        E = E,
        BGN = cbs_info['BGN'],
        CRC = cbs_info['CRC']
    )
    sib1_symb = nrSymbolModulate(sib1_bits, 'QPSK')

    assert len(sib1_symb) == len(sib1_idx)
    # DMRS
    dmrs_symb = nrSIBDMRS_param(ncellid,len(dmrs_idx))
    # print(dmrs_symb)
    assert len(dmrs_symb) == len(dmrs_idx)

    nrSetResources(sib1_idx, grid, sib1_symb)
    nrSetResources(dmrs_idx, grid, dmrs_symb)
    meta2 = dict(sib1_idx=sib1_idx, dmrs_idx=dmrs_idx, E=E)
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

    if normalize:
        p = np.sum(np.abs(h)**2)
        if p > 0:
            h /= np.sqrt(p)

    return h

# def apply_tdl_c(x, fs_hz, seed=None):
#     h = tdl_c_impulse_response(fs_hz, seed)
#     # 用“same”保持长度和对齐，不引入整体时移（避免起点错位）
#     y = np.convolve(x, h, mode='same')
#     #print('h=',h)
#     return y, h
def apply_tdl_c(x, fs_hz, seed=None, ds=300e-9, fc_hz=7e9, speed_kmh=3.0, nfft=4096, normalize=True):
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
def cdl_c_impulse_response(fs_hz, ds=300e-9, fc_GHz=3.5, seed=None):
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

def apply_cdl_c(x, fs_hz, ds=300e-9, fc_GHz=7, seed=None):
    """
    用 CDL-C 冲激响应卷积输入波形，保持与 TDL 版本相同的接口与对齐策略。
    返回:
        y, h
    """
    h = cdl_c_impulse_response(fs_hz, ds=ds, fc_GHz=fc_GHz, seed=seed)
    y = np.convolve(x, h, mode='same')  # 与你现在的 TDL 实现保持一致

    return y, h

def cdl_c_impulse_response_64x1(
    fs_hz,
    ds=300e-9,
    fc_GHz=3.5,
    seed=None,
    tx_array=(1, 1, 2, 4, 1),   # [Mg, Ng, M, N, P] → 1×1 面板, 面板内 8×8, 单极化 → 64 元
    rx_array=(1, 1, 1, 1, 1),   # 1 根收天线
    w_tx=None,                  # 发射波束（逐阵元），shape = [Ntx_elems_total,]
    w_rx=None                   # 接收合成（逐阵元），shape = [Nrx_elems_total,]，1 根收天线默认 [1]
):
    """
    生成 64Tx×1Rx（单流）的等效 SISO 冲激响应 h。
    默认用 CDL-C、静态（无多普勒），单位功率归一化。
    """
    if seed is not None:
        np.random.seed(seed)

    # 阵列规模
    tx_ant = list(tx_array)
    rx_ant = list(rx_array)
    Ntx_elems = tx_ant[0]*tx_ant[1]*tx_ant[2]*tx_ant[3]*tx_ant[4]
    Nrx_elems = rx_ant[0]*rx_ant[1]*rx_ant[2]*rx_ant[3]*rx_ant[4]  # 这里=1

    # 默认波束：均匀加权（broadside），单位范数
    if w_tx is None:
        w_tx = np.ones(Ntx_elems, dtype=complex) / np.sqrt(Ntx_elems)
    else:
        w_tx = np.asarray(w_tx).reshape(-1)
        assert w_tx.size == Ntx_elems, "w_tx 长度需等于发射阵元总数"
        # 归一化建议
        norm = np.linalg.norm(w_tx)
        if norm > 0:
            w_tx = w_tx / norm

    if w_rx is None:
        w_rx = np.ones(Nrx_elems, dtype=complex)  # 单根天线就是 [1]
    else:
        w_rx = np.asarray(w_rx).reshape(-1)
        assert w_rx.size == Nrx_elems, "w_rx 长度需等于接收阵元总数"
        # 归一化（单根为 1，无所谓）
        norm = np.linalg.norm(w_rx)
        if norm > 0:
            w_rx = w_rx / norm

    # 初始化信道：用 “beamforming='yes'” 走波束端口路径，但我们会覆盖 precoder
    channel, _, _ = channel_initialize(
        cdl_type='CDLC',
        delay_spread=ds,
        Kf=[],                 # NLOS 无 Kf
        tx_antenna=tx_ant,
        rx_antenna=rx_ant,
        ue_speed=0.0,          # 静态（无多普勒）
        fc_GHz=fc_GHz,
        fs=fs_hz,
        num_tti=0,
        tti_length=1e-3,
        angle_gap=0.0,
        power_gap=0.0,
        beamforming='yes',
        chan_method='online'
    )

    # 覆盖波束成形矩阵：单流 → [1, Ntx_elems] / [1, Nrx_elems]
    # 注意：channel_virtualization 期望的是 “行=端口数”
    # channel['tx_precoding_vector'] = w_tx.reshape(1, -1)
    # channel['rx_precoding_vector'] = w_rx.reshape(1, -1)
    # channel['tx_dimension'] = 1
    # channel['rx_dimension'] = 1


    # 生成元素级通道（各 cluster），并投影到波束端口（单流 → 每径得到一个标量）
    h_elem_clusters, delays = channel_response_generate(channel, initial_time=0.0)   # [cluster, Nrx_elem, Ntx_elem]
    h_beam_clusters = channel_virtualization(h_elem_clusters,  # → [rx_dim(=1), tx_dim(=1), cluster]
                                             channel['rx_precoding_vector'],
                                             channel['tx_precoding_vector'])

    # 压成每径系数向量（长度 = num_cluster）
    coeff_per_cluster = h_beam_clusters[0, 0, :]  # 单流 rx=0, tx=0

    # 把每径的连续时延/系数做 sinc 内插，得到离散冲激响应 h[n]
    h = channel_time_interpolation(delays, coeff_per_cluster, ts=1.0/fs_hz, num_ext=4)

    # 单位功率归一化（与 TDL 的习惯一致）：∑|h|^2 = 1
    power = np.sum(np.abs(h)**2)
    if power > 0:
        h = h / np.sqrt(power)

    return h


def apply_cdl_c_64x1(x, fs_hz, ds=300e-9, fc_GHz=7, seed=None,
                     tx_array=(1,1,8,8,1), rx_array=(1,1,1,1,1),
                     w_tx=None, w_rx=None):
    """
    用 64×1（单流）CDL-C 冲激响应卷积输入波形，保持与 TDL 版本相同的接口与对齐策略。
    返回: y, h
    """
    h = cdl_c_impulse_response_64x1(
        fs_hz, ds=ds, fc_GHz=fc_GHz, seed=seed,
        tx_array=tx_array, rx_array=rx_array,
        w_tx=w_tx, w_rx=w_rx
    )
    y = np.convolve(x, h, mode='same')
    # import matplotlib
    # matplotlib.use("Agg")
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8,4))
    # plt.plot(1e9 * np.arange(len(h)) / fs_hz, np.abs(h)**2, marker='o', lw=1)
    # plt.title("64x1 CDL-C Channel Impulse Response Power")
    # plt.xlabel("Delay (ns)")
    # plt.ylabel("Power |h(t)|²")
    # plt.grid(True, which='both')
    # plt.tight_layout()
    # plt.savefig("cdl_c_64x1_impulse_response.png", dpi=150)
    # print("信道冲激响应功率已保存到 cdl_c_64x1_impulse_response.png")
    return y, h

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
    nrb_sib=15, nsym_sib=12, rng_seed=None,
    N_comb=4,payload = 1256
):
    if rng_seed is not None:
        np.random.seed(rng_seed)

    A = payload  # Transport block length
    rate = (A + 16) / ((11 * 15 * 12 + 90) * 2)  # Target code rate
    rv = 0  # Redundancy version, 0-3
    dmrs_step = 2
    dmrs_loc = 0  # PDSCH起始位置 实际为第2个符号
    sib1_pdsch_bits = np.random.randint(0, 2, A).astype(int)
    cbs_info = nrDLSCHInfo(A, rate)

    sib_grid, meta, meta2, sib1_symb_in = build_SIB1pdsch_grid_only(
        ncellid=ncellid, trblk_bits=sib1_pdsch_bits,
        nrb_sib=nrb_sib, nsym_sib=nsym_sib, dmrs_loc=dmrs_loc, k_step=dmrs_step, cbs_info=cbs_info
    )
    sib_idx = meta2['sib1_idx'];
    dmrs_idx = meta2['dmrs_idx'];
    E = meta2['E']
    # 多轮接收，逐轮得 LLR
    llr_sum = None
    for r in range(N_comb):
        llr_r = _rx_round_get_llr(
            snr_db, ncellid, sib_grid, sib_idx, dmrs_idx, E,
            fs=fs, scs_khz=scs_khz, nrb_sib=nrb_sib, nsym_sib=nsym_sib,meta=meta,
        )
        if llr_sum is None:
            llr_sum = llr_r.astype(np.float64, copy=True)
        else:
            llr_sum += llr_r
    modulation = 'QPSK'
    nlayers = 1
    raterec = nrRateRecoverLDPC(llr_sum, A, rate, rv, modulation, nlayers)

    dec_bits, iters = nrLDPCDecode(raterec, cbs_info['BGN'], 25,blklen=A)
    blk, blk_err = nrCodeBlockDesegmentLDPC(dec_bits, cbs_info['BGN'], A + cbs_info['L'])
    out, tb_err = nrCRCDecode(blk, cbs_info['CRC'])
    crc = tb_err
    return (crc == 0), dict(N_comb=N_comb, E=E)

def _rx_round_get_llr(
    snr_db, ncellid,
    sib_grid,               # 发端已映射好的 grid（含 SIBH+DMRS）
    sib_idx, dmrs_idx, E,  # 索引 & 速率匹配长度
    fs=30.72e6, scs_khz=15, nrb_sib=15, nsym_sib=12,meta=None
):
    # 调制
    ##
    tx_wave, _ = nrOFDMModulate(carrier=None, grid=sib_grid, scs=scs_khz, SampleRate=fs)
    if NO_CHANNEL:
        ch_out = tx_wave
    else:
        if CDL_CHANNEL:
            ch_out_cdl, H_cdl_timedomain = apply_cdl_c_64x1(tx_wave, fs, ds=300e-9,fc_GHz=7)
            ch_out = ch_out_cdl
        else:
            ch_out, H_tdl_timedomain = apply_tdl_c(tx_wave, fs, ds=300e-9)
    rxGrid = nrOFDMDemodulate(waveform=ch_out, nrb=nrb_sib, scs=scs_khz, initialNSlot=0, SampleRate=fs,
                              CyclicPrefixFraction=0.5)[:, :nsym_sib]

    # 在频域网格按 RE-SNR 加噪（避免 RB 依赖）
    active_idx = np.sort(np.concatenate([sib_idx, dmrs_idx]))
    rxGrid, nVar_ref = add_awgn_on_gri_per_symbol(rxGrid, sib_grid, snr_db, active_idx)
    dmrs_symb = nrSIBDMRS_param(ncellid, len(dmrs_idx))
    dmrs_y = nrExtractResources(dmrs_idx, rxGrid)
    refGridH = np.zeros_like(rxGrid, dtype=complex)

    # print(dmrs_syms)
    nrSetResources(dmrs_idx, refGridH, dmrs_symb)
    if NO_CHANNEL:
        err = dmrs_y - dmrs_symb
        nVar_est = float(np.mean(np.abs(err) ** 2))
        nVar = nVar_ref
        y_sib = nrExtractResources(sib_idx, rxGrid)
        sib_eq = y_sib

        # pbch_eq, csi = nrEqualizeMMSE_persym(y_sib, h_pbch, nVar, pbch_idx, K=nrb_ssb * 12)
        # llr = nrSymbolDemodulate_param_persym(sib_eq, 'QPSK', nVar, sib_idx, nrb_sib * 12, 1, 'soft')
        llr = nrSymbolDemodulate(sib_eq, 'QPSK', np.mean(nVar), 'soft')
        # llr = nrSymbolDemodulate_param(sib_eq, 'QPSK', np.mean(nVar), 'soft')
        scr_rx = meta['scr'].astype(int)
        llr_descrambled = llr * (1 - 2 * scr_rx)
    else:
        if ONLY_TDLC:
            txGrid_id = nrOFDMDemodulate(waveform=tx_wave, nrb=nrb_sib, scs=scs_khz,
        initialNSlot=0, SampleRate=fs#, CyclicPrefixFraction=0.5
    )[:, :nsym_sib]
            ch_only =  nrOFDMDemodulate(waveform=ch_out, nrb=nrb_sib, scs=scs_khz,
        initialNSlot=0, SampleRate=fs#, CyclicPrefixFraction=0.5
    )[:, :nsym_sib]
            H_true = np.ones_like(ch_only, dtype=complex)
            mask = (np.abs(txGrid_id) > 0)  # 只在实际映射了符号的 RE 上求比值
            H_true[mask] = ch_only[mask] / txGrid_id[mask]
            H_power =  np.abs(H_true) ** 2
            nVar = nVar_ref

            # 提取→均衡→软解
            y_sib = nrExtractResources(sib_idx, rxGrid)
            h_sib = H_true.ravel(order="F")[sib_idx]
            sib_eq, csi = nrEqualizeMMSE(y_sib, h_sib, nVar)
            llr = nrSymbolDemodulate_param(sib_eq, 'QPSK', nVar, csi, 'soft') # 长度应为 E
            scr_rx = meta['scr'].astype(int)
            llr_descrambled = llr * (1 - 2 * scr_rx)
        else:
            # nrSetResources(dmrs_idx, refGridH, dmrs_symb)
            # H_old, nVar_est = nrChannelEstimate(rxGrid=rxGrid, refGrid=refGridH)

            # scs_kHz = 15
            # num_fft = 2048
            # RB_start = 0  # 举例：SSB 起始 RB index，依场景确定
            # IdxSC = (np.arange(nrb_sib * 12) + RB_start * 12).reshape(-1, 1)
            # H =myChannelEstimate(
            #     rxGrid=rxGrid, refGrid=refGridH,IdxSC=IdxSC,td_denoise=True  # 有效带宽 RE 下标（列向量或一维数组），如形如 [[644],[645],...,[1383]]
            # )
            # print(np.sum(np.abs(H ** 2)))

            H = myChannelEstimate_std(rxGrid=rxGrid, refSym=dmrs_symb, refGrid=refGridH, EST_TFDOMAIN=False,
                                      EST_PBCH=False, method='denoising',rb_size = 2, N_sym=1,dmrs_l = 0)
            # print(np.sum(np.abs(H ** 2)))
            nVar = nVar_ref
            # 提取→均衡→软解
            y_sib = nrExtractResources(sib_idx, rxGrid)
            h_sib = nrExtractResources(sib_idx, H)
            sib_eq, csi = nrEqualizeMMSE_persym(y_sib, h_sib, nVar, sib_idx, K=nrb_sib * 12)
            llr = nrSymbolDemodulate_param_persym(sib_eq, 'QPSK', nVar, sib_idx, nrb_sib * 12, csi, 'soft')
            scr_rx = meta['scr'].astype(int)
            llr_descrambled = llr * (1 - 2 * scr_rx)




    # llr = nrSymbolDemodulate_param(pbch_eq, 'QPSK', nVar, csi, 'soft')  # 长度应为 E
    # llr = nrSymbolDemodulate(pbch_eq, 'QPSK', nVar,  'soft')
    # llr = llr * np.repeat(csi, 2)                            # 保持与单次实现一致
    assert len(llr_descrambled) == E
    return llr_descrambled

if __name__ == "__main__":
    # 建议用稍“温和”的 SNR 段，便于拉开曲线
    # print_nmse_compare_db(snr_db=-2.0, ncellid=208, fs=30.72e6, scs_khz=30, trials_each=300)
    snr_points =  [-11,-10.5,-10,-9.5,-9,-8.5,-8,-7.5,-7]  # 也可以换回你之前的段, -13,   `-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,0,1,2,3,4,5,6,8,10

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
        ("48RB × 12sym 1 comb 200 bit payload", 48, 12, 1, 200),
        # ("48RB × 12sym 1 comb 1200 bit payload", 48, 12, 1, 1200),

    ]

    curves = {}
    for label, nrb_cfg, nsym_cfg, comb_cfg, payload_cfg in configs:
        bler_curve = simulate_bler_config(
            snr_points,
            n_trials=3000,          # 你可以调大/调小
            ncellid=208,
            fs=30.72e6,
            scs_khz=15, # TODO fit 30 kHz
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