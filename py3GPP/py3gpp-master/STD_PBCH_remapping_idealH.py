# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 14:23:13 2025

 File name: STD_PBCH_CDL_8_8_rxSNR
 Author: m00829866	Version: 0.1	Date: 2025-11-04
 Description:
     1. standard PBCH Link Level Simulation

 修改记录：
 date name line xxx
"""

import numpy as np
from py3gpp import *
import matplotlib.pyplot as plt

'''
(
    nrOFDMModulate, nrOFDMDemodulate,
    nrSetResources, nrExtractResources,
    nrPBCHIndices, nrPBCHDMRS, nrPBCHDMRSIndices,
    nrPBCHPRBS, nrChannelEstimate, nrEqualizeMMSE,
    nrSymbolModulate, nrSymbolDemodulate, nrPBCHIndices_param,
    nrRateRecoverPolar, nrPolarDecode, nrCRCDecode,
    nrBCH
)
'''
from channel_initialize import channel_initialize,channel_settings_initialize,channel_cdl_initialize,rotate_angles_max_direction,scaling_angles,calculate_delay_spread,calculate_angular_spread
from channel_functions import channel_fading_signal, channel_response_generate, calculate_antenna_pattern, get_dft_codebook, channel_virtualization, channel_time_interpolation, channel_frequency_response
from nrPBCHIndicesVariable import nrPBCHIndices_param,nrPBCHDMRSIndices_param,nrPBCHDMRS_param,nrBCH_param,nrSymbolDemodulate_param,nrSymbolDemodulate_param_persym,nrRateRecoverPolar_mh,nrRateMatchPolar_mh
from tqdm import tqdm
from ext_scl.scl_adapter import polar_decode_scl_llr
from GenTDLChannel import ChannelInfo
from scipy import sparse
from myChannelEstimation import DMRSFilterGenerate_v2_explicit,myChannelEstimate,myChannelEstimatev2,myChannelEstimate_std

# ---------------------------
# 开关 & 辅助
# ---------------------------
NO_CHANNEL = False          # 先用纯 AWGN 验证链路 ——> True
CDL_CHANNEL = False
USE_KNOWN_NVAR = False      # 用 SNR 推噪声方差 ——> True
PRINT_DIAG = False #True          # 打印关键诊断信息
POWER_NORM_PER_RE = False
REF_NRB_FOR_POWER = 20
awgn_fast_path = False
USE_SCL = True
ONLY_TDLC = True
SNR_OFFSET_DB = 8
USE_RX_SNR = True

### FOR  DEBUG
def _idx_to_kl(idx, Nsc):
    # 线性索引按列优先（Fortran）展开：idx = k + l*Nsc
    k = idx % Nsc
    l = idx // Nsc
    return k, l


####
def add_awgn(x, snr_db):
    # snr_db = snr_db - SNR_OFFSET_DB
    p_sig = np.mean(np.abs(x)**2)
    p_noise = p_sig / (10**(snr_db/10.0))
    n = (np.random.randn(*x.shape) + 1j*np.random.randn(*x.shape)) * np.sqrt(p_noise/2.0)
    return x + n, p_noise  # 返回噪声功率，便于 nVar 计算
def add_awgn_on_grid(rxGrid_clean, ssb_grid,  snr_db_re, active_idx):
    """
    在频域网格上直接加噪，snr_db_re 是“每 RE 的目标 SNR (Es/N0，复符号功率口径)”
    """
    snr_lin = 10**(snr_db_re/10.0)
    # 以“每个 RE 的平均信号功率”为口径配噪
    flat = ssb_grid.ravel(order="F")
    # flat = rxGrid_clean.ravel(order="F")
    p_sig = np.mean(np.abs(flat[active_idx]) ** 2)
    nvar_complex = p_sig / snr_lin                          # 复符号噪声功率
    noise = (np.random.randn(*ssb_grid.shape) + 1j*np.random.randn(*ssb_grid.shape)) \
            * np.sqrt(nvar_complex/2.0)
    return rxGrid_clean + noise, nvar_complex
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
    return rxGrid_clean + noise, nvar_cols
# def tdl_c_impulse_response(fs_hz, seed=None):
#     """
#     生成符合 3GPP TR 38.901 TDL-C profile 的多径信道冲激响应。
#
#     参数:
#         fs_hz : float
#             系统采样率 (Hz)，例如 30.72e6。
#         ds : float
#             Delay Spread (秒)，默认 300 ns。
#         seed : int or None
#             随机种子，便于复现。
#
#     返回:
#         h : ndarray(complex)
#             复数基带离散脉冲响应（长度为最大延时采样 + 1）。
#     """
#     ds = 300e-9
#
#     # ---- (1) Path delays & powers from 3GPP TR 38.901 Table 7.7.2-4 (TDL-C)
#     delays_per_path = np.array([
#         0, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366, 0.6448, 0.6560,
#         0.6584, 0.7935, 0.8213, 0.9336, 1.2285, 1.3083, 2.1704, 2.7105,
#         4.2589, 4.6003, 5.4902, 5.6077, 6.3065, 6.6374, 7.0427, 8.6523
#     ]) * ds  # scale by DS (delay spread)
#
#     power_per_path_db = np.array([
#         -4.4, -1.2, -3.5, -5.2, -2.5, 0, -2.2, -3.9,
#         -7.4, -7.1, -10.7, -11.1, -5.1, -6.8, -8.7, -13.2,
#         -13.9, -13.9, -15.8, -17.1, -16.0, -15.7, -21.6, -22.8
#     ])
#
#     # ---- (2) Convert power(dB) → linear, normalize
#     p_lin = 10 ** (power_per_path_db / 10.0)
#     p_lin /= np.sum(p_lin)
#
#     # ---- (3) Generate complex Gaussian fading taps
#     if seed is not None:
#         np.random.seed(seed)
#     taps = (np.random.randn(len(p_lin)) + 1j * np.random.randn(len(p_lin))) / np.sqrt(2)
#     taps *= np.sqrt(p_lin)
#
#     # ---- (4) Convert continuous delays (s) to discrete sample delays
#     d_samp = np.round(delays_per_path * fs_hz).astype(int)
#     L = d_samp.max() + 1
#     h = np.zeros(L, dtype=complex)
#     for d, tap in zip(d_samp, taps):
#         h[d] += tap
#     if USE_RX_SNR:
#         # ---- (5) Normalize to unit total power
#         h /= np.sqrt(np.sum(np.abs(h) ** 2))
#         return h
#     else:
#         return h

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

def build_time_varying_H_sparse(h_tap_t, L):
    num_taps, T = h_tap_t.shape
    assert T == L

    col = np.arange(L)  # n
    tap = np.arange(num_taps)[:, None]  # k (列向量)

    rows = col[None, :] - tap  # n - k
    valid = rows >= 0

    row_idx = rows[valid].ravel()
    col_idx = np.broadcast_to(col, rows.shape)[valid].ravel()
    data = h_tap_t[valid].ravel()

    H = sparse.coo_matrix((data, (row_idx, col_idx)), shape=(L, L)).tocsr()
    return H
def apply_tdl_c_time_variable(tx_wave, fs,
                speed_kmh,        # 速度（km/h）
                fc_hz,            # 载频（Hz）
                tdl_type='TDLC',  # 'TDLA'..'TDLE'
                tdl_ds_ns=300,     # RMS DS（ns），匹配你的 ChannelInfo.DS 的用法
                nfft_hint=2048    # 仅作 ChannelInfo 初始化需要的 NFFT 提示
                ):
    # 1) 用 ChannelInfo 构造信道对象（不改内部实现）
    ch = ChannelInfo(
        nTx=1, nRx=1,
        Speed=speed_kmh,   # 你的类里内部会 /3.6 变成 m/s
        Fc=fc_hz,
        Fs=fs,
        ChanType=tdl_type,
        DS=tdl_ds_ns,
        NFFT=nfft_hint,
        TO=0, FO=0
    )

    # 2) 用现有的 Jakes 加速法生成“随时间变化”的抽头序列（不改实现）
    num_samples = tx_wave.shape[0]
    t = np.arange(num_samples) / fs
    ch.gen_Jakes_Accelerate(time_samples=t)   # 生成 ch.time_channel: [nRx, nTx, DS_tap, T]

    # 3) 把 time_channel 作为时变 FIR 做卷积（胶水层）
    #    SISO：取 [0,0,:,:]；多天线可按需扩展
    h_tap_t = ch.time_channel[0, 0, :, :]     # [DS_tap, T]
    ds_tap = h_tap_t.shape[0]
    T = tx_wave.shape[0]

    # -------------------------------
    # 新增部分：时不变 vs 时变 卷积
    # -------------------------------
    USE_TIME_VARYING = True   # ✅ True: 时变信道  False: 时不变信道

    if not USE_TIME_VARYING:
        USE_TIME_VARYING
    else:
        L = num_samples  # 波形长度 4096
        num_taps = h_tap_t.shape[0]  # 通道 tap 数 81


        # H_eff = np.zeros((L, L))
        #
        # for n in range(L):  # 对于输出的每一个采样点 n
        #     for k in range(num_taps):  # 遍历所有 tap
        #         row = n - k  # tx_wav 的对应位置
        #
        #         if row >= 0:  # 超出范围的 index 视为 0，不填
        #             H_eff[row, n] = h_tap_t[k, n]
        #
        # y = tx_wave @ H_eff
        # 列索引: 0~4095
        # col = np.arange(L)
        #
        # # tap 索引: 0~80
        # tap = np.arange(num_taps)
        #
        # # 计算所有 (row = n - k)，生成 81×4096 的矩阵
        # rows = col[None, :] - tap[:, None]  # shape: (81, 4096)
        #
        # # 合法位置 mask（只保留 row ≥ 0）
        # valid = rows >= 0
        #
        # # 构造等效矩阵 H_eff：4096×4096
        # H_eff = np.zeros((L, L))
        #
        # # 把所有合法位置一次性填进去（关键优化）
        # H_eff[rows[valid], col[None, :].repeat(num_taps, axis=0)[valid]] = h_tap_t[valid]
        H_eff = build_time_varying_H_sparse(h_tap_t, L)
        # 最终输出
        y = tx_wave @ H_eff


    return y, ch  # 返回波形与信道对象（你后续若想观测 doppler/ch.funs 可用）

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
def nrEqualizeMMSE_persym(rxSym, hest, nVar,pbch_idx,K):
    # 每个 RE 属于哪一列（符号索引 ℓ）
    sym_ids = (pbch_idx // K).astype(np.int64)  # ℓ in [0, N-1]
    # 给每个 RE 取对应列的噪声方差
    nVar_re = nVar[sym_ids]  # (M,)

    # MMSE 等化
    a = np.abs(hest * np.conj(hest))
    csi = a + nVar_re  # 分母
    # 防止极端零分母
    csi = np.where(csi == 0, 1e-30, csi)
    rxEq = rxSym * np.conj(hest) / csi

    return rxEq, csi
# ---------------------------
# 发端：只构造 PBCH + PBCH DMRS
# ---------------------------
def build_ssb_grid_pbch_only(ncellid, ibar_ssb, trblk_bits, nrb_ssb=20, nsym_ssb=4, nrb_ss=12,REMAPPING = None):
    grid = np.zeros((nrb_ssb*12, nsym_ssb), dtype=complex)
    # 统一索引
    pbch_idx = nrPBCHIndices_param(ncellid, nrb_ssb, nsym_ssb, nrb_ss)
    dmrs_idx = nrPBCHDMRSIndices_param(ncellid, nrb_ssb, nsym_ssb, nrb_ss)

    E = len(pbch_idx)*2

    # 编码 -> 速配(E) -> QPSK
    pbch_bits = nrBCH_param(trblk_bits, sfn=0, hrf=0, lssb=0, idxoffset=0, ncellid=ncellid, E=E)

        # === 这里做 bit 重排 ===
    pbch_bits_remap = remapping_pbch_bits(
            pbch_bits,
            pbch_idx=pbch_idx,
            nrb_ssb=nrb_ssb,
            nsym_ssb=nsym_ssb,
            mode=REMAPPING  # 或 "straight"
    )
    pbch_symb = nrSymbolModulate(pbch_bits_remap, 'QPSK')


    # assert len(pbch_symb) == len(pbch_idx)
    # DMRS
    dmrs_symb = nrPBCHDMRS_param(ncellid, ibar_ssb,len(dmrs_idx))
    # print(dmrs_symb)
    # assert len(dmrs_symb) == len(dmrs_idx)

    nrSetResources(pbch_idx, grid, pbch_symb)
    # nrSetResources(nrPBCHIndices(ncellid), grid, pbch_symb)
    nrSetResources(dmrs_idx, grid, dmrs_symb)
    # nrSetResources(nrPBCHDMRSIndices(ncellid), grid, nrPBCHDMRS(ncellid, ibar_ssb))
    # if POWER_NORM_PER_RE and nrb_ssb > 0:
    #     scale = np.sqrt(REF_NRB_FOR_POWER / float(nrb_ssb))
    #     grid *= scale
    meta = dict(pbch_idx=pbch_idx, dmrs_idx=dmrs_idx, E=E)
    return grid, meta


def simulate_bler_config(
    snr_db_list,
    n_trials=200,
    ncellid=210,
    fs=30.72e6,
    scs_khz=15,
    seed=2025,
    nrb_ssb=20,
    nsym_ssb=4,
    nrb_ss=12,
    use_combining=False,
    N_comb=4,
    REMAPPING=None,

):
    """对单个配置 (nrb_ssb, nsym_ssb, nrb_ss) 扫 SNR 生成 BLER 曲线"""

    # 早停配置
    p_goal = 5*1e-3
    min_trials = 500
    bler = []
    rng = np.random.RandomState(seed)
    for snr_db in snr_db_list:

        n_err = 0
        round_total = 0
        mse_sum = None
        with tqdm(total=n_trials, desc=f"SNR={snr_db:>5.1f} dB", ncols=70, leave=False) as pbar:
            for _ in range(n_trials):
                if use_combining:
                    ok, info,mse = one_shot_pbch_min_combined(
                        snr_db, ncellid=ncellid, fs=fs, scs_khz=scs_khz,
                        nrb_ssb=nrb_ssb, nsym_ssb=nsym_ssb, nrb_ss=nrb_ss,
                        ibar_tx=None, rng_seed=rng.randint(1 << 31),
                        N_comb=N_comb,REMAPPING=REMAPPING
                    )
                    if mse_sum is None:
                        mse_sum = mse
                    else:
                        mse_sum += mse
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
                if round_total >= 1000 and n_err <= 1:
                    break
                pbar.update(1)
        bler.append(n_err / round_total)
        print(f"[CFG nrb={nrb_ssb}, nsym={nsym_ssb}, comb={use_combining}, N={N_comb}] "
              f"SNR={snr_db:>5.1f} dB  BLER={bler[-1]:.3f}")
        # mse_sum_48 = mse_sum[0:48,:]
        # mse_per_sym = np.nanmean(mse_sum_48, axis=0) / n_trials # shape = [Nsym]
        #
        # plt.figure(figsize=(8, 4))
        # plt.plot(mse_per_sym, marker='o')
        # plt.xlabel("OFDM Symbol index (l)")
        # plt.ylabel("MSE (1000-run average)")
        # plt.title("PBCH Channel Estimation MSE per symbol (averaged over 1000 runs)")
        #
        # for l, val in enumerate(mse_per_sym):
        #     plt.text(l, val, f"{val:.1e}", ha='center', va='bottom', fontsize=8)
        #
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()
    return np.array(bler)

def _rx_round_get_llr(
    snr_db, ncellid, ibar_tx,
    ssb_grid,               # 发端已映射好的 grid（含 PBCH+DMRS）
    pbch_idx, dmrs_idx, E,  # 索引 & 速率匹配长度
    fs=30.72e6, scs_khz=15, nrb_ssb=20, nsym_ssb=4,
    use_known_nvar=False
):
    # 调制
    ##
    # active_idx = np.sort(np.concatenate([pbch_idx, dmrs_idx]))
    # tx_grid, nVar_ref = add_awgn_on_grid(ssb_grid, snr_db, active_idx)

    # tx_wave, _ = nrOFDMModulate(carrier=None, grid=tx_grid, scs=scs_khz, SampleRate=fs)
    tx_wave, _ = nrOFDMModulate(carrier=None, grid=ssb_grid, scs=scs_khz, SampleRate=fs)

    # 信道
    if NO_CHANNEL:
        ch_out = tx_wave
    else:
        if CDL_CHANNEL:
            ch_out_cdl, H_cdl_timedomain = apply_cdl_c_64x1(tx_wave, fs, ds=300e-9,fc_GHz=7)
            ch_out = ch_out_cdl
        else:
            ch_out, H_tdl_timedomain = apply_tdl_c_time_variable(tx_wave, fs,
                                         speed_kmh=3,
                                         fc_hz=7e9,
                                         tdl_type='TDLC',
                                         tdl_ds_ns=300,
                                         nfft_hint=2048)

    #2048*4+160+144*3

    # # 反变换（干净网格）
    rxGrid_clean = nrOFDMDemodulate(
        waveform=ch_out, nrb=nrb_ssb, scs=scs_khz,
        initialNSlot=0, SampleRate=fs#, CyclicPrefixFraction=0.5
    )[:, :nsym_ssb]
    # rxGrid = rxGrid_clean
    #
    # 在频域网格按 RE-SNR 加噪（避免 RB 依赖）
    active_idx = np.sort(np.concatenate([pbch_idx, dmrs_idx]))
    # rxGrid, nVar_ref = add_awgn_on_grid(rxGrid_clean,ssb_grid, snr_db,active_idx)
    rxGrid, nVar_ref = add_awgn_on_gri_per_symbol(rxGrid_clean, ssb_grid, snr_db, active_idx)
    # 参考导频网格（CE）
    refGridH = np.zeros_like(rxGrid_clean, dtype=complex)
    dmrs_syms = nrPBCHDMRS_param(ncellid, ibar_tx, len(dmrs_idx))
    # print(dmrs_syms)
    nrSetResources(dmrs_idx, refGridH, dmrs_syms)
    # H, nVar_est = nrChannelEstimate(rxGrid=rxGrid, refGrid=refGridH)
    if NO_CHANNEL:
        dmrs_y = nrExtractResources(dmrs_idx, rxGrid)
        err = dmrs_y - dmrs_syms
        nVar_est = float(np.mean(np.abs(err) ** 2))
        nVar = nVar_ref if use_known_nvar else nVar_est
        y_pbch = nrExtractResources(pbch_idx, rxGrid)
        pbch_eq = y_pbch
        llr = nrSymbolDemodulate(pbch_eq, 'QPSK', nVar,  'soft')
        # llr = llr * np.repeat(csi, 2)                            # 保持与单次实现一致

        # nrSetResources(dmrs_idx, refGridH, dmrs_syms)
        # H, nVar_est = nrChannelEstimate(rxGrid=rxGrid, refGrid=refGridH)
        # nVar = nVar_ref if use_known_nvar else nVar_ref
        # # 提取→均衡→软解
        # y_pbch = nrExtractResources(pbch_idx, rxGrid)
        # h_pbch = nrExtractResources(pbch_idx, H)
        # # pbch_eq, csi = nrEqualizeMMSE(y_pbch, h_pbch, nVar)
        # pbch_eq, csi = nrEqualizeMMSE_persym(y_pbch, h_pbch, nVar, pbch_idx, K=nrb_ssb * 12)
        # llr = nrSymbolDemodulate_param_persym(pbch_eq, 'QPSK', nVar, pbch_idx, nrb_ssb * 12, csi, 'soft')
    else:
        if ONLY_TDLC:
            txGrid_id = nrOFDMDemodulate(waveform=tx_wave, nrb=nrb_ssb, scs=scs_khz,
        initialNSlot=0, SampleRate=fs#, CyclicPrefixFraction=0.5
    )[:, :nsym_ssb]
            ch_only =  nrOFDMDemodulate(waveform=ch_out, nrb=nrb_ssb, scs=scs_khz,
        initialNSlot=0, SampleRate=fs#, CyclicPrefixFraction=0.5
    )[:, :nsym_ssb]
            H_true = np.ones_like(ch_only, dtype=complex)
            mask = (np.abs(txGrid_id) > 0)  # 只在实际映射了符号的 RE 上求比值
            H_true[mask] = ch_only[mask] / txGrid_id[mask]
            # H_power =  np.abs(H_true) ** 2
            nVar = nVar_ref if use_known_nvar else nVar_ref

            # 提取→均衡→软解
            y_pbch = nrExtractResources(pbch_idx, rxGrid)
            h_pbch = H_true.ravel(order="F")[pbch_idx]
            pbch_eq, csi = nrEqualizeMMSE_persym(y_pbch, h_pbch, nVar, pbch_idx, K=nrb_ssb * 12)
            # pbch_eq = y_pbch / np.where(np.abs(h_pbch) > 0, h_pbch, 1.0)
            llr = nrSymbolDemodulate_param_persym(pbch_eq, 'QPSK', nVar, pbch_idx, nrb_ssb * 12, csi, 'soft')
            # llr = nrSymbolDemodulate(pbch_eq, 'QPSK', np.mean(nVar),  'soft')
            mse_re = 1
        else:
            nrSetResources(dmrs_idx, refGridH, dmrs_syms)

            # H =myChannelEstimate(
            #     rxGrid=rxGrid, refGrid=refGridH,IdxSC=IdxSC,td_denoise=True  # 有效带宽 RE 下标（列向量或一维数组），如形如 [[644],[645],...,[1383]]
            # )
            # print(np.sum(np.abs(H**2)))
            # nRB_SSB = 20  # 20 RB = 240 subcarriers
            # scs_kHz = 15
            # num_fft = 2048
            # RB_start = 0  # 举例：SSB 起始 RB index，依场景确定
            # IdxSC = (np.arange(nRB_SSB * 12) + RB_start * 12).reshape(-1, 1)
            # H = myChannelEstimatev2(
            #         rxGrid=rxGrid, refSym=dmrs_syms, refGrid=refGridH, carrier=None, snr= snr_db,
            #         # === 新增：LMMSE 相关参数 ===
            #         deltaF=scs_khz*1000,  # 子载波间隔(Hz)
            #         tau_rms=300e-9,  # RMS 时延扩展(秒)
            #         lmmse_model='one_pole',  # 'one_pole' 或 'gau ss'
            #         prg_gran=24,  # 手动指定子带粒度(如 24)。None=自动根据 DMRS 间距推断
            #         # === 时域去噪参数（保留你的原逻辑） ===
            #         td_denoise=True,
            #         IdxSC=IdxSC
            # )

            H = myChannelEstimate_std(rxGrid=rxGrid, refSym=dmrs_syms, refGrid=refGridH, EST_TFDOMAIN=False,
                                      EST_PBCH=True)
            txGrid_id = nrOFDMDemodulate(waveform=tx_wave, nrb=nrb_ssb, scs=scs_khz,
                                         initialNSlot=0, SampleRate=fs  # , CyclicPrefixFraction=0.5
                                         )[:, :nsym_ssb]
            ch_only = nrOFDMDemodulate(waveform=ch_out, nrb=nrb_ssb, scs=scs_khz,
                                       initialNSlot=0, SampleRate=fs  # , CyclicPrefixFraction=0.5
                                       )[:, :nsym_ssb]
            valid_mask = (np.abs(refGridH) > 0) | (np.abs(txGrid_id) > 1e-12)
            H_true = np.ones_like(ch_only, dtype=complex)
            mse_re_full = np.abs(H - H_true) ** 2
            mse_re = np.where(valid_mask, mse_re_full, np.nan)
            # 初始化累计变量

            # H = myChannelEstimate_std(
            #         rxGrid=rxGrid, refSym=dmrs_syms, refGrid=refGridH, carrier=None
            # )
            nVar = nVar_ref
            # 提取→均衡→软解
            y_pbch = nrExtractResources(pbch_idx, rxGrid)
            h_pbch = nrExtractResources(pbch_idx, H)
            # pbch_eq, csi = nrEqualizeMMSE(y_pbch, h_pbch, nVar)
            pbch_eq, csi = nrEqualizeMMSE_persym(y_pbch, h_pbch, nVar, pbch_idx, K=nrb_ssb * 12)
            llr = nrSymbolDemodulate_param_persym(pbch_eq, 'QPSK', nVar, pbch_idx, nrb_ssb * 12, csi, 'soft')


    # llr = nrSymbolDemodulate_param(pbch_eq, 'QPSK', nVar, csi, 'soft')  # 长度应为 E
    # llr = nrSymbolDemodulate(pbch_eq, 'QPSK', nVar,  'soft')
    # llr = llr * np.repeat(csi, 2)                            # 保持与单次实现一致
    # assert len(llr) == E
    return llr,mse_re

def _split_pbch_positions(pbch_idx, nrb_ssb, nsym_ssb, nrb_center=12):
    """
    给定 PBCH 线性索引 pbch_idx（列优先：idx = k + l*K）,
    按 RB 位置划分为：
      - center_pos: 属于“中间 nrb_center 个 RB”的那些 RE 在 pbch_idx 中的下标集合
      - edge_pos  : 属于两边 RB 的那些 RE 在 pbch_idx 中的下标集合

    center_pos / edge_pos 中存的是“第几个 PBCH RE”，即 pbch_idx 自身的 index，而不是频域 index。
    """
    pbch_idx = np.asarray(pbch_idx, dtype=np.int64)
    K = nrb_ssb * 12          # 每个符号子载波数

    # 线性索引 idx = k + l*K -> k
    k = pbch_idx % K          # 子载波索引 0..K-1

    # RB 索引：每 12 个子载波一个 RB
    rb = k // 12              # 0..(nrb_ssb-1)

    # 中间 nrb_center 个 RB，例如 nrb_ssb=20 时 center=4..15
    center_start = (nrb_ssb - nrb_center)//2
    center_end   = center_start + nrb_center

    center_mask = (rb >= center_start) & (rb < center_end)
    center_pos  = np.where(center_mask)[0]
    edge_pos    = np.where(~center_mask)[0]

    return center_pos, edge_pos

def remapping_pbch_bits(pbch_bits, pbch_idx, nrb_ssb, nsym_ssb, mode="edge_first"):
    """
    对 PBCH 编码后的比特做重排。

    假设：
      - QPSK 调制：每个 RE 承载 2 bit。
      - pbch_bits 长度 E = 2*M, 其中 M = len(pbch_idx) 为 PBCH RE 数。

    mode:
      - "straight": 不做重排，原样返回。
      - "reverse" : 将原始比特流的前半段映射到“中间 12RB”上的 RE，
                    后半段映射到“两侧 RB”上的 RE。
                    （中心/边缘的 RE 的几何位置由 pbch_idx + nrb_ssb 决定）
    """
    pbch_bits = np.asarray(pbch_bits).copy()
    E = pbch_bits.size
    assert E % 2 == 0, "目前实现假定 QPSK，每个 RE 2bit"
    M = E // 2

    if mode == "straight":
        return pbch_bits

    pbch_idx = np.asarray(pbch_idx, dtype=np.int64)
    assert pbch_idx.size == M, "pbch_bits 长度与 pbch_idx 数量不匹配"

    center_pos, edge_pos = _split_pbch_positions(pbch_idx, nrb_ssb, nsym_ssb, nrb_center=12)
    nC = center_pos.size
    nE = edge_pos.size
    assert nC > 0 and nE > 0 and nC + nE == M

    # 原始比特流视作：
    #   前 2*nC bit  -> 将要分配给“中间 RB”的那些 RE
    #   后 2*(M-nC)  -> 将要分配给“两侧 RB”的那些 RE
    bits_out = np.empty_like(pbch_bits)

    if mode == "center_first":
        # --- 前半段 -> center_pos ---
        for i, p in enumerate(center_pos):
            src = 2 * i
            dst = 2 * p
            bits_out[dst    ] = pbch_bits[src    ]
            bits_out[dst + 1] = pbch_bits[src + 1]

        # --- 后半段 -> edge_pos ---
        for j, p in enumerate(edge_pos):
            src = 2 * (nC + j)
            dst = 2 * p
            bits_out[dst    ] = pbch_bits[src    ]
            bits_out[dst + 1] = pbch_bits[src + 1]

    elif mode == "edge_first":
        # --- 前半段 -> edge_pos ---
        for i, p in enumerate(edge_pos):
            src = 2 * i
            dst = 2 * p
            bits_out[dst    ] = pbch_bits[src    ]
            bits_out[dst + 1] = pbch_bits[src + 1]

        # --- 后半段 -> center_pos ---
        for j, p in enumerate(center_pos):
            src = 2 * (nE + j)
            dst = 2 * p
            bits_out[dst    ] = pbch_bits[src    ]
            bits_out[dst + 1] = pbch_bits[src + 1]

    return bits_out

def deremapping_pbch_bits(llr_in, pbch_idx, nrb_ssb, nsym_ssb, mode="edge_first"):
    """
    将按 pbch_idx 顺序组织的 LLR 向量，反变换回“原始 PBCH 比特顺序”。

    假设：
      - QPSK: 每 RE 对应 2 个 LLR
      - llr_in 长度 E=2*M, M=len(pbch_idx)

    mode:
      - "straight": 不做重排，原样返回
      - "reverse" : 与 remapping_pbch_bits("reverse") 完全逆运算：
                    前 2*nC 比特对应 center_pos 那些 RE,
                    后 2*(M-nC) 对应 edge_pos 那些 RE。
    """
    llr_in = np.asarray(llr_in)
    E = llr_in.size
    assert E % 2 == 0, "QPSK 场景下 LLR 长度必须是偶数"
    M = E // 2

    if mode == "straight":
        return llr_in.copy()

    pbch_idx = np.asarray(pbch_idx, dtype=np.int64)
    assert pbch_idx.size == M

    center_pos, edge_pos = _split_pbch_positions(pbch_idx, nrb_ssb, nsym_ssb, nrb_center=12)
    nC = center_pos.size
    nE = edge_pos.size

    assert nC > 0 and nC < M

    llr_out = np.empty_like(llr_in)

    if mode == "center_first":
        # --- 中间 RB 的 RE → 前 2*nC 个比特 ---
        for i, p in enumerate(center_pos):
            src = 2 * p
            dst = 2 * i
            llr_out[dst] = llr_in[src]
            llr_out[dst + 1] = llr_in[src + 1]

        # --- 两侧 RB 的 RE → 后半段 ---
        for j, p in enumerate(edge_pos):
            src = 2 * p
            dst = 2 * (nC + j)
            llr_out[dst] = llr_in[src]
            llr_out[dst + 1] = llr_in[src + 1]

    elif mode == "edge_first":
        # --- 两侧 RB 的 RE → 前 2*nE 个比特 ---
        for i, p in enumerate(edge_pos):
            src = 2 * p
            dst = 2 * i
            llr_out[dst] = llr_in[src]
            llr_out[dst + 1] = llr_in[src + 1]

        # --- 中间 RB 的 RE → 后半段 ---
        for j, p in enumerate(center_pos):
            src = 2 * p
            dst = 2 * (nE + j)
            llr_out[dst] = llr_in[src]
            llr_out[dst + 1] = llr_in[src + 1]

    else:
        raise ValueError(f"Unknown deremap mode: {mode}")

    return llr_out


def one_shot_pbch_min_combined(
    snr_db, ncellid,
    fs=30.72e6, scs_khz=15,
    nrb_ssb=20, nsym_ssb=4, nrb_ss=12,
    ibar_tx=None, rng_seed=None,
    N_comb=4, REMAPPING = False
):
    if rng_seed is not None:
        np.random.seed(rng_seed)
    if ibar_tx is None:
        ibar_tx = np.random.randint(0,8)

    # 生成 32 比特 TB（一次，保证每轮发同一块，Chase Combining）
    A = 32
    trblk_bits = np.random.randint(0,2,A).astype(int)

    # 发端网格（含 PBCH & DMRS），以及索引/E
    ssb_grid, meta = build_ssb_grid_pbch_only(
        ncellid, ibar_tx, trblk_bits,
        nrb_ssb=nrb_ssb, nsym_ssb=nsym_ssb, nrb_ss=nrb_ss, REMAPPING=REMAPPING
    )
    pbch_idx = meta['pbch_idx']; dmrs_idx = meta['dmrs_idx']; E = meta['E']

    # assert len(np.intersect1d(pbch_idx, dmrs_idx)) == 0
    # assert np.all(np.diff(pbch_idx) > 0)
    # assert np.all(np.diff(dmrs_idx) > 0)


    # 多轮接收，逐轮得 LLR
    llr_sum = None
    mse_sum = None
    for r in range(N_comb):
        llr_r,mse = _rx_round_get_llr(
            snr_db, ncellid, ibar_tx,
            ssb_grid, pbch_idx, dmrs_idx, E,
            fs=fs, scs_khz=scs_khz, nrb_ssb=nrb_ssb, nsym_ssb=nsym_ssb,
            use_known_nvar=USE_KNOWN_NVAR
        )
        if llr_sum is None:
            llr_sum = llr_r.astype(np.float64, copy=True)
            mse_sum = mse
        else:
            llr_sum += llr_r
            mse_sum += mse

    # Polar 逆速配 & 译码（与单次保持一致）
    P = 24; K = 32 + P
    # 你的实现里 N 用 2**min(floor(log2(E)),9)；保持一致
    N = 2 ** min(int(np.log2(E)), 9)

    # === 在逆速配之前做 LLR 反重排 ===
    LLR_deremap = deremapping_pbch_bits(
            llr_sum,
            pbch_idx=pbch_idx,
            nrb_ssb=nrb_ssb,
            nsym_ssb=nsym_ssb,
            mode=REMAPPING
    )

    # decIn1 = nrRateRecoverPolar(llr_sum, K, N, False, discardRepetition=False)
    decIn = nrRateRecoverPolar_mh(LLR_deremap, K, N, False, discardRepetition=False) ##


    # print(len(pbch_idx))
    # print(len(llr_descr))
    if USE_SCL:
        decoded_bits,crc_1 = polar_decode_scl_llr(decIn, K=K, N=N, list_size=8, crc_degree="CRC24C")
        _, crc = nrCRCDecode(decoded_bits, '24C')
    else:
        decoded = nrPolarDecode(decIn, K, 0, 0, nmax=min(int(np.log2(E)), 9))
        _, crc = nrCRCDecode(decoded, '24C')
        # crc_ok = (crc == 0)

    # decIn = nrRateRecoverPolar(llr_sum, K, N, False, discardRepetition=False)
    # decoded = nrPolarDecode(decIn, K, 0, 0, nmax=min(int(np.log2(E)), 9))
    # _, crc = nrCRCDecode(decoded, '24C')
    return (crc == 0), dict(N_comb=N_comb, E=E, ibar_tx=ibar_tx),mse_sum

# ---------------------------
# 直接执行：多配置仿真 + 存图
# ---------------------------
if __name__ == "__main__":

    snr_points =  [-8.5,-8,-7.5]  # 也可以换回你之前的段, -13,   `-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,0,1,2,3,4,5,6,8,10

    # 要对比的配置（标签, nrb_ssb, nsym_ssb, nrb_ss）
    # 注意：nrb_ss 不要大于 nrb_ssb；这里统一用 12 个 RB 的中心带宽，
    # 在 nrb_ssb = 12 的场景会让 PSS/SSS 覆盖整个带宽（合法，只是 PBCH 在对应符号上被挖空）
    configs = [
        # ("12RB × 4sym 1 comb", 12, 4, 12, 1),
        # ("20RB × 4sym 1 comb", 20, 4, 12, 1),
        # ("20RB × 4sym 2 comb" , 20,  4, 12, 2),
        # ("12RB × 6sym 1 comb" , 12,  6, 12, 1),
        # ("12RB × 12sym", 12, 12, 12),
        # ("50RB × 4sym" , 50,  4, 12),
        # ("50RB × 12sym", 50, 12, 12),

        # ("24RB × 4sym 1 comb", 24, 4, 12, 1),

        # ("24RB × 4sym 1 comb", 24, 4, 24, 1),
        # ("22RB × 4sym 1 comb", 22, 4, 18, 1),
        # ("20RB × 4sym 1 comb", 20, 4, 12, 1),
        # ("18RB × 4sym 1 comb", 18, 4, 6, 1),
        # ("12RB × 6sym 1 comb", 12, 6, 12, 1),
        #
        ("20RB × 4sym 1 comb straight", 20, 4, 12, 1, "straight"),
        ("20RB × 4sym 1 comb edge_first", 20, 4, 12, 1, "edge_first"),
        ("20RB × 4sym 1 comb center_first", 20, 4, 12, 1, "center_first"),
        # ("18RB × 4sym 1 comb", 18, 4, 6, 1),
        # ("24RB × 4sym 1 comb", 24, 4, 24, 1),
        # ("16RB × 5sym 1 comb", 16, 5, 16, 1),
        # ("12RB × 6sym 1 comb", 12, 6, 12, 1),
        # # ("8RB × 8sym 1 comb", 8, 8, 8, 1),
        # # ("6RB × 10sym 1 comb", 6, 10, 6, 1),
        # ("12RB × 4sym 1 comb", 12, 4, 12, 1),
        # # ("12RB × 6sym 1 comb", 12, 6, 12, 1),
        # ("20RB × 4sym 2 comb", 20,  4, 12, 2),

    ]

    curves = {}
    for label, nrb_cfg, nsym_cfg, nrb_ss_cfg, comb_cfg,rmp_cfg in configs:
        bler_curve = simulate_bler_config(
            snr_points,
            n_trials=10000,          # 你可以调大/调小
            ncellid=208,
            fs=30.72e6,
            scs_khz=15,
            seed=3055,
            nrb_ssb=nrb_cfg,
            nsym_ssb=nsym_cfg,
            nrb_ss=nrb_ss_cfg,
            use_combining=True,
            N_comb=comb_cfg,
            REMAPPING=rmp_cfg,
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
    plt.savefig("pbch_test_CE_rx_SNR_remapping_idealtdl.png", dpi=150)
    print("多配置 BLER 曲线已保存到 pbch_test_CE_rx_SNR_remapping_idealtdl.png")
    # === 保存数值结果 ===
    import csv
    with open("pbch_test_CE_rx_SNR_remapping_idealtdl.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Config", "SNR(dB)", "BLER"])
        for label, bler_curve in curves.items():
            for snr, bler_val in zip(snr_points, bler_curve):
                writer.writerow([label, snr, bler_val])
    print("数值结果已保存到 pbch_test_CE_rx_SNR_remapping_idealtdl.csv")