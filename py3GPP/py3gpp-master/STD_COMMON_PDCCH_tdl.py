# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 14:23:13 2025

 File name: STD_COMMON_PDCCH_tdl
 Author: m00829866	Version: 0.1	Date: 2025-11-07
 Description:
     1. standard common PDCCH Link Level Simulation

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
    nrpdcchIndices, nrpdcchDMRS, nrpdcchDMRSIndices,
    nrpdcchPRBS, nrChannelEstimate, nrEqualizeMMSE,
    nrSymbolModulate, nrSymbolDemodulate, nrpdcchIndices_param,
    nrRateRecoverPolar, nrPolarDecode, nrCRCDecode,
    nrBCH
)
'''
from channel_initialize import channel_initialize,channel_settings_initialize,channel_cdl_initialize,rotate_angles_max_direction,scaling_angles,calculate_delay_spread,calculate_angular_spread
from channel_functions import channel_fading_signal, channel_response_generate, calculate_antenna_pattern, get_dft_codebook, channel_virtualization, channel_time_interpolation, channel_frequency_response
from nrPDCCHIndicesVariable import nrPDCCHIndices_param,nrPDCCHDMRSIndices_param,nrPDCCHDMRS_param,nrPDCCH_param,nrSymbolDemodulate_param,nrSymbolDemodulate_param_persym,_dci_crc24c_with_rnti_mask,nrRateRecoverPolar_mh
from tqdm import tqdm
from ext_scl2.scl_adapter import polar_decode_scl_llr
from GenTDLChannel import ChannelInfo
from myChannelEstimation import DMRSFilterGenerate_v2_explicit,myChannelEstimate,myChannelEstimate_std
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
ONLY_TDLC = False
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
def add_awgn_on_grid(rxGrid_clean, pdcch_grid,  snr_db_re, active_idx):
    """
    在频域网格上直接加噪，snr_db_re 是“每 RE 的目标 SNR (Es/N0，复符号功率口径)”
    """
    snr_lin = 10**(snr_db_re/10.0)
    # 以“每个 RE 的平均信号功率”为口径配噪
    flat = pdcch_grid.ravel(order="F")
    # flat = rxGrid_clean.ravel(order="F")
    p_sig = np.mean(np.abs(flat[active_idx]) ** 2)
    nvar_complex = p_sig / snr_lin                          # 复符号噪声功率
    noise = (np.random.randn(*pdcch_grid.shape) + 1j*np.random.randn(*pdcch_grid.shape)) \
            * np.sqrt(nvar_complex/2.0)
    return rxGrid_clean + noise, nvar_complex
def add_awgn_on_gri_per_symbol(rxGrid_clean, pdcch_grid,  snr_db_re, active_idx):
    """
    逐 OFDM 符号统计信号功率并加噪：
      - snr_db_re: 以“每 RE 的发射口径 Es/N0 (dB)”定义
      - active_idx: 列优先的一维索引（如 pdcch∪DMRS），用于该符号内 p_sig 估计
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
        flat = pdcch_grid.ravel(order="F")
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
    # noise = (np.random.randn(*pdcch_grid.shape) + 1j*np.random.randn(*pdcch_grid.shape)) \
    #         * np.sqrt(nvar_complex/2.0)
    return rxGrid_clean + noise, nvar_cols

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
def nrEqualizeMMSE_persym(rxSym, hest, nVar,pdcch_idx,K):
    # 每个 RE 属于哪一列（符号索引 ℓ）
    sym_ids = (pdcch_idx // K).astype(np.int64)  # ℓ in [0, N-1]
    # 给每个 RE 取对应列的噪声方差
    nVar_re = nVar[sym_ids]  # (M,)

    # MMSE 等化
    a = np.abs(hest * np.conj(hest))
    csi = a + nVar_re  # 分母
    # 防止极端零分母
    csi = np.where(csi == 0, 1e-30, csi)
    rxEq = rxSym * np.conj(hest) / csi

    return rxEq, csi
def pdcch_llr_descramble(llr: np.ndarray, n_id_scram: int, n_rnti_scram: int) -> np.ndarray:
    """
    LLR 域解扰码：c_init = (n_rnti_scram<<16) + n_id_scram
    llr: shape=(E,) or (E,1)
    """
    E = int(llr.size)
    c_init = (int(n_rnti_scram) << 16) + int(n_id_scram)
    c = nrPRBS(c_init, E).astype(np.uint8)        # 0/1
    s = (1 - 2*c).astype(np.int8)                # 0->+1, 1->-1
    return (llr.reshape(-1) * s).astype(np.float32)
def _crc24_only(bits: np.ndarray) -> np.ndarray:
    out = nrCRCEncode(bits.astype(np.int8), "24C").reshape(-1).astype(np.uint8)
    return out[-24:] if out.size != 24 else out
def crc_unmask_and_check(payload_hat: np.ndarray, crc_masked_hat: np.ndarray, rnti_crc: int) -> bool:
    rnti_bits16 = np.array([(rnti_crc >> i) & 1 for i in range(16)], dtype=np.uint8)
    mask24 = np.concatenate([rnti_bits16, rnti_bits16[:8]]).astype(np.uint8)
    crc_hat  = np.bitwise_xor(crc_masked_hat.astype(np.uint8), mask24)   # 解掩码后的 24bit CRC
    crc_calc = _crc24_only(payload_hat.astype(np.uint8))                 # **保证就是24位**
    return np.array_equal(crc_hat, crc_calc)
# ---------------------------
# 发端：只构造 pdcch + pdcch DMRS
# ---------------------------
def build_sib_grid_pdcch_only(ncellid, trblk_bits, nrb_pdcch=48, nsym_pdcch=2):
    grid = np.zeros((nrb_pdcch*12, nsym_pdcch), dtype=complex)
    # 统一索引
    pdcch_idx = nrPDCCHIndices_param(nrb_pdcch=nrb_pdcch, nsym_pdcch=nsym_pdcch, comb=4)
    dmrs_idx = nrPDCCHDMRSIndices_param(nrb_pdcch=nrb_pdcch, nsym_pdcch=nsym_pdcch, comb=4)

    E = len(pdcch_idx)*2

    # 编码 -> 速配(E) -> QPSK

    pdcch_bits = nrPDCCH_param(trblk_bits, rnti_crc=0xFFFF, n_id_scram=ncellid, n_rnti_scram=0, E=E)
    pdcch_symb = nrSymbolModulate(pdcch_bits, 'QPSK')
    assert len(pdcch_symb) == len(pdcch_idx)
    # DMRS
    dmrs_symb = nrPDCCHDMRS_param(ncellid,len(dmrs_idx))
    # print(dmrs_symb)
    assert len(dmrs_symb) == len(dmrs_idx)

    nrSetResources(pdcch_idx, grid, pdcch_symb)
    # nrSetResources(nrpdcchIndices(ncellid), grid, pdcch_symb)
    nrSetResources(dmrs_idx, grid, dmrs_symb)
    # nrSetResources(nrpdcchDMRSIndices(ncellid), grid, nrpdcchDMRS(ncellid, ibar_ssb))
    # if POWER_NORM_PER_RE and nrb_pdcch > 0:
    #     scale = np.sqrt(REF_NRB_FOR_POWER / float(nrb_pdcch))
    #     grid *= scale
    meta = dict(pdcch_idx=pdcch_idx, dmrs_idx=dmrs_idx, E=E,pdcch_bits=pdcch_bits)
    return grid, meta

## TODO: check PDCCH and PDSCH procedure in SPEC, and find all the difference between our code and spec
def simulate_bler_config(
    snr_db_list,
    n_trials=200,
    ncellid=210,
    fs=30.72e6,
    scs_khz=15,
    seed=2025,
    nrb_pdcch=20,
    nsym_pdcch=4,
    use_combining=False,
    N_comb=4
):
    """对单个配置 (nrb_pdcch, nsym_pdcch, nrb_ss) 扫 SNR 生成 BLER 曲线"""

    # 早停配置
    p_goal = 5*1e-3
    min_trials = 500
    bler = []
    rng = np.random.RandomState(seed)
    for snr_db in snr_db_list:
        n_err = 0
        round_total = 0
        with tqdm(total=n_trials, desc=f"SNR={snr_db:>5.1f} dB", ncols=70, leave=False) as pbar:
            for _ in range(n_trials):
                if use_combining:
                    ok, info = one_shot_pdcch_min_combined(
                        snr_db, ncellid=ncellid, fs=fs, scs_khz=scs_khz,
                        nrb_pdcch=nrb_pdcch, nsym_pdcch=nsym_pdcch,
                        ibar_tx=None, rng_seed=rng.randint(1 << 31),
                        N_comb=N_comb
                    )
                else:
                    # ok, info = one_shot_pdcch_min(
                    #     snr_db, ncellid=ncellid, fs=fs, scs_khz=scs_khz,
                    #     nrb_pdcch=nrb_pdcch, nsym_pdcch=nsym_pdcch, nrb_ss=nrb_ss,
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
        print(f"[CFG nrb={nrb_pdcch}, nsym={nsym_pdcch}, comb={use_combining}, N={N_comb}] "
              f"SNR={snr_db:>5.1f} dB  BLER={bler[-1]:.3f}")
    return np.array(bler)

def _rx_round_get_llr(
    snr_db, ncellid, ibar_tx,
    pdcch_grid,               # 发端已映射好的 grid（含 pdcch+DMRS）
    pdcch_idx, dmrs_idx, E,  # 索引 & 速率匹配长度
    fs=30.72e6, scs_khz=15, nrb_pdcch=48, nsym_pdcch=2,
    use_known_nvar=False
):
    # 调制
    ##
    # active_idx = np.sort(np.concatenate([pdcch_idx, dmrs_idx]))
    # tx_grid, nVar_ref = add_awgn_on_grid(pdcch_grid, snr_db, active_idx)

    # tx_wave, _ = nrOFDMModulate(carrier=None, grid=tx_grid, scs=scs_khz, SampleRate=fs)
    tx_wave, _ = nrOFDMModulate(carrier=None, grid=pdcch_grid, scs=scs_khz, SampleRate=fs)

    # 信道
    if NO_CHANNEL:
        ch_out = tx_wave
    else:
        if CDL_CHANNEL:
            ch_out_cdl, H_cdl_timedomain = apply_cdl_c_64x1(tx_wave, fs, ds=300e-9,fc_GHz=7)
            ch_out = ch_out_cdl
        else:
            ch_out, H_tdl_timedomain = apply_tdl_c(tx_wave, ds=300e-9,fs_hz=fs)
    # Nfft=2048
    # Nsc = 240
    # H_tdl_timedomain_padded = np.pad(H_tdl_timedomain, (0, Nfft - len(H_tdl_timedomain)), mode='constant')
    # H_fd_full = np.fft.fft(H_tdl_timedomain_padded, Nfft)
    # H_fd_shift = np.fft.fftshift(H_fd_full)
    # start = Nfft // 2 - Nsc // 2
    # H_fd_sel = H_fd_shift[start:start + Nsc]

    #2048*4+160+144*3

    # # 反变换（干净网格）
    rxGrid_clean = nrOFDMDemodulate(
        waveform=ch_out, nrb=nrb_pdcch, scs=scs_khz,
        initialNSlot=0, SampleRate=fs#, CyclicPrefixFraction=0.5
    )[:, :nsym_pdcch]
    # rxGrid = rxGrid_clean
    #
    # 在频域网格按 RE-SNR 加噪（避免 RB 依赖）
    active_idx = np.sort(np.concatenate([pdcch_idx, dmrs_idx]))
    # rxGrid, nVar_ref = add_awgn_on_grid(rxGrid_clean,pdcch_grid, snr_db,active_idx)
    rxGrid, nVar_ref = add_awgn_on_gri_per_symbol(rxGrid_clean, pdcch_grid, snr_db, active_idx)
    # 参考导频网格（CE）
    refGridH = np.zeros_like(rxGrid_clean, dtype=complex)
    dmrs_syms = nrPDCCHDMRS_param(ncellid=ncellid, num_re=len(dmrs_idx))
    # print(dmrs_syms)
    nrSetResources(dmrs_idx, refGridH, dmrs_syms)
    # H, nVar_est = nrChannelEstimate(rxGrid=rxGrid, refGrid=refGridH)
    if NO_CHANNEL:
        dmrs_y = nrExtractResources(dmrs_idx, rxGrid)
        err = dmrs_y - dmrs_syms
        nVar_est = float(np.mean(np.abs(err) ** 2))
        nVar = nVar_ref if use_known_nvar else nVar_est
        y_pdcch = nrExtractResources(pdcch_idx, rxGrid)
        pdcch_eq = y_pdcch
        llr = nrSymbolDemodulate(pdcch_eq, 'QPSK', nVar,  'soft')
        c_init = (int(0) << 16) + int(ncellid) % (1 << 31)
        scr = nrPRBS(c_init, E).astype(np.uint8)
        scr_bits = nrPRBS(c_init, E).astype(np.int8)  # 0/1
        s = 1 - 2 * scr_bits  # 1/-1
        llr_descr = llr * s
        # llr_descr = pdcch_llr_descramble(llr[:E], n_id_scram=ncellid, n_rnti_scram=0)
        # llr = llr * np.repeat(csi, 2)                            # 保持与单次实现一致

        # nrSetResources(dmrs_idx, refGridH, dmrs_syms)
        # H, nVar_est = nrChannelEstimate(rxGrid=rxGrid, refGrid=refGridH)
        # nVar = nVar_ref if use_known_nvar else nVar_ref
        # # 提取→均衡→软解
        # y_pdcch = nrExtractResources(pdcch_idx, rxGrid)
        # h_pdcch = nrExtractResources(pdcch_idx, H)
        # # pdcch_eq, csi = nrEqualizeMMSE(y_pdcch, h_pdcch, nVar)
        # pdcch_eq, csi = nrEqualizeMMSE_persym(y_pdcch, h_pdcch, nVar, pdcch_idx, K=nrb_pdcch * 12)
        # llr = nrSymbolDemodulate_param_persym(pdcch_eq, 'QPSK', nVar, pdcch_idx, nrb_pdcch * 12, csi, 'soft')
    else:
        if ONLY_TDLC:
            txGrid_id = nrOFDMDemodulate(waveform=tx_wave, nrb=nrb_pdcch, scs=scs_khz,
        initialNSlot=0, SampleRate=fs#, CyclicPrefixFraction=0.5
    )[:, :nsym_pdcch]
            ch_only =  nrOFDMDemodulate(waveform=ch_out, nrb=nrb_pdcch, scs=scs_khz,
        initialNSlot=0, SampleRate=fs#, CyclicPrefixFraction=0.5
    )[:, :nsym_pdcch]
            H_true = np.ones_like(ch_only, dtype=complex)
            mask = (np.abs(txGrid_id) > 0)  # 只在实际映射了符号的 RE 上求比值
            H_true[mask] = ch_only[mask] / txGrid_id[mask]
            H_power =  np.abs(H_true) ** 2
            nVar = nVar_ref if use_known_nvar else nVar_ref

            # 提取→均衡→软解
            y_pdcch = nrExtractResources(pdcch_idx, rxGrid)
            h_pdcch = H_true.ravel(order="F")[pdcch_idx]
            pdcch_eq, csi = nrEqualizeMMSE_persym(y_pdcch, h_pdcch, nVar, pdcch_idx, K=nrb_pdcch * 12)
            llr = nrSymbolDemodulate_param_persym(pdcch_eq, 'QPSK', nVar, pdcch_idx, nrb_pdcch * 12, csi, 'soft')
            c_init = (int(0) << 16) + int(ncellid) % (1 << 31)
            scr_bits = nrPRBS(c_init, E).astype(np.int8)  # 0/1
            s = 1 - 2 * scr_bits  # 1/-1
            llr_descr = llr * s
        else:

            nrSetResources(dmrs_idx, refGridH, dmrs_syms)
            # H_old, nVar_est = nrChannelEstimate(rxGrid=rxGrid, refGrid=refGridH)
            # filters_per_port = DMRSFilterGenerate_v2_explicit(
            nRB_pdcch = 20  # 20 RB = 240 subcarriers
            scs_kHz = 15
            num_fft = 2048
            RB_start = 0  # 举例：SSB 起始 RB index，依场景确定
            IdxSC = (np.arange(nRB_pdcch * 12) + RB_start * 12).reshape(-1, 1)
            # H =myChannelEstimate(
            #     rxGrid=rxGrid, refGrid=refGridH,IdxSC=IdxSC,td_denoise=True  # 有效带宽 RE 下标（列向量或一维数组），如形如 [[644],[645],...,[1383]]
            # )
            H = myChannelEstimate_std(rxGrid=rxGrid, refSym=dmrs_syms, refGrid=refGridH,EST_TFDOMAIN = True, EST_PBCH = False, method = 'interpolation',rb_size = 3, N_sym=2)
            # H = myChannelEstimate_std(
            #     rxGrid=rxGrid, refSym=dmrs_syms, refGrid=refGridH, carrier=None
            # )
            #         scs_khz=scs_khz, dmrs_ports=(0,), delay_spread_ns=300.0, snr_db=snr_db, spans=(12, 24, 36, 48)
            #     )
            # dmrs_filter = []
            # num_layer = 1
            # for _ in range(num_layer):
            #         dmrs_filter.append(filters_per_port[0])
            # H, nVar_est = myChannelEstimate(
            #     rxGrid=rxGrid,
            #     refGrid=refGridH,
            #     dmrs_filter=dmrs_filter,  # 你之前生成好的滤波器
            #     HestGran=24  # 要和 dmrs_filter 的档位一致；常用 12/24/36/48
            # )
            nVar = nVar_ref
            # 提取→均衡→软解
            y_pdcch = nrExtractResources(pdcch_idx, rxGrid)
            h_pdcch = nrExtractResources(pdcch_idx, H)
            # pdcch_eq, csi = nrEqualizeMMSE(y_pdcch, h_pdcch, nVar)
            pdcch_eq, csi = nrEqualizeMMSE_persym(y_pdcch, h_pdcch, nVar, pdcch_idx, K=nrb_pdcch * 12)
            llr = nrSymbolDemodulate_param_persym(pdcch_eq, 'QPSK', nVar, pdcch_idx, nrb_pdcch * 12, csi, 'soft')
            c_init = (int(0) << 16) + int(ncellid) % (1 << 31)
            scr_bits = nrPRBS(c_init, E).astype(np.int8)  # 0/1
            s = 1 - 2 * scr_bits  # 1/-1
            llr_descr = llr * s

    # llr = nrSymbolDemodulate_param(pdcch_eq, 'QPSK', nVar, csi, 'soft')  # 长度应为 E
    # llr = nrSymbolDemodulate(pdcch_eq, 'QPSK', nVar,  'soft')
    # llr = llr * np.repeat(csi, 2)                            # 保持与单次实现一致
    assert len(llr) == E
    return llr_descr

def one_shot_pdcch_min_combined(
    snr_db, ncellid,
    fs=30.72e6, scs_khz=15,
    nrb_pdcch=20, nsym_pdcch=4,
    ibar_tx=None, rng_seed=None,
    N_comb=4
):
    if rng_seed is not None:
        np.random.seed(rng_seed)
    if ibar_tx is None:
        ibar_tx = np.random.randint(0,8)

    # 生成 32 比特 TB（一次，保证每轮发同一块，Chase Combining）
    A = 40
    trblk_bits = np.random.randint(0,2,A).astype(int)

    # 发端网格（含 pdcch & DMRS），以及索引/E
    pdcch_grid, meta = build_sib_grid_pdcch_only(
        ncellid, trblk_bits,
        nrb_pdcch=nrb_pdcch, nsym_pdcch=nsym_pdcch
    )
    pdcch_idx = meta['pdcch_idx']; dmrs_idx = meta['dmrs_idx']; E = meta['E']
    pdcch_bits = meta['pdcch_bits']

    assert len(np.intersect1d(pdcch_idx, dmrs_idx)) == 0
    assert np.all(np.diff(pdcch_idx) > 0)
    assert np.all(np.diff(dmrs_idx) > 0)


    # 多轮接收，逐轮得 LLR
    llr_sum = None
    for r in range(N_comb):
        llr_r = _rx_round_get_llr(
            snr_db, ncellid, ibar_tx,
            pdcch_grid, pdcch_idx, dmrs_idx, E,
            fs=fs, scs_khz=scs_khz, nrb_pdcch=nrb_pdcch, nsym_pdcch=nsym_pdcch,
            use_known_nvar=USE_KNOWN_NVAR
        )
        if llr_sum is None:
            llr_sum = llr_r.astype(np.float64, copy=True)
        else:
            llr_sum += llr_r

    # Polar 逆速配 & 译码（与单次保持一致）
    P = 24; K = A + P
    # 你的实现里 N 用 2**min(floor(log2(E)),9)；保持一致
    N = 2 ** min(int(np.log2(E)), 9)


    # decIn = nrRateRecoverPolar(llr_sum, K, N, False, discardRepetition=False)
    decIn = nrRateRecoverPolar_mh(llr_sum, K, N, False, discardRepetition=False)
    # print(len(pdcch_idx))
    # print(len(llr_descr))
    if USE_SCL:
        decoded_bits,crc_1 = polar_decode_scl_llr(decIn, K=K, N=N, list_size=8, crc_degree="CRC24C")
        _, crc = nrCRCDecode(decoded_bits, '24C')
        ok = crc_unmask_and_check(decoded_bits[:40], decoded_bits[40:40+24], rnti_crc=0xFFFF)
    else:
        decoded = nrPolarDecode(decIn, K, 0, 0, nmax=min(int(np.log2(E)), 9))
        _, crc = nrCRCDecode(decoded, '24C')
        # crc_ok = (crc == 0)

    # decIn = nrRateRecoverPolar(llr_sum, K, N, False, discardRepetition=False)
    # decoded = nrPolarDecode(decIn, K, 0, 0, nmax=min(int(np.log2(E)), 9))
    # _, crc = nrCRCDecode(decoded, '24C')
    return ok, dict(N_comb=N_comb, E=E, ibar_tx=ibar_tx)

# ---------------------------
# 直接执行：多配置仿真 + 存图
# ---------------------------
if __name__ == "__main__":
    # 建议用稍“温和”的 SNR 段，便于拉开曲线
    # print_nmse_compare_db(snr_db=-2.0, ncellid=208, fs=30.72e6, scs_khz=30, trials_each=300)
    snr_points =  [-8, -7, -6, -5]  # 也可以换回你之前的段, -13,   `-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,0,1,2,3,4,5,6,8,10

    # 要对比的 7 组配置（标签, nrb_pdcch, nsym_pdcch, nrb_ss）
    # 注意：nrb_ss 不要大于 nrb_pdcch；这里统一用 12 个 RB 的中心带宽，
    # 在 nrb_pdcch = 12 的场景会让 PSS/SSS 覆盖整个带宽（合法，只是 pdcch 在对应符号上被挖空）
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
        ("48RB × 2sym 1 comb", 48, 2, 12, 1),
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
    for label, nrb_cfg, nsym_cfg, nrb_ss_cfg, comb_cfg in configs:
        bler_curve = simulate_bler_config(
            snr_points,
            n_trials=5000,          # 你可以调大/调小
            ncellid=208,
            fs=30.72e6,
            scs_khz=30,
            seed=2029,
            nrb_pdcch=nrb_cfg,
            nsym_pdcch=nsym_cfg,
            use_combining=True,
            N_comb=comb_cfg
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
    plt.title(f"pdcch BLER vs SNR  (NO_CHANNEL={NO_CHANNEL}, KNOWN_nVar={USE_KNOWN_NVAR})")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("pdcch_test_CE_rx_SNR_tdl_CE_denoise.png", dpi=150)
    print("多配置 BLER 曲线已保存到 pdcch_test_CE_rx_SNR_tdl_CE_denoise.png")
    # === 保存数值结果 ===
    import csv
    with open("pdcch_test_CE_rx_SNR_tdl_CE_denoise.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Config", "SNR(dB)", "BLER"])
        for label, bler_curve in curves.items():
            for snr, bler_val in zip(snr_points, bler_curve):
                writer.writerow([label, snr, bler_val])
    print("数值结果已保存到 pdcch_test_CE_rx_SNR_tdl_CE_denoise.csv")