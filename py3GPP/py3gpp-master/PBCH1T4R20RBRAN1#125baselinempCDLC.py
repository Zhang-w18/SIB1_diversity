import os
# 🚨 屏蔽杀手：必须放在所有 import 之前！
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 彻底蒙住程序的眼睛，让它以为这台机器没有 GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # 强行闭嘴：屏蔽所有底层的 Info/Warning/Error 日志
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
from py3gpp import *
import matplotlib.pyplot as plt
import time

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
from nrPBCHIndicesVariable import nrPBCHIndices_param, nrPBCHDMRSIndices_param, nrPBCHDMRS_param, nrBCH_param, \
    nrSymbolDemodulate_param, nrSymbolDemodulate_param_persym, nrRateRecoverPolar_mh, nrRateMatchPolar_mh
from tqdm import tqdm
from channel_initialize import channel_initialize
from channel_functions import channel_response_generate, channel_time_interpolation
from myChannelEstimation4 import myChannelEstimate_std
from channel_functions import get_dft_codebook
from ext_scl2.scl_adapter import polar_decode_scl_llr
from ext_scl.encoding import PolarEncoder, Polar5GEncoder
from ext_scl.decoding import PolarSCLDecoder, Polar5GDecoder
# from myChannelEstimation41 import DMRSFilterGenerate_v2_explicit, myChannelEstimate, myChannelEstimatev2, \
#     myChannelEstimate_std
import GenTDLChannel
from GenTDLChannel import ChannelInfo
from numba import njit, prange
import concurrent.futures
# ---------------------------
# 开关 & 辅助
# ---------------------------
NO_CHANNEL = False  # 先用纯 AWGN 验证链路 ——> True
USE_KNOWN_NVAR = False  # 用 SNR 推噪声方差 ——> True
PRINT_DIAG = False  # True          # 打印关键诊断信息
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


def add_awgn_on_gri_per_symbol(rxGrid_clean, ssb_grid, snr_db_re, active_idx):
    """
    1T4R 物理正确版：基于发端 (Tx) 能量，逐符号 (per symbol) 计算底噪
    """
    K, N, nRx = rxGrid_clean.shape
    snr_lin = 10 ** (snr_db_re / 10.0)
    
    # 将无衰落的发射网格 ssb_grid 展平，用于统计真实的发射功率
    flat_tx = ssb_grid.ravel(order="F")
    
    # 预计算全局发射功率（当某列无 active RE 时做回退）
    p_sig_global_tx = np.mean(np.abs(flat_tx[active_idx]) ** 2)
    
    nvar_cols = np.zeros(N, dtype=float)
    
    for l in range(N):
        # 寻找该符号列 (symbol l) 内的 active RE
        in_col = (active_idx >= l * K) & (active_idx < (l + 1) * K)
        
        if np.any(in_col):
            # 统计 发送端 ssb_grid 在该符号有效 RE 上的能量
            p_sig_l_tx = np.mean(np.abs(flat_tx[active_idx[in_col]]) ** 2)
        else:
            p_sig_l_tx = p_sig_global_tx
            
        # 真实的物理底噪，由发送端能量唯一确定
        nvar_cols[l] = p_sig_l_tx / snr_lin

    # 扩充维度，为 4 根天线生成独立同分布的噪声
    nvar_expanded = nvar_cols[None, :, None]
    noise = (np.random.randn(K, N, nRx) + 1j * np.random.randn(K, N, nRx)) * np.sqrt(nvar_expanded / 2.0)
    
    return rxGrid_clean + noise, nvar_cols

# --- §5.4.1.1 Sub-block interleaving ---
def nrEqualizeMMSE_persym(rxSym, hest, nVar, pbch_idx, K):
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



def build_wbf_from_channel_codebook(qv=0, qh=0, M=4, N=4, P=2):
    """
    Build dual-polarized DFT beamforming matrix aligned with get_dft_codebook().

    TX antenna order:
      [pol0 M*N elements, pol1 M*N elements]

    Spatial order within each polarization:
      antenna index = mv * N + mh

    Beam row order:
      beam index = qv * N + qh
    """
    assert P == 2, "This function assumes dual polarization P=2."

    qv = int(qv) % M
    qh = int(qh) % N

    C = get_dft_codebook(M, N, P, dft_offset=0)

    num_spatial = M * N

    beam_idx = qv * N + qh

    pol0_row = beam_idx
    pol1_row = beam_idx + num_spatial

    w0 = C[pol0_row, :].astype(np.complex128)
    w1 = C[pol1_row, :].astype(np.complex128)

    w0 = w0 / np.linalg.norm(w0)
    w1 = w1 / np.linalg.norm(w1)

    Wbf = np.column_stack([w0, w1])

    assert Wbf.shape == (M * N * P, 2), f"Unexpected Wbf shape: {Wbf.shape}"
    assert np.allclose(Wbf.conj().T @ Wbf, np.eye(2), atol=1e-10), \
        "Wbf columns are not orthonormal."

    return Wbf


def bb_precoder_traditional():
    return np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)


def bb_precoder_cyclic(prg_idx, phase_table=(0, np.pi / 2, np.pi, 3 * np.pi / 2)):
    phi = phase_table[prg_idx % len(phase_table)]
    return np.array([1.0, np.exp(1j * phi)], dtype=np.complex128) / np.sqrt(2.0)


def build_precoded_tx_grids(base_grid, nrb_sib, prg_size_rb=4, mode="traditional", Wbf=None,
                            phase_table=(0, np.pi / 2, np.pi, 3 * np.pi / 2)):
    if Wbf is None: Wbf = build_wbf_from_channel_codebook(qv=0, qh=0, M=4, N=4, P=2)
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


def cdl_c_impulse_response_48t4r_full(
        fs_hz, ds_ns=300, fc_GHz=7, seed=None,
        initial_time=0.0, ue_speed=3
):
    if seed is not None:
        np.random.seed(seed)

    ds = ds_ns * 1e-9

    # 🚨 核心修改 1：发射天线配置 [Mg, Ng, M, N, P]
    # M=6 (垂直6振子), N=4 (水平4振子), P=2 (双极化)
    tx_ant = [1, 1, 6, 4, 2]
    rx_ant = [1, 1, 1, 2, 2] # 保持 4Rx 不变

    # 💡 提示：检查你的 channel_initialize 底层实现是否支持传入振子间距
    # 如果支持，你需要把 dV=0.8, dH=0.5 传进去。如果底层写死了 0.5，对理论 BLER 影响不大，仅影响波束物理指向角。
    channel, _, _ = channel_initialize(
        cdl_type='CDLC',
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

    n_rx = 4
    n_tx = 48 # 🚨 核心修改 2：物理信道维度扩张到 48

    H_mimo = np.zeros((n_rx, n_tx, len(h00)), dtype=np.complex128)

    for rx in range(n_rx):
        for tx in range(n_tx):
            H_mimo[rx, tx, :] = channel_time_interpolation(
                delays, h_paths[:, rx, tx], ts=1.0 / fs_hz, num_ext=4
            )

    return H_mimo

# 配套的 apply 函数也要改用 48T 的发生器
def apply_cdl_c_mimo_tx_48t(tx_waves, fs_hz, seed=None, ds_ns=300, fc_hz=7e9, speed_kmh=3, initial_time=0.0):
    H_mimo = cdl_c_impulse_response_48t4r_full(fs_hz, ds_ns, fc_hz / 1e9, seed, initial_time, speed_kmh)
    Ns, Nt = tx_waves.shape
    y = np.zeros((Ns, H_mimo.shape[0]), dtype=np.complex128)
    for rx in range(H_mimo.shape[0]):
        acc = np.zeros(Ns, dtype=np.complex128)
        for tx in range(Nt):
            acc += np.convolve(tx_waves[:, tx], H_mimo[rx, tx, :], mode='same')
        y[:, rx] = acc
    return y, H_mimo


def apply_cdl_c_mimo_tx(tx_waves, fs_hz, seed=None, ds_ns=300, fc_hz=7e9, speed_kmh=3, initial_time=0.0):
    H_mimo = cdl_c_impulse_response_32t4r_full(fs_hz, ds_ns, fc_hz / 1e9, seed, initial_time, speed_kmh)
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


# ---------------------------
# 发端：只构造 PBCH + PBCH DMRS
# ---------------------------

def nrBCH_paramrevised(trblk, sfn, hrf, lssb, idxoffset, ncellid, E=864):
    # interleaving according to TS38.212 7.1.1
    # fmt: off
    G = [16, 23, 18, 17, 8, 30, 10, 6, 24, 7, 0, 5, 3, 2, 1, 4, 9, 11, 12, 13, 14, 15, 19, 20, 21, 22, 25, 26, 27, 28,
         29, 31]
    # fmt: on
    SFN_PAYLOAD_BEGIN = 1
    SFN_PAYLOAD_LENGTH = 6
    SFN_2ND_LSB = SFN_PAYLOAD_LENGTH + 2
    SFN_3RD_LSB = SFN_PAYLOAD_LENGTH + 1
    # v = 2 * scrblk[G[SFN_3RD_LSB]] + scrblk[G[SFN_2ND_LSB]]
    j_sfn = 0
    j_other = 14
    payload = trblk
    A = 32
    a = np.zeros(A, "int")
    for i in range(24):
        if (i >= SFN_PAYLOAD_BEGIN) and (i < (SFN_PAYLOAD_BEGIN + SFN_PAYLOAD_LENGTH)):
            a[G[j_sfn]] = payload[i]
            j_sfn += 1
        else:
            a[G[j_other]] = payload[i]
            j_other += 1
    a[G[10]] = hrf
    a[G[j_sfn + 0]] = 1 if sfn & 0x08 != 0 else 0
    a[G[j_sfn + 1]] = 1 if sfn & 0x04 != 0 else 0
    a[G[j_sfn + 2]] = 1 if sfn & 0x02 != 0 else 0
    a[G[j_sfn + 3]] = 1 if sfn & 0x01 != 0 else 0

    # scramble
    tmp_seq = nrPBCHPRBS(ncellid, 0, len(a) * 100)
    if lssb in (4, 8):
        M = A - 3
    else:
        M = A - 6
    v = 2 * a[G[SFN_3RD_LSB]] + a[G[SFN_2ND_LSB]]
    n = v * M
    scrambling_seq = tmp_seq[n:][:A]
    scrambling_seq_final = np.zeros(A, "int")
    j = 0
    for i in range(A):
        is_ssb_idx = (i in (G[11], G[12], G[13])) and lssb == 64
        if is_ssb_idx or i == G[10] or i == G[SFN_2ND_LSB] or i == G[SFN_3RD_LSB]:
            scrambling_seq_final[i] = 0
        else:
            scrambling_seq_final[i] = scrambling_seq[j]
            j += 1
    scrblk_scrambled = np.bitwise_xor(scrambling_seq_final, a)

    # attach CRC
    bits = nrCRCEncode(scrblk_scrambled, "24C")[:, 0]

    # polar encoding + rate matching
    NMAX = min(int(np.log2(E)), 9)
    # NMAX = 9
    # enc = Polar5GEncoder(k=len(bits), n=2**NMAX,channel_type="downlink")
    # encoded = enc(bits)
    # encoded = np.array(encoded, dtype=np.int32)
    encoded = nrPolarEncode(bits, 0, nmax=NMAX, iil=True)

    rate_matched = nrRateMatchPolar_mh(
        enc_bits=encoded,  # d[]
        K=len(bits),  # K = 32(payload after mapping) + 24(CRC) = 56
        E=E,  # 典型 PBCH：864；若做“12RB 中心映射”实验则传 432
        I_BIL=False  # PBCH 不用 coded-bit interleaver
    )
    return rate_matched


def build_ssb_grid_pbch_only(ncellid, ibar_ssb, trblk_bits, nrb_ssb=20, nsym_ssb=4, nrb_ss=12,start_rb=0,end_rb=20):
    grid = np.zeros((nrb_ssb * 12, nsym_ssb), dtype=complex)
    # 统一索引
    pbch_idx = nrPBCHIndices_param(ncellid, nrb_ssb, nsym_ssb, nrb_ss)
    dmrs_idx = nrPBCHDMRSIndices_param(ncellid, nrb_ssb, nsym_ssb, nrb_ss)

    E = len(pbch_idx) * 2

    # 编码 -> 速配(E) -> QPSK
    pbch_bits = nrBCH_paramrevised(trblk_bits, sfn=0, hrf=0, lssb=0, idxoffset=0, ncellid=ncellid, E=E)
    pbch_symb = nrSymbolModulate(pbch_bits, 'QPSK')
    assert len(pbch_symb) == len(pbch_idx)
    # DMRS
    dmrs_symb = nrPBCHDMRS_param(ncellid, ibar_ssb, len(dmrs_idx))
    # print(dmrs_symb)
    assert len(dmrs_symb) == len(dmrs_idx)

    # start_rb = 4  # 中间12个RB开始的RB编号
    # end_rb = 16  # 中间12个RB结束后的RB编号

    # 先归一化索引：把大于240的索引限制到240以内
    normalized_idx = pbch_idx % 240

    # 计算每个 normalized_idx 所在的RB
    rb_index = normalized_idx // 12  # 每个子载波所在的RB编号

    # 判断该RB是否在中间12个RB范围内
    mask = (rb_index >= start_rb) & (rb_index < end_rb)
    normalized_idx1 = dmrs_idx % 240

    # 计算每个 normalized_idx 所在的RB
    rb_index1 = normalized_idx1 // 12  # 每个子载波所在的RB编号

    # 判断该RB是否在中间12个RB范围内
    mask1 = (rb_index1 >= start_rb) & (rb_index1 < end_rb)

    nrSetResources(pbch_idx[mask], grid, pbch_symb[mask])
    # nrSetResources(nrPBCHIndices(ncellid), grid, pbch_symb)
    nrSetResources(dmrs_idx[mask1], grid, dmrs_symb[mask1])
    # nrSetResources(nrPBCHDMRSIndices(ncellid), grid, nrPBCHDMRS(ncellid, ibar_ssb))
    # if POWER_NORM_PER_RE and nrb_ssb > 0:
    #     scale = np.sqrt(REF_NRB_FOR_POWER / float(nrb_ssb))
    #     grid *= scale
    meta = dict(pbch_idx=pbch_idx, dmrs_idx=dmrs_idx, E=E, pbch_mask=mask, dmrs_mask=mask1)
    return grid, meta


# ---------------------------
# 单次试验（已知 NID；仅 DMRS 检测 + CE + 解码）


def estimate_ibar_once(rxGrid, ncellid, nrb_ssb, nsym_ssb, nrb_ss, ibar_tx):
    # ---- 变换回时域加窗，统计窗内能量，序列判决 ----优化版本
    K = rxGrid.shape[0]  # 240
    dmrs_idx_all = nrPBCHDMRSIndices_param(ncellid, nrb_ssb, nsym_ssb, nrb_ss)  # (Ndmrs_total,)
    dmrs_rx_all = nrExtractResources(dmrs_idx_all, rxGrid)  # (Ndmrs_total,)

    # 参考 DMRS：一次性拿到所有 v=0..7（8 × Ndmrs）
    refs = np.vstack([nrPBCHDMRS_param(ncellid, v, len(dmrs_idx_all)) for v in range(8)])  # complex64/128
    ref_conj = np.conj(refs)  # (8, Ndmrs_total)

    # 按 4 个符号做 mask/kpos，避免重复计算
    sym_masks = []
    sym_kpos = []
    for s in range(4):
        base = s * K
        mask = (dmrs_idx_all >= base) & (dmrs_idx_all < base + K)
        sym_masks.append(mask)
        # 该符号内的子载波位置 0..K-1
        kpos = (dmrs_idx_all[mask] - base).astype(int)
        sym_kpos.append(kpos)

    # 窗口长度
    ratio = 0.10
    L = max(1, int(np.ceil(K * ratio)))

    corr = np.zeros(8, dtype=float)

    for s in range(4):
        mask = sym_masks[s]
        if not np.any(mask):
            continue

        # 频域匹配：对所有 v 一次性做逐元素乘
        #   prod.shape = (8, Ns), Ns 为该符号的 DMRS RE 数
        y_s = dmrs_rx_all[mask]  # (Ns,)
        prod = ref_conj[:, mask] * y_s[None, :]  # (8, Ns)

        # 把 prod“散射”到长度 K 的频域向量（每行是一个候选 v）
        Ys_all = np.zeros((8, K), dtype=complex)
        kpos = sym_kpos[s]  # (Ns,)
        # 高级索引批量赋值：每个 v 在相同 kpos 上写入对应 prod
        Ys_all[np.arange(8)[:, None], kpos[None, :]] = prod

        # IFFT 到时域
        ys_td = np.fft.ifft(Ys_all, axis=1) * np.sqrt(K)
        e = np.abs(ys_td) ** 2  # (8, K)

        # 环形滑窗最大值（向量化）：
        # 拼接 e|e → 8×(2K)，做沿 axis=1 的前缀和，再取每个起点长度 L 的窗和
        e_dbl = np.concatenate([e, e], axis=1)  # (8, 2K)
        csum = np.cumsum(np.concatenate([np.zeros((8, 1)), e_dbl], axis=1), axis=1)
        # window_sums[:, i] = sum(e_dbl[:, i:i+L])
        window_sums = csum[:, L:L + K] - csum[:, 0:K]  # (8, K)
        corr += np.max(window_sums, axis=1)  # (8,)

    ibar_hat = int(np.argmax(corr))
    return ibar_hat




# 提取单次试算为一个纯函数，方便被多进程调用
def worker_trial(args):
    snr_db, ncellid, fs, scs_khz, nrb_ssb, nsym_ssb, nrb_ss, use_combining, N_comb, start_rb, end_rb, seed, qv, qh = args
    
    # 🚨 所有的 print 统统删掉，保持绝对静默
    try:
        ok, _ = one_shot_pbch_min_combined(
            snr_db, ncellid=ncellid, fs=fs, scs_khz=scs_khz,
            nrb_ssb=nrb_ssb, nsym_ssb=nsym_ssb, nrb_ss=nrb_ss,
            ibar_tx=None, rng_seed=seed,
            N_comb=N_comb, start_rb=start_rb, end_rb=end_rb,
            qv=qv, qh=qh
        )
        return ok
    except Exception as e:
        # 只保留报错时的 print，防止黑盒崩溃
        print(f"❌ 发生崩溃: {e}")
        return False
import multiprocessing as mp

def simulate_bler_config_mp(
        snr_db_list, n_trials=5000, ncellid=208, fs=61.44e6, scs_khz=30,
        seed=2025, nrb_ssb=20, nsym_ssb=4, nrb_ss=12,
        use_combining=False, N_comb=1, start_rb=0, end_rb=20, 
        max_workers=10, 
        qv=0, qh=0 
):
    bler = []
    rng = np.random.RandomState(seed)
    
    for snr_db in snr_db_list:
        n_err = 0
        round_total = 0
        
        # 准备任务参数列表
        tasks = [
            (snr_db, ncellid, fs, scs_khz, nrb_ssb, nsym_ssb, nrb_ss, 
             use_combining, N_comb, start_rb, end_rb, rng.randint(1 << 31), qv, qh)
            for _ in range(n_trials)
        ]
        
        # 🚨 换用更底层的 Pool，规避 futures 的死锁问题
        with mp.Pool(processes=max_workers) as pool:
            # imap_unordered: 哪个进程先算完，就立刻吐出结果，不阻塞排队
            results_iterator = pool.imap_unordered(worker_trial, tasks)
            
            with tqdm(total=n_trials, desc=f"SNR={snr_db:>5.1f} dB", ncols=70, leave=False) as pbar:
                try:
                    for ok in results_iterator:
                        round_total += 1
                        if not ok:
                            n_err += 1
                            
                        pbar.update(1)
                        
                        # 早停机制
                        if n_err >= 500:
                            # 比 future.cancel() 更暴力的强制停止：直接发送 SIGTERM 杀掉所有子进程
                            pool.terminate() 
                            break
                            
                except Exception as e:
                    print(f"\n[主进程错误] 收集结果时发生异常: {e}")
                    pool.terminate()
        
        # 计算该 SNR 的最终 BLER
        current_bler = n_err / round_total if round_total > 0 else 1.0
        bler.append(current_bler)
        print(f"[CFG nrb={nrb_ssb}, nsym={nsym_ssb}, comb={use_combining}, N={N_comb}] "
              f"SNR={snr_db:>5.1f} dB  BLER={current_bler:.4f} (测试了 {round_total} 次)")
              
    return np.array(bler)

def detect_ssb_index_from_dmrs_4x_window_aligned(rxGrid, dmrs_idx, ncellid, fs=61.44e6, Lmax=8):
    """
    终极版盲检：全局多径谱对齐累加 (兼容带有 SSS 挖洞的 20RB 标准结构)
    """
    K, N, nRx = rxGrid.shape
    candidate_metrics = np.zeros(Lmax, dtype=float)
    
    sym_indices = dmrs_idx // K
    unique_syms = np.unique(sym_indices)
    
    # 🚨 核心修复 1：基于全局子载波总数 K 确定统一的 IFFT 长度，彻底告别长度不匹配
    N_fft = 2 ** int(np.ceil(np.log2(K))) * 4
    
    Y_rx = np.zeros((len(dmrs_idx), nRx), dtype=complex)
    for r in range(nRx):
        Y_rx[:, r] = nrExtractResources(dmrs_idx, rxGrid[:, :, r])
        
    for cand_idx in range(Lmax):
        ref_dmrs = nrPBCHDMRS_param(ncellid, cand_idx, len(dmrs_idx))
        H_comp_all = Y_rx * np.conj(ref_dmrs)[:, None]
        
        # 初始化固定长度的全局能量谱
        total_power_smoothed = np.zeros(N_fft, dtype=float)
        
        for sym in unique_syms:
            mask = (sym_indices == sym)
            H_comp_sym = H_comp_all[mask, :]
            
            # 🚨 核心修复 2：把导频放回长度为 K 的全频带真实物理坐标上！
            # 这样哪怕符号 2 中间被 SSS 挖空了，它的多径相位也能和符号 1 完美对齐
            H_grid = np.zeros((K, nRx), dtype=complex)
            kpos = (dmrs_idx[mask] % K).astype(int)
            H_grid[kpos, :] = H_comp_sym
            
            # 对映射好的全频带网格做 IFFT
            h_td = np.fft.ifft(H_grid, n=N_fft, axis=0, norm="ortho")
            
            power = np.abs(h_td) ** 2  
            power_combined = np.sum(power, axis=1)
            
            # 🚨 核心修复 3：滑窗长度严格取 IFFT 总长度的 10%
            # 这完美复刻了你最初版本 `ratio = 0.10` 的宽容大滑窗，专门对付 TDL-C 的大时延
            win_len_points = max(1, int(np.ceil(N_fft * 0.08)))
            window = np.ones(win_len_points)
            
            # 环形滑窗累加
            power_smoothed = np.fft.ifft(
                np.fft.fft(power_combined) * np.fft.fft(window, n=N_fft)
            ).real
            
            # 完美对齐，安全叠加
            total_power_smoothed += power_smoothed
            
        # 提取全局多径谱中的最强聚集能量
        candidate_metrics[cand_idx] = float(np.max(total_power_smoothed))
        
    ibar_hat = int(np.argmax(candidate_metrics))
    
    return ibar_hat, candidate_metrics

def _rx_round_get_llr(
        snr_db, ncellid, ibar_tx,
        ssb_grid,  # 发端已映射好的 1-port grid（含 PBCH+DMRS）
        pbch_idx, dmrs_idx, E, pbch_mask, dmrs_mask,
        fs=61.44e6, scs_khz=30, nrb_ssb=20, nsym_ssb=4,
        use_known_nvar=False,
        start_rb=0, end_rb=20,
        qv=0, qh=0, cdl_seed=None, initial_time=0.0  # 🚨 新增：波束方向与CDL信道参数
):
    K_grid = nrb_ssb * 12
    N_grid = nsym_ssb
    # ==========================================
    # 🚨 阶段 1：PBCH 频域波束赋形 (1 Port -> 48 Ports)
    # ==========================================
    # 修改点：将 M 设为 6，完美对应你的 6x4x2 阵列
    Wbf = build_wbf_from_channel_codebook(qv=qv, qh=qh, M=6, N=4, P=2)
    
    wbb = bb_precoder_traditional() 
    w_eff = Wbf @ wbb
    w_eff = w_eff / np.linalg.norm(w_eff)  # (48,) 的复数权重向量

    # 将单端口 ssb_grid 复制并加权，生成 48 根天线的 Tx Grids
    tx_grids = np.zeros((48, K_grid, N_grid), dtype=np.complex128)
    for t in range(48):
        tx_grids[t, :, :] = w_eff[t] * ssb_grid

    # ==========================================
    # 🚨 阶段 2：48端口独立 OFDM 调制 & 过 48T4R CDL 信道
    # ==========================================
    if NO_CHANNEL:
        rxGrid_clean = np.zeros((K_grid, N_grid, 4), dtype=complex)
        for r in range(4):
            rxGrid_clean[:, :, r] = ssb_grid
        nRx = 4
    else:
        tx_waves = nrOFDMModulate_multi_tx(tx_grids, scs_khz, fs)
        
        # 🚨 调用刚刚写好的 48T4R 信道
        ch_out, _ = apply_cdl_c_mimo_tx_48t(
            tx_waves, fs_hz=fs, seed=cdl_seed, ds_ns=300, 
            fc_hz=7e9, speed_kmh=120, initial_time=initial_time
        )
        
        rxGrids_nRx_K_N = ofdm_demodulate_multi_rx(
            ch_out, nrb=nrb_ssb, scs=scs_khz, initialNSlot=0, 
            SampleRate=fs, nsym_keep=nsym_ssb, CyclicPrefixFraction=0.5
        )
        rxGrid_clean = np.transpose(rxGrids_nRx_K_N, (1, 2, 0))
        nRx = rxGrid_clean.shape[2]
    # ==========================================
    # 🚨 阶段 3：加噪与 DMRS 盲检 (完美兼容原有逻辑)
    # ==========================================
    active_idx = np.sort(np.concatenate([pbch_idx[pbch_mask], dmrs_idx[dmrs_mask]]))
    
    # 直接复用你原有的加噪函数，它支持 (K, N, nRx) 的三维矩阵
    rxGrid, nVar_ref = add_awgn_on_gri_per_symbol(rxGrid_clean, ssb_grid, snr_db, active_idx)

    # 接入真实的 DMRS 盲检算法 (基于 4 根接收天线的全局谱对齐)
    ibar_hat, cand_metrics = detect_ssb_index_from_dmrs_4x_window_aligned(
        rxGrid, dmrs_idx[dmrs_mask], ncellid, fs=fs, Lmax=8
    )

    # ==========================================
    # 阶段 4：等效信道估计与 1T4R 完美 MRC 软解调
    # (接收端视作 1T4R，自动吃掉发端的 32Tx 赋形增益)
    # ==========================================
    refGridH = np.zeros((K_grid, N_grid), dtype=complex)
    dmrs_syms = nrPBCHDMRS_param(ncellid, ibar_hat, len(dmrs_idx))
    nrSetResources(dmrs_idx[dmrs_mask], refGridH, dmrs_syms[dmrs_mask])

    N_pbch_re = len(pbch_idx)
    Y_all = np.zeros((N_pbch_re, nRx), dtype=complex)
    H_all = np.zeros((N_pbch_re, nRx), dtype=complex)

    for r in range(nRx):
        rxGrid_r = rxGrid[:, :, r]
        
        if NO_CHANNEL:
            nVar = nVar_ref
            Y_all[:, r] = nrExtractResources(pbch_idx, rxGrid_r)
            H_all[:, r] = np.ones(N_pbch_re, dtype=complex)
        else:
            # 这里的 myChannelEstimate_std 估出来的是包含了 32Tx 波束权重的“等效信道”
            H = myChannelEstimate_std(
                rxGrid=rxGrid_r, refSym=dmrs_syms[dmrs_mask], refGrid=refGridH, 
                EST_TFDOMAIN=False, EST_PBCH=True, start_rb=start_rb, end_rb=end_rb
            )
            nVar = nVar_ref 
            Y_all[:, r] = nrExtractResources(pbch_idx, rxGrid_r)
            H_all[:, r] = nrExtractResources(pbch_idx, H)

    # 1T4R MRC 联合均衡 (基于 ZF 架构的无损权重)
    nVar_sym = nVar_ref if use_known_nvar else nVar
    H_power = np.sum(np.abs(H_all)**2, axis=1)  
    eps = 1e-12
    pbch_eq_zf = np.sum(H_all.conj() * Y_all, axis=1) / (H_power + eps)
    
    csi_optimal = H_power
    
    # 统一软解调
    llr_sum = nrSymbolDemodulate_param_persym(
        pbch_eq_zf, 'QPSK', nVar_sym, pbch_idx, nrb_ssb * 12, csi_optimal, 'soft'
    ) 
    
    # 对齐掩码未使用的 RE 置零
    bit_mask = np.repeat(pbch_mask.astype(bool), 2)
    llr_sum[~bit_mask] = 0.0

    assert len(llr_sum) == E
    return llr_sum


def one_shot_pbch_min_combined(
        snr_db, ncellid,
        fs=61.44e6, scs_khz=30,
        nrb_ssb=20, nsym_ssb=4, nrb_ss=12,
        ibar_tx=None, rng_seed=None,
        N_comb=1, start_rb=0, end_rb=20,
        qv=0, qh=0  # 🚨 新增：接收从 worker 传来的波束方向
):
    if rng_seed is not None:
        np.random.seed(rng_seed)
    if ibar_tx is None:
        ibar_tx = np.random.randint(0, 8)

    # 生成 32 比特 TB
    A = 32
    trblk_bits = np.random.randint(0, 2, A).astype(int)

    # 发端网格（含 PBCH & DMRS）
    ssb_grid, meta = build_ssb_grid_pbch_only(
        ncellid, ibar_tx, trblk_bits,
        nrb_ssb=nrb_ssb, nsym_ssb=nsym_ssb, nrb_ss=nrb_ss, start_rb=start_rb, end_rb=end_rb
    )
    pbch_idx = meta['pbch_idx']
    dmrs_idx = meta['dmrs_idx']
    E = meta['E']
    pbch_mask = meta['pbch_mask']
    dmrs_mask = meta['dmrs_mask']

    # 多轮接收，逐轮得 LLR
    llr_sum = None
    for r in range(N_comb):
        # 🚨 修复点：使用 rng_seed 作为 CDL 信道的种子基础
        c_seed = (rng_seed + r) if rng_seed is not None else None
        
        llr_r = _rx_round_get_llr(
            snr_db, ncellid, ibar_tx,
            ssb_grid, pbch_idx, dmrs_idx, E, pbch_mask, dmrs_mask,
            fs=fs, scs_khz=scs_khz, nrb_ssb=nrb_ssb, nsym_ssb=nsym_ssb,
            use_known_nvar=False, start_rb=start_rb, end_rb=end_rb,
            qv=qv, qh=qh, cdl_seed=c_seed, initial_time=r * 0.005 # 假设重传间隔5ms
        )
        if llr_sum is None:
            llr_sum = llr_r.astype(np.float64, copy=True)
        else:
            llr_sum += llr_r

    # Polar 逆速配 & 译码
    P = 24
    K = 32 + P
    N = 2 ** min(int(np.log2(E)), 9)

    decIn = nrRateRecoverPolar_mh(llr_sum, K, N, False, discardRepetition=False)
    
    # 译码 (这里假设你之前开了 USE_SCL)
    decoded_bits, crc_1 = polar_decode_scl_llr(decIn, K=K, N=N, list_size=8, crc_degree="CRC24C")
    _, crc = nrCRCDecode(decoded_bits, '24C')
    
    return (crc == 0), dict(N_comb=N_comb, E=E, ibar_tx=ibar_tx)


# ---------------------------
# 直接执行：多配置仿真 + 存图
# ---------------------------
if __name__ == "__main__":
    import multiprocessing
    import time
    # 强制使用 spawn
    multiprocessing.set_start_method('spawn', force=True)
    
    # ==========================================
    # 🚨 终极杀招：主进程 Numba 预热
    # 在拉起任何多进程之前，先在绝对单进程环境下跑一次完整的试算。
    # 这一步会强迫 Numba 把所有 @njit 函数全部编译成机器码并写好缓存。
    # ==========================================
    print(">>> [系统] 正在主进程预热 Numba 编译器，请等待约 1~2 秒...")
    warmup_args = (-14.5, 208, 61.44e6, 30, 20, 4, 12, True, 1, 0, 20, 889, 0, 0)
    try:
        worker_trial(warmup_args)
        print(">>> [系统] Numba 预热完毕！所有底层机器码已安全写入缓存。")
    except Exception as e:
        print(f">>> [系统] 预热时发生错误（不影响主流程）: {e}")
    # ==========================================


    np.random.seed(2026)
    snr_points = [-14.5, -14, -13.5, -13, -12.5, -12]

    configs = [
         ("48T4R 20RB 4sym Puncture 0", 20, 4, 12, 1, 0, 20),
    ]

    curves = {}
    target_qv = 0
    target_qh = 0

    for label, nrb_cfg, nsym_cfg, nrb_ss_cfg, comb_cfg, start_rb_cfg, end_rb_cfg in configs:
        print(f"\n>>> 开始并发仿真 [单波束正对 UE]: qv={target_qv}, qh={target_qh} <<<")
        bler_curve = simulate_bler_config_mp(
            snr_points,
            n_trials=8000, 
            ncellid=208,
            fs=61.44e6,
            scs_khz=30,
            seed=889,
            nrb_ssb=nrb_cfg,
            nsym_ssb=nsym_cfg,
            nrb_ss=nrb_ss_cfg,
            use_combining=True,
            N_comb=comb_cfg,
            start_rb=start_rb_cfg, 
            end_rb=end_rb_cfg,
            qv=target_qv, 
            qh=target_qh, 
            max_workers=50  # 预热完后，这里开 10 个跑
        )
        curves[label] = bler_curve


    # --- 画图与保存 ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5.5))
    for label, bler in curves.items():
        plt.semilogy(snr_points, np.maximum(bler, 1e-4), label=label, marker='o')
        
    plt.grid(True, which='both')
    plt.xlabel("SNR (dB)")
    plt.ylabel("BLER")
    plt.title("PBCH BLER vs SNR (48T4R, Single Boresight Beam)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("pbch_single_beam_48T4R.png", dpi=150)
    print("\n单波束正对 BLER 曲线已保存到 pbch_single_beam_48T4R.png")
