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
from nrPBCHIndicesVariable import nrPBCHIndices_param, nrPBCHDMRSIndices_param, nrPBCHDMRS_param, nrBCH_param, \
    nrSymbolDemodulate_param, nrSymbolDemodulate_param_persym, nrRateRecoverPolar_mh, nrRateMatchPolar_mh
from tqdm import tqdm
from ext_scl2.scl_adapter import polar_decode_scl_llr
from ext_scl.encoding import PolarEncoder, Polar5GEncoder
from ext_scl.decoding import PolarSCLDecoder, Polar5GDecoder
from myChannelEstimation1 import DMRSFilterGenerate_v2_explicit, myChannelEstimate, myChannelEstimatev2, \
    myChannelEstimate_std
import GenTDLChannel
from GenTDLChannel import ChannelInfo

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


####
def add_awgn(x, snr_db):
    # snr_db = snr_db - SNR_OFFSET_DB
    p_sig = np.mean(np.abs(x) ** 2)
    p_noise = p_sig / (10 ** (snr_db / 10.0))
    n = (np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape)) * np.sqrt(p_noise / 2.0)
    return x + n, p_noise  # 返回噪声功率，便于 nVar 计算


def add_awgn_on_grid(rxGrid_clean, ssb_grid, snr_db_re, active_idx):
    """
    在频域网格上直接加噪，snr_db_re 是“每 RE 的目标 SNR (Es/N0，复符号功率口径)”
    """
    snr_lin = 10 ** (snr_db_re / 10.0)
    # 以“每个 RE 的平均信号功率”为口径配噪
    # flat = ssb_grid.ravel(order="F")
    flat = rxGrid_clean.ravel(order="F")
    p_sig = np.mean(np.abs(flat[active_idx]) ** 2)
    # print(p_sig)
    nvar_complex = p_sig / snr_lin  # 复符号噪声功率
    noise = (np.random.randn(*ssb_grid.shape) + 1j * np.random.randn(*ssb_grid.shape)) \
            * np.sqrt(nvar_complex / 2.0)
    return rxGrid_clean + noise, nvar_complex


def add_awgn_on_gri_per_symbol(rxGrid_clean, ssb_grid, snr_db_re, active_idx):
    """
    逐 OFDM 符号统计信号功率并加噪：
      - snr_db_re: 以“每 RE 的发射口径 Es/N0 (dB)”定义
      - active_idx: 列优先的一维索引（如 PBCH∪DMRS），用于该符号内 p_sig 估计
    返回:
      rxGrid_noisy: 加噪后的频域网格（与 rxGrid_clean 同形状）
      nvar_cols:    每个符号列对应的复符号噪声功率 (长度 = Ns)
    """
    K, N = rxGrid_clean.shape
    snr_lin = 10 ** (snr_db_re / 10.0)
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
        in_col = (active_idx >= l * K) & (active_idx < (l + 1) * K)
        if np.any(in_col):
            p_sig_l = np.mean(np.abs(flat[active_idx[in_col]]) ** 2)
        else:
            p_sig_l = p_sig_global
        nvar_cols[l] = p_sig_l / snr_lin
    noise = (np.random.randn(K, N) + 1j * np.random.randn(K, N)) * np.sqrt(nvar_cols[None, :] / 2.0)
    # p_sig = np.mean(np.abs(flat[active_idx]) ** 2)
    # nvar_complex = np.mean(nvar_cols)                        # 复符号噪声功率
    # noise = (np.random.randn(*ssb_grid.shape) + 1j*np.random.randn(*ssb_grid.shape)) \
    #         * np.sqrt(nvar_complex/2.0)
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


def tdl_c_impulse_response(fs_hz, ds=300e-9, fc_hz=3.5e9, speed_kmh=3.0, nfft=2048, seed=None, normalize=True):
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
        Speed=speed_kmh,  # km/h
        Fc=fc_hz,  # Hz
        Fs=fs_hz,  # Hz
        ChanType='TDLC',
        DS=DS_ns,  # ns
        NFFT=nfft,
        TO=0, FO=0
    )

    # 生成一次静态 Rayleigh 多径（不随时间变化）
    ch.gen_Rayleigh(num_samples=1)  # time_channel shape = [nRx, nTx, DS_tap, 1]

    # 取 1x1 MIMO 的时域冲激响应
    h = ch.time_channel[0, 0, :, 0].astype(np.complex128, copy=True)

    if normalize:
        p = np.sum(np.abs(h) ** 2)
        if p > 0:
            h /= np.sqrt(p)

    return h


# def apply_tdl_c(x, fs_hz, seed=None):
#     h = tdl_c_impulse_response(fs_hz, seed)
#     # 用“same”保持长度和对齐，不引入整体时移（避免起点错位）
#     y = np.convolve(x, h, mode='same')
#     #print('h=',h)
#     return y, h
def apply_tdl_c(x, fs_hz, seed=None, ds=300e-9, fc_hz=3.5e9, speed_kmh=3.0, nfft=2048, normalize=True):
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


def simulate_bler_config(
        snr_db_list,
        n_trials=5000,
        ncellid=208,
        fs=30.72e6,
        scs_khz=15,
        seed=2025,
        nrb_ssb=20,
        nsym_ssb=4,
        nrb_ss=12,
        use_combining=False,
        N_comb=1,
        start_rb=0,end_rb=20
):
    """对单个配置 (nrb_ssb, nsym_ssb, nrb_ss) 扫 SNR 生成 BLER 曲线"""

    # 早停配置
    p_goal = 5 * 1e-3
    min_trials = 500
    bler = []
    rng = np.random.RandomState(seed)
    for snr_db in snr_db_list:
        n_err = 0
        round_total = 0
        with tqdm(total=n_trials, desc=f"SNR={snr_db:>5.1f} dB", ncols=70, leave=False) as pbar:
            for _ in range(n_trials):
                if _==10:
                    print('s')
                if use_combining:
                    ok, info = one_shot_pbch_min_combined(
                        snr_db, ncellid=ncellid, fs=fs, scs_khz=scs_khz,
                        nrb_ssb=nrb_ssb, nsym_ssb=nsym_ssb, nrb_ss=nrb_ss,
                        ibar_tx=None, rng_seed=rng.randint(1 << 31),
                        N_comb=N_comb,start_rb=start_rb,end_rb=end_rb
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
                if n_err >= 100:
                    # 早停：出现 10 次错误就提前结束，提高速度
                    break
                if round_total >= min_trials and n_err <= 2 and (3.0 / round_total <= p_goal):
                    # 成功率高且试验够多 → 早停
                    break
                if round_total >= 3000 and n_err <= 1:
                    break
                pbar.update(1)
        bler.append(n_err / round_total)
        print(f"[CFG nrb={nrb_ssb}, nsym={nsym_ssb}, comb={use_combining}, N={N_comb}] "
              f"SNR={snr_db:>5.1f} dB  BLER={bler[-1]:.4f}")
    return np.array(bler)


def _rx_round_get_llr(
        snr_db, ncellid, ibar_tx,
        ssb_grid,  # 发端已映射好的 grid（含 PBCH+DMRS）
        pbch_idx, dmrs_idx, E, pbch_mask, dmrs_mask,  # 索引 & 速率匹配长度
        fs=30.72e6, scs_khz=15, nrb_ssb=20, nsym_ssb=4,
        use_known_nvar=False,
        start_rb=0,end_rb=20
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
        ch_out, ch_obj = apply_tdl_c(tx_wave, fs,
                                     speed_kmh=3,
                                     fc_hz=3.5e9,
                                     ds=300,
                                     nfft=2048,
                                     normalize=True)
    # Nfft=2048
    # Nsc = 240
    # H_tdl_timedomain_padded = np.pad(H_tdl_timedomain, (0, Nfft - len(H_tdl_timedomain)), mode='constant')
    # H_fd_full = np.fft.fft(H_tdl_timedomain_padded, Nfft)
    # H_fd_shift = np.fft.fftshift(H_fd_full)
    # start = Nfft // 2 - Nsc // 2
    # H_fd_sel = H_fd_shift[start:start + Nsc]

    # 2048*4+160+144*3

    # # 反变换（干净网格）
    rxGrid_clean = nrOFDMDemodulate(
        waveform=ch_out, nrb=nrb_ssb, scs=scs_khz,
        initialNSlot=0, SampleRate=fs  # , CyclicPrefixFraction=0.5
    )[:, :nsym_ssb]
    # rxGrid = rxGrid_clean
    #
    # 在频域网格按 RE-SNR 加噪（避免 RB 依赖）
    active_idx = np.sort(np.concatenate([pbch_idx[pbch_mask], dmrs_idx[dmrs_mask]]))
    # rxGrid, nVar_ref = add_awgn_on_grid(rxGrid_clean,ssb_grid, snr_db,active_idx)
    rxGrid, nVar_ref = add_awgn_on_gri_per_symbol(rxGrid_clean, ssb_grid, snr_db, active_idx)
    nrb_ss = 12
    # ibar_hat = estimate_ibar_once(rxGrid, ncellid, nrb_ssb, nsym_ssb, nrb_ss, ibar_tx)
    ibar_hat = ibar_tx
    # 参考导频网格（CE）
    refGridH = np.zeros_like(rxGrid_clean, dtype=complex)
    dmrs_syms = nrPBCHDMRS_param(ncellid, ibar_hat, len(dmrs_idx))
    # print(dmrs_syms)
    nrSetResources(dmrs_idx[dmrs_mask], refGridH, dmrs_syms[dmrs_mask])
    # H, nVar_est = nrChannelEstimate(rxGrid=rxGrid, refGrid=refGridH)
    if NO_CHANNEL:
        dmrs_y = nrExtractResources(dmrs_idx, rxGrid)

        # dmrs_h = nrExtractResources(dmrs_idx, H)
        # err = dmrs_y - dmrs_h* dmrs_syms
        err = dmrs_y - dmrs_syms

        nVar_est = float(np.mean(np.abs(err) ** 2))
        nVar = nVar_ref if use_known_nvar else nVar_est
        y_pbch = nrExtractResources(pbch_idx, rxGrid)
        # h_pbch = nrExtractResources(pbch_idx, H)
        # pbch_eq, csi = nrEqualizeMMSE(y_pbch, h_pbch, nVar)
        pbch_eq = y_pbch
        llr = nrSymbolDemodulate(pbch_eq, 'QPSK', nVar, 'soft')
        # llr = llr * np.repeat(csi, 2)                            # 保持与单次实现一致
        bit_mask = np.repeat(pbch_mask.astype(bool), 2)  # QPSK → 每RE两比特
        if llr.shape[0] != bit_mask.shape[0]:
            raise RuntimeError(f"LLR长度({len(llr)})与掩码长度({len(bit_mask)})不一致")
        llr[~bit_mask] = 0.0
    else:
        # ONLY_TDLC=True
        if ONLY_TDLC:
            txGrid_id = nrOFDMDemodulate(waveform=tx_wave, nrb=nrb_ssb, scs=scs_khz,
                                         initialNSlot=0, SampleRate=fs  # , CyclicPrefixFraction=0.5
                                         )[:, :nsym_ssb]
            ch_only = nrOFDMDemodulate(waveform=ch_out, nrb=nrb_ssb, scs=scs_khz,
                                       initialNSlot=0, SampleRate=fs  # , CyclicPrefixFraction=0.5
                                       )[:, :nsym_ssb]
            H_true = np.ones_like(ch_only, dtype=complex)
            mask = (np.abs(txGrid_id) > 0)  # 只在实际映射了符号的 RE 上求比值
            H_true[mask] = ch_only[mask] / txGrid_id[mask]
            H_power = np.abs(H_true) ** 2
            nVar = nVar_ref  # if use_known_nvar else nVar_est

            # 提取→均衡→软解
            # y_pbch = nrExtractResources(pbch_idx, rxGrid)
            # h_pbch = H_true.ravel(order="F")[pbch_idx]
            # pbch_eq, csi = nrEqualizeMMSE(y_pbch, h_pbch, nVar)  # 与 pbch_idx 一一对应
            # llr_full = nrSymbolDemodulate_param_persym(pbch_eq, 'QPSK', nVar, pbch_idx, nrb_ssb * 12, csi, 'soft')
            y_pbch = nrExtractResources(pbch_idx, rxGrid)
            h_pbch = nrExtractResources(pbch_idx, H_true)
            # pbch_eq, csi = nrEqualizeMMSE(y_pbch, h_pbch, nVar)
            pbch_eq, csi = nrEqualizeMMSE_persym(y_pbch, h_pbch, nVar, pbch_idx, K=nrb_ssb * 12)
            llr = nrSymbolDemodulate_param_persym(pbch_eq, 'QPSK', nVar, pbch_idx, nrb_ssb * 12, csi, 'soft')  # 长度应为 E
            bit_mask = np.repeat(pbch_mask.astype(bool), 2)  # QPSK → 每RE两比特
            if llr.shape[0] != bit_mask.shape[0]:
                raise RuntimeError(f"LLR长度({len(llr)})与掩码长度({len(bit_mask)})不一致")
            llr[~bit_mask] = 0.0
        else:
            # dmrs_y = nrExtractResources(dmrs_idx, rxGrid)
            # dmrs_h = nrExtractResources(dmrs_idx, H)
            # err = dmrs_y - dmrs_h * dmrs_syms
            # nVar_est = float(np.mean(np.abs(err) ** 2))
            # nrSetResources(dmrs_idx, refGridH, dmrs_syms)
            H = myChannelEstimate_std(rxGrid=rxGrid, refSym=dmrs_syms[dmrs_mask], refGrid=refGridH, EST_TFDOMAIN=False,
                                      EST_PBCH=True,start_rb=start_rb,end_rb=end_rb)
            txGrid_id = nrOFDMDemodulate(waveform=tx_wave, nrb=nrb_ssb, scs=scs_khz,
                                         initialNSlot=0, SampleRate=fs  # , CyclicPrefixFraction=0.5
                                         )[:, :nsym_ssb]
            ch_only = nrOFDMDemodulate(waveform=ch_out, nrb=nrb_ssb, scs=scs_khz,
                                       initialNSlot=0, SampleRate=fs  # , CyclicPrefixFraction=0.5
                                       )[:, :nsym_ssb]
            H_true = np.ones_like(ch_only, dtype=complex)
            deltaH = np.ones_like(ch_only, dtype=complex)
            delta = np.ones_like(ch_only, dtype=complex)
            mask = (np.abs(txGrid_id) > 0)  # 只在实际映射了符号的 RE 上求比值
            H_true[mask] = ch_only[mask] / txGrid_id[mask]
            deltaH = np.abs((H-H_true)**2)
            delta[48:192,1] = deltaH[48:192,1]
            delta[48:192, 3] = deltaH[48:192, 3]
            nmse = np.mean(np.abs(delta) ** 2) / np.mean(np.abs(H_true[48:192,:]) ** 2)
            # print(nmse)
            # H=ch_obj
            nVar = nVar_ref
            # nVar = nVar_est
            # 提取→均衡→软解
            y_pbch = nrExtractResources(pbch_idx, rxGrid)
            h_pbch = nrExtractResources(pbch_idx, H)
            # pbch_eq, csi = nrEqualizeMMSE(y_pbch, h_pbch, nVar)
            pbch_eq, csi = nrEqualizeMMSE_persym(y_pbch, h_pbch, nVar, pbch_idx, K=nrb_ssb * 12)
            llr = nrSymbolDemodulate_param_persym(pbch_eq, 'QPSK', nVar, pbch_idx, nrb_ssb * 12, csi, 'soft')  # 长度应为 E
            bit_mask = np.repeat(pbch_mask.astype(bool), 2)  # QPSK → 每RE两比特
            if llr.shape[0] != bit_mask.shape[0]:
                raise RuntimeError(f"LLR长度({len(llr)})与掩码长度({len(bit_mask)})不一致")
            llr[~bit_mask] = 0.0

    # llr = nrSymbolDemodulate_param(pbch_eq, 'QPSK', nVar, csi, 'soft')  # 长度应为 E
    # llr = nrSymbolDemodulate(pbch_eq, 'QPSK', nVar,  'soft')
    # llr = llr * np.repeat(csi, 2)                            # 保持与单次实现一致
    assert len(llr) == E
    return llr


def one_shot_pbch_min_combined(
        snr_db, ncellid,
        fs=30.72e6, scs_khz=15,
        nrb_ssb=20, nsym_ssb=4, nrb_ss=12,
        ibar_tx=None, rng_seed=None,
        N_comb=4,start_rb=0,end_rb=20
):
    if rng_seed is not None:
        np.random.seed(rng_seed)
    if ibar_tx is None:
        ibar_tx = np.random.randint(0, 8)

    # 生成 32 比特 TB（一次，保证每轮发同一块，Chase Combining）
    A = 32
    trblk_bits = np.random.randint(0, 2, A).astype(int)

    # 发端网格（含 PBCH & DMRS），以及索引/E

    ssb_grid, meta = build_ssb_grid_pbch_only(
        ncellid, ibar_tx, trblk_bits,
        nrb_ssb=nrb_ssb, nsym_ssb=nsym_ssb, nrb_ss=nrb_ss,start_rb=start_rb,end_rb=end_rb
    )
    pbch_idx = meta['pbch_idx'];
    dmrs_idx = meta['dmrs_idx'];
    E = meta['E'];
    pbch_mask = meta['pbch_mask'];
    dmrs_mask = meta['dmrs_mask']

    assert len(np.intersect1d(pbch_idx, dmrs_idx)) == 0
    assert np.all(np.diff(pbch_idx) > 0)
    assert np.all(np.diff(dmrs_idx) > 0)

    # 多轮接收，逐轮得 LLR
    llr_sum = None
    for r in range(N_comb):
        llr_r = _rx_round_get_llr(
            snr_db, ncellid, ibar_tx,
            ssb_grid, pbch_idx, dmrs_idx, E, pbch_mask, dmrs_mask,
            fs=fs, scs_khz=scs_khz, nrb_ssb=nrb_ssb, nsym_ssb=nsym_ssb,
            use_known_nvar=USE_KNOWN_NVAR,start_rb=start_rb,end_rb=end_rb
        )
        if llr_sum is None:
            llr_sum = llr_r.astype(np.float64, copy=True)
        else:
            llr_sum += llr_r

    # Polar 逆速配 & 译码（与单次保持一致）
    P = 24;
    K = 32 + P
    # 你的实现里 N 用 2**min(floor(log2(E)),9)；保持一致
    N = 2 ** min(int(np.log2(E)), 9)

    decIn = nrRateRecoverPolar_mh(llr_sum, K, N, False, discardRepetition=False)
    # print(len(pbch_idx))
    # print(len(llr_descr))
    if USE_SCL:
        decoded_bits, crc_1 = polar_decode_scl_llr(decIn, K=K, N=N, list_size=8, iil=True,crc_degree="CRC24C")
        _, crc = nrCRCDecode(decoded_bits, '24C')
    else:
        decoded = nrPolarDecode(decIn, K, 0, 0, nmax=min(int(np.log2(E)), 9))
        _, crc = nrCRCDecode(decoded, '24C')
        # crc_ok = (crc == 0)

    # decIn = nrRateRecoverPolar(llr_sum, K, N, False, discardRepetition=False)
    # decoded = nrPolarDecode(decIn, K, 0, 0, nmax=min(int(np.log2(E)), 9))
    # _, crc = nrCRCDecode(decoded, '24C')
    return (crc == 0), dict(N_comb=N_comb, E=E, ibar_tx=ibar_tx)


# ---------------------------
# 直接执行：多配置仿真 + 存图
# ---------------------------
if __name__ == "__main__":
    # 建议用稍“温和”的 SNR 段，便于拉开曲线
    # print_nmse_compare_db(snr_db=-2.0, ncellid=208, fs=30.72e6, scs_khz=30, trials_each=300)
    # snr_points =  [-10,-9,-8,-7,-6,-5,-4,-3]
    # snr_points =  [4.5,5,5.5,6,6.5,7,7.5]
    snr_points = [-7,-6,-5,-4,-3]  # 也可以换回你之前的段, -13,   `-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,0,1,2,3,4,5,6,8,10
    # snr_points1 =  [-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9]
    # 要对比的 7 组配置（标签, nrb_ssb, nsym_ssb, nrb_ss）
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
        ("20RB × 4sym 1 comb puncture 0 rb", 20, 4, 12, 1, 0, 20 - 0),
        # ("20RB × 4sym 1 comb puncture 1 rb", 20, 4, 12, 1, 1, 20 - 1),
        # ("20RB × 4sym 1 comb puncture 2 rb", 20, 4, 12, 1, 2, 20 - 2),
        # ("20RB × 4sym 1 comb puncture 3 rb", 20, 4, 12, 1, 3, 20 - 3),
        # ("20RB × 4sym 1 comb puncture 4 rb", 20, 4, 12, 1, 4, 20 - 4),
        # ("20RB × 4sym 1 comb", 20, 4, 12, 1),
        # ("18RB × 4sym 1 comb", 18, 4, 6, 1),
        # ("24RB × 4sym 1 comb", 24, 4, 24, 1),
        # ("16RB × 5sym 1 comb", 16, 5, 16, 1),
        # ("12RB × 6sym 1 comb", 12, 6, 12, 1),
        # ("8RB × 8sym 1 comb", 8, 8, 8, 1),
        # ("6RB × 10sym 1 comb", 6, 10, 6, 1),
        # ("12RB × 4sym 1 comb", 12, 4, 12, 1),
        # ("12RB × 6sym 1 comb", 12, 6, 12, 1),
        # ("20RB × 4sym 2 comb", 20,  4, 12, 2),

    ]

    curves = {}
    for label, nrb_cfg, nsym_cfg, nrb_ss_cfg, comb_cfg,start_rb_cfg, end_rb_cfg in configs:
        bler_curve = simulate_bler_config(
            snr_points,
            n_trials=10000,  # 你可以调大/调小
            ncellid=208,
            fs=30.72e6,
            scs_khz=15,
            seed=1938,
            nrb_ssb=nrb_cfg,
            nsym_ssb=nsym_cfg,
            nrb_ss=nrb_ss_cfg,
            use_combining=True,
            N_comb=comb_cfg,
            start_rb = start_rb_cfg, end_rb = end_rb_cfg
        )
        curves[label] = bler_curve

    # 画图
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5.5))
    bler = []
    for label, bler in curves.items():
        plt.semilogy(snr_points, np.maximum(bler, 1e-3), label='normal', marker='o')
    plt.grid(True, which='both')
    plt.xlabel("SNR (dB)")
    plt.ylabel("BLER")
    plt.title(f"PBCH BLER vs SNR, (Comb=1),  (NO_CHANNEL={NO_CHANNEL}, KNOWN_nVar={USE_KNOWN_NVAR})")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("pbch_remapping_py3gpp_SCL_highSNR.png", dpi=150)
    print("已保存")