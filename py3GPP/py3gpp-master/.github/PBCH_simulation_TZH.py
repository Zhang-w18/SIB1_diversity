import numpy as np
from py3gpp import *
'''
(
    nrOFDMModulate, nrOFDMDemodulate,
    nrSetResources, nrExtractResources,
    nrPBCHIndices, nrPBCHDMRS, nrPBCHDMRSIndices,
    nrPBCHPRBS, nrChannelEstimate, nrEqualizeMMSE,
    nrSymbolModulate, nrSymbolDemodulate,
    nrRateRecoverPolar, nrPolarDecode, nrCRCDecode,
    nrBCH
)
'''

# ---------------------------
# 开关 & 辅助
# ---------------------------
NO_CHANNEL = False          # 先用纯 AWGN 验证链路 ——> True
USE_KNOWN_NVAR = False      # 用 SNR 推噪声方差 ——> True
PRINT_DIAG = False #True          # 打印关键诊断信息

def add_awgn(x, snr_db):
    p_sig = np.mean(np.abs(x)**2)
    p_noise = p_sig / (10**(snr_db/10.0))
    n = (np.random.randn(*x.shape) + 1j*np.random.randn(*x.shape)) * np.sqrt(p_noise/2.0)
    return x + n, p_noise  # 返回噪声功率，便于 nVar 计算

def tdl_c_impulse_response(fs_hz, seed=None):
    """
    生成符合 3GPP TR 38.901 TDL-C profile 的多径信道冲激响应。

    参数:
        fs_hz : float
            系统采样率 (Hz)，例如 30.72e6。
        ds : float
            Delay Spread (秒)，默认 300 ns。
        seed : int or None
            随机种子，便于复现。

    返回:
        h : ndarray(complex)
            复数基带离散脉冲响应（长度为最大延时采样 + 1）。
    """
    ds = 300e-9

    # ---- (1) Path delays & powers from 3GPP TR 38.901 Table 7.7.2-4 (TDL-C)
    delays_per_path = np.array([
        0, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366, 0.6448, 0.6560,
        0.6584, 0.7935, 0.8213, 0.9336, 1.2285, 1.3083, 2.1704, 2.7105,
        4.2589, 4.6003, 5.4902, 5.6077, 6.3065, 6.6374, 7.0427, 8.6523
    ]) * ds  # scale by DS (delay spread)

    power_per_path_db = np.array([
        -4.4, -1.2, -3.5, -5.2, -2.5, 0, -2.2, -3.9,
        -7.4, -7.1, -10.7, -11.1, -5.1, -6.8, -8.7, -13.2,
        -13.9, -13.9, -15.8, -17.1, -16.0, -15.7, -21.6, -22.8
    ])

    # ---- (2) Convert power(dB) → linear, normalize
    p_lin = 10 ** (power_per_path_db / 10.0)
    p_lin /= np.sum(p_lin)

    # ---- (3) Generate complex Gaussian fading taps
    if seed is not None:
        np.random.seed(seed)
    taps = (np.random.randn(len(p_lin)) + 1j * np.random.randn(len(p_lin))) / np.sqrt(2)
    taps *= np.sqrt(p_lin)

    # ---- (4) Convert continuous delays (s) to discrete sample delays
    d_samp = np.round(delays_per_path * fs_hz).astype(int)
    L = d_samp.max() + 1
    h = np.zeros(L, dtype=complex)
    for d, tap in zip(d_samp, taps):
        h[d] += tap

    # ---- (5) Normalize to unit total power
    h /= np.sqrt(np.sum(np.abs(h) ** 2))
    return h


def apply_tdl_c(x, fs_hz, seed=None):
    h = tdl_c_impulse_response(fs_hz, seed)
    # 用“same”保持长度和对齐，不引入整体时移（避免起点错位）
    y = np.convolve(x, h, mode='same')
    #print('h=',h)
    return y, h

# ---------------------------
# 发端：只构造 PBCH + PBCH DMRS
# ---------------------------
def build_ssb_grid_pbch_only(ncellid, ibar_ssb, trblk_bits, nrb_ssb=20):
    grid = np.zeros((nrb_ssb*12, 4), dtype=complex)
    pbch_bits = nrBCH(trblk_bits, sfn=0, hrf=0, lssb=0, idxoffset=0, ncellid=ncellid)
    pbch_symb = nrSymbolModulate(pbch_bits, 'QPSK')
    nrSetResources(nrPBCHIndices(ncellid), grid, pbch_symb)
    nrSetResources(nrPBCHDMRSIndices(ncellid), grid, nrPBCHDMRS(ncellid, ibar_ssb))
    return grid

# ---------------------------
# 单次试验（已知 NID；仅 DMRS 检测 + CE + 解码）
# ---------------------------
def one_shot_pbch_min(snr_db, ncellid, fs=30.72e6, scs_khz=15, nrb_ssb=20, ibar_tx=None, rng_seed=None):
    if rng_seed is not None: np.random.seed(rng_seed)
    if ibar_tx is None: ibar_tx = np.random.randint(0,8)

    # 随机 32 比特 PBCH 传输块
    A = 32
    trblk_bits = np.random.randint(0,2,A).astype(int)

    # 发端：网格→OFDM
    ssb_grid = build_ssb_grid_pbch_only(ncellid, ibar_tx, trblk_bits, nrb_ssb=nrb_ssb)
    tx_wave, info = nrOFDMModulate(carrier=None, grid=ssb_grid, scs=scs_khz, SampleRate=fs)

    #print('info: ', info)

    # 信道
    if NO_CHANNEL:
        ch_out = tx_wave.copy()
    else:
        ch_out, _ = apply_tdl_c(tx_wave, fs)

    rx_wave, p_noise = add_awgn(ch_out, snr_db)

    # 直接整段反变换，确保至少4列
    rxGrid_full = nrOFDMDemodulate(waveform=rx_wave, nrb=nrb_ssb, scs=scs_khz,
                                   initialNSlot=0, SampleRate=fs, CyclicPrefixFraction=0.5)
    if PRINT_DIAG:
        print("rxGrid_full shape:", rxGrid_full.shape)
    if rxGrid_full.shape[1] < 4:
        return False, dict(reason=f"Demodulated symbols={rxGrid_full.shape[1]} < 4")
    rxGrid = rxGrid_full[:, 0:4]


    # DMRS 检测：搜索 ibar 0..7
    dmrs_idx = nrPBCHDMRSIndices(ncellid)

    '''
    G = np.zeros_like(rxGrid)
    v_test = 0
    nrSetResources(dmrs_idx, G, nrPBCHDMRS(ncellid, v_test))
    assert np.allclose(nrExtractResources(dmrs_idx, G), nrPBCHDMRS(ncellid, v_test))
    '''


    '''
    # 相干叠加，这个算法有问题，实现不了DMRS检测
    dmrs_rx = nrExtractResources(dmrs_idx, rxGrid)
    corr = np.zeros(8)
    for v in range(8):
        dmrs_ref = nrPBCHDMRS(ncellid, v)
        corr[v] = np.abs(np.vdot(dmrs_ref, dmrs_rx))# / (np.linalg.norm(dmrs_ref)*np.linalg.norm(dmrs_rx) + 1e-12)
    ibar_hat = int(np.argmax(corr))
    '''


    '''
    # ---- 变换回时域加窗，统计窗内能量，序列判决 ----
    K = rxGrid.shape[0]  # 子载波数（240）
    dmrs_idx_all = nrPBCHDMRSIndices(ncellid)  # 全局 DMRS 索引（按列优先）
    dmrs_rx_all = nrExtractResources(dmrs_idx_all, rxGrid)

    # 按 4 个 PBCH 符号切分 DMRS 索引
    sym_masks = []
    for s in range(4):
        base = s * K
        mask = (dmrs_idx_all >= base) & (dmrs_idx_all < base + K)
        sym_masks.append(mask)

    # 时域窗口长度（经验 5%~15% 都行；信道 DS=300ns 远小于符号时长，这个范围比较稳健）
    ratio = 0.10
    L = max(1, int(np.ceil(K * ratio)))

    corr = np.zeros(8)
    for v in range(8):
        dmrs_ref_all = nrPBCHDMRS(ncellid, v)
        score_v = 0.0

        for s in range(4):
            mask = sym_masks[s]
            if not np.any(mask):
                continue

            # 频域匹配滤波：只在 DMRS RE 上做 y*conj(ref)，其他子载波置零
            Ys = np.zeros(K, dtype=complex)
            kpos = (dmrs_idx_all[mask] - s * K).astype(int)  # 该符号内的子载波位置
            Ys[kpos] = dmrs_rx_all[mask] * np.conj(dmrs_ref_all[mask])

            # IFFT 到时域（除以sqrt K保能量一致）
            ys_td = np.fft.ifft(Ys) * np.sqrt(K)
            e = np.abs(ys_td) ** 2  # 时域能量包络（长度 K）

            # ——做“环形”滑窗能量最大值——
            # 用拼接实现环形卷积的滑窗和：对 e||e 做长度 L 的窗口求和，取前 K 个起点的最大值
            e_dbl = np.concatenate([e, e])
            csum = np.cumsum(np.concatenate([[0.0], e_dbl]))
            # window_sums[i] = sum(e_dbl[i : i+L])
            window_sums = csum[L:L + K] - csum[0:K]
            score_v += float(np.max(window_sums))

        corr[v] = score_v

    ibar_hat = int(np.argmax(corr))
    '''

    '''
    # ---- 变换回时域加窗，统计窗内能量，序列判决 ----优化版本
    K = rxGrid.shape[0]  # 240
    dmrs_idx_all = nrPBCHDMRSIndices(ncellid)  # (Ndmrs_total,)
    dmrs_rx_all = nrExtractResources(dmrs_idx_all, rxGrid)  # (Ndmrs_total,)

    # 参考 DMRS：一次性拿到所有 v=0..7（8 × Ndmrs）
    refs = np.vstack([nrPBCHDMRS(ncellid, v) for v in range(8)])  # complex64/128
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

    print("ibar_tx:", ibar_tx, "ibar_hat:", ibar_hat, "DMRS corr:", np.round(corr,3))
    '''

    # 二维信道估计
    refGridH = np.zeros((nrb_ssb*12, 4), dtype=complex)
    nrSetResources(dmrs_idx, refGridH, nrPBCHDMRS(ncellid, ibar_tx))   #如果做了dmrs检测，这里用ibar_hat
    H, _ = nrChannelEstimate(rxGrid=rxGrid, refGrid=refGridH)

    # 频域 DMRS 残差估噪（推荐）
    dmrs_idx = nrPBCHDMRSIndices(ncellid)
    dmrs_y = nrExtractResources(dmrs_idx, rxGrid)  # 观测
    dmrs_x = nrPBCHDMRS(ncellid, ibar_tx)  # 参考（若用 dmrs 检测结果就传 ibar_hat）
    dmrs_h = nrExtractResources(dmrs_idx, H)  # 估计的信道
    err = dmrs_y - dmrs_h * dmrs_x
    nVar_est = float(np.mean(np.abs(err) ** 2))  # 频域/当前尺度下的噪声方差


    # MMSE 均衡
    pbch_idx = nrPBCHIndices(ncellid)
    y_pbch = nrExtractResources(pbch_idx, rxGrid)
    h_pbch = nrExtractResources(pbch_idx, H)

    # nVar 选择：用已知SNR推导 or 用估计值
    # LLR 的 nVar 通常指“每复符号噪声功率”，与 nrSymbolDemodulate 的定义保持一致
    if USE_KNOWN_NVAR:
        nVar = p_noise      # = 平均符号功率 / SNR_linear
    else:
        nVar = nVar_est

    pbch_eq, csi = nrEqualizeMMSE(y_pbch, h_pbch, nVar)

    # 简单 EVM 看看均衡效果（可注释）
    #ref_hard = nrSymbolModulate(nrSymbolDemodulate(pbch_eq, 'QPSK', 1, 'hard'), 'QPSK')
    #evm = float(np.mean(np.abs(pbch_eq - ref_hard)))
    #if PRINT_DIAG:
    #    print("Mean |err| (EVM proxy):", round(evm,4))

    # 软解调 → 解扰 → CSI 加权
    llr = nrSymbolDemodulate(pbch_eq, 'QPSK', nVar, 'soft')  # 长度 864
    E = 864

    #scram = nrPBCHPRBS(ncellid, ibar_hat, E)
    #llr_descr = llr * (1 - 2*scram)
    llr_descr = llr
    llr_descr_csi = llr_descr * np.repeat(csi, 2)    #csi加权之后，译码性能会好很多

    # 逆率匹配 → 极化译码 → CRC24C
    P = 24; K = 32 + P; N = 512
    decIn = nrRateRecoverPolar(llr_descr, K, N, False, discardRepetition=False)
    decoded = nrPolarDecode(decIn, K, 0, 0)
    _, crc = nrCRCDecode(decoded, '24C')
    crc_ok = (crc == 0)
    return crc_ok, dict(ibar_tx=ibar_tx, ibar_hat=ibar_tx)

# ---------------------------
# Monte Carlo：BLER
# ---------------------------
def simulate_bler_min(snr_db_list, n_trials=20, ncellid=210, fs=30.72e6, scs_khz=15, seed=2025):
    bler = []
    rng = np.random.RandomState(seed)
    for snr_db in snr_db_list:
        #print('SNR=', snr_db)
        n_err = 0
        round_total = 0

        for _ in range(n_trials):
            #print('round=', round_total)
            #print('n_error=', n_err)
            ok, info = one_shot_pbch_min(
                snr_db, ncellid=ncellid, fs=fs, scs_khz=scs_khz,
                nrb_ssb=20, ibar_tx=None, rng_seed=rng.randint(1<<31)
            )
            round_total += 1
            #print('round_total=', round_total)
            if not ok:
                n_err += 1
                if n_err >= 10:
                    break
        bler.append(n_err / round_total)

        print(f"SNR={snr_db:>4.1f} dB  BLER={bler[-1]:.3f}")
    return np.array(bler)

# ---------------------------
# 直接执行：仿真 + 存图
# ---------------------------
snr_points = [-16, -14, -12, -10,-8, -6]
bler_curve = simulate_bler_min(snr_points, n_trials=1000, ncellid=210, fs=30.72e6, scs_khz=15, seed=2025)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.figure()
plt.semilogy(snr_points, np.maximum(bler_curve, 1e-3), marker='o')
plt.grid(True, which='both')
plt.xlabel("SNR (dB)"); plt.ylabel("BLER")

title = f"PBCH (DMRS detect + CE + Decode)  NO_CHANNEL={NO_CHANNEL}  KNOWN_nVar={USE_KNOWN_NVAR}"
plt.title(title)
plt.savefig("pbch_bler_min_no_dmrs_detection.png", dpi=150, bbox_inches='tight')
print("BLER 曲线已保存到 pbch_bler_min.png")
