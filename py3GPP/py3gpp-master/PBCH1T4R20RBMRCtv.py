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
from myChannelEstimation4 import DMRSFilterGenerate_v2_explicit, myChannelEstimate, myChannelEstimatev2, \
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


import numpy as np
from scipy import sparse
# 假设 ChannelInfo 已经在你的环境中定义好了

def apply_tdl_c(x, fs_hz, seed=None, ds=300e-9, fc_hz=7e9, speed_kmh=3, nfft=2048, normalize=False, nRx=4, use_time_varying=True):
    """
    统一的 1T4R TDL-C 过信道接口，完美支持极速时变（Jakes）与时不变（Static）切换。
    返回:
      y: 形状为 (len(x), nRx) 的时域接收信号
      h_time: 形状为 (nRx, DS_tap, len(x)) 的冲激响应 (如果是时不变，时间维度为冗余展开)
    """
    if seed is not None:
        np.random.seed(seed)

    L = len(x)

    # ==========================================
    # 1. 构造信道对象
    # ==========================================
    # 注意：这里的 ds 传入如果是秒，确保 ChannelInfo 内部处理是对的 (如你之前的 DS_ns = ds)
    # 如果 ChannelInfo 需要 ns，请在这里乘以 1e9，比如 DS=ds*1e9。此处按你原本逻辑传入。
    ch = ChannelInfo(
        nTx=1, nRx=nRx,
        Speed=speed_kmh,
        Fc=fc_hz,
        Fs=fs_hz,
        ChanType='TDLC',
        DS=ds,  
        NFFT=nfft,
        TO=0, FO=0
    )

    # ==========================================
    # 2. 生成 1T4R 的冲激响应
    # ==========================================
    if use_time_varying:
        # 🏎️ 时变信道：生成每一秒都在变化的 Jakes 多普勒衰落
        t = np.arange(L) / fs_hz
        ch.gen_Jakes_Accelerate(time_samples=t)  
        # 取 1 发 nRx 收: [nRx, 1(nTx), DS_tap, T] -> [nRx, DS_tap, T]
        h_time = ch.time_channel[:, 0, :, :]
    else:
        # 🧱 时不变信道：只生成 1 个静态采样点
        ch.gen_Rayleigh(num_samples=1)
        h_static = ch.time_channel[:, 0, :, 0]  # [nRx, DS_tap]
        # 为了输出维度统一，将静态抽头在时间维度上复制 L 份 -> [nRx, DS_tap, T]
        h_time = np.repeat(h_static[:, :, np.newaxis], L, axis=2)

    # ==========================================
    # 3. 严格的物理能量归一化
    # ==========================================
    if normalize:
        # 计算所有天线、所有时间点的平均总能量
        total_power = np.sum(np.abs(h_time)**2) / L
        if total_power > 0:
            avg_power_per_ant = total_power / nRx
            h_time /= np.sqrt(avg_power_per_ant)

    # ==========================================
    # 4. 过信道 (时变稀疏矩阵乘法 vs 时不变一维卷积)
    # ==========================================
    y = np.zeros((L, nRx), dtype=np.complex128)
    num_taps = h_time.shape[1]

    for r in range(nRx):
        h_tap_t = h_time[r, :, :]  # 取出当前第 r 根天线的时变抽头矩阵: [num_taps, L]

        if not use_time_varying:
            # 时不变降级：直接用高速的 np.convolve
            y[:, r] = np.convolve(x, h_tap_t[:, 0], mode='same')
        
        else:
            # 高级时变模式：利用 scipy.sparse 构建时间位移矩阵 (0 内存溢出风险)
            diags = []
            offsets = []
            for k in range(num_taps):
                if k >= L: 
                    break
                # h_tap_t[k, :] 是第 k 径在所有时间点的值
                # 对应方程 y[n] = sum_k h[k, n] * x[n-k]
                diags.append(h_tap_t[k, k:L])  
                offsets.append(k)              

            H_eff_sparse = sparse.diags(
                diagonals=diags,
                offsets=offsets,
                shape=(L, L),
                dtype=np.complex128,
                format="csr"
            )

            # 稀疏矩阵转置右乘，完美等价于物理世界的时变因果卷积
            y[:, r] = H_eff_sparse.T @ x

    return y, h_time


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
        fs=61.44e6,
        scs_khz=30,
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
                if use_combining:
                    ok, info = one_shot_pbch_min_combined(
                        snr_db, ncellid=ncellid, fs=fs, scs_khz=scs_khz,
                        nrb_ssb=nrb_ssb, nsym_ssb=nsym_ssb, nrb_ss=nrb_ss,
                        ibar_tx=None, rng_seed=rng.randint(1 << 31),
                        N_comb=N_comb,start_rb=start_rb,end_rb=end_rb
                    )
                else:
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
        fs=61.44e6, scs_khz=30, nrb_ssb=20, nsym_ssb=4,
        use_known_nvar=False,
        start_rb=0,end_rb=20
):

    tx_wave, _ = nrOFDMModulate(carrier=None, grid=ssb_grid, scs=scs_khz, SampleRate=fs)

    # 信道
    if NO_CHANNEL:
        ch_out = tx_wave
    else:
        ch_out, ch_obj = apply_tdl_c(tx_wave, fs,
                                     speed_kmh=3,
                                     fc_hz=7e9,
                                     ds=300,
                                     nfft=2048,
                                     normalize=False)

    nRx = ch_out.shape[1] if ch_out.ndim > 1 else 1
    
    # 获取频域网格的尺寸 K 和 N
    K = nrb_ssb * 12
    N = nsym_ssb
    
    # 初始化三维网格 rxGrid_clean: (子载波, OFDM符号, 天线数)
    rxGrid_clean = np.zeros((K, N, nRx), dtype=complex)
    
    for r in range(nRx):
        # 对每根天线独立做 OFDM 解调
        rxGrid_clean[:, :, r] = nrOFDMDemodulate(
            waveform=ch_out[:, r], nrb=nrb_ssb, scs=scs_khz,
            initialNSlot=0, SampleRate=fs
        )[:, :nsym_ssb]

    # 获取有效 RE 的索引
    active_idx = np.sort(np.concatenate([pbch_idx[pbch_mask], dmrs_idx[dmrs_mask]]))
    
    # 调用更新后的 1T4R 加噪函数
    rxGrid, nVar_ref = add_awgn_on_gri_per_symbol(rxGrid_clean, ssb_grid, snr_db, active_idx)
    nrb_ss = 12
    # ibar_hat = estimate_ibar_once(rxGrid, ncellid, nrb_ssb, nsym_ssb, nrb_ss, ibar_tx)
    ibar_hat = ibar_tx
    
    # 获取网格的维度信息（此时 rxGrid 是三维的: K x N x nRx）
    K_grid, N_grid, nRx = rxGrid.shape
    
    # 参考导频网格（CE），发送端是单天线，所以 refGridH 保持二维 (K, N) 即可
    refGridH = np.zeros((K_grid, N_grid), dtype=complex)
    dmrs_syms = nrPBCHDMRS_param(ncellid, ibar_hat, len(dmrs_idx))
    nrSetResources(dmrs_idx[dmrs_mask], refGridH, dmrs_syms[dmrs_mask])
    
    # 提取 QPSK 的比特掩码（每 RE 对应两比特）
    bit_mask = np.repeat(pbch_mask.astype(bool), 2)
    
    # 初始化最终累加的 LLR
    llr_sum = np.zeros(E, dtype=float)

    # 提前解调发送端的 txGrid_id (用于后续计算 H_true 和 NMSE)，避免在多天线循环中重复计算
    if not NO_CHANNEL:
        txGrid_id = nrOFDMDemodulate(
            waveform=tx_wave, nrb=nrb_ssb, scs=scs_khz,
            initialNSlot=0, SampleRate=fs
        )[:, :nsym_ssb]
        tx_mask = (np.abs(txGrid_id) > 0)

    # ==========================================
    # 核心修改点：准备多天线数据容器
    # ==========================================
    N_pbch_re = len(pbch_idx)
    Y_all = np.zeros((N_pbch_re, nRx), dtype=complex)
    H_all = np.zeros((N_pbch_re, nRx), dtype=complex)

    # 1. 独立信道估计与符号提取 (仅提数据，不在这里做均衡和解调)
    for r in range(nRx):
        rxGrid_r = rxGrid[:, :, r]
        
        if NO_CHANNEL:
            dmrs_y = nrExtractResources(dmrs_idx, rxGrid_r)
            err = dmrs_y - dmrs_syms
            nVar_est = float(np.mean(np.abs(err) ** 2))
            nVar = nVar_ref if use_known_nvar else nVar_est
            
            Y_all[:, r] = nrExtractResources(pbch_idx, rxGrid_r)
            H_all[:, r] = np.ones(N_pbch_re, dtype=complex)
            
        elif ONLY_TDLC:
            ch_only = nrOFDMDemodulate(waveform=ch_out[:, r], nrb=nrb_ssb, scs=scs_khz, initialNSlot=0, SampleRate=fs)[:, :nsym_ssb]
            H_true = np.ones_like(ch_only, dtype=complex)
            H_true[tx_mask] = ch_only[tx_mask] / txGrid_id[tx_mask]
            
            nVar = nVar_ref
            Y_all[:, r] = nrExtractResources(pbch_idx, rxGrid_r)
            H_all[:, r] = nrExtractResources(pbch_idx, H_true)
            
        else:
            H = myChannelEstimate_std(
                rxGrid=rxGrid_r, refSym=dmrs_syms[dmrs_mask], refGrid=refGridH, 
                EST_TFDOMAIN=False, EST_PBCH=True, start_rb=start_rb, end_rb=end_rb
            )
            nVar = nVar_ref 
            Y_all[:, r] = nrExtractResources(pbch_idx, rxGrid_r)
            H_all[:, r] = nrExtractResources(pbch_idx, H)

    # ==========================================
    # 2. 1T4R 完美 MRC 联合均衡 (基于 ZF 架构的无损权重)
    # ==========================================
    # 恢复原始的按符号噪声方差 (Shape: N_sym,) 供解调器使用
    nVar_sym = nVar_ref if use_known_nvar else nVar

    # 沿着天线维度(axis=1)求信道能量: H_aveg = H^H * H (Shape: 432,)
    H_power = np.sum(np.abs(H_all)**2, axis=1)  
    
    # 采用 ZF 迫零方式求出信号幅度严格归一化为 1 的符号
    # 加上 eps 防止在极端深衰落的子载波上除以 0
    eps = 1e-12
    pbch_eq_zf = np.sum(H_all.conj() * Y_all, axis=1) / (H_power + eps)
    
    # 🚨 核心改动：直接将信道能量 H_power 作为 CSI 喂给解调器
    # 这样底层软解调器在计算 LLR 时，会严格执行 LLR ∝ real(pbch_eq_zf) * H_power / nVar
    # 完美实现深衰落抑制，榨干每一滴频率分集增益！
    csi_optimal = H_power
    
    # ==========================================
    # 3. 统一做一次软解调
    # ==========================================
    llr_sum = nrSymbolDemodulate_param_persym(
        pbch_eq_zf, 'QPSK', nVar_sym, pbch_idx, nrb_ssb * 12, csi_optimal, 'soft'
    ) 
    
    # 4. 对齐掩码并把未使用的 RE 位置置零
    bit_mask = np.repeat(pbch_mask.astype(bool), 2)
    llr_sum[~bit_mask] = 0.0

    assert len(llr_sum) == E
    return llr_sum


def one_shot_pbch_min_combined(
        snr_db, ncellid,
        fs=61.44e6, scs_khz=30,
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
        decoded_bits, crc_1 = polar_decode_scl_llr(decIn, K=K, N=N, list_size=8, crc_degree="CRC24C")
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
    np.random.seed(2026)
    # 建议用稍“温和”的 SNR 段，便于拉开曲线
    # print_nmse_compare_db(snr_db=-2.0, ncellid=208, fs=30.72e6, scs_khz=30, trials_each=300)
    snr_points =  [-8.5,-8,-7.5,-7,-6.5,-6,-5.5,-5,-4.5]
    # snr_points =  [4.5,5,5.5,6,6.5,7,7.5]
    #snr_points = [ -4, -3, -2, -1,0,1,2,3,4,5]  # 也可以换回你之前的段, -13,   `-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,0,1,2,3,4,5,6,8,10
    # snr_points1 =  [-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9]
    # 要对比的 7 组配置（标签, nrb_ssb, nsym_ssb, nrb_ss）
    # 注意：nrb_ss 不要大于 nrb_ssb；这里统一用 12 个 RB 的中心带宽，
    # 在 nrb_ssb = 12 的场景会让 PSS/SSS 覆盖整个带宽（合法，只是 PBCH 在对应符号上被挖空）
    configs = [

        #
         ("20RB × 4sym 1 comb puncture 0 rb", 20, 4, 12, 1, 0, 20 - 0),


    ]

    curves = {}
    for label, nrb_cfg, nsym_cfg, nrb_ss_cfg, comb_cfg,start_rb_cfg, end_rb_cfg in configs:
        bler_curve = simulate_bler_config(
            snr_points,
            n_trials=500,  # 你可以调大/调小
            ncellid=208,
            fs=61.44e6,
            scs_khz=30,
            seed=889,
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
    plt.title(f"PBCH BLER vs SNR (1T4R, Comb=1), NO_CHANNEL={NO_CHANNEL}")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("pbch_remapping_py3gpp_SCL_1T4R.png", dpi=150)
    print("已保存")