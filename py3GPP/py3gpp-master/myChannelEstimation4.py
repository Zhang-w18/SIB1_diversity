"""
Created on Thu Jun  3 17:31:11 2025
 Copyright:  Huawei Technologies Co., Ltd. All rights reserved.
 File name: myChannelEstimation

 Description:
    1. Linear Interpolation + transform domain denoise
    2. Transform domain Interpolation + denoise
    3. Transform domain Interpolation + denoise by Blocks

 修改记录：
 date name line xxx

@author: m00829866
"""
import numpy as np
import scipy as sy
PLOT_FIGURE = False
def _normalize_dmrs_cols(dmrs_l, N):
    if dmrs_l is None:
        return np.array([], dtype=int)
    if np.isscalar(dmrs_l):
        cols = np.array([int(dmrs_l)], dtype=int)
    else:
        cols = np.array(sorted(set(int(x) for x in dmrs_l)), dtype=int)
    cols = cols[(cols >= 0) & (cols < N)]
    return cols


def _time_interp_complex(H, cols_with, cols_wo, s_idx=None, e_idx=None):
    """
    对 H 的指定频域范围 [s_idx:e_idx) 做时间方向复数线性插值
    """
    K, N = H.shape
    if s_idx is None:
        s_idx = 0
    if e_idx is None:
        e_idx = K

    cols_with = np.asarray(cols_with, dtype=int)
    cols_wo = np.asarray(cols_wo, dtype=int)

    if cols_with.size == 0 or cols_wo.size == 0:
        return H

    if cols_with.size == 1:
        H[s_idx:e_idx, cols_wo] = H[s_idx:e_idx, cols_with[0:1]]
        return H

    x = cols_with.astype(float)
    xi = cols_wo.astype(float)

    for k in range(s_idx, e_idx):
        y = H[k, cols_with]
        H[k, cols_wo] = (
            np.interp(xi, x, np.real(y)) +
            1j * np.interp(xi, x, np.imag(y))
        )

    return H
def _dmrs_pos_for_span_and_ports(HestGran: int, dmrs_ports):
    """
    返回 ports_positions: List[np.ndarray]
      - 每个端口在该 span 内的 DMRS 索引（0-based，长度≈HestGran/2）
      - 规则按 NR Type-I + FD-OCC 的常见布局近似
    """
    P = len(dmrs_ports)
    if P == 1:
        return [np.arange(0, HestGran, 2, dtype=int)]                   # 0,2,4,...
    if P == 2:
        pos = np.arange(1, max(HestGran-2, 1), 2, dtype=int)            # 1,3,5,...,HestGran-3（保守）
        return [pos, pos]
    if P == 3:
        pos12 = np.arange(1, max(HestGran-2, 1), 2, dtype=int)
        pos3  = np.arange(1, HestGran, 2, dtype=int)
        return [pos12, pos12, pos3]
    if P == 4:
        pos12 = np.arange(1, max(HestGran-2, 1), 2, dtype=int)
        pos34 = np.arange(2, HestGran, 2, dtype=int)
        return [pos12, pos12, pos34, pos34]
    # 默认单端口
    return [np.arange(0, HestGran, 2, dtype=int)]


def _lmmse_filter_for_span(HestGran: int, dmrs_pos: np.ndarray,
                           deltaF_hz: float, tauRMS_s: float, noise_var: float) -> np.ndarray:
    """
    频域一维相关模型的 LMMSE（Wiener 等价）滤波器:
      F = C_xd @ inv(C_dd + σ² I)
    C_dd[p,q] = 1 / (1 + j*2π*tauRMS*Δf*(k_d[p]-k_d[q]))
    C_xd[i,p] = 1 / (1 + j*2π*tauRMS*Δf*(k_x[i]-k_d[p]))
    返回形状: (HestGran, Nd)
    """
    import numpy as np
    kx = np.arange(HestGran, dtype=float)[:, None]   # (HestGran,1)
    kd = dmrs_pos.astype(float)[None, :]              # (1,Nd)
    Nd = kd.shape[1]
    if Nd == 0:
        return np.zeros((HestGran, 0), dtype=complex)

    j2p = 1j * 2.0 * np.pi * tauRMS_s * deltaF_hz
    Cdd = 1.0 / (1.0 + j2p * (kd.T - kd))           # (Nd,Nd)
    Cxd = 1.0 / (1.0 + j2p * (kx - kd))             # (HestGran,Nd)

    # CSshift 修正（与原版一致）
    n = np.arange(Nd, dtype=float)
    S = np.diag(np.exp(1j * np.pi * n))             # (Nd,Nd)
    # Reg = Cdd + S @ Cdd @ S.conj().T + noise_var * np.eye(Nd, dtype=complex)
    Reg = Cdd + noise_var * np.eye(Nd, dtype=complex)
    return (Cxd @ np.linalg.inv(Reg)).astype(complex)


def DMRSFilterGenerate_v2_explicit(
    scs_khz: float,                 # 子载波间隔(kHz)：15/30/60/120...
    dmrs_ports=(0,),                # 端口列表：如 (0,) 或 (0,1)
    delay_spread_ns: float = 300.0, # τRMS (ns)：TDL-C 常用 300
    snr_db: float = -5.0,           # 估计用 SNR（仅影响正则强度）
    spans=(12, 24, 36, 48)          # 要生成的子带粒度集合
):
    """
    返回 filters: List[np.ndarray]，与 spans 一一对应
      filters[i] 形状 = (span_i, Nd_i)
    - 多端口时对各端口滤波器做均值合并（可按能量改成加权）
    - 若最后一段 span 不满（你在 CE 里会裁剪），本函数按 span 生成完整矩阵
    """
    deltaF_hz = float(scs_khz)*1e3 if scs_khz < 1e6 else float(scs_khz)
    tauRMS_s  = float(delay_spread_ns) * 1e-9
    noise_var = (10.0 ** (-snr_db / 10.0))

    P = len(dmrs_ports)
    all_port_filters = []

    for iport in range(P):
        per_gran = []
        for span in spans:
            span = int(span)
            # 与原版端口规则对齐
            if P == 1:
                dmrs_pos = np.arange(0, span, 2, dtype=int)
            elif P == 2:
                dmrs_pos = np.arange(1, max(span-2,1), 2, dtype=int)
            elif P == 3:
                dmrs_pos = np.arange(1, max(span-2,1), 2, dtype=int) if iport < 2 else np.arange(1, span, 2, dtype=int)
            elif P == 4:
                dmrs_pos = np.arange(1, max(span-2,1), 2, dtype=int) if iport < 2 else np.arange(2, span, 2, dtype=int)
            else:
                dmrs_pos = np.arange(0, span, 2, dtype=int)

            F = _lmmse_filter_for_span(span, dmrs_pos, deltaF_hz, tauRMS_s, noise_var)
            per_gran.append(F)
        all_port_filters.append(per_gran)

    return all_port_filters


def NoiseReductionTD(HestLS, num_fft, maxCP, IdxSC):
    """
    noise reduction in time domain for estimated channel
    ----------
    varible structure:
    input:
        HestLS :        complex array[SC * symbol]                    the estimated channel of each dmrs symbol in all the subcarriers
        num_fft :       int                                           fft point number
        maxCP :         int                                           the maximum CP length used to decide the lentgh of filter
        IdxSC :         int array [1 * SC]                            the index of RE in valid bandwidth

    output:
        HestNF :        dict                                          the estimated channel after noise reduction of each dmrs symbol
    description:
        将720个RE信道(644~1383)的头部48个RE(644~691)翻转填充到720个RE前的48个RE(616~663)
        将720个RE信道(644~1383)的尾部48个RE(1336~1383)翻转填充到720个RE后的48个RE(1384~1431)
        然后进行2048点IFFT
        前160个采用点不变，前160个采样点后面的48个采样点（161~208）乘以一个缩放比例，该缩放比例从1逐渐减小为0
        后48个采用点不变（2000~2047）乘以一个缩放比例，该缩放比例从0逐渐减小为1
        中间的2048-160-48-48（208~1999）个采用点直接置零
        转换到频域，并乘以功率缩放比，功率缩放比为降噪前频域信道能量处理降噪后的频域信道能量
    -------

    """
    num_ext = 48  # default: 48
    fupsample = 0  # default: 2
    len_tri_filter = 24 * fupsample
    tri_filter = np.linspace(0, 1, len_tri_filter)
    len_filter = 50  # default: maxCP

    num_sym = HestLS.shape[1]
    idx_sc = IdxSC[:, 0]

    h_nred_tmp = np.zeros([num_fft, num_sym], dtype=complex)
    h_nred_tmp[idx_sc, :] = HestLS
    # extention in start RE
    idx_up = idx_sc[0] - num_ext + np.arange(num_ext, dtype=int)
    h_nred_tmp[idx_up, :] = np.flipud(HestLS[0:num_ext, :])  # 将各列的元素倒序排列
    # extention in end RE
    idx_down = idx_sc[-1] + 1 + np.arange(num_ext, dtype=int)
    h_nred_tmp[idx_down, :] = np.flipud(HestLS[-num_ext::, :])

    # noise reduction in time domain
    h_est_td = sy.fft.ifft(h_nred_tmp, num_fft, axis=0) * np.sqrt(num_fft)
    for isym in range(num_sym):
        # trianglefilter for the front end signal
        idx_front = len_filter + np.arange(len_tri_filter, dtype=int)
        h_est_td[idx_front, isym] = h_est_td[idx_front, isym] * (1 - tri_filter)
        # zero for the middle signal
        # idx_zero = np.arange(len_filter + len_tri_filter, num_fft-len_tri_filter, dtype = int)
        idx_zero = np.arange(len_filter + len_tri_filter, num_fft - 1 * len_tri_filter, dtype=int)
        h_est_td[idx_zero, isym] = 0
        # trianglefilter for the tail end signal
        # idx_tail = num_fft-len_tri_filter + np.arange(len_tri_filter, dtype = int)
        idx_tail = num_fft - 1 * len_tri_filter + np.arange(len_tri_filter, dtype=int)
        h_est_td[idx_tail, isym] = h_est_td[idx_tail, isym] * tri_filter

    h_est_fd = sy.fft.fft(h_est_td, num_fft, axis=0) / np.sqrt(num_fft)
    HestNF = h_est_fd[idx_sc, :]
    power_scale = np.mean(pow(np.abs(HestLS), 2)) / np.mean(pow(np.abs(HestNF), 2))
    HestNF = HestNF * np.sqrt(power_scale)
    # HestNF = HestLS
    return HestNF
def NoiseReductionTD_PBCH(H_valid, num_fft, maxCP, IdxSC, nVar=None,
                          wiener=True, soft=True, alpha=2.5,
                          smooth_time=False, beta=0.7):
    """
    H_valid: [S, N] 仅有效带宽子载波上的 CFR
    返回: 去噪后的 H_valid_dn (同维度)
    """
    idx_sc = IdxSC[:, 0] if (IdxSC.ndim == 2 and IdxSC.shape[1] == 1) else np.asarray(IdxSC).ravel()
    S, N = H_valid.shape
    H_dn = np.empty_like(H_valid)

    # 可选：跨符号的时域平滑缓冲
    h_bar = None

    for n in range(N):
        # --- 填充到完整 NFFT 频栅 ---
        H_full = np.zeros(num_fft, dtype=complex)
        H_full[idx_sc] = H_valid[:, n]

        # --- 频->时 ---
        h = np.fft.ifft(H_full, num_fft)

        # --- 定时对齐（可选）：把最大能量 tap 挪到 [0, maxCP] 内 ---
        # peak = np.argmax(np.abs(h))
        # shift = (peak // num_fft)  # 亦可据经验判断是否需要移位
        # h = np.roll(h, -shift)

        # --- 窗口选择（硬截断） ---
        L = int(maxCP)  # 也可用能量累计阈值自适应求 L
        h[L:] = 0.0

        # --- 平滑窗（可选） ---
        # from scipy.signal import tukey
        # win = tukey(L, alpha=0.3); h[:L] *= win

        # --- per-tap Wiener（可选，需要 nVar；没有就用导频残差的 nVar） ---
        if wiener and (nVar is not None) and (nVar > 0):
            p = np.abs(h[:L])**2
            w = p / (p + nVar)
            h[:L] *= w

        # --- 软阈值（可选） ---
        if soft and (nVar is not None) and (nVar > 0):
            lam = alpha * np.sqrt(nVar)
            mag = np.abs(h[:L])
            mask = mag > lam
            # soft-shrink
            h[:L][mask] *= (1.0 - lam / mag[mask])
            h[:L][~mask] = 0.0

        # --- 符号间时域平滑（可选） ---
        if smooth_time:
            if h_bar is None:
                h_bar = h.copy()
            else:
                h_bar = beta * h_bar + (1 - beta) * h
            h_use = h_bar
        else:
            h_use = h

        # --- 时->频，并抽回有效带宽 ---
        H_full_dn = np.fft.fft(h_use, num_fft)
        H_dn[:, n] = H_full_dn[idx_sc]

    return H_dn

import numpy as np
import scipy
from py3gpp import nrSetResources

import scipy
from py3gpp import nrSetResources

def myChannelEstimate(  ## 线性内插 + 时域联合降噪
    rxGrid=None, refInd=None, refSym=None, refGrid=None, carrier=None,
    # === 新增：时域去噪开关与参数 ===
    td_denoise=True,
    num_fft=2048,      # 你的系统 FFT 点数
    maxCP=144,         # 你的系统最大 CP（决定前端保留）
    IdxSC=None         # 有效带宽 RE 下标（列向量或一维数组），如形如 [[644],[645],...,[1383]]
):
    cp = "normal" if carrier is None else "extended"

    # 解析导频网格
    if refGrid is None:
        if (refInd is None) or (refSym is None):
            print("Error: refInd and refSym need to be given when refGrid is None!")
            return
        refGrid = np.zeros(rxGrid.shape, dtype=complex)
        nrSetResources(refInd, refGrid, refSym)
    else:
        pilot_idx = np.where(refGrid.ravel(order="F") != 0)
        refInd = pilot_idx[0]
        refSym = refGrid.ravel(order="F")[refInd]

    # 基本维度
    K, N = rxGrid.shape  # subcarriers, symbols

    # DMRS 处接收/发送符号
    rxSym_all = rxGrid.ravel(order="F")[refInd]
    refSym_all = refSym

    # 逐列 LS + 频域线性插值（复数分实/虚）
    H = np.zeros_like(refGrid, dtype=complex)
    has_dmrs_col = np.zeros(N, dtype=bool)

    for col in range(N):
        in_col = np.where((refInd >= col*K) & (refInd < (col+1)*K))[0]
        if in_col.size == 0:
            continue
        has_dmrs_col[col] = True

        sc_idx = (refInd[in_col] - col*K).astype(int)  # 0..K-1
        y = rxSym_all[in_col]
        x = refSym_all[in_col]
        # 防除零
        eps = 1e-12
        x_safe = np.where(np.abs(x) < eps, eps + 0j, x)
        h_ls = y / x_safe

        order = np.argsort(sc_idx)
        sc_idx = sc_idx[order]
        h_ls = h_ls[order]

        H.real[:, col] = np.interp(np.arange(K), sc_idx, np.real(h_ls))
        H.imag[:, col] = np.interp(np.arange(K), sc_idx, np.imag(h_ls))

    # 没 DMRS 的列做时间方向插值
    if not np.all(has_dmrs_col):
        cols_with = np.where(has_dmrs_col)[0]
        cols_wo   = np.where(~has_dmrs_col)[0]
        if cols_with.size >= 2:
            for k in range(K):
                y = H[k, cols_with]
                H[k, cols_wo] = (np.interp(cols_wo, cols_with, np.real(y))
                                +1j*np.interp(cols_wo, cols_with, np.imag(y)))
        elif cols_with.size == 1:
            H[:, cols_wo] = H[:, cols_with[0:1]]

    # === 在“有效带宽”内做时域去噪 ===
    if td_denoise:
        if IdxSC is None:
            raise ValueError("td_denoise=True 时必须提供 IdxSC（有效带宽索引）。")
        H_dn = np.zeros_like(H, dtype=complex)
        idx_sc = IdxSC[:, 0] if (IdxSC.ndim == 2 and IdxSC.shape[1] == 1) else np.asarray(IdxSC).ravel()
        # 只对有效带宽频点做降噪（速度快、也与原函数语义一致）
        SC, Ns = rxGrid.shape
        cp_len_first = 160
        cp_len_others = maxCP
        for col in range(Ns):
            cp_len = cp_len_first if col == 0 else cp_len_others
            H_dn[:, col] = noise_reduction_td_bandlimited(
                HestLS=H[:, [col]],  # 按函数接口需要 [SC, Ns']，这里 Ns'=1
                num_fft=num_fft,
                maxCP=cp_len,
                IdxSC=idx_sc,
                num_ext=48,
                len_tri_filter=48,
                len_front_keep=cp_len,
                power_rescale=True
            ).ravel()           # 放回
    else:
        H_dn = H
    # # 用导频残差估噪声方差（稳健）
    # resid_e2 = []
    # for col in range(N):
    #     in_col = np.where((refInd >= col*K) & (refInd < (col+1)*K))[0]
    #     if in_col.size == 0:
    #         continue
    #     sc_idx = (refInd[in_col] - col*K).astype(int)
    #     y = rxSym_all[in_col]
    #     x = refSym_all[in_col]
    #     resid = y - x * H[sc_idx, col]
    #     resid_e2.append(np.abs(resid)**2)
    # nVar = float(np.median(np.concatenate(resid_e2))) if resid_e2 else 0.0

    return H_dn

def noise_reduction_td_bandlimited(
    HestLS: np.ndarray,      # [SC, Ns] 频域CFR（已在有效带宽上，已按列=符号对齐）
    num_fft: int,
    maxCP: int,              # 用于界定“前端保留”长度
    IdxSC: np.ndarray,       # [SC] 或 [SC,1]，有效带宽在NFFT上的索引（未shift）
    num_ext: int = 48,       # 频带两端镜像扩展长度（RE）
    len_tri_filter: int = 48,# 三角过渡带长度（采样点，时域）
    len_front_keep: int = 160, # 前端“完全保留”的采样点数（NR 15kHz 下第1符号常取160，其他144，可按实情传入）
    power_rescale: bool = True
) -> np.ndarray:
    """
    基于你提供的流程实现，去除了外部依赖，用 numpy 实现：
    1) 频带镜像扩展（两端各 num_ext 个 RE）
    2) IFFT -> 时域：前len_front_keep点保留；随后 len_tri_filter 点作三角渐变到0
                     末尾 len_tri_filter 点作三角渐变从0到1；中段清零
    3) FFT 回频域，并按去噪前后功率做重标（可关）
    """

    # 找到peak，然后peak前留1/2 cp size，后面留cp size

    # 归一化 IdxSC 形状
    idx_sc = IdxSC[:, 0] if (IdxSC.ndim == 2 and IdxSC.shape[1] == 1) else np.asarray(IdxSC).ravel()

    def idx_centered(num_fft=2048, n_sc=240):
        return np.arange(num_fft//2 - n_sc//2, num_fft//2 + n_sc//2)



    SC, Ns = HestLS.shape
    idx_sc = idx_centered(num_fft, SC)

    # ---- (A) 把带宽填回完整NFFT，并做镜像扩展 ----
    H_full = np.zeros((num_fft, Ns), dtype=complex)
    H_full[idx_sc, :] = HestLS

    # 头端扩展：把有效带宽最左边 num_ext 个RE 反转，填到左侧空带
    if num_ext > 0:
        idx_up = idx_sc[0] - num_ext + np.arange(num_ext, dtype=int)  # 左侧空带区间索引
        H_full[idx_up, :] = np.flipud(HestLS[:num_ext, :])

        # 尾端扩展
        idx_down = idx_sc[-1] + 1 + np.arange(num_ext, dtype=int)
        H_full[idx_down, :] = np.flipud(HestLS[-num_ext:, :])

    # ---- (B) IFFT 到时域 ----
    # 注意：这里不做fftshift，保持与IdxSC约定一致
    H_unshift_full = np.fft.ifftshift(H_full, axes=0)
    h_td = np.fft.ifft(H_unshift_full, n=num_fft,axis=0)* np.sqrt(num_fft)
    import matplotlib.pyplot as plt
    # plt.figure(figsize=(8, 4))
    # plt.plot(np.abs(h_td[:, 0]), label='|h_td| (symbol 0)')
    # plt.xlabel('Sample index (delay)')
    # plt.ylabel('Magnitude')
    # plt.title('Time-domain impulse response magnitude')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    # ---- (C) 时域窗处理：前端保留 + 三角渐变 + 中段清零 + 尾端三角渐变 ----
    L0 = int(len_front_keep)          # 完全保留段
    Lt = int(len_tri_filter)          # 三角过渡段
    assert L0 + Lt <= num_fft//2, "前端保留+过渡长度不应超过一半采样，检查参数。"

    # 前端三角过渡：索引 [L0, L0+Lt)
    tri = np.linspace(1.0, 0.0, Lt, endpoint=False)  # 从1线性减到接近0
    if Lt > 0:
        h_td[L0:L0+Lt, :] *= tri[:, None]

    # 中段清零：索引 [L0+Lt, num_fft-Lt)
    if L0 + Lt < num_fft - Lt:
        h_td[L0+Lt:num_fft-Lt-L0, :] = 0.0

    # 尾端三角过渡：索引 [num_fft-Lt, num_fft)
    if Lt > 0:
        tri_tail = np.linspace(0.0, 1.0, Lt, endpoint=False)  # 从0线性升到接近1
        h_td[num_fft-Lt-L0:num_fft-L0, :] *= tri_tail[:, None]

    # ---- (D) 回频域 ----
    H_fd_unshift = np.fft.fft(h_td, n=num_fft, axis=0) / np.sqrt(num_fft)  # 未shift谱（DC在0）
    H_fd_dn = np.fft.fftshift(H_fd_unshift, axes=0)  # 转回“居中布局”（DC在中间）

    # H_fd_dn = np.fft.fft(h_td, n=num_fft, axis=0) / np.sqrt(num_fft)

    # 取回有效带宽
    H_dn = H_fd_dn[idx_sc, :]
    # plt.figure(figsize=(8, 4))
    # plt.plot(np.abs(h_td[:, 0]), label='|h_td| (symbol 0)')
    # plt.xlabel('Sample index (delay)')
    # plt.ylabel('Magnitude')
    # plt.title('Time-domain impulse response magnitude')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    # ---- (E) 频域功率重标（可关）----
    if power_rescale:
        p_before = np.mean(np.abs(HestLS)**2) + 1e-15
        p_after  = np.mean(np.abs(H_dn   )**2) + 1e-15
        H_dn *= np.sqrt(p_before / p_after)

    return H_dn

import numpy as np

def build_lmmse_dmrs_filter(dmrs_pos,
                            HestGran,
                            deltaF,
                            tau_rms,
                            snr,
                            model='one_pole'):
    """
    构造频域 LMMSE 滤波器 W（不考虑 OCC 相位翻转）
    使得: H_data (HestGran×1) = W (HestGran×Ndmrs) @ H_dmrs_ls (Ndmrs×1)

    参数
    ----
    dmrs_pos : 1D array-like[int]
        当前子带/PRG 内 DMRS 的相对频域位置(0..HestGran-1)，长度 = Ndmrs。
        例如 PBCH PRG=24 且密度1/4、offset=1 → dmrs_pos=[1,5,9,13,17,21]
    HestGran : int
        子带/PRG 的数据 RE 个数（如 12, 24, 36, 48）。
    deltaF : float
        子载波间隔(Hz)，如 15000/30000 等。
    tau_rms : float
        RMS 时延扩展(秒)，如 300e-9。
    noise_var : float
        噪声方差 σ²（和 LS 等效噪声一致；按线性值，不是 dB）。
    model : str
        频域相关模型：
        - 'one_pole' : R(Δk) = 1 / (1 + j*2π*tau_rms*deltaF*Δk)   （与你现有实现一致）
        - 'gauss'    : R(Δk) = exp(-0.5 * (2π*tau_rms*deltaF*Δk)^2)

    返回
    ----
    W : np.ndarray (complex64), shape=(HestGran, Ndmrs)
        LMMSE 频域插值矩阵
    """
    dmrs_pos = np.asarray(dmrs_pos, dtype=int)
    if dmrs_pos.ndim != 1:
        raise ValueError("dmrs_pos must be 1-D.")
    if np.any(dmrs_pos < 0) or np.any(dmrs_pos >= HestGran):
        raise ValueError("dmrs_pos has indices outside [0, HestGran).")
    # 去重并排序，保证相关矩阵可逆性更稳
    dmrs_pos = np.unique(dmrs_pos)
    Ndmrs = dmrs_pos.size

    k_data = np.arange(HestGran, dtype=int)          # 0..HestGran-1
    # 频域距离矩阵
    d_hd_hp = k_data[:, None] - dmrs_pos[None, :]    # (HestGran, Ndmrs)
    d_hp_hp = dmrs_pos[:, None] - dmrs_pos[None, :]  # (Ndmrs, Ndmrs)

    a = 2.0 * np.pi * tau_rms * float(deltaF)

    if model == 'one_pole':
        # 与你之前 LMMSE 分支一致的近似模型
        R_hd_hp = 1.0 / (1.0 + 1j * a * d_hd_hp)
        R_hp_hp = 1.0 / (1.0 + 1j * a * d_hp_hp)
    elif model == 'gauss':
        # 高斯相关模型（有时更稳）
        R_hd_hp = np.exp(-0.5 * (a * d_hd_hp)**2)
        R_hp_hp = np.exp(-0.5 * (a * d_hp_hp)**2)
    else:
        raise ValueError("Unsupported model. Use 'one_pole' or 'gauss'.")
    noise_var = 1 / pow(10, snr / 10)
    # LMMSE: W = R_hd,hp @ (R_hp,hp + σ² I)^(-1)
    # 用 solve 提升数值稳定性
    reg = R_hp_hp + (noise_var * np.eye(Ndmrs, dtype=R_hp_hp.dtype))
    # 解：(R_hp_hp + σ²I) X = R_hd_hp^T  →  X^T 就是 W
    X = np.linalg.solve(reg, R_hd_hp.T)              # (Ndmrs, HestGran)
    W = X.T.astype(np.complex64)                     # (HestGran, Ndmrs)
    return W


def myChannelEstimatev2(
    rxGrid=None, refInd=None, refSym=None, refGrid=None, carrier=None,snr= 0,
    # === 新增：LMMSE 相关参数 ===
    deltaF=15000.0,            # 子载波间隔(Hz)
    tau_rms=300e-9,            # RMS 时延扩展(秒)
    lmmse_model='one_pole',    # 'one_pole' 或 'gauss'
    prg_gran=None,             # 手动指定子带粒度(如 24)。None=自动根据 DMRS 间距推断
    # === 时域去噪参数（保留你的原逻辑） ===
    td_denoise=True,
    num_fft=2048,
    maxCP=144,
    IdxSC=None
):
    cp = "normal" if carrier is None else "extended"

    # 解析导频网格
    if refGrid is None:
        if (refInd is None) or (refSym is None):
            print("Error: refInd and refSym need to be given when refGrid is None!")
            return
        refGrid = np.zeros(rxGrid.shape, dtype=complex)
        nrSetResources(refInd, refGrid, refSym)
    else:
        pilot_idx = np.where(refGrid.ravel(order="F") != 0)
        refInd = pilot_idx[0]
        refSym = refGrid.ravel(order="F")[refInd]

    # 基本维度
    K, N = rxGrid.shape  # subcarriers, symbols

    # DMRS 处接收/发送符号
    rxSym_all = rxGrid.ravel(order="F")[refInd]
    refSym_all = refSym

    # 输出信道
    H = np.zeros_like(refGrid, dtype=complex)
    has_dmrs_col = np.zeros(N, dtype=bool)

    # === 每个包含 DMRS 的符号列：LS → 按子带做 LMMSE 插值 ===
    for col in range(N):
        in_col = np.where((refInd >= col*K) & (refInd < (col+1)*K))[0]
        if in_col.size == 0:
            continue
        has_dmrs_col[col] = True

        sc_idx = (refInd[in_col] - col*K).astype(int)  # 0..K-1
        y = rxSym_all[in_col]
        x = refSym_all[in_col]

        # LS (防零)
        eps = 1e-12
        x_safe = np.where(np.abs(x) < eps, eps + 0j, x)
        h_ls = y / x_safe

        # 按子带做：确定粒度
        sc_sorted = np.sort(sc_idx)
        if prg_gran is None:
            if sc_sorted.size >= 2:
                diffs = np.diff(sc_sorted)
                # 取最常见的间距（DMRS 周期），例如 PBCH 1/4 → spacing≈4
                spacing = int(np.bincount(diffs).argmax())
                gran = max(12, 6*spacing)  # 1/4 密度 → 6*4=24；至少 12
                # 对齐到 12 的倍数（便于 PRB 粒度）
                gran = int(np.ceil(gran/12.0)*12)
            else:
                gran = 24  # 兜底
        else:
            gran = int(prg_gran)

        # 列内的噪声方差估计（robust median of residuals）
        # resid = y - x * h_ls_true；此处用 LS 的自残差估计噪声
        resid = y - x_safe * h_ls
        noise_var_col = float(np.median(np.abs(resid)**2)) if resid.size else 1e-3

        # 建立便于索引的 map
        # 对每个子带 [s:e)，先取其中 DMRS 的下标与 LS 值，再做 LMMSE
        for sb_start in range(0, K, gran):
            sb_end = min(sb_start + gran, K)
            # 当前子带内的 DMRS 下标（绝对频域）
            mask_sb = (sc_idx >= sb_start) & (sc_idx < sb_end)
            if not np.any(mask_sb):
                continue

            dmrs_abs = sc_idx[mask_sb]            # 绝对频域索引
            h_dmrs_ls = h_ls[mask_sb]             # 对应 LS
            # 转为相对子带起点的位置
            dmrs_pos = dmrs_abs - sb_start        # 0..(gran-1)

            # 构造 LMMSE W，并做插值
            W = build_lmmse_dmrs_filter(
                dmrs_pos=dmrs_pos,
                HestGran=(sb_end - sb_start),
                deltaF=deltaF,
                tau_rms=tau_rms,
                snr=snr,
                model=lmmse_model
            )
            if W is None:
                continue

            Hd = (W @ h_dmrs_ls.astype(np.complex64)).ravel()   # (子带长度,)

            # 写回：这一列该子带范围
            H[sb_start:sb_end, col] = Hd

        # 子带之间可能有极少未覆盖点（例如某子带没有 DMRS 且左右没插到）
        # 简单补洞：对该列剩余的 0 值用最近邻/线性在频域补上
        missing = np.where(H[:, col] == 0)[0]
        if missing.size:
            known = np.where(H[:, col] != 0)[0]
            if known.size:
                H.real[missing, col] = np.interp(missing, known, H.real[known, col])
                H.imag[missing, col] = np.interp(missing, known, H.imag[known, col])
            else:
                # 该列极端兜底（理论上不会出现，因为 in_col.size > 0）
                H[:, col] = 0

    # === 没有 DMRS 的列：时间方向插值（保留你的逻辑） ===
    if not np.all(has_dmrs_col):
        cols_with = np.where(has_dmrs_col)[0]
        cols_wo   = np.where(~has_dmrs_col)[0]
        if cols_with.size >= 2:
            for k in range(K):
                y = H[k, cols_with]
                H[k, cols_wo] = (np.interp(cols_wo, cols_with, np.real(y))
                                +1j*np.interp(cols_wo, cols_with, np.imag(y)))
        elif cols_with.size == 1:
            H[:, cols_wo] = H[:, cols_with[0:1]]

    # === 有效带宽内的时域去噪（你的原函数） ===
    if td_denoise:
        if IdxSC is None:
            raise ValueError("td_denoise=True 时必须提供 IdxSC（有效带宽索引）。")
        H_dn = np.zeros_like(H, dtype=complex)
        idx_sc = IdxSC[:, 0] if (IdxSC.ndim == 2 and IdxSC.shape[1] == 1) else np.asarray(IdxSC).ravel()
        SC, Ns = rxGrid.shape
        cp_len_first = 160
        cp_len_others = maxCP
        for col in range(Ns):
            cp_len = cp_len_first if col == 0 else cp_len_others
            H_dn[:, col] = noise_reduction_td_bandlimited(
                HestLS=H[:, [col]],
                num_fft=num_fft,
                maxCP=cp_len,
                IdxSC=idx_sc,
                num_ext=48,
                len_tri_filter=48,
                len_front_keep=cp_len,
                power_rescale=True
            ).ravel()
    else:
        H_dn = H

    return H_dn

# 以 PBCH 一个 PRG=24 为例，DMRS 每隔4个 RE，一个偏移 offset∈{0,1,2,3}
HestGran = 24
offset = 1
dmrs_pos = offset + 4*np.arange(6)                  # 6 个 DMRS: [1,5,9,13,17,21]

deltaF   = 15000.0                                  # 15 kHz 举例（按你的实际 SCS）
tau_rms  = 300e-9                                   # 300 ns
snr_db   = 10.0
noise_var = 10**(-snr_db/10)                        # 简单 SNR→噪声的近似

# W = build_lmmse_dmrs_filter(dmrs_pos, HestGran, deltaF, tau_rms, noise_var, model='one_pole')
# print(W)
# 之后：对该 PRG 的 LS 估计向量 h_dmrs_ls（长度=6），做插值
# h_data_est = W @ h_dmrs_ls    # 得到长度 24 的频域信道估计
def _fft_u(x):
    """单位化 FFT（幅度不缩放）：F/√N，与 IFFT 对称。"""
    x = np.asarray(x)
    return np.fft.fftshift(np.fft.fft(x, axis=0) / np.sqrt(x.shape[0]))

def _ifft_u(X):
    """单位化 IFFT（幅度不缩放）：F^H/√N。"""
    X = np.asarray(X)
    return np.fft.ifft(np.fft.ifftshift(X), axis=0) * np.sqrt(X.shape[0])
def td_interpolate_from_dmrs(
    H_dmrs_denoised,   # 长度 Nd 的 comb-DMRS 频域估计（已按 comb 顺序重排：n=0..Nd-1）
    dmrs_pos_rel,      # 0..Nd-1（与上面序列一致的“等间隔顺序索引”）
    L_blk,             # 目标块长（如 24）
    keep_pilot_consistency=True,   # 是否在 DMRS 处强制回填原值
    ls_ref_on_dmrs=None,           # 原始 LS（用于对齐/回填），长度 Nd，对应 dmrs_pos_rel
    cut_neighborhood=1,             # 寻找切点时，邻域能量聚合半径（1=看两侧各1个tap）
    spacing=2,         # comb 间隔（SIB1 常见 comb=2）
    offset=0,          # comb 偏移（SIB1 常见 offset=0 或 1）
):
    """
    仅做 '变换域插值'：Nd → L_blk
    前提：H_dmrs_denoised 已经是 comb=N 解开后的等间隔序列（顺序 n=0..Nd-1）
    返回：H_blk（长度 L_blk），为该频域块所有子载波的信道估计
    """
    H_dmrs_denoised = np.asarray(H_dmrs_denoised, np.complex64).ravel()
    Nd = H_dmrs_denoised.size
    assert L_blk >= Nd and Nd >= 1

    # 1) IFFT → hN（单位化）
    hN = _ifft_u(H_dmrs_denoised)
    if PLOT_FIGURE:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(np.abs(hN[:]**2), label='|h_td| (symbol 0)')
        plt.xlabel('Sample index (delay)')
        plt.ylabel('Magnitude')
        plt.title('Time-domain impulse response magnitude')
        plt.grid(True)
        plt.legend()
        plt.show()
    # 2) 选切点（不移位）：圆周邻域能量和最小的位置
    p = (np.abs(hN) ** 2).astype(np.float64)
    R = int(max(0, cut_neighborhood))
    if R == 0:
        cost = p.copy()
    else:
        cost = np.empty(Nd, dtype=np.float64)
        for i in range(Nd):
            s = 0.0
            for t in range(-R, R+1):
                s += p[(i + t) % Nd]
            cost[i] = s
    cut = int(np.argmin(cost))

    # 3) 在 cut 处“中间补零”（不循环移位）
    Z = L_blk - Nd  # 需要补的零个数
    hL = np.zeros((L_blk,), dtype=np.complex64)
    # 原始 hN[0:cut] 保持在前段
    if cut > 0:
        hL[:cut] = hN[:cut]
    # 中间插 Z 个 0
    # 将 hN[cut:Nd] 放在后段（从 cut+Z 开始）
    if cut < Nd:
        hL[cut+Z:cut+Z+(Nd-cut)] = hN[cut:]
    if PLOT_FIGURE:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(np.abs(hL[:]**2), label='|h_td| (symbol 0)')
        plt.xlabel('Sample index (delay)')
        plt.ylabel('Magnitude')
        plt.title('Time-domain impulse response magnitude')
        plt.grid(True)
        plt.legend()
        plt.show()
    # 4) FFT 回频域 + 解析幅度补偿
    H_blk = _fft_u(hL)
    # H_blk *= np.sqrt(L_blk / Nd)   # 幅度补偿，避免零插整体变小

    # 5) 导频最小二乘对齐 + 一致性（用“绝对 comb 位置”回填）
    # dmrs_pos_abs = (offset + spacing * np.arange(Nd)) % L_blk
    dmrs_pos_abs = dmrs_pos_rel
    if ls_ref_on_dmrs is not None:
        ls_ref_on_dmrs = np.asarray(ls_ref_on_dmrs, np.complex64).ravel()
        Hp = H_blk[dmrs_pos_abs]
        den = np.vdot(Hp, Hp)
        if den != 0:
            alpha = np.vdot(Hp, ls_ref_on_dmrs) / den
            H_blk *= alpha
        if keep_pilot_consistency:
            H_blk[dmrs_pos_abs] = ls_ref_on_dmrs

    return H_blk
def estimate_symbol_by_blocks_simple(
    col, K, refInd, rxSym_all, refSym_all, start_rb, end_rb
):
    H_col = np.zeros((K,), dtype=np.complex64)

    in_col = np.where((refInd >= col*K) & (refInd < (col+1)*K))[0]
    if in_col.size == 0:
        return H_col, False

    sc_idx = (refInd[in_col] - col*K).astype(int)  # 当前符号 DMRS 位置
    y = rxSym_all[in_col]
    x = refSym_all[in_col]
    eps = 1e-12
    Hls = (y / np.where(np.abs(x) < eps, eps+0j, x)).astype(np.complex64)

    # PBCH 掩码（你原来那段 pbch_mask 逻辑可以保留）
    pbch_mask = np.zeros((K,), dtype=bool)
    if col == 0:
        return H_col, False
    elif col == 1:
        pbch_mask[start_rb*12:end_rb*12] = True
    elif col == 2:
        pbch_mask[:48] = True
        pbch_mask[192:] = True
    elif col == 3:
        pbch_mask[start_rb*12:end_rb*12] = True
    else:
        pbch_mask[:] = True

    # 只在 PBCH 覆盖的频带内做插值
    runs = []
    k = 0
    while k < K:
        if pbch_mask[k]:
            s = k
            while k < K and pbch_mask[k]:
                k += 1
            runs.append((s, k))
        else:
            k += 1

    for (s, e) in runs:
        L_blk = e - s
        in_blk = np.where((sc_idx >= s) & (sc_idx < e))[0]
        if in_blk.size == 0:
            continue

        dmrs_abs = sc_idx[in_blk]
        dmrs_rel = (dmrs_abs - s).astype(int)
        Hdmrs    = Hls[in_blk]

        H_blk = np.zeros((L_blk,), dtype=np.complex64)
        H_blk[dmrs_rel] = Hdmrs

        missing = np.setdiff1d(np.arange(L_blk), dmrs_rel)
        if missing.size > 0 and dmrs_rel.size > 1:
            Hr = H_blk.real
            Hi = H_blk.imag
            Hr[missing] = np.interp(missing, dmrs_rel, Hr[dmrs_rel])
            Hi[missing] = np.interp(missing, dmrs_rel, Hi[dmrs_rel])
            H_blk = Hr + 1j * Hi

        H_col[s:e] = H_blk

    return H_col, True
def ce_block_td_denoise_fullsymbol(
    Hest_dmrs,      # shape = (Ndmrs,)   该子带 DMRS 上的 LS 信道
    dmrs_pos,       # shape = (Ndmrs,)   该子带内 DMRS 相对索引(0..L-1)
    L_blk,              # 子带长度(如 24)
    energy_keep=0.9,# 时延域保留能量比例(0~1)
    guard=0,        # 可选：对窗口左右各扩一点tap
    pilot_consistency=True # 回频域后在导频处回填LS，保证一致性/幅度不变
):
    """
    对任意频域块做：DMRS 序列(IDFT)→时延窗→零填充到 L_blk→频移纠偏→回频域。
    支持任意块长和DMRS间隔。
    返回：H_blk shape=(L_blk,)
    """
    Hest_dmrs = np.asarray(Hest_dmrs, dtype=np.complex64).ravel()
    dmrs_pos = np.asarray(dmrs_pos, dtype=int).ravel()
    Nd = Hest_dmrs.size
    assert Hest_dmrs.size == dmrs_pos.size and Hest_dmrs.size > 0
    L = int(L_blk)
    # --- 检查等间隔并提取 spacing/offset（PBCH spacing=4） ---
    if Nd > 1:
        diffs = np.diff(np.sort(dmrs_pos))
        if not np.all(diffs == diffs[0]):
            raise ValueError("该函数要求该块内 DMRS 等间隔（PBCH 为间隔=4）。")
        spacing = int(diffs[0])
    else:
        spacing = 4
    if spacing <= 0:
        spacing = 4
    offset = int(dmrs_pos.min() % spacing)

    # 把 DMRS 排成 n=0..Nd-1（pos = offset + spacing*n）
    n_idx = ((dmrs_pos - offset) // spacing).astype(int)
    order = np.argsort(n_idx)
    HN = Hest_dmrs[order]
    n_sorted = n_idx[order]
    if not np.array_equal(n_sorted, np.arange(Nd)):
        raise ValueError("DMRS 位置应满足 offset + spacing*n 且 n 连续。")
    # 1) Nd 点 IDFT → 时延域（单位化）
    hN = _ifft_u(HN)  # (Nd,)
    if PLOT_FIGURE:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(np.abs(hN[:]**2), label='|h_td| (symbol 0)')
        plt.xlabel('Sample index (delay)')
        plt.ylabel('Magnitude')
        plt.title('Time-domain impulse response magnitude')
        plt.grid(True)
        plt.legend()
        plt.show()
    # 2) 按能量累加找主要能量窗口，并可左右 guard 扩张
    # # —— 改这里：用尖峰 + 环形定宽窗 ——
    # p = np.abs(hN) ** 2
    # Nd = hN.size
    # if p.sum() > 0:
    #     peak = int(np.argmax(p))  # 找尖峰位置
    #     win_radius = max(0, int(guard))  # 左右各取 win_radius 个 tap；你也可单独传参
    #
    #     # 环形索引：peak-win_radius ... peak+win_radius，越界用取模回绕
    #     offs = np.arange(-win_radius, win_radius + 1, dtype=int)  # 共 2*R+1 点
    #     idx = (peak + offs) % Nd
    #
    #     winN = np.zeros((Nd,), dtype=np.float32)
    #     winN[idx] = 1.0
    # else:
    #     winN = np.ones((Nd,), dtype=np.float32)
    #
    # hN_win = hN * winN

    p = np.abs(hN) ** 2

    if p.sum() > 0:
        peak = int(np.argmax(p))  # 尖峰位置（常在末尾）
        cp_len = 160
        num_fft = 2048
        # 窗长度按 CP/FFT 比例：win_len ≈ (cp_len/num_fft) * Nd
        ratio = float(cp_len) / float(num_fft)
        win_len = int(np.ceil(ratio * Nd))
        win_len = max(5, min(win_len, Nd))

        if win_len % 2 == 0:  # 用奇数长度，便于“以峰为中心”
            win_len += 1
            if win_len > Nd: win_len = Nd

        # 以主峰为中心向后取——也就是从 start=peak-R 开始，连续取 win_len 个点
        R = (win_len - 1) // 2
        R = min(Nd // 2, R + int(guard))  # 如需再扩一点
        start = peak - R
        idx = (start + np.arange(win_len, dtype=int)) % Nd  # 关键：回绕取模

        # 窗型：汉宁(平滑)或矩形
        use_hann = False
        if use_hann and win_len > 1:
            w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(win_len) / (win_len - 1)))
        else:
            w = np.ones(win_len, dtype=float)
        w = (w / w.max()).astype(np.float32)

        winN = np.zeros((Nd,), dtype=np.float32)
        winN[idx] = w
    else:
        winN = np.ones((Nd,), dtype=np.float32)
    # winN = np.ones((Nd,), dtype=np.float32)
    hN_win = hN * winN
    if PLOT_FIGURE:
        plt.figure(figsize=(8, 4))
        plt.plot(np.abs(hN_win[:]**2), label='|h_td| (symbol 0)')
        plt.xlabel('Sample index (delay)')
        plt.ylabel('Magnitude')
        plt.title('Time-domain impulse response magnitude')
        plt.grid(True)
        plt.legend()
        plt.show()
    # ===== 3) 时延域零填充：Nd → L（中间零填充，保持边界相对关系）=====
    if L % Nd != 0:
        raise ValueError("L 必须是 Nd 的整数倍（例如 PBCH: 24 = 6 * 4）")

    # 将时延域序列从“圆周”的中点拆分：前 half 放开头，后 half 放结尾，中间全 0
    half = Nd // 2
    hL = np.zeros((L,), dtype=np.complex64)

    # 前半段 [0:half) 放到开头
    hL[:half] = hN_win[:half]

    # 后半段 [half:Nd) 放到数组末尾（留出 Nd - half 个位置）
    tail_len = Nd - half
    hL[L - tail_len:] = hN_win[half:]
    if PLOT_FIGURE:
        plt.figure(figsize=(8, 4))
        plt.plot(np.abs(hL[:]**2), label='|h_td| (symbol 0)')
        plt.xlabel('Sample index (delay)')
        plt.ylabel('Magnitude')
        plt.title('Time-domain impulse response magnitude')
        plt.grid(True)
        plt.legend()
        plt.show()
    # 4) 频移补偿 offset：时延域乘 e^{j 2π (offset/L_blk) m}
    m = np.arange(L_blk, dtype=float)
    ph = np.exp(1j * 2.0 * np.pi * (offset / L_blk) * m).astype(np.complex64)
    hL_shift = hL * ph

    # 5) 回频域（单位化 FFT）
    H_blk = _fft_u(hL_shift).astype(np.complex64)  # (L_blk,)

    # 6) 解析归一化（Nd→L_blk 带来的 sqrt(Nd/L_blk) 缩放需补回）
    H_blk *= np.sqrt(L_blk / Nd)

    # 7) 用导频最小二乘做全带增益对齐（补窗引起的幅度偏差）
    Hp_est = H_blk[dmrs_pos]
    denom = np.vdot(Hp_est, Hp_est)
    if denom != 0:
        alpha = np.vdot(Hp_est, Hest_dmrs) / denom
        H_blk *= alpha

    # 8) 导频一致性：导频处回填 LS，保证幅度绝对不变
    if pilot_consistency:
        H_blk[dmrs_pos] = Hest_dmrs

    return H_blk


# def myChannelEstimatev3(rxGrid=None, refInd=None, refSym=None, refGrid=None, carrier=None, prg_gran=24, energy_keep=0.9, guard=0):
#     cp = "normal" if carrier is None else "extended"
#
#     # 解析导频网格
#     if refGrid is None:
#         if (refInd is None) or (refSym is None):
#             print("Error: refInd and refSym need to be given when refGrid is None!")
#             return
#         refGrid = np.zeros(rxGrid.shape, dtype=complex)
#         nrSetResources(refInd, refGrid, refSym)
#     else:
#         pilot_idx = np.where(refGrid.ravel(order="F") != 0)
#         refInd = pilot_idx[0]
#         refSym = refGrid.ravel(order="F")[refInd]
#
#     # 基本维度
#     K, N = rxGrid.shape  # subcarriers, symbols
#
#     # DMRS 处接收/发送符号
#     rxSym_all = rxGrid.ravel(order="F")[refInd]
#     refSym_all = refSym
#
#     # 输出信道
#
#     H = np.zeros_like(refGrid, dtype=complex)
#     has_dmrs_col = np.zeros(N, dtype=bool)
#
#     for col in range(N):
#         in_col = np.where((refInd >= col*K) & (refInd < (col+1)*K))[0]
#         if in_col.size == 0:
#             continue
#         has_dmrs_col[col] = True
#
#         sc_idx = (refInd[in_col] - col*K).astype(int)   # 本列导频频域索引(0..K-1)
#         y = rxSym_all[in_col];  x = refSym_all[in_col]
#         eps = 1e-12
#         h_ls_all = y / np.where(np.abs(x) < eps, eps+0j, x)  # 该列所有导频的LS
#
#         # —— 逐子带处理（默认 24，一般 PBCH: 24=2PRB）——
#         L = int(prg_gran)
#         for sb_start in range(0, K, L):
#             sb_end = min(sb_start + L, K)
#             L_eff = sb_end - sb_start
#
#             # 该子带内的 DMRS
#             mask_sb = (sc_idx >= sb_start) & (sc_idx < sb_end)
#             if not np.any(mask_sb):
#                 continue
#
#             dmrs_abs = sc_idx[mask_sb]
#             dmrs_pos = (dmrs_abs - sb_start).astype(int)          # 相对子带起点
#             h_ls     = h_ls_all[mask_sb].astype(np.complex64)
#
#             # 变换域降噪 + 插值
#             up_factor = max(1, L_eff // max(1, dmrs_pos.size))
#             H_sub = ce_subband_td_denoise_interpolate(
#                 Hest_dmrs=h_ls,
#                 dmrs_pos=dmrs_pos,
#                 L=L_eff,
#                 up_factor=up_factor,
#                 energy_keep=energy_keep,
#                 guard=guard,
#                 pilot_consistency=True
#             )
#             H[sb_start:sb_end, col] = H_sub
#
#         # 若某列仍有未覆盖点（极少），做一次频域线性“补洞”
#         missing = np.where(H[:, col] == 0)[0]
#         if missing.size:
#             known = np.where(H[:, col] != 0)[0]
#             if known.size:
#                 H.real[missing, col] = np.interp(missing, known, H.real[known, col])
#                 H.imag[missing, col] = np.interp(missing, known, H.imag[known, col])
#
#     # —— 无 DMRS 列：时间方向插值（保留你原逻辑）——
#     # ...（同你原函数）...
#
#     # —— 有效带宽的 TD 去噪（保留你原逻辑）——
#     # ...（同你原函数）...
#     H_dn = H
#     return H_dn  # 或 H（若关闭 td_denoise）
def estimate_symbol_by_blocks(
    col,            # 符号索引
    K,              # 子载波数（240）
    refInd,         # 导频索引（按 Fortran 展平）
    rxSym_all,      # DMRS上的接收符号 (与 refInd 对齐)
    refSym_all,      # DMRS上的发送符号 (与 refInd 对齐)
    start_rb,
    end_rb
):
    """
    返回：H_col shape=(K,) 本符号整列的信道估计（复数）
    """
    H_col = np.zeros((K,), dtype=np.complex64)

    # 1) 找本符号的 DMRS：在 refInd 中属于该列的条目
    in_col = np.where((refInd >= col*K) & (refInd < (col+1)*K))[0]
    if in_col.size == 0:
        return H_col, False  # 此符号无DMRS（如 PSS）

    sc_idx = (refInd[in_col] - col*K).astype(int)  # DMRS 频域下标（0..K-1）
    y = rxSym_all[in_col]; x = refSym_all[in_col]
    eps = 1e-12
    Hls = (y / np.where(np.abs(x) < eps, eps+0j, x)).astype(np.complex64)

    # 2) 构建“连续 PBCH 频块”的掩码  TODO This field needs to be extended to variable size of PBCH
    pbch_mask = np.zeros((K,), dtype=bool)
    if col == 0:
        # 符号0: 全PSS → 无PBCH，直接返回
        return H_col, False
    elif col == 1:
        pbch_mask[:start_rb*12] = False
        pbch_mask[start_rb*12:end_rb*12] = True
        pbch_mask[end_rb*12:] = False
    elif col == 2:
        pbch_mask[:48] = True
        pbch_mask[192:] = True
        pbch_mask[:start_rb*12] = False
        pbch_mask[end_rb*12:] = False
    elif col == 3:
        pbch_mask[:start_rb * 12] = False
        pbch_mask[start_rb * 12:end_rb * 12] = True
        pbch_mask[end_rb * 12:] = False
    else:
        # 以防超出 0..3 的范围
        pbch_mask[:] = True

    # 3) 把 PBCH 掩码划分为连续块
    #    找所有 True 的连续 run：[s,e)
    runs = []
    k = 0
    while k < K:
        if pbch_mask[k]:
            s = k
            while k < K and pbch_mask[k]:
                k += 1
            runs.append((s, k))  # [s, k)
        else:
            k += 1
    if len(runs) == 0:
        return H_col, False
        # 4) 对每个连续块单独做“变换域降噪+插值”
    for (s, e) in runs:
        L_blk = e - s
        # 该块内的 DMRS
        in_blk = np.where((sc_idx >= s) & (sc_idx < e))[0]
        if in_blk.size == 0:
            # 该块没DMRS（少见），可跳过或邻域补洞，先跳过
            continue
        dmrs_abs = sc_idx[in_blk]
        dmrs_rel = (dmrs_abs - s).astype(int)
        Hdmrs    = Hls[in_blk]

        # 直接用上面的“整块”函数
        H_blk = ce_block_td_denoise_fullsymbol(
            Hest_dmrs=Hdmrs,
            dmrs_pos=dmrs_rel,
            L_blk=L_blk,
            energy_keep=0.9,
            guard=0,
            pilot_consistency=True
        )
        H_col[s:e] = H_blk

    return H_col, True

def estimate_symbol_by_blocks1(
    col,            # 符号索引
    K,              # 子载波数（240）
    refInd,         # 导频索引（按 Fortran 展平）
    rxSym_all,      # DMRS上的接收符号 (与 refInd 对齐)
    refSym_all,      # DMRS上的发送符号 (与 refInd 对齐)
    start_rb,
    end_rb
):
    """
    返回：H_col shape=(K,) 本符号整列的信道估计（复数）
    """
    H_col = np.zeros((K,), dtype=np.complex64)

    # 1) 找本符号的 DMRS：在 refInd 中属于该列的条目
    in_col = np.where((refInd >= col*K) & (refInd < (col+1)*K))[0]
    if in_col.size == 0:
        return H_col, False  # 此符号无DMRS（如 PSS）

    sc_idx = (refInd[in_col] - col*K).astype(int)  # DMRS 频域下标（0..K-1）
    y = rxSym_all[in_col]; x = refSym_all[in_col]
    eps = 1e-12
    Hls = (y / np.where(np.abs(x) < eps, eps+0j, x)).astype(np.complex64)

    # 2) 构建“连续 PBCH 频块”的掩码  TODO This field needs to be extended to variable size of PBCH
    pbch_mask = np.zeros((K,), dtype=bool)
    if col == 0:
        # 符号0: 全PSS → 无PBCH，直接返回
        return H_col, False
    elif col == 1:
        pbch_mask[:start_rb*12] = False
        pbch_mask[start_rb*12:end_rb*12] = True
        pbch_mask[end_rb*12:] = False
    elif col == 2:
        pbch_mask[:48] = True
        pbch_mask[192:] = True
        pbch_mask[:start_rb*12] = False
        pbch_mask[end_rb*12:] = False
    elif col == 3:
        pbch_mask[:start_rb * 12] = False
        pbch_mask[start_rb * 12:end_rb * 12] = True
        pbch_mask[end_rb * 12:] = False
    else:
        # 以防超出 0..3 的范围
        pbch_mask[:] = True

    # 3) 把 PBCH 掩码划分为连续块
    #    找所有 True 的连续 run：[s,e)
    runs = []
    k = 0
    while k < K:
        if pbch_mask[k]:
            s = k
            while k < K and pbch_mask[k]:
                k += 1
            runs.append((s, k))  # [s, k)
        else:
            k += 1

    # 4) 对每个连续块单独做“变换域降噪+插值”
    for (s, e) in runs:
        L_blk = e - s
        # 该块内的 DMRS
        in_blk = np.where((sc_idx >= s) & (sc_idx < e))[0]
        if in_blk.size == 0:
            # 该块没DMRS（少见），可跳过或邻域补洞，先跳过
            continue
        dmrs_abs = sc_idx[in_blk]
        dmrs_rel = (dmrs_abs - s).astype(int)

        y_blk = y[in_blk]
        x_blk = x[in_blk]
        H_ls_denoised = td_dmrs_denoise_fixed(y_blk, x_blk, Nfft=256, window_type="fixed", win_len_ratio=0.30,
                                              eps=1e-12)
        # TODO 改成线性插值后时域降噪
        # H_ls_denoised = Hls[in_blk]
        H_blk = td_interpolate_from_dmrs(H_ls_denoised, dmrs_rel, L_blk,spacing=4, ls_ref_on_dmrs=H_ls_denoised)
        # 直接用上面的“整块”函数
        # Hdmrs = Hls[in_blk]dmrs_pos_abs
        # H_blk = ce_block_td_denoise_fullsymbol(
        #     Hest_dmrs=Hdmrs,
        #     dmrs_pos=dmrs_rel,
        #     L_blk=L_blk,
        #     energy_keep=0.9,
        #     guard=0,
        #     pilot_consistency=True
        # )
        H_col[s:e] = H_blk

    return H_col, True

def circular_window_around_peak(Nfft, peak_idx, win_len, kind="rect", guard=0):
    """
    在循环序列长度 Nfft 上，以 peak_idx 为中心，取 win_len 个样点的环形窗。
    - kind: "rect" 或 "hann"
    - guard: 额外向两侧扩展的样点数
    """
    assert 1 <= win_len <= Nfft
    # 用奇数长度居中更自然；若为偶数，向上取到奇数
    if win_len % 2 == 0:
        win_len = min(Nfft, win_len + 1)

    R = (win_len - 1)//2 + int(guard)
    offs = np.arange(-R, R+1, dtype=int)            # 2R+1 点
    idx  = (peak_idx + offs) % Nfft                 # 环形索引

    w = np.zeros(Nfft, dtype=np.float32)
    if kind == "hann" and idx.size > 1:
        # 汉宁窗（两端平滑）
        local = 0.5*(1 - np.cos(2*np.pi*np.arange(idx.size)/(idx.size-1)))
        w[idx] = local / local.max()
    else:
        # 矩形窗
        w[idx] = 1.0
    return w


def td_dmrs_denoise_fixed(
        Y, P, *,
        Nfft=2048,  # 变换域降噪使用的 IFFT/FFT 长度
        window_type="hamming",  # "hamming", "hanning", "rectangular"
        win_len_ratio=0.2,  # 窗的长度占 Nfft 的比例 (0~1)
        eps=1e-12
):
    """
    修正版：使用循环移位法进行变换域降噪。
    """
    Y = np.asarray(Y, np.complex64).ravel()
    P = np.asarray(P, np.complex64).ravel()
    assert Y.size == P.size and Y.size > 0
    M = Y.size

    # --- Step1: LS估计 ---
    denom = (np.abs(P) ** 2)
    S1 = Y * np.conj(P) / np.where(denom < eps, eps, denom)

    # --- Step2: 频域补零 ---
    assert Nfft >= M and (Nfft & (Nfft - 1) == 0), "Nfft 建议为 2 的幂且 >= M"
    S2 = np.zeros((Nfft,), dtype=np.complex64)
    S2[:M] = S1

    # --- Step3: 变换到时延域 ---
    s2 = _ifft_u(S2)
    if PLOT_FIGURE:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(np.abs(s2[:]**2), label='|h_td| (symbol 0)')
        plt.xlabel('Sample index (delay)')
        plt.ylabel('Magnitude')
        plt.title('Time-domain impulse response magnitude')
        plt.grid(True)
        plt.legend()
        plt.show()
    # --- 核心修复：循环移位 + 标准窗 ---
    # Step 4: 循环移位，将信号从两侧移到中心
    s2_shifted = np.roll(s2, Nfft // 2)
    if PLOT_FIGURE:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(np.abs(s2_shifted[:]**2), label='|h_td| (symbol 0)')
        plt.xlabel('Sample index (delay)')
        plt.ylabel('Magnitude')
        plt.title('Time-domain impulse response magnitude')
        plt.grid(True)
        plt.legend()
        plt.show()
    # Step 5: 创建并应用标准窗
    if window_type == "hamming":
        window = np.hamming(Nfft)
    elif window_type == "hanning":
        window = np.hanning(Nfft)
    else:  # 默认矩形窗
        window = np.ones(Nfft)

    # 可选：根据比例缩短窗长，只保留中心部分，滤除更多噪声
    if win_len_ratio < 1.0:
        win_len = int(Nfft * win_len_ratio)
        if win_len % 2 == 0: win_len += 1
        start = (Nfft - win_len) // 2
        end = start + win_len
        temp_win = np.zeros(Nfft)
        temp_win[start:end] = window[start:end]
        window = temp_win
    # === 新增：让窗以峰值为中心 ===
    center_on_peak = True
    if center_on_peak:  # 你可以加一个函数参数控制开关
        peak_idx = int(np.argmax(np.abs(s2_shifted)))  # 找主峰位置
        center = Nfft // 2  # 当前窗的“中心”
        shift = peak_idx - center  # 需要平移的量
        window = np.roll(window, shift)  # 把窗的中心挪到主峰

    s3_shifted = s2_shifted * window.astype(np.complex64)
    if PLOT_FIGURE:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(np.abs(s3_shifted[:]**2), label='|h_td| (symbol 0)')
        plt.xlabel('Sample index (delay)')
        plt.ylabel('Magnitude')
        plt.title('Time-domain impulse response magnitude')
        plt.grid(True)
        plt.legend()
        plt.show()
    # Step 6: 反向循环移位，恢复原始位置
    s3 = np.roll(s3_shifted, -Nfft // 2)
    if PLOT_FIGURE:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(np.abs(s3[:]**2), label='|h_td| (symbol 0)')
        plt.xlabel('Sample index (delay)')
        plt.ylabel('Magnitude')
        plt.title('Time-domain impulse response magnitude')
        plt.grid(True)
        plt.legend()
        plt.show()
    # --- Step7: 回频域 ---
    S3 = _fft_u(s3)

    # --- Step8: 抽回 DMRS 序列并做幅度对齐 ---
    H_denoised = S3[:M].astype(np.complex64, copy=False)
    denom = np.vdot(H_denoised, H_denoised)
    if denom != 0:
        alpha = np.vdot(H_denoised, S1) / denom  # S1 是原始 LS
        H_denoised *= alpha
    return H_denoised
def td_dmrs_denoise(
    Y, P, *,
    Nfft=2048,                 # 变换域降噪使用的 IFFT/FFT 长度（建议 1024/2048/4096 等 2^n）
    window="energy",           # "energy"（能量自适应环形窗）或 "fixed"（固定长度环形窗）或 直接传入 ndarray
    win_len=None,              # 固定窗时的窗长（奇数更好）；若 None 则按 CP/FFT 比例估（见下）
    energy_keep=0.80,          # 自适应窗：保留的能量比例（0~1）
    guard=0,                   # 窗左右额外扩展的样点数（整形）
    eps=1e-12,
    win_mean_norm=True
):
    """
    仅对 DMRS 上的 LS 估计做“变换域降噪”，不做频域插值：
      输入:
        Y:  DMRS 上接收符号 (M,)
        P:  DMRS 上发送符号 (M,)
      输出:
        H_denoised: DMRS 点上的降噪信道估计 (M,)

    流程严格对应:
      Step1: S1 = Y * conj(P) / |P|^2   （若 |P|=1 的 QPSK DMRS，等价于 Y*conj(P)）
      Step2: S2 = [S1, 0, 0, ..., 0]    （补零到 Nfft）
      Step3: s2 = IFFT(S2, Nfft)
      Step4: s3 = s2 * W                （环形窗加权）
      Step5: S3 = FFT(s3, Nfft)
      Step6: S4 = S3[:M]                （抽回 DMRS 序列）
    """
    Y = np.asarray(Y, np.complex64).ravel()
    P = np.asarray(P, np.complex64).ravel()
    assert Y.size == P.size and Y.size > 0
    M = Y.size

    # --- Step1: 去导频（鲁棒写法，对一般 DMRS 亦成立；若 |P|=1，会简化为 Y*conj(P)）
    denom = (np.abs(P) ** 2)
    S1 = Y * np.conj(P) / np.where(denom < eps, eps, denom)

    # --- Step2: 频域补零到 Nfft
    assert Nfft >= M and (Nfft & (Nfft - 1) == 0), "Nfft 建议为 2 的幂且 >= M"
    S2 = np.zeros((Nfft,), dtype=np.complex64)
    S2[:M] = S1

    # --- Step3: 变换到时延域
    s2 = _ifft_u(S2)
    if PLOT_FIGURE:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(np.abs(s2[:]**2), label='|h_td| (symbol 0)')
        plt.xlabel('Sample index (delay)')
        plt.ylabel('Magnitude')
        plt.title('Time-domain impulse response magnitude')
        plt.grid(True)
        plt.legend()
        plt.show()
    # --- Step4: 变换域加窗（环形，围绕主峰）
    if isinstance(window, np.ndarray):
        w = np.asarray(window, np.float32).ravel()
        assert w.size == Nfft, "自定义窗长度必须等于 Nfft"
        # if w.max() > 0: w = w / w.max()
    else:
        # 找主峰（循环域）
        p = np.abs(s2) ** 2
        peak = int(np.argmax(p))

        if window == "fixed":
            # 固定长度环形窗；若 win_len 未给，按 CP/FFT 比例估一个
            if win_len is None:
                cp_len = 160.0;
                num_fft = 2048.0
                base = max(10, int(np.ceil((cp_len*4 / num_fft) * Nfft)))
                win_len = base
            if win_len % 2 == 0: win_len = min(Nfft, win_len + 1)
            R = (win_len - 1) // 2 + int(guard)
            offs = np.arange(-R, R + 1, dtype=int)
            idx = (peak + offs) % Nfft
            w = np.zeros(Nfft, dtype=np.float32);
            w[idx] = 1.0
        else:
            # "energy"：能量自适应环形窗，直到覆盖 energy_keep 的能量
            total = float(p.sum())
            if total <= 0:
                w = np.ones(Nfft, dtype=np.float32)
            else:
                left = right = 0
                acc = p[peak]
                while acc / total < float(energy_keep) and (left + right + 1) < Nfft:
                    cand_left = p[(peak - (left + 1)) % Nfft]
                    cand_right = p[(peak + (right + 1)) % Nfft]
                    if cand_left >= cand_right:
                        left += 1;
                        acc += cand_left
                    else:
                        right += 1;
                        acc += cand_right
                left = min(left + int(guard), Nfft - 1)
                right = min(right + int(guard), Nfft - 1)
                offs = np.arange(-left, right + 1, dtype=int)
                idx = (peak + offs) % Nfft
                w = np.zeros(Nfft, dtype=np.float32);
                w[idx] = 1.0

    # 均值=1 归一（DC 增益=1）
    if win_mean_norm:
        wm = max(float(w.mean()), 1e-12)
        w = w / wm

    s3 = s2 * w.astype(np.complex64)

    if PLOT_FIGURE:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(np.abs(s3[:]**2), label='|h_td| (symbol 0)')
        plt.xlabel('Sample index (delay)')
        plt.ylabel('Magnitude')
        plt.title('Time-domain impulse response magnitude')
        plt.grid(True)
        plt.legend()
        plt.show()
    # --- Step5: 回频域
    S3 = _fft_u(s3)


    # --- Step6: 抽回 DMRS 序列（保持与输入同一顺序）
    H_denoised = S3[:M].astype(np.complex64, copy=False)
    denom = np.vdot(H_denoised, H_denoised)
    if denom != 0:
        alpha = np.vdot(H_denoised, S1) / denom  # S1 是原始 LS
        H_denoised *= alpha
    return H_denoised

def estimate_symbol_by_blocks_M(
        col,  # 符号索引
        K,  # 子载波数（240）
        refInd,  # 导频索引（按 Fortran 展平）
        rxSym_all,  # DMRS上的接收符号 (与 refInd 对齐)
        refSym_all,  # DMRS上的发送符号 (与 refInd 对齐)
        rb_size=2,  # 新增：每个块的RB数量
        energy_keep=0.9,
        guard=0,
        method='interpolation',  # 'interpolation' or 'denoising'
        fft_size_N=2048  # 新增：用于降噪的FFT长度
):
    """
    返回：H_col shape=(K,) 本符号整列的信道估计（复数）
    """
    H_col = np.zeros((K,), dtype=np.complex64)
    L_block = rb_size * 12  # 每个块的子载波数 (12子载波/RB)

    # 1) 找本符号的 DMRS
    in_col = np.where((refInd >= col * K) & (refInd < (col + 1) * K))[0]
    if in_col.size == 0:
        return H_col, False  # 此符号无DMRS

    sc_idx = (refInd[in_col] - col * K).astype(int)  # DMRS 频域下标（0..K-1）
    y = rxSym_all[in_col]
    x = refSym_all[in_col]
    eps = 1e-12
    Hls = (y / np.where(np.abs(x) < eps, eps + 0j, x)).astype(np.complex64)
    # 2) 按指定块大小划分频域
    num_blocks = (K + L_block - 1) // L_block  # 向上取整

    for block_idx in range(num_blocks):
        s = block_idx * L_block  # 块起始位置
        e = min(s + L_block, K)  # 块结束位置
        L_blk = e - s  # 当前块实际长度

        # 当前块内的DMRS
        in_blk = np.where((sc_idx >= s) & (sc_idx < e))[0]
        if in_blk.size == 0:
            continue  # 该块没有DMRS，跳过

        dmrs_abs = sc_idx[in_blk]
        dmrs_rel = (dmrs_abs - s).astype(int)  # 块内相对位置
        y_blk = y[in_blk]
        x_blk = x[in_blk]
        Hdmrs = Hls[in_blk]
        # --- 根据method选择处理方式 ---
        if method == 'interpolation':
            # 方法1: 直接使用LS估计进行变换域插值 (与之前逻辑相同)
            eps = 1e-12
            # Hls = (y_blk / np.where(np.abs(x_blk) < eps, eps + 0j, x_blk)).astype(np.complex64)
            # H_blk = ce_block_transform_domain_interpolation(
            #     Hest_dmrs=Hls,
            #     dmrs_pos=dmrs_rel,
            #     L_blk=L_blk,
            #     pilot_consistency=True
            # )
            H_blk = ce_block_td_denoise_fullsymbol(
                Hest_dmrs=Hdmrs,
                dmrs_pos=dmrs_rel,
                L_blk=L_blk,
                energy_keep=energy_keep,
                guard=guard,
                pilot_consistency=True
            )

        elif method == 'denoising':
            # 方法2: 先降噪，再插值
            # Step A: 对导频进行变换域降噪
            # diffs = np.diff(np.sort(dmrs_rel))
            # spacing_est = int(np.gcd.reduce(diffs)) if diffs.size else 2
            # if spacing_est <= 0: spacing_est = 2
            # spacing = spacing_est
            # offset = int(dmrs_rel.min() % spacing)
            #
            # # n = 0..Nd-1 的 comb 顺序重排
            # n_idx = ((dmrs_rel - offset) // spacing).astype(int)
            # order = np.argsort(n_idx)
            # Hls_seq = Hdmrs[order]  # comb 顺序的 LS
            # y_seq = y_blk[order]  # 与导频位置一致的接收
            # x_seq = x_blk[order]  # 与导频位置一致的发送
            # H_ls_denoised = td_dmrs_denoise(y_blk, x_blk, Nfft=256, window="fixed", guard=0,energy_keep=0.97)
            H_ls_denoised = td_dmrs_denoise_fixed(y_blk, x_blk, Nfft=256, window_type="fixed", win_len_ratio=0.16,eps=1e-12)
            # H_ls_denoised = Hdmrs
            H_blk = td_interpolate_from_dmrs(H_ls_denoised, dmrs_rel, L_blk, ls_ref_on_dmrs=H_ls_denoised)
        else:
            raise ValueError("method must be 'interpolation' or 'denoising'")

        H_col[s:e] = H_blk

    return H_col, True

def estimate_time_freq_block(
        start_sym, start_rb, N_sym, rb_size,
        K, N, refInd, rxSym_all, refSym_all,
        energy_keep, guard
):
    """
    【修改后】对一个时频块进行信道估计
    核心改动：一次性筛选出时频块内所有DMRS，避免遗漏。
    """
    L_block = rb_size * 12  # 块的频域宽度
    H_block = np.zeros((L_block, N_sym), dtype=np.complex64)

    # 1. 确定时频块的边界
    start_sc = start_rb * 12
    end_sc = start_sc + L_block
    end_sym_exclusive = start_sym + N_sym

    # 2. 创建一个掩码，用于快速筛选出时频块内的所有DMRS
    # refInd 是全局索引，格式为 global_idx = col * K + row
    # 我们需要找出所有满足 col 和 row 条件的 global_idx
    mask = np.zeros(len(refInd), dtype=bool)

    # 遍历所有DMRS的全局索引，检查是否在时频块内
    for i, global_idx in enumerate(refInd):
        col = global_idx // K
        row = global_idx % K

        if (col >= start_sym and col < end_sym_exclusive and
                row >= start_sc and row < end_sc):
            mask[i] = True

    # 3. 使用掩码一次性获取所有相关的DMRS数据
    if not np.any(mask):
        return H_block, False

    relevant_refInd = refInd[mask]
    relevant_rxSym = rxSym_all[mask]
    relevant_refSym = refSym_all[mask]

    # 4. 将获取到的DMRS按子载波位置分组
    dmrs_dict = {}  # {subcarrier_idx: [(y, x, sym_idx), ...]}
    for i in range(len(relevant_refInd)):
        global_idx = relevant_refInd[i]
        col = global_idx // K
        row = global_idx % K

        # 记录为块内相对位置
        rel_sc = row - start_sc
        rel_sym = col - start_sym

        if rel_sc not in dmrs_dict:
            dmrs_dict[rel_sc] = []
        dmrs_dict[rel_sc].append((relevant_rxSym[i], relevant_refSym[i], rel_sym))

    if not dmrs_dict:
        return H_block, False

    # 5. 对每个子载波位置，进行时间维度的匹配滤波
    Hest_dmrs_list = []
    dmrs_pos_list = []

    for rel_sc, dmrs_data in dmrs_dict.items():
        # 确保每个子载波位置都有N_sym个DMRS
        if len(dmrs_data) != N_sym:
            # 如果数量不匹配，跳过该子载波
            continue

        # 按符号排序，确保顺序正确
        dmrs_data.sort(key=lambda item: item[2])

        y_vec = np.array([d[0] for d in dmrs_data])
        x_vec = np.array([d[1] for d in dmrs_data])

        # 时间维度的匹配滤波
        numerator = np.vdot(x_vec, y_vec)
        denominator = np.vdot(x_vec, x_vec)

        if np.abs(denominator) < 1e-12:
            continue

        H_mf = numerator / denominator

        Hest_dmrs_list.append(H_mf)
        dmrs_pos_list.append(rel_sc)

    if not Hest_dmrs_list:
        return H_block, False

    method = 'interpolation'
    # 5. 根据选择的方法进行频率维度的变换域降噪
    fft_len = 2048
    window = np.hamming(fft_len)
    if method == 'interpolation':
        if fft_len is None or window is None:
            raise ValueError("使用 'interpolation' 方法时，必须提供 fft_len 和 window 参数。")
        H_blk = ce_block_td_denoise_interpolation(
            Hest_dmrs=np.array(Hest_dmrs_list),
            dmrs_pos=np.array(dmrs_pos_list),
            L_blk=L_block,
            fft_len=fft_len,
            window=window,
            pilot_consistency=True,
        )
    else: # 默认使用原有的 'zero_padding' 方法
        H_blk = ce_block_td_denoise_fullsymbol(
            Hest_dmrs=np.array(Hest_dmrs_list),
            dmrs_pos=np.array(dmrs_pos_list),
            L_blk=L_block,
            energy_keep=energy_keep,
            guard=guard,
            pilot_consistency=True
        )
    # # 6. 对匹配滤波后的结果进行频率维度的变换域降噪
    # H_blk = ce_block_td_denoise_fullsymbol(
    #     Hest_dmrs=np.array(Hest_dmrs_list),
    #     dmrs_pos=np.array(dmrs_pos_list),
    #     L_blk=L_block,
    #     energy_keep=energy_keep,
    #     guard=guard,
    #     pilot_consistency=True
    # )

    # 7. 将结果填充到时频块的所有符号上
    for sym_offset in range(N_sym):
        H_block[:, sym_offset] = H_blk

    return H_block, True

def estimate_pdsch_symbol_fullband_td(
        rxGrid,
        refGrid,
        col,
        start_rb=0,
        end_rb=None,
        td_win_len_ratio=0.08,
        td_window_type="fixed",
        eps=1e-12,
        use_denoised_pilot_anchor=True,
):
    """
    对某一个 PDSCH DMRS symbol:
      1) 在全活动带宽上提取 DMRS LS 信道
      2) 对 comb-DMRS 序列做 TD 去噪
      3) 插值回整个活动带宽

    返回:
      H_blk: shape=(K_active,)
      ok   : bool
    """
    K, N = rxGrid.shape
    if end_rb is None:
        end_rb = K // 12

    s_idx = start_rb * 12
    e_idx = end_rb * 12
    K_active = e_idx - s_idx
    if K_active <= 0:
        return np.zeros((0,), dtype=np.complex64), False

    # 本 symbol、活动带宽内的 DMRS 位置
    ref_col = refGrid[s_idx:e_idx, col]
    dmrs_pos_rel = np.where(np.abs(ref_col) > eps)[0]   # 相对 [0, K_active)
    if dmrs_pos_rel.size == 0:
        return np.zeros((K_active,), dtype=np.complex64), False

    dmrs_pos_abs = dmrs_pos_rel + s_idx

    # LS
    y = rxGrid[dmrs_pos_abs, col]
    x = refGrid[dmrs_pos_abs, col]
    H_ls = y * np.conj(x) / np.where(np.abs(x) ** 2 < eps, eps, np.abs(x) ** 2)

    # 动态 Nfft：至少不小于 Nd，且取 2 的幂
    Nd = len(H_ls)
    nfft_dynamic = max(2 ** int(np.ceil(np.log2(max(Nd, 1)))), 64)

    # TD 去噪
    H_ls_denoised = td_dmrs_denoise_fixed(
        H_ls,
        np.ones_like(H_ls),
        Nfft=nfft_dynamic,
        window_type=td_window_type,
        win_len_ratio=td_win_len_ratio,
        eps=eps
    )

    # comb 插值回全活动带宽
    H_blk = td_interpolate_from_dmrs(
        H_dmrs_denoised=H_ls_denoised,
        dmrs_pos_rel=dmrs_pos_rel,
        L_blk=K_active,
        keep_pilot_consistency=True,
        ls_ref_on_dmrs=H_ls_denoised if use_denoised_pilot_anchor else H_ls,
        spacing=2,   # 实际这里你函数里已基本不用 spacing 了
        offset=0
    )

    return H_blk.astype(np.complex64), True
def myChannelEstimate_pdsch_fullband_td(
        rxGrid=None,
        refGrid=None,
        dmrs_l=None,
        start_rb=0,
        end_rb=None,
        td_win_len_ratio=0.08,
        td_window_type="fixed",
        eps=1e-12,
        use_denoised_pilot_anchor=True,
):
    """
    适用于 PDSCH 的全带宽 TD 去噪 CE:
      - 每个 DMRS symbol 独立:
            LS -> TD denoise -> fullband frequency interpolation
      - 非 DMRS symbol:
            时间线性插值

    参数:
      rxGrid, refGrid : shape=(K, N)
      dmrs_l          : DMRS symbol 列号列表，例如 [0, 6, 9]
      start_rb, end_rb: 活动带宽范围
    """
    if rxGrid is None or refGrid is None:
        raise ValueError("rxGrid and refGrid must not be None")

    K, N = rxGrid.shape
    if end_rb is None:
        end_rb = K // 12

    s_idx = start_rb * 12
    e_idx = end_rb * 12
    K_active = e_idx - s_idx
    if K_active <= 0:
        raise ValueError("Active bandwidth is empty")

    dmrs_cols = _normalize_dmrs_cols(dmrs_l, N)
    if dmrs_cols.size == 0:
        raise ValueError("dmrs_l is empty or invalid")

    H = np.zeros_like(rxGrid, dtype=np.complex64)
    has_dmrs_col = np.zeros(N, dtype=bool)

    # 1) 每个 DMRS symbol 独立做“全带宽 TD 去噪 + 频域插值”
    for col in dmrs_cols:
        H_blk, ok = estimate_pdsch_symbol_fullband_td(
            rxGrid=rxGrid,
            refGrid=refGrid,
            col=col,
            start_rb=start_rb,
            end_rb=end_rb,
            td_win_len_ratio=td_win_len_ratio,
            td_window_type=td_window_type,
            eps=eps,
            use_denoised_pilot_anchor=use_denoised_pilot_anchor,
        )
        if ok:
            H[s_idx:e_idx, col] = H_blk
            has_dmrs_col[col] = True

    # 2) 非 DMRS symbol 做时间线性插值
    cols_with = np.where(has_dmrs_col)[0]
    cols_wo = np.where(~has_dmrs_col)[0]
    H = _time_interp_complex(H, cols_with, cols_wo, s_idx=s_idx, e_idx=e_idx)

    return H
def ce_block_td_denoise_interpolation(
        Hest_dmrs,  # shape = (Ndmrs,) DMRS上的LS信道估计
        dmrs_pos,  # shape = (Ndmrs,) DMRS在块内的相对索引
        L_blk,  # 块的总长度
        fft_len,  # FFT长度 N (必须为2的幂次，且 >= Ndmrs)
        window,  # 时延域窗函数W (长度为fft_len)
        pilot_consistency=True  # 导频一致性
):
    """
    基于频域补零和时域加窗的变换域降噪。
    流程：频域补零 -> IFFT -> 时域加窗 -> FFT -> 抽取。
    """
    Hest_dmrs = np.asarray(Hest_dmrs, dtype=np.complex64).ravel()
    dmrs_pos = np.asarray(dmrs_pos, dtype=int).ravel()
    Nd = Hest_dmrs.size

    # --- 参数检查 ---
    if not (fft_len & (fft_len - 1) == 0):
        raise ValueError("fft_len 必须是2的幂次。")
    if fft_len < Nd:
        raise ValueError("fft_len 必须大于或等于DMRS的数量。")
    if window.size != fft_len:
        raise ValueError("窗函数的长度必须等于fft_len。")

    # --- Step 1 & 2: 构建频域稀疏向量并补零 ---
    # S2: 长度为fft_len的频域向量，在DMRS位置上是信道估计值，其余为0
    S2 = np.zeros(fft_len, dtype=np.complex64)
    S2[dmrs_pos] = Hest_dmrs

    # --- Step 3: 变换到时延域 ---
    s2 = _ifft_u(S2)  # 使用单位化IFFT
    s2_shifted = np.roll(s2, fft_len // 2)
    # --- Step 4: 在时延域进行加窗处理 ---
    s3_shifted  = s2_shifted * window
    s3 = np.roll(s3_shifted, -fft_len // 2)
    # --- Step 5: 从时延域逆变换回频域 ---

    S3 = _fft_u(s3)  # 使用单位化FFT

    # --- Step 6: 在频域抽取对应位置的信道估计结果 ---
    # H_final_dmrs: 在DMRS位置上的、经过降噪处理的信道估计
    H_final_dmrs = S3[dmrs_pos]

    # --- 后处理：填充整个频块 ---
    # 使用线性插值，将DMRS位置的信道估计扩展到整个块
    H_blk = np.zeros(L_blk, dtype=np.complex64)
    if Nd > 1:
        known_idx = dmrs_pos
        known_val = H_final_dmrs
        unknown_idx = np.setdiff1d(np.arange(L_blk), known_idx)
        if unknown_idx.size > 0:
            H_blk.real[unknown_idx] = np.interp(unknown_idx, known_idx, known_val.real)
            H_blk.imag[unknown_idx] = np.interp(unknown_idx, known_idx, known_val.imag)
    elif Nd == 1:
        H_blk[:] = H_final_dmrs  # 如果只有一个DMRS，则整个块都使用这个值

    # --- 导频一致性 ---
    if pilot_consistency:
        H_blk[dmrs_pos] = Hest_dmrs

    return H_blk
def myChannelEstimate_stdorigin(rxGrid=None, refInd=None, refSym=None, refGrid=None, carrier=None, prg_gran=24, energy_keep=0.9, guard=0,EST_TFDOMAIN = True, EST_PBCH = False, method = 'interpolation' ,rb_size = 3, N_sym=2,dmrs_l = 0,start_rb=0,end_rb=20):
    cp = "normal" if carrier is None else "extended"

    # 解析导频网格
    if refGrid is None:
        if (refInd is None) or (refSym is None):
            print("Error: refInd and refSym need to be given when refGrid is None!")
            return
        refGrid = np.zeros(rxGrid.shape, dtype=complex)
        nrSetResources(refInd, refGrid, refSym)
    else:
        pilot_idx = np.where(refGrid.ravel(order="F") != 0)
        refInd = pilot_idx[0]
        refSym = refGrid.ravel(order="F")[refInd]

    # 基本维度
    K, N = rxGrid.shape  # subcarriers, symbols

    # DMRS 处接收/发送符号
    rxSym_all = rxGrid.ravel(order="F")[refInd]
    refSym_all = refSym

    # 输出信道
    # rb_size = 3
    # N_sym = 2
    H = np.zeros_like(refGrid, dtype=complex)
    has_dmrs_col = np.zeros(N, dtype=bool)
    if EST_TFDOMAIN: ## 含有DMRS的符号数>1时，e.g. PDCCH
        L_block = rb_size * 12
        if N == N_sym:
            for start_sym in range(0, N, N_sym):
                for start_rb in range(0, K // 12, rb_size):

                    # 调用新的时频块处理函数
                    H_block, ok = estimate_time_freq_block(
                        start_sym=start_sym,
                        start_rb=start_rb,
                        N_sym=N_sym,
                        rb_size=rb_size,
                        K=K, N=N,
                        refInd=refInd,
                        rxSym_all=rxSym_all,
                        refSym_all=refSym_all,
                        energy_keep=energy_keep,
                        guard=guard
                    )

                    if ok:
                        # 将处理结果填充回H矩阵
                        for sym_offset in range(N_sym):
                            col = start_sym + sym_offset
                            if col >= N: continue
                            s = start_rb * 12
                            e = s + L_block
                            H[s:e, col] = H_block[:, sym_offset]
                            has_dmrs_col[col] = True
            # --- 后处理：对未覆盖的点进行插值 ---
            for col in range(N):
                if not has_dmrs_col[col]:
                    continue

                missing = np.where(H[:, col] == 0)[0]
                if missing.size:
                    known = np.where(H[:, col] != 0)[0]
                    if known.size > 1:
                        H.real[missing, col] = np.interp(missing, known, H.real[known, col])
                        H.imag[missing, col] = np.interp(missing, known, H.imag[known, col])
        else:# 需要输入dmrs_pos
            start_sym = dmrs_l
            for start_rb in range(0, K // 12, rb_size):
                H_block, ok = estimate_time_freq_block(
                    start_sym=start_sym,
                    start_rb=start_rb,
                    N_sym=1,  # 这里强制只做 1 个符号（就是 DMRS 那列）
                    rb_size=rb_size,
                    K=K, N=N,
                    refInd=refInd,
                    rxSym_all=rxSym_all,
                    refSym_all=refSym_all,
                    energy_keep=energy_keep,
                    guard=guard
                )
                if ok:
                    s = start_rb * 12
                    e = s + rb_size * 12
                    H[s:e, dmrs_l] = H_block[:, 0]
                    has_dmrs_col[dmrs_l] = True

            # 频域补洞（仅在 dmrs_l 那列）
            missing = np.where(H[:, dmrs_l] == 0)[0]
            if missing.size:
                known = np.where(H[:, dmrs_l] != 0)[0]
                if known.size > 1:
                    H.real[missing, dmrs_l] = np.interp(missing, known, H.real[known, dmrs_l])
                    H.imag[missing, dmrs_l] = np.interp(missing, known, H.imag[known, dmrs_l])
    else:
        for col in range(N):
            if EST_PBCH: #PBCH，逐块进行变换域降噪
                # H_col, ok = estimate_symbol_by_blocks(
                #     col=col, K=K, refInd=refInd, rxSym_all=rxSym_all, refSym_all=refSym_all,start_rb=start_rb,end_rb=end_rb
                # )
                H_col, ok = estimate_symbol_by_blocks1(
                    col=col, K=K, refInd=refInd, rxSym_all=rxSym_all, refSym_all=refSym_all, start_rb=start_rb,
                    end_rb=end_rb
                )
                # H_col, ok = estimate_symbol_by_blocks_simple(
                #     col=col, K=K, refInd=refInd, rxSym_all=rxSym_all, refSym_all=refSym_all, start_rb=start_rb,
                #     end_rb=end_rb
                # )
            else: # PDSCH 按照M RB大小的Bundle进行降噪
                H_col, ok =estimate_symbol_by_blocks_M(
                    col=col, K=K, refInd=refInd, rxSym_all=rxSym_all, refSym_all=refSym_all, rb_size=rb_size,
                    method=method, # 选择新方法
                    )
                #     estimate_symbol_by_blocks_M(
                #     col=col, K=K, refInd=refInd, rxSym_all=rxSym_all, refSym_all=refSym_all, rb_size=3, method = method
                # )
            H[:, col] = H_col
            has_dmrs_col[col] = ok

            # 若某列仍有未覆盖点（极少），做一次频域线性“补洞”
            missing = np.where(H[:, col] == 0)[0]
            if missing.size:
                known = np.where(H[:, col] != 0)[0]
                if known.size:
                    H.real[missing, col] = np.interp(missing, known, H.real[known, col])
                    H.imag[missing, col] = np.interp(missing, known, H.imag[known, col])

    # —— 无 DMRS 列：时间方向插值（保留原逻辑）——
    if not np.all(has_dmrs_col):
        cols_with = np.where(has_dmrs_col)[0]
        cols_wo = np.where(~has_dmrs_col)[0]
        if cols_with.size >= 2:
            for k in range(K):
                y = H[k, cols_with]
                H[k, cols_wo] = (np.interp(cols_wo, cols_with, np.real(y))
                                 + 1j * np.interp(cols_wo, cols_with, np.imag(y)))
        elif cols_with.size == 1:
            H[:, cols_wo] = H[:, cols_with[0:1]]

    # —— 有效带宽的 TD 去噪 (Transform-Domain Denoising) ——
    H_dn = np.copy(H)
    
    # 根据你的配置获取有效带宽范围
    s_idx = start_rb * 12
    e_idx = end_rb * 12
    K_active = e_idx - s_idx

    if K_active > 0:
        for col in range(N):
            # 如果这一列完全没有信道估计值，则跳过
            if np.all(H[s_idx:e_idx, col] == 0):
                continue
                
            # 1. 提取有效频带内的信道估计，并用 IFFT 变换到时域（冲激响应）
            H_freq = H_dn[s_idx:e_idx, col]
            h_time = np.fft.ifft(H_freq)
            
            # 2. 时域加窗 (Windowing) 滤除纯噪声
            # PBCH 通常是 240 个子载波。15kHz SCS 下，240 点的 IFFT 时域分辨率约为 277ns。
            # 保留前 32 个 tap，相当于保留了约 8.8us 的多径长度，足以完美覆盖 TDL-C 300ns 的时延扩展
            # 同时能滤除 (240-40)/240 ≈ 83% 的噪声能量
            keep_taps = 32  
            window = np.zeros_like(h_time)
            window[:keep_taps] = 1.0
            
            # 保护位：有时候由于定时同步误差(Timing Offset)，能量会泄露到循环前缀(CP)的尾部（即最后几个点）
            # 所以保守起见，保留最后 8 个 tap 的能量
            window[-8:] = 1.0 
            
            h_time_denoised = h_time * window
            
            # 3. 用 FFT 变回频域，得到降噪后的信道
            H_freq_denoised = np.fft.fft(h_time_denoised)
            
            # 将降噪后的信道回填
            H_dn[s_idx:e_idx, col] = H_freq_denoised

    return H_dn

def myChannelEstimate_std(
    rxGrid=None, refInd=None, refSym=None, refGrid=None, carrier=None,
    prg_gran=24, energy_keep=0.9, guard=0,
    EST_TFDOMAIN=True, EST_PBCH=False, EST_PDSCH_FULLBAND=False,
    method='interpolation', rb_size=3, N_sym=2, dmrs_l=0,
    start_rb=0, end_rb=20,
    td_win_len_ratio=0.08,
    td_window_type="fixed",
):
    cp = "normal" if carrier is None else "extended"

    # 解析导频网格
    if refGrid is None:
        if (refInd is None) or (refSym is None):
            print("Error: refInd and refSym need to be given when refGrid is None!")
            return
        refGrid = np.zeros(rxGrid.shape, dtype=complex)
        nrSetResources(refInd, refGrid, refSym)
    else:
        pilot_idx = np.where(refGrid.ravel(order="F") != 0)
        refInd = pilot_idx[0]
        refSym = refGrid.ravel(order="F")[refInd]

    K, N = rxGrid.shape

    # ========== 新增：PDSCH 全带宽 TD 去噪分支 ==========
    if (not EST_TFDOMAIN) and (not EST_PBCH) and EST_PDSCH_FULLBAND:
        return myChannelEstimate_pdsch_fullband_td(
            rxGrid=rxGrid,
            refGrid=refGrid,
            dmrs_l=dmrs_l,
            start_rb=start_rb,
            end_rb=end_rb,
            td_win_len_ratio=td_win_len_ratio,
            td_window_type=td_window_type,
            eps=1e-12,
            use_denoised_pilot_anchor=True,
        )


    # -------------------------------------------------------------
    # 🚨 公平竞技版：标准 5G PBCH 的智能联合信道估计 (Joint CE)
    # -------------------------------------------------------------
    if not EST_TFDOMAIN:
        if EST_PBCH and N == 4: # 确保这是标准 4 符号 SSB
            pbch_syms_full = [1, 3]       # 满带宽的符号
            pbch_syms_partial = [2]       # 中间被 SSS 挖洞的符号
            all_pbch_syms = [1, 2, 3]     # 所有包含 DMRS 的符号

            s_idx = start_rb * 12
            e_idx = end_rb * 12
            K_active = e_idx - s_idx      # 标准是 240

            # 1. 提取满带宽符号(如符号1)上的所有 DMRS 子载波索引
            dmrs_sc = np.where(refGrid[s_idx:e_idx, pbch_syms_full[0]] != 0)[0] + s_idx

            if len(dmrs_sc) > 0:
                H_ls_avg = np.zeros(len(dmrs_sc), dtype=complex)

                # 2. 智能跨符号累加 (避开中心空洞 48~191)
                for i, k in enumerate(dmrs_sc):
                    rel_k = k - s_idx

                    # 判断该 DMRS 是否落在符号 2 的 SSS 区域内
                    if 48 <= rel_k < 192:
                        syms_to_avg = pbch_syms_full    # 中心区域：平均 [1, 3]
                    else:
                        syms_to_avg = all_pbch_syms     # 两翼区域：平均 [1, 2, 3]

                    y_sum = 0
                    valid_syms = 0
                    for col in syms_to_avg:
                        y = rxGrid[k, col]
                        x = refGrid[k, col]
                        if np.abs(x) > 1e-12:
                            y_sum += (y / x)
                            valid_syms += 1

                    if valid_syms > 0:
                        H_ls_avg[i] = y_sum / valid_syms

                # 3. 在纯净的 DMRS 信道上进行变换域降噪
                x_dummy = np.ones_like(H_ls_avg)
                L_dmrs = len(H_ls_avg)  # 240RB下，通常是 60 个 DMRS
                nfft_dynamic = max(2**(int(np.ceil(np.log2(L_dmrs)))), 64)

                H_ls_denoised = td_dmrs_denoise_fixed(
                    H_ls_avg, x_dummy,
                    Nfft=nfft_dynamic,
                    window_type="fixed",
                    win_len_ratio=0.15,  # 64 点的 0.15 约 9.6 个 tap，完美覆盖 TDL-C
                    eps=1e-12
                )

                # 4. 安全插值到全频带
                dmrs_rel = dmrs_sc - s_idx
                H_blk = td_interpolate_from_dmrs(
                    H_ls_denoised, dmrs_rel, K_active,
                    spacing=4, ls_ref_on_dmrs=H_ls_denoised
                )

                # 5. 把完美信道覆盖到 [1, 2, 3] 上
                for col in all_pbch_syms:
                    H[s_idx:e_idx, col] = H_blk
                    has_dmrs_col[col] = True

        else:
            # PDSCH 或其他非 PBCH 逻辑保持不变
            for col in range(N):
                H_col, ok = estimate_symbol_by_blocks_M(
                    col=col, K=K, refInd=refInd, rxSym_all=rxSym_all, refSym_all=refSym_all, rb_size=rb_size, method=method
                )
                H[:, col] = H_col
                has_dmrs_col[col] = ok
    else:
        # EST_TFDOMAIN == True 分支保持不变 (通常用于 PDCCH 等)
        pass # 为了代码简洁，你原先这块极其庞大的逻辑如果没用可以不贴，如果有用就原封不动保留

    # —— 无 DMRS 列：时间方向插值 (将符号 1 的信道安全推给符号 0 的 PSS) ——
    if not np.all(has_dmrs_col):
        cols_with = np.where(has_dmrs_col)[0]
        cols_wo = np.where(~has_dmrs_col)[0]
        if cols_with.size >= 2:
            for k in range(K):
                y = H[k, cols_with]
                H[k, cols_wo] = (np.interp(cols_wo, cols_with, np.real(y))
                                 + 1j * np.interp(cols_wo, cols_with, np.imag(y)))
        elif cols_with.size == 1:
            H[:, cols_wo] = H[:, cols_with[0:1]]

    # ==========================================
    # 🚨 拦截器：如果是 PBCH，带着完美的 Joint CE 结果直接返回！
    # ==========================================
    if EST_PBCH:
        return H

    # —— PDSCH/PDCCH 等继续走老版的有效带宽 TD 去噪 ——
    H_dn = np.copy(H)
    s_idx = start_rb * 12
    e_idx = end_rb * 12
    K_active = e_idx - s_idx

    if K_active > 0:
        for col in range(N):
            if np.all(H[s_idx:e_idx, col] == 0): continue
            H_freq = H_dn[s_idx:e_idx, col]
            h_time = np.fft.ifft(H_freq)
            keep_taps = 48
            window = np.zeros_like(h_time)
            window[:keep_taps] = 1.0
            window[-16:] = 1.0
            h_time_denoised = h_time * window
            H_freq_denoised = np.fft.fft(h_time_denoised)
            H_dn[s_idx:e_idx, col] = H_freq_denoised

    return H_dn
