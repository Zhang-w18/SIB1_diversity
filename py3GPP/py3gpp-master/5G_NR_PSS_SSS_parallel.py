import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor

# 5G PSS/SSS 检测，改成了 Tx SNR, 不显示进度条

# =============================================================================
# User-provided channel model (fixed for DS unit consistency)
# =============================================================================
class ChannelInfo:
    def __init__(self, nTx, nRx, Speed, Fc, Fs, ChanType, DS, NFFT, TO=0, FO=0):
        assert nTx > 0
        assert nRx > 0
        assert Speed >= 0

        self.nTx = nTx
        self.nRx = nRx
        self.Speed = Speed / 3.6              # m/s
        self.Fc = Fc                          # Hz
        self.Fs = Fs                          # Hz
        self.NFFT = NFFT
        self.timeOff = TO
        self.CFO = FO
        self.ChanType = ChanType.upper()
        self.DS = DS                          # ns
        self.ChannelTypePara()

    def ChannelTypePara(self):
        if self.ChanType.find('TDLC') >= 0:
            self.DelaysperPath = np.array([
                0, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366, 0.6448, 0.6560,
                0.6584, 0.7935, 0.8213, 0.9336, 1.2285, 1.3083, 2.1704, 2.7105,
                4.2589, 4.6003, 5.4902, 5.6077, 6.3065, 6.6374, 7.0427, 8.6523
            ]) * self.DS
            self.PowerperPath = np.array([
                -4.4, -1.2, -3.5, -5.2, -2.5, 0, -2.2, -3.9, -7.4, -7.1,
                -10.7, -11.1, -5.1, -6.8, -8.7, -13.2, -13.9, -13.9, -15.8,
                -17.1, -16, -15.7, -21.6, -22.8
            ])
            self.first_path_los = 0
        elif self.ChanType == 'AWGN':
            self.DelaysperPath = np.array([0.0])
            self.PowerperPath = np.array([0.0])
            self.first_path_los = 0
        else:
            raise ValueError("This example keeps only TDLC and AWGN.")

        self.taps = np.round(self.DelaysperPath * 1e-9 * self.Fs).astype(int)

        power = 10 ** (self.PowerperPath / 10)
        power = power / np.sum(power)
        self.taps_power = np.sqrt(power)
        self.num_taps = self.taps.size
        self.conv_points = np.unique(self.taps)
        self.DS_tap = int(np.max(self.taps)) + 1
        self.doppler = self.Speed * self.Fc / 3e8

    def gen_Jakes_Accelerate(self, time_samples):
        M = 8
        hf_samples = np.zeros([self.nTx * self.nRx, self.DS_tap, time_samples.size], dtype=complex)
        wd_t = 2 * np.pi * self.doppler * time_samples
        kIndex = np.arange(0, M).reshape(M, 1)

        for iant in range(self.nTx * self.nRx):
            for itap in range(self.num_taps):
                alpha = (np.arange(0, M) + np.random.rand(M)) * np.pi / (2 * M)
                phi = (2 * np.random.rand(M) - 1) * np.pi
                psi = (2 * np.random.rand(M) - 1) * np.pi

                hf_samples[iant][self.taps[itap]][:] += self.taps_power[itap] * (
                    np.sum(
                        np.cos(np.cos(alpha[kIndex]) * wd_t + phi[kIndex]) +
                        1j * np.cos(np.sin(alpha[kIndex]) * wd_t + psi[kIndex]),
                        axis=0
                    )
                )

        hf_samples *= np.sqrt(1 / M)
        self.time_channel = hf_samples.reshape(self.nRx, self.nTx, self.DS_tap, time_samples.size)


# =============================================================================
# Simulation config
# =============================================================================
@dataclass
class NRSimConfig:
    pci: Optional[int] = None
    fc_hz: float = 3.5e9
    scs_hz: float = 30e3
    nfft: int = 512

    chan_type: str = "TDLC"
    delay_spread_ns: float = 300.0
    speed_kmh: float = 3.0
    system_cfo_ppm: float = 5.0
    snr_db: float = 0.0
    timing_offset_samp: int = 80

    # 必须为正奇数：
    # 1 -> [0] * scs_hz
    # 3 -> [-1, 0, 1] * scs_hz
    # 5 -> [-2, -1, 0, 1, 2] * scs_hz
    # 7 -> [-3, -2, -1, 0, 1, 2, 3] * scs_hz
    fo_num_candidates: int = 5

    ce_delay_guard_bins: int = 2
    ce_wrap_bins: int = 1
    # Oracle RSRP: SSS信道估计后在频域平滑的窗口长度（建议奇数）
    oracle_rsrp_smooth_win: int = 9
    seed: int = 1


# =============================================================================
# Numerology / OFDM
# =============================================================================
@lru_cache(maxsize=None)
def normal_cp_lengths(nfft: int) -> np.ndarray:
    cp_others = int(round(144 * nfft / 2048))
    cp_first = nfft - 13 * cp_others
    return np.array([cp_first] + [cp_others] * 13, dtype=int)


@lru_cache(maxsize=None)
def centered_bin_range(width: int) -> np.ndarray:
    return np.arange(-width // 2, width // 2, dtype=int)


def ofdm_modulate_symbol(grid_shifted: np.ndarray, cp_len: int) -> np.ndarray:
    x_fd = np.fft.ifftshift(grid_shifted)
    x_td = np.fft.ifft(x_fd, norm="ortho")
    return np.concatenate([x_td[-cp_len:], x_td])


def ofdm_demodulate_symbol(y_td: np.ndarray, nfft: int, cp_len: int) -> np.ndarray:
    y_no_cp = y_td[cp_len:cp_len + nfft]
    Y = np.fft.fft(y_no_cp, norm="ortho")
    return np.fft.fftshift(Y)


# =============================================================================
# 38.211-compliant NR PSS / SSS generation
# =============================================================================
@lru_cache(maxsize=1)
def mseq_pss_base() -> np.ndarray:
    x = np.zeros(127, dtype=int)
    x[:7] = np.array([0, 1, 1, 0, 1, 1, 1], dtype=int)
    for n in range(120):
        x[n + 7] = (x[n + 4] + x[n]) % 2
    return x


def nr_pss(nid2: int) -> np.ndarray:
    x = mseq_pss_base()
    m = (np.arange(127) + 43 * nid2) % 127
    return (1 - 2 * x[m]).astype(complex)


@lru_cache(maxsize=1)
def mseq_sss_x0() -> np.ndarray:
    x = np.zeros(127, dtype=int)
    x[0] = 1
    for n in range(120):
        x[n + 7] = (x[n + 4] + x[n]) % 2
    return x


@lru_cache(maxsize=1)
def mseq_sss_x1() -> np.ndarray:
    x = np.zeros(127, dtype=int)
    x[0] = 1
    for n in range(120):
        x[n + 7] = (x[n + 1] + x[n]) % 2
    return x


def nr_sss(nid1: int, nid2: int) -> np.ndarray:
    x0 = mseq_sss_x0()
    x1 = mseq_sss_x1()
    m0 = 15 * (nid1 // 112) + 5 * nid2
    m1 = nid1 % 112
    n = np.arange(127)
    s0 = 1 - 2 * x0[(n + m0) % 127]
    s1 = 1 - 2 * x1[(n + m1) % 127]
    return (s0 * s1).astype(complex)


@lru_cache(maxsize=1)
def build_sss_bank():
    """
    预生成所有 SSS 序列：
        shape = [nid2=3, nid1=336, 127]
    这样 detect_sss() 就不需要每个 trial 再循环生成 336 条序列。
    """
    bank = np.zeros((3, 336, 127), dtype=complex)
    for nid2 in range(3):
        for nid1 in range(336):
            bank[nid2, nid1] = nr_sss(nid1, nid2)
    return bank


# =============================================================================
# SSB mapping
# =============================================================================
def build_ssb_grid(pci: int, nfft: int) -> Tuple[np.ndarray, Dict]:
    nid2 = pci % 3
    nid1 = pci // 3

    grid = np.zeros((4, nfft), dtype=complex)
    k_ssb = centered_bin_range(240)
    sync_local = np.arange(56, 183)
    pss_k = k_ssb[sync_local]
    sss_k = k_ssb[sync_local]

    grid[0, pss_k + nfft // 2] = nr_pss(nid2)
    grid[2, sss_k + nfft // 2] = build_sss_bank()[nid2, nid1]

    info = {
        "nid1": nid1,
        "nid2": nid2,
        "k_ssb": k_ssb,
        "pss_k": pss_k,
        "sss_k": sss_k,
    }
    return grid, info


def build_ssb_waveform(pci: int, nfft: int):
    grid, info = build_ssb_grid(pci, nfft)
    cp = normal_cp_lengths(nfft)
    tx_syms = [ofdm_modulate_symbol(grid[l], cp[l]) for l in range(4)]
    tx_ssb = np.concatenate(tx_syms)
    return tx_ssb, grid, info, cp


def get_reference_re_energy(grid: np.ndarray) -> float:
    """
    统计参考 RE 集合上的平均能量。
    当前这份代码只有 PSS / SSS 是非零 RE，因此直接取 grid 中全部非零 RE。
    若以后扩展到 PBCH BLER 仿真，建议改成只统计 PBCH + PBCH DMRS 的 RE。
    """
    ref_re = grid[np.abs(grid) > 0]
    if ref_re.size == 0:
        raise ValueError("No non-zero reference RE found in grid.")
    return float(np.mean(np.abs(ref_re) ** 2))


@lru_cache(maxsize=None)
def build_pss_reference_waveforms(nfft: int):
    cp = normal_cp_lengths(nfft)
    refs = []
    k_ssb = centered_bin_range(240)
    sync_local = np.arange(56, 183)
    pss_k = k_ssb[sync_local]

    for nid2 in range(3):
        g = np.zeros(nfft, dtype=complex)
        g[pss_k + nfft // 2] = nr_pss(nid2)
        refs.append(ofdm_modulate_symbol(g, cp[0]))
    return tuple(refs)


@lru_cache(maxsize=None)
def build_pss_reference_useful_symbols(nfft: int):
    refs = []
    k_ssb = centered_bin_range(240)
    sync_local = np.arange(56, 183)
    pss_k = k_ssb[sync_local]

    for nid2 in range(3):
        g = np.zeros(nfft, dtype=complex)
        g[pss_k + nfft // 2] = nr_pss(nid2)
        x_fd = np.fft.ifftshift(g)
        x_td = np.fft.ifft(x_fd, norm="ortho")
        refs.append(x_td)
    return tuple(refs)


@lru_cache(maxsize=None)
def get_pss_ref_energy(nfft: int) -> float:
    ref = build_pss_reference_waveforms(nfft)[0]
    return float(np.vdot(ref, ref).real)


# =============================================================================
# Channel / impairments
# =============================================================================
def apply_time_varying_channel(x: np.ndarray, cfg: NRSimConfig):
    ch = ChannelInfo(
        nTx=1,
        nRx=1,
        Speed=cfg.speed_kmh,
        Fc=cfg.fc_hz,
        Fs=cfg.nfft * cfg.scs_hz,
        ChanType=cfg.chan_type,
        DS=cfg.delay_spread_ns,
        NFFT=cfg.nfft,
    )

    nout = len(x) + ch.DS_tap - 1
    time_samples = np.arange(nout) / (cfg.nfft * cfg.scs_hz)
    ch.gen_Jakes_Accelerate(time_samples)
    h = ch.time_channel[0, 0, :, :]

    y = np.zeros(nout, dtype=complex)
    x_len = len(x)
    for tap in range(ch.DS_tap):
        m = np.arange(tap, tap + x_len)
        valid = m < nout
        y[m[valid]] += h[tap, m[valid]] * x[:np.sum(valid)]
    return y, ch


def apply_cfo_and_noise(x: np.ndarray, cfg: NRSimConfig, ref_re_energy: float):
    fs = cfg.nfft * cfg.scs_hz
    cfo_hz = cfg.system_cfo_ppm * 1e-6 * cfg.fc_hz
    n = np.arange(len(x))
    y = x * np.exp(1j * 2 * np.pi * cfo_hz * n / fs)

    snr_lin = 10 ** (cfg.snr_db / 10.0)

    # 这份代码的 OFDM 采用 norm="ortho"，FFT/IFFT 为 unitary 变换，
    # 因此时域复采样点噪声方差 == 频域每个 RE 的噪声方差。
    # 目标定义：reference-RE average energy / noise variance = snr_lin
    noise_var = ref_re_energy / snr_lin

    w = np.sqrt(noise_var / 2.0) * (
        np.random.randn(len(y)) + 1j * np.random.randn(len(y))
    )

    return y + w, cfo_hz, noise_var


# =============================================================================
# PSS joint timing / NID2 / integer CFO
# =============================================================================
def build_integer_fo_grid(cfg: NRSimConfig) -> np.ndarray:
    n = int(cfg.fo_num_candidates)
    if n <= 0 or (n % 2) == 0:
        raise ValueError("fo_num_candidates 必须为正奇数，例如 1、3、5、7。")
    half = n // 2
    integer_bins = np.arange(-half, half + 1, dtype=int)
    return integer_bins.astype(float) * cfg.scs_hz


def joint_pss_search(rx: np.ndarray, nfft: int, cfg: NRSimConfig, fs: float):
    """
    相比 v4 的优化点：
    - PSS 参考波形和参考能量全部缓存
    - 对同一个整数倍频偏假设，只计算一次 rx_win_energy
    - 避免在每个 nid2 上重复做相同的窗口能量卷积
    """
    refs = build_pss_reference_waveforms(nfft)
    ref_energy = get_pss_ref_energy(nfft)
    fo_grid = build_integer_fo_grid(cfg)
    n = np.arange(len(rx))
    ones_ref = np.ones(len(refs[0]))

    best = {
        "metric": -np.inf,
        "nid2": None,
        "ifo_hz": None,
        "ifo_bins": None,
        "timing": None,
        "corr_at_peak": None,
        "fo_grid": fo_grid,
    }

    for fo in fo_grid:
        rx_c = rx * np.exp(-1j * 2 * np.pi * fo * n / fs)
        rx_win_energy = np.convolve(np.abs(rx_c) ** 2, ones_ref, mode="valid")
        denom = rx_win_energy * ref_energy + 1e-12

        for nid2, ref in enumerate(refs):
            corr = np.convolve(rx_c, np.conj(ref[::-1]), mode="valid")
            metric = np.abs(corr) ** 2 / denom
            idx = int(np.argmax(metric))
            val = float(metric[idx])
            if val > best["metric"]:
                best = {
                    "metric": val,
                    "nid2": nid2,
                    "ifo_hz": float(fo),
                    "ifo_bins": int(round(fo / cfg.scs_hz)),
                    "timing": idx,
                    "corr_at_peak": corr[idx],
                    "fo_grid": fo_grid,
                }
    return best


# =============================================================================
# Symbol extraction / fine fractional CFO / channel estimation / SSS detection
# =============================================================================
def extract_symbols_from_rx(rx: np.ndarray, start0: int, cp: np.ndarray, nfft: int):
    syms = []
    ptr = start0
    for l in range(4):
        slen = cp[l] + nfft
        seg = rx[ptr:ptr + slen]
        if len(seg) < slen:
            seg = np.pad(seg, (0, slen - len(seg)))
        syms.append(seg)
        ptr += slen
    return syms


def get_useful_part_from_symbol(sym_with_cp: np.ndarray, cp_len: int, nfft: int):
    useful = sym_with_cp[cp_len:cp_len + nfft]
    if len(useful) < nfft:
        useful = np.pad(useful, (0, nfft - len(useful)))
    return useful


def estimate_fractional_cfo_from_pss_halves(
    rx_pss_sym_with_cp: np.ndarray,
    pss_ref_no_cp: np.ndarray,
    cp_len: int,
    nfft: int,
    fs: float,
):
    y = get_useful_part_from_symbol(rx_pss_sym_with_cp, cp_len, nfft)

    half = nfft // 2
    y0 = y[:half]
    y1 = y[half:half + half]
    d0 = pss_ref_no_cp[:half]
    d1 = pss_ref_no_cp[half:half + half]

    C0 = np.vdot(d0, y0)
    C1 = np.vdot(d1, y1)
    prod = C1 * np.conj(C0)

    if np.abs(C0) < 1e-15 or np.abs(C1) < 1e-15 or np.abs(prod) < 1e-15:
        return 0.0, {
            "C0": C0,
            "C1": C1,
            "phase_diff_rad": 0.0,
            "quality": 0.0,
        }

    phase_diff = np.angle(prod)
    ffo_hz_hat = phase_diff * fs / (np.pi * nfft)

    return float(ffo_hz_hat), {
        "C0": C0,
        "C1": C1,
        "phase_diff_rad": float(phase_diff),
        "quality": float(np.abs(prod)),
    }


def estimate_fractional_cfo_from_cp(rx_syms, cp: np.ndarray, nfft: int, use_symbol_idx=(0, 2)):
    acc = 0.0j
    used = 0
    for l in use_symbol_idx:
        cp_len = int(cp[l])
        if cp_len <= 0:
            continue
        sym = rx_syms[l]
        r_cp = sym[:cp_len]
        r_tail = sym[nfft:nfft + cp_len]
        if len(r_cp) == cp_len and len(r_tail) == cp_len:
            acc += np.vdot(r_cp, r_tail)
            used += 1

    if used == 0 or np.abs(acc) < 1e-15:
        return 0.0

    return float(np.angle(acc) / (2 * np.pi))


def estimate_delay_support_bins(H_ls: np.ndarray, guard_bins: int, wrap_bins: int) -> np.ndarray:
    nfft = len(H_ls)
    h_td = np.fft.ifft(H_ls, norm="ortho")
    power = np.abs(h_td) ** 2

    peak = int(np.argmax(power))
    total_power = np.sum(power) + 1e-15
    peak_power = power[peak]
    threshold = max(0.01 * peak_power, 0.001 * total_power)

    valid = np.where(power >= threshold)[0]
    if valid.size == 0:
        valid = np.array([peak], dtype=int)

    start = max(0, int(valid.min()) - int(guard_bins))
    end = min(nfft - 1, int(valid.max()) + int(guard_bins))

    keep = np.zeros(nfft, dtype=bool)
    keep[start:end + 1] = True

    if wrap_bins > 0:
        keep[:wrap_bins] = True
        keep[-wrap_bins:] = True

    return keep


def pss_channel_estimation(
    Y0_shifted: np.ndarray,
    nid2_hat: int,
    info: Dict,
    noise_var: float,
    guard_bins: int = 2,
    wrap_bins: int = 1,
):
    pss_seq = nr_pss(nid2_hat)
    Y_pss = Y0_shifted[info["pss_k"] + len(Y0_shifted) // 2]
    H_ls = Y_pss / (pss_seq + 1e-12)

    keep_mask = estimate_delay_support_bins(H_ls, guard_bins, wrap_bins)
    h_td = np.fft.ifft(H_ls, norm="ortho")
    h_td_denoised = h_td * keep_mask.astype(float)
    H_est = np.fft.fft(h_td_denoised, norm="ortho")

    W_mmse = np.conj(H_est) / (np.abs(H_est) ** 2 + noise_var + 1e-12)

    debug = {
        "Y_pss": Y_pss,
        "pss_seq": pss_seq,
        "H_ls": H_ls,
        "h_td_ls": h_td,
        "h_td_denoised": h_td_denoised,
        "delay_keep_mask": keep_mask,
        "W_mmse": W_mmse,
    }
    return H_est, debug


def detect_sss(
    Y2_shifted: np.ndarray,
    H_est: np.ndarray,
    nid2_hat: int,
    info: Dict,
    noise_var: float,
):
    """
    相比 v4 的优化点：
    - 不再 for nid1 in range(336) 循环逐条生成 SSS
    - 直接使用预生成 SSS bank 做矩阵乘法，一次性得到 336 路相关 metric
    """
    Y_sss = Y2_shifted[info["sss_k"] + len(Y2_shifted) // 2]
    W_mmse = np.conj(H_est) / (np.abs(H_est) ** 2 + noise_var + 1e-12)
    Xeq = Y_sss * W_mmse

    sss_bank = build_sss_bank()[nid2_hat]      # [336, 127]
    metrics = np.abs(sss_bank @ Xeq)           # 一次性得到 336 路 metric
    nid1_hat = int(np.argmax(metrics))
    return nid1_hat, metrics, Xeq

# =============================================================================
# Oracle RSRP (SSS-based) measurement
# =============================================================================
# =============================================================================
# Oracle RSRP (SSS-based, channel-estimation-assisted)
# =============================================================================
def linear_to_db(x: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(x, 1e-15))


def moving_average_same_complex(x: np.ndarray, win: int):
    """
    对复数序列做 same 模式移动平均。
    返回：
      x_smooth: 平滑后的复数序列
      denom: 每个位置实际参与平均的样本数
    """
    win = int(win)
    if win <= 1:
        return x.copy(), np.ones(len(x), dtype=float)

    kernel = np.ones(win, dtype=float)
    num = np.convolve(x, kernel, mode="same")
    den = np.convolve(np.ones(len(x), dtype=float), kernel, mode="same")
    return num / den, den


def extract_oracle_sss_re(
    rx: np.ndarray,
    start0: int,
    cp: np.ndarray,
    nfft: int,
    info: Dict,
    fs: float,
    cfo_hz: float = 0.0,
):
    """
    Oracle方式提取 SSS RE：
    - 已知真 timing start0
    - 已知 CFO，先做理想补偿
    - 然后直接提取 symbol#2 的 SSS RE
    """
    n = np.arange(len(rx))
    if abs(cfo_hz) > 0:
        rx = rx * np.exp(-1j * 2 * np.pi * cfo_hz * n / fs)

    rx_syms = extract_symbols_from_rx(rx, start0, cp, nfft)
    Y2 = ofdm_demodulate_symbol(rx_syms[2], nfft, cp[2])

    sss_re = Y2[info["sss_k"] + nfft // 2]
    return sss_re, Y2


def estimate_oracle_rsrp_from_sss_channel(
    sss_re: np.ndarray,
    sss_seq: np.ndarray,
    noise_var: float,
    smooth_win: int = 9,
):
    """
    基于 SSS 的信道估计辅助 RSRP 估计：
      1) H_ls = Y_sss / S_sss
      2) 对 H_ls 做频域平滑，降低噪声
      3) 对 |H_smooth|^2 做噪声校正后求平均

    说明：
    - 这里估的是接收端参考信号功率，对应 |H|^2（因为 |S_sss|=1）
    - 不是把信道完全均衡掉后再算符号功率
    """
    H_ls = sss_re / (sss_seq + 1e-12)

    H_smooth, denom = moving_average_same_complex(H_ls, smooth_win)

    # 平滑后每个bin的等效噪声方差近似降低为 noise_var / denom
    noise_var_after_smooth = noise_var / denom

    # 先按RE得到去噪后的功率估计，再整体平均
    p_re_est = np.abs(H_smooth) ** 2 - noise_var_after_smooth
    rsrp_est_lin_preclip = float(np.mean(p_re_est))
    rsrp_est_lin = max(rsrp_est_lin_preclip, 1e-15)

    dbg = {
        "H_ls": H_ls,
        "H_smooth": H_smooth,
        "denom": denom,
        "noise_var_after_smooth": noise_var_after_smooth,
        "p_re_est": p_re_est,
        "rsrp_est_lin_preclip": rsrp_est_lin_preclip,
    }
    return rsrp_est_lin, dbg
def run_single_trial_oracle_rsrp(
    cfg: NRSimConfig,
    pci: Optional[int] = None,
    seed: Optional[int] = None,
):
    """
    单次 trial：
    - 生成单个 SSB
    - 经过信道 + CFO + 噪声
    - 用 oracle timing + oracle CFO 提取 SSS RE
    - 用 SSS 做 LS 信道估计 + 频域平滑
    - 计算 true / estimated RSRP
    """
    if seed is not None:
        np.random.seed(seed)
    elif cfg.seed is not None:
        np.random.seed(cfg.seed)

    fs = cfg.nfft * cfg.scs_hz

    if pci is None:
        pci = int(np.random.randint(0, 1008)) if cfg.pci is None else int(cfg.pci)

    tx_ssb, grid, info, cp = build_ssb_waveform(pci, cfg.nfft)

    tx = np.concatenate([np.zeros(cfg.timing_offset_samp, dtype=complex), tx_ssb])

    # noiseless channel output
    y_ch, ch = apply_time_varying_channel(tx, cfg)

    # noisy rx
    ref_re_energy = get_reference_re_energy(grid)
    rx, cfo_true_hz, noise_var = apply_cfo_and_noise(
        y_ch, cfg, ref_re_energy=ref_re_energy
    )

    # oracle true SSS RE（无噪声、无CFO）
    sss_true_re, _ = extract_oracle_sss_re(
        rx=y_ch,
        start0=cfg.timing_offset_samp,
        cp=cp,
        nfft=cfg.nfft,
        info=info,
        fs=fs,
        cfo_hz=0.0,
    )

    # oracle estimated SSS RE（带噪声，但用真CFO理想补偿）
    sss_est_re, _ = extract_oracle_sss_re(
        rx=rx,
        start0=cfg.timing_offset_samp,
        cp=cp,
        nfft=cfg.nfft,
        info=info,
        fs=fs,
        cfo_hz=cfo_true_hz,
    )

    sss_seq = build_sss_bank()[info["nid2"], info["nid1"]]

    # true RSRP：无噪声条件下的真实接收功率
    H_true = sss_true_re / (sss_seq + 1e-12)
    rsrp_true_lin = float(np.mean(np.abs(H_true) ** 2))

    # estimated RSRP：SSS信道估计 + 平滑 + 去噪
    rsrp_est_lin, rsrp_dbg = estimate_oracle_rsrp_from_sss_channel(
        sss_re=sss_est_re,
        sss_seq=sss_seq,
        noise_var=noise_var,
        smooth_win=cfg.oracle_rsrp_smooth_win,
    )

    rsrp_true_db = float(linear_to_db(np.array([rsrp_true_lin]))[0])
    rsrp_est_db = float(linear_to_db(np.array([rsrp_est_lin]))[0])
    rsrp_err_db = rsrp_est_db - rsrp_true_db

    return {
        "cfg": cfg,
        "true": {
            "pci": pci,
            "nid1": pci // 3,
            "nid2": pci % 3,
            "cfo_hz": cfo_true_hz,
            "doppler_hz": ch.doppler,
            "timing_start": cfg.timing_offset_samp,
            "rsrp_lin": rsrp_true_lin,
            "rsrp_db": rsrp_true_db,
        },
        "est": {
            "rsrp_lin": rsrp_est_lin,
            "rsrp_db": rsrp_est_db,
        },
        "stats": {
            "rsrp_err_db": float(rsrp_err_db),
            "abs_rsrp_err_db": float(abs(rsrp_err_db)),
            "noise_var": float(noise_var),
            "rsrp_est_lin_preclip": float(rsrp_dbg["rsrp_est_lin_preclip"]),
            "preclip_nonpositive_flag": bool(rsrp_dbg["rsrp_est_lin_preclip"] <= 0),
        },
        "debug": {
            "cp": cp,
            "ref_re_energy": ref_re_energy,
            "sss_true_re": sss_true_re,
            "sss_est_re": sss_est_re,
            "sss_seq": sss_seq,
            "H_true": H_true,
            "H_ls": rsrp_dbg["H_ls"],
            "H_smooth": rsrp_dbg["H_smooth"],
            "smooth_denom": rsrp_dbg["denom"],
            "noise_var_after_smooth": rsrp_dbg["noise_var_after_smooth"],
            "p_re_est": rsrp_dbg["p_re_est"],
            "rx": rx,
            "y_ch": y_ch,
        },
    }

def run_oracle_rsrp_sim(
    cfg: NRSimConfig,
    snr_db: float,
    n_trials: int = 10000,
    fixed_pci: Optional[int] = 0,
    verbose: bool = True,
):
    """
    单个 SNR 下的 Oracle RSRP Monte Carlo
    """
    cfg_local = NRSimConfig(**vars(cfg))
    cfg_local.snr_db = float(snr_db)

    rng = np.random.default_rng(cfg.seed)

    rsrp_true_db_list = []
    rsrp_est_db_list = []
    rsrp_err_db_list = []
    abs_rsrp_err_db_list = []
    rsrp_est_lin_preclip_list = []
    preclip_nonpositive_count = 0

    for i in range(n_trials):
        trial_seed = int(rng.integers(0, 2**31 - 1))
        trial_pci = int(fixed_pci) if fixed_pci is not None else int(rng.integers(0, 1008))

        trial_res = run_single_trial_oracle_rsrp(
            cfg_local,
            pci=trial_pci,
            seed=trial_seed,
        )

        rsrp_true_db_list.append(trial_res["true"]["rsrp_db"])
        rsrp_est_db_list.append(trial_res["est"]["rsrp_db"])
        rsrp_err_db_list.append(trial_res["stats"]["rsrp_err_db"])
        abs_rsrp_err_db_list.append(trial_res["stats"]["abs_rsrp_err_db"])
        rsrp_est_lin_preclip_list.append(trial_res["stats"]["rsrp_est_lin_preclip"])

        if trial_res["stats"]["preclip_nonpositive_flag"]:
            preclip_nonpositive_count += 1

        if verbose and ((i + 1) % 1000 == 0 or (i + 1) == n_trials):
            print(f"SNR={snr_db:.1f} dB, finished {i+1}/{n_trials}")

    rsrp_true_db_arr = np.array(rsrp_true_db_list, dtype=float)
    rsrp_est_db_arr = np.array(rsrp_est_db_list, dtype=float)
    rsrp_err_db_arr = np.array(rsrp_err_db_list, dtype=float)
    abs_rsrp_err_db_arr = np.array(abs_rsrp_err_db_list, dtype=float)
    rsrp_est_lin_preclip_arr = np.array(rsrp_est_lin_preclip_list, dtype=float)

    q90_abs_err = float(np.percentile(abs_rsrp_err_db_arr, 90))
    pass_prob_3p5 = float(np.mean(abs_rsrp_err_db_arr <= 3.5))
    preclip_nonpositive_ratio = float(preclip_nonpositive_count / n_trials)

    return {
        "snr_db": float(snr_db),
        "n_trials": int(n_trials),
        "oracle_true_rsrp_db_samples": rsrp_true_db_arr,
        "oracle_est_rsrp_db_samples": rsrp_est_db_arr,
        "oracle_rsrp_error_db_samples": rsrp_err_db_arr,
        "oracle_abs_rsrp_error_db_samples": abs_rsrp_err_db_arr,
        "oracle_rsrp_est_lin_preclip_samples": rsrp_est_lin_preclip_arr,
        "q90_abs_error_db": q90_abs_err,
        "pass_prob_within_3p5_db": pass_prob_3p5,
        "preclip_nonpositive_ratio": preclip_nonpositive_ratio,
        "mean_true_rsrp_db": float(np.mean(rsrp_true_db_arr)),
        "mean_est_rsrp_db": float(np.mean(rsrp_est_db_arr)),
        "mean_error_db": float(np.mean(rsrp_err_db_arr)),
        "std_error_db": float(np.std(rsrp_err_db_arr, ddof=0)),
    }
# =============================================================================
# Proxy for multiple-SSB work point: Y-sample non-coherent combining
# Still evaluates one target SSB only
# =============================================================================
def run_single_trial_oracle_rsrp_multi_sample_proxy(
    cfg: NRSimConfig,
    pci: Optional[int] = None,
    seed: Optional[int] = None,
    num_samples: int = 4,
):
    """
    多个SSB工作点的 proxy 版本：
    - 仍然只评估一个目标 SSB
    - 同一 trial 内生成 num_samples 次 measurement samples
    - 每次 sample:
        * 同一 PCI / 目标SSB
        * 同一大尺度条件（同一 cfg）
        * 不同噪声
        * 不同小尺度快衰落 realization（通过不同 seed 近似）
    - 对每次 sample 得到的线性域 RSRP 做平均（非相干合并）
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    # 用一个 base seed 生成本 trial 下各个 sample 的子种子
    if seed is not None:
        base_seed = int(seed)
    elif cfg.seed is not None:
        base_seed = int(cfg.seed)
    else:
        base_seed = 1

    rng = np.random.default_rng(base_seed)

    if pci is None:
        pci = int(np.random.randint(0, 1008)) if cfg.pci is None else int(cfg.pci)

    true_lin_list = []
    est_lin_list = []
    true_db_list = []
    est_db_list = []
    err_db_list = []
    preclip_nonpositive_flags = []

    sample_seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(num_samples)]

    for sample_seed in sample_seeds:
        one = run_single_trial_oracle_rsrp(
            cfg=cfg,
            pci=pci,
            seed=sample_seed,
        )

        true_lin_list.append(one["true"]["rsrp_lin"])
        est_lin_list.append(one["est"]["rsrp_lin"])
        true_db_list.append(one["true"]["rsrp_db"])
        est_db_list.append(one["est"]["rsrp_db"])
        err_db_list.append(one["stats"]["rsrp_err_db"])
        preclip_nonpositive_flags.append(one["stats"]["preclip_nonpositive_flag"])

    true_lin_arr = np.array(true_lin_list, dtype=float)
    est_lin_arr = np.array(est_lin_list, dtype=float)
    true_db_arr = np.array(true_db_list, dtype=float)
    est_db_arr = np.array(est_db_list, dtype=float)
    err_db_arr = np.array(err_db_list, dtype=float)
    preclip_nonpositive_flags = np.array(preclip_nonpositive_flags, dtype=bool)

    # 非相干合并：线性域功率平均
    rsrp_true_comb_lin = float(np.mean(true_lin_arr))
    rsrp_est_comb_lin = float(np.mean(est_lin_arr))

    rsrp_true_comb_db = float(linear_to_db(np.array([rsrp_true_comb_lin]))[0])
    rsrp_est_comb_db = float(linear_to_db(np.array([rsrp_est_comb_lin]))[0])
    rsrp_comb_err_db = rsrp_est_comb_db - rsrp_true_comb_db

    return {
        "cfg": cfg,
        "num_samples": int(num_samples),
        "true": {
            "pci": int(pci),
            "rsrp_lin_per_sample": true_lin_arr,
            "rsrp_db_per_sample": true_db_arr,
            "rsrp_comb_lin": rsrp_true_comb_lin,
            "rsrp_comb_db": rsrp_true_comb_db,
        },
        "est": {
            "rsrp_lin_per_sample": est_lin_arr,
            "rsrp_db_per_sample": est_db_arr,
            "rsrp_comb_lin": rsrp_est_comb_lin,
            "rsrp_comb_db": rsrp_est_comb_db,
        },
        "stats": {
            "rsrp_err_db_per_sample": err_db_arr,
            "rsrp_comb_err_db": float(rsrp_comb_err_db),
            "abs_rsrp_comb_err_db": float(abs(rsrp_comb_err_db)),
            "sample_preclip_nonpositive_flags": preclip_nonpositive_flags,
            "sample_preclip_nonpositive_ratio": float(np.mean(preclip_nonpositive_flags)),
            "trial_preclip_any_flag": bool(np.any(preclip_nonpositive_flags)),
        },
        "debug": {
            "sample_seeds": sample_seeds,
        },
    }


def run_oracle_rsrp_multi_sample_proxy_sim(
    cfg: NRSimConfig,
    snr_db: float,
    n_trials: int = 10000,
    fixed_pci: Optional[int] = 0,
    num_samples: int = 4,
    verbose: bool = True,
):
    """
    multiple-SSB work point 的 proxy 仿真：
    - 仍然只测一个目标 SSB
    - 每个 trial 内做 num_samples 次 measurement samples
    - 对线性域 RSRP 做非相干平均
    - 输出合并后的 CDF
    """
    cfg_local = NRSimConfig(**vars(cfg))
    cfg_local.snr_db = float(snr_db)

    rng = np.random.default_rng(cfg.seed)

    comb_true_db_list = []
    comb_est_db_list = []
    comb_err_db_list = []
    comb_abs_err_db_list = []

    sample_preclip_ratio_list = []
    trial_preclip_any_count = 0

    for i in range(n_trials):
        trial_seed = int(rng.integers(0, 2**31 - 1))
        trial_pci = int(fixed_pci) if fixed_pci is not None else int(rng.integers(0, 1008))

        trial_res = run_single_trial_oracle_rsrp_multi_sample_proxy(
            cfg=cfg_local,
            pci=trial_pci,
            seed=trial_seed,
            num_samples=num_samples,
        )

        comb_true_db_list.append(trial_res["true"]["rsrp_comb_db"])
        comb_est_db_list.append(trial_res["est"]["rsrp_comb_db"])
        comb_err_db_list.append(trial_res["stats"]["rsrp_comb_err_db"])
        comb_abs_err_db_list.append(trial_res["stats"]["abs_rsrp_comb_err_db"])
        sample_preclip_ratio_list.append(trial_res["stats"]["sample_preclip_nonpositive_ratio"])

        if trial_res["stats"]["trial_preclip_any_flag"]:
            trial_preclip_any_count += 1

        if verbose and ((i + 1) % 1000 == 0 or (i + 1) == n_trials):
            print(f"SNR={snr_db:.1f} dB, finished {i+1}/{n_trials}")

    comb_true_db_arr = np.array(comb_true_db_list, dtype=float)
    comb_est_db_arr = np.array(comb_est_db_list, dtype=float)
    comb_err_db_arr = np.array(comb_err_db_list, dtype=float)
    comb_abs_err_db_arr = np.array(comb_abs_err_db_list, dtype=float)
    sample_preclip_ratio_arr = np.array(sample_preclip_ratio_list, dtype=float)

    q90_abs_err = float(np.percentile(comb_abs_err_db_arr, 90))
    pass_prob_3p5 = float(np.mean(comb_abs_err_db_arr <= 3.5))

    return {
        "snr_db": float(snr_db),
        "n_trials": int(n_trials),
        "num_samples": int(num_samples),
        "oracle_true_rsrp_db_samples": comb_true_db_arr,
        "oracle_est_rsrp_db_samples": comb_est_db_arr,
        "oracle_rsrp_error_db_samples": comb_err_db_arr,
        "oracle_abs_rsrp_error_db_samples": comb_abs_err_db_arr,
        "q90_abs_error_db": q90_abs_err,
        "pass_prob_within_3p5_db": pass_prob_3p5,
        "mean_true_rsrp_db": float(np.mean(comb_true_db_arr)),
        "mean_est_rsrp_db": float(np.mean(comb_est_db_arr)),
        "mean_error_db": float(np.mean(comb_err_db_arr)),
        "std_error_db": float(np.std(comb_err_db_arr, ddof=0)),
        "mean_sample_preclip_nonpositive_ratio": float(np.mean(sample_preclip_ratio_arr)),
        "trial_preclip_any_ratio": float(trial_preclip_any_count / n_trials),
    }
def plot_oracle_rsrp_multi_sample_proxy_cdf(
    res,
    save_dir="figures",
    prefix="oracle_multi_sample_proxy"
):
    os.makedirs(save_dir, exist_ok=True)

    est_rsrp = res["oracle_est_rsrp_db_samples"]
    abs_err = res["oracle_abs_rsrp_error_db_samples"]

    x1, y1 = ecdf(est_rsrp)
    x2, y2 = ecdf(abs_err)

    # 图1：合并后估计RSRP的CDF
    plt.figure(figsize=(7, 5))
    plt.plot(x1, y1)
    plt.grid(True)
    plt.xlabel("Combined Oracle estimated SSS-RSRP (dB)")
    plt.ylabel("CDF")
    plt.title(
        f"Proxy multiple-SSB Oracle RSRP CDF @ SNR = {res['snr_db']:.1f} dB, "
        f"Y={res['num_samples']}"
    )

    fig1_path = os.path.join(
        save_dir,
        f"{prefix}_rsrp_cdf_snr_{res['snr_db']:.1f}dB_Y{res['num_samples']}.png"
    )
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 图2：合并后|误差|的CDF
    plt.figure(figsize=(7, 5))
    plt.plot(x2, y2, label="|Combined RSRP error| CDF")
    plt.axvline(3.5, linestyle="--", label="3.5 dB")
    plt.axhline(0.9, linestyle="--", label="90%")
    plt.grid(True)
    plt.xlabel("|Combined RSRP error| (dB)")
    plt.ylabel("CDF")
    plt.title(
        f"Proxy multiple-SSB RSRP Accuracy CDF @ SNR = {res['snr_db']:.1f} dB, "
        f"Y={res['num_samples']}"
    )
    plt.legend()

    fig2_path = os.path.join(
        save_dir,
        f"{prefix}_accuracy_cdf_snr_{res['snr_db']:.1f}dB_Y{res['num_samples']}.png"
    )
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {fig1_path}")
    print(f"Saved figure: {fig2_path}")
    print(f"mean_sample_preclip_nonpositive_ratio = {res['mean_sample_preclip_nonpositive_ratio']:.4f}")
    print(f"trial_preclip_any_ratio = {res['trial_preclip_any_ratio']:.4f}")
def ecdf(x: np.ndarray):
    x = np.sort(np.asarray(x))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

import os

def plot_oracle_rsrp_cdf(res, save_dir="figures", prefix="oracle_single_ssb"):
    os.makedirs(save_dir, exist_ok=True)

    est_rsrp = res["oracle_est_rsrp_db_samples"]
    abs_err = res["oracle_abs_rsrp_error_db_samples"]

    x1, y1 = ecdf(est_rsrp)
    x2, y2 = ecdf(abs_err)

    # 图1：估计RSRP的CDF
    plt.figure(figsize=(7, 5))
    plt.plot(x1, y1)
    plt.grid(True)
    plt.xlabel("Oracle estimated SSS-RSRP (dB)")
    plt.ylabel("CDF")
    plt.title(f"Single SSB Oracle RSRP CDF @ SNR = {res['snr_db']:.1f} dB")

    fig1_path = os.path.join(
        save_dir,
        f"{prefix}_rsrp_cdf_snr_{res['snr_db']:.1f}dB.png"
    )
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 图2：|误差|的CDF
    plt.figure(figsize=(7, 5))
    plt.plot(x2, y2, label="|RSRP error| CDF")
    plt.axvline(3.5, linestyle="--", label="3.5 dB")
    plt.axhline(0.9, linestyle="--", label="90%")
    plt.grid(True)
    plt.xlabel("|RSRP error| (dB)")
    plt.ylabel("CDF")
    plt.title(f"Single SSB Oracle RSRP Accuracy CDF @ SNR = {res['snr_db']:.1f} dB")
    plt.legend()

    fig2_path = os.path.join(
        save_dir,
        f"{prefix}_accuracy_cdf_snr_{res['snr_db']:.1f}dB.png"
    )
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {fig1_path}")
    print(f"Saved figure: {fig2_path}")

    if "preclip_nonpositive_ratio" in res:
        print(f"preclip_nonpositive_ratio = {res['preclip_nonpositive_ratio']:.4f}")

# =============================================================================
# Single-trial simulation
# =============================================================================
def run_single_trial(cfg: NRSimConfig, pci: Optional[int] = None, seed: Optional[int] = None):
    if seed is not None:
        np.random.seed(seed)
    elif cfg.seed is not None:
        np.random.seed(cfg.seed)

    fs = cfg.nfft * cfg.scs_hz

    if pci is None:
        pci = int(np.random.randint(0, 1008)) if cfg.pci is None else int(cfg.pci)

    tx_ssb, grid, info, cp = build_ssb_waveform(pci, cfg.nfft)
    tx = np.concatenate([np.zeros(cfg.timing_offset_samp, dtype=complex), tx_ssb])

    y_ch, ch = apply_time_varying_channel(tx, cfg)

    ref_re_energy = get_reference_re_energy(grid)
    rx, cfo_true_hz, noise_var = apply_cfo_and_noise(
        y_ch, cfg, ref_re_energy=ref_re_energy
    )

    pss_est = joint_pss_search(rx=rx, nfft=cfg.nfft, cfg=cfg, fs=fs)

    n = np.arange(len(rx))
    rx_ifo_comp = rx * np.exp(-1j * 2 * np.pi * pss_est["ifo_hz"] * n / fs)
    rx_syms_ifo = extract_symbols_from_rx(rx_ifo_comp, pss_est["timing"], cp, cfg.nfft)

    pss_ref_no_cp = build_pss_reference_useful_symbols(cfg.nfft)[pss_est["nid2"]]
    ffo_hz_hat, ffo_pss_dbg = estimate_fractional_cfo_from_pss_halves(
        rx_pss_sym_with_cp=rx_syms_ifo[0],
        pss_ref_no_cp=pss_ref_no_cp,
        cp_len=cp[0],
        nfft=cfg.nfft,
        fs=fs,
    )

    ffo_cp_eps_hat = estimate_fractional_cfo_from_cp(rx_syms_ifo, cp, cfg.nfft, use_symbol_idx=(0, 2))
    ffo_cp_hz_hat = ffo_cp_eps_hat * cfg.scs_hz

    cfo_est_hz = pss_est["ifo_hz"] + ffo_hz_hat

    rx_cfo_comp = rx * np.exp(-1j * 2 * np.pi * cfo_est_hz * n / fs)
    rx_syms = extract_symbols_from_rx(rx_cfo_comp, pss_est["timing"], cp, cfg.nfft)

    Y0 = ofdm_demodulate_symbol(rx_syms[0], cfg.nfft, cp[0])
    Y2 = ofdm_demodulate_symbol(rx_syms[2], cfg.nfft, cp[2])

    H_est, ce_dbg = pss_channel_estimation(
        Y0_shifted=Y0,
        nid2_hat=pss_est["nid2"],
        info=info,
        noise_var=noise_var,
        guard_bins=cfg.ce_delay_guard_bins,
        wrap_bins=cfg.ce_wrap_bins,
    )

    nid1_hat, sss_metrics, Xeq = detect_sss(
        Y2_shifted=Y2,
        H_est=H_est,
        nid2_hat=pss_est["nid2"],
        info=info,
        noise_var=noise_var,
    )
    pci_hat = 3 * nid1_hat + pss_est["nid2"]

    residual_cfo_hz = cfo_true_hz - cfo_est_hz
    timing_error_samp = pss_est["timing"] - cfg.timing_offset_samp

    return {
        "cfg": cfg,
        "true": {
            "pci": pci,
            "nid1": pci // 3,
            "nid2": pci % 3,
            "cfo_hz": cfo_true_hz,
            "doppler_hz": ch.doppler,
            "timing_start": cfg.timing_offset_samp,
        },
        "est": {
            "timing_start": pss_est["timing"],
            "nid2": pss_est["nid2"],
            "ifo_hz": pss_est["ifo_hz"],
            "ffo_hz": ffo_hz_hat,
            "fo_hz": cfo_est_hz,
            "nid1": nid1_hat,
            "pci": pci_hat,
            "pss_metric": pss_est["metric"],
        },
        "stats": {
            "residual_cfo_hz": residual_cfo_hz,
            "timing_error_samp": timing_error_samp,
            "pci_error": int(pci_hat != pci),
        },
        "debug": {
            "cp": cp,
            "channel_ds_tap": ch.DS_tap,
            "tx_ssb": tx_ssb,
            "tx": tx,
            "rx": rx,
            "rx_ifo_comp": rx_ifo_comp,
            "rx_cfo_comp": rx_cfo_comp,
            "Y0": Y0,
            "Y2": Y2,
            "H_est_pss": H_est,
            "H_ls_pss": ce_dbg["H_ls"],
            "h_td_ls_pss": ce_dbg["h_td_ls"],
            "h_td_denoised_pss": ce_dbg["h_td_denoised"],
            "delay_keep_mask_pss": ce_dbg["delay_keep_mask"],
            "Xeq_sss": Xeq,
            "sss_metrics": sss_metrics,
            "noise_var": noise_var,
            "ifo_grid": pss_est["fo_grid"],
            "ffo_pss_phase_diff_rad": ffo_pss_dbg["phase_diff_rad"],
            "ffo_pss_quality": ffo_pss_dbg["quality"],
            "ffo_cp_hz_hat": ffo_cp_hz_hat,
            "ffo_pss_C0": ffo_pss_dbg["C0"],
            "ffo_pss_C1": ffo_pss_dbg["C1"],
            "ref_re_energy": ref_re_energy,
        },
    }


# =============================================================================
# Monte Carlo over multiple SNRs
# =============================================================================
def _run_one_snr_worker(args):
    """
    单个 SNR 的 Monte Carlo 仿真 worker。
    保持和 v4 一致：在某个 SNR 下，一旦 PCI 错误次数 > max_pci_errors_per_snr，就提前终止。
    """
    cfg_dict, snr_db, trial_seeds, trial_pcis, max_pci_errors_per_snr = args
    cfg = NRSimConfig(**cfg_dict)
    cfg.snr_db = float(snr_db)

    residual_cfo_list = []
    timing_error_list = []
    pci_error_count = 0
    n_run = 0

    for trial_seed, trial_pci in zip(trial_seeds, trial_pcis):
        cfg.seed = int(trial_seed)
        cfg.pci = int(trial_pci)

        try:
            trial_res = run_single_trial(cfg, pci=trial_pci, seed=trial_seed)
            residual_cfo_list.append(trial_res["stats"]["residual_cfo_hz"])
            timing_error_list.append(trial_res["stats"]["timing_error_samp"])
            if trial_res["stats"]["pci_error"]:
                pci_error_count += 1
        except Exception:
            pci_error_count += 1

        n_run += 1

        if pci_error_count > max_pci_errors_per_snr:
            break

    return {
        "snr_db": float(snr_db),
        "n_run": int(n_run),
        "pci_error_count": int(pci_error_count),
        "residual_cfo": np.array(residual_cfo_list, dtype=float),
        "timing_error": np.array(timing_error_list, dtype=float),
        "early_stop": bool(pci_error_count > max_pci_errors_per_snr),
    }


def run_sim(
    cfg: NRSimConfig,
    snr_db_list,
    n_trials: int = 1,
    max_pci_errors_per_snr: int = 100,
    verbose: bool = True,
    parallel: bool = False,
    num_workers: Optional[int] = None,
):
    """
    优化点：
    - 串行模式：逻辑与 v4 等价
    - 并行模式：按 SNR 维度多进程并行
    - 为了保证串行/并行的随机输入完全一致，先在主进程预生成每个 SNR 下所有 trial 的 seed / PCI 计划
    """
    snr_db_arr = np.atleast_1d(np.array(snr_db_list, dtype=float))
    assert n_trials > 0

    master_rng = np.random.default_rng(cfg.seed)

    # 先在主进程生成所有随机计划，确保串行/并行的一致性
    trial_seed_plan = []
    trial_pci_plan = []
    for _ in snr_db_arr:
        trial_seed_plan.append([int(master_rng.integers(0, 2**31 - 1)) for _ in range(n_trials)])
        trial_pci_plan.append([int(master_rng.integers(0, 1008)) for _ in range(n_trials)])

    results_by_snr = []

    if parallel and len(snr_db_arr) > 1:
        if num_workers is None:
            num_workers = min(len(snr_db_arr), os.cpu_count() or 1)

        worker_args = [
            (vars(cfg).copy(), snr_db_arr[i], trial_seed_plan[i], trial_pci_plan[i], max_pci_errors_per_snr)
            for i in range(len(snr_db_arr))
        ]

        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            results_by_snr = list(ex.map(_run_one_snr_worker, worker_args))
    else:
        for i, snr_db in enumerate(snr_db_arr):
            snr_res = _run_one_snr_worker(
                (vars(cfg).copy(), snr_db, trial_seed_plan[i], trial_pci_plan[i], max_pci_errors_per_snr)
            )
            results_by_snr.append(snr_res)

            if verbose:
                print(
                    f"SNR = {snr_db:6.2f} dB, "
                    f"executed {snr_res['n_run']:5d}/{n_trials}, "
                    f"PCI errors = {snr_res['pci_error_count']:4d}"
                )

    n_run_per_snr = np.array([r["n_run"] for r in results_by_snr], dtype=int)
    pci_error_count_arr = np.array([r["pci_error_count"] for r in results_by_snr], dtype=int)
    early_stop_flag = np.array([r["early_stop"] for r in results_by_snr], dtype=bool)

    residual_cfo_mean = np.full(len(snr_db_arr), np.nan, dtype=float)
    residual_cfo_std = np.full(len(snr_db_arr), np.nan, dtype=float)
    timing_error_mean = np.full(len(snr_db_arr), np.nan, dtype=float)
    timing_error_std = np.full(len(snr_db_arr), np.nan, dtype=float)

    per_snr_residual_cfo = []
    per_snr_timing_error = []

    for i, r in enumerate(results_by_snr):
        per_snr_residual_cfo.append(r["residual_cfo"])
        per_snr_timing_error.append(r["timing_error"])

        if r["residual_cfo"].size > 0:
            residual_cfo_mean[i] = np.mean(r["residual_cfo"])
            residual_cfo_std[i] = np.std(r["residual_cfo"], ddof=0)

        if r["timing_error"].size > 0:
            timing_error_mean[i] = np.mean(r["timing_error"])
            timing_error_std[i] = np.std(r["timing_error"], ddof=0)

    pci_error_prob = pci_error_count_arr / np.maximum(n_run_per_snr, 1)

    return {
        "snr_db_list": snr_db_arr,
        "n_trials_requested": int(n_trials),
        "n_run_per_snr": n_run_per_snr,
        "pci_error_count": pci_error_count_arr,
        "pci_error_prob": pci_error_prob,
        "early_stop_flag": early_stop_flag,
        "residual_cfo_mean_hz": residual_cfo_mean,
        "residual_cfo_std_hz": residual_cfo_std,
        "timing_error_mean_samp": timing_error_mean,
        "timing_error_std_samp": timing_error_std,
        "per_snr_residual_cfo_hz": per_snr_residual_cfo,
        "per_snr_timing_error_samp": per_snr_timing_error,
        "parallel": bool(parallel),
        "num_workers": int(num_workers) if (parallel and num_workers is not None) else None,
    }


def pretty_print_result(res):
    t = res["true"]
    e = res["est"]
    d = res["debug"]
    s = res["stats"]
    print("=== 5G NR PSS/SSS link simulation ===")
    print(f"Nfft = {res['cfg'].nfft}, Fs = {res['cfg'].nfft * res['cfg'].scs_hz / 1e6:.2f} MHz")
    print(f"CP lengths = {d['cp']}")
    print(f"TDL-C DS_tap = {d['channel_ds_tap']}")
    print(f"Reference RE avg energy = {d['ref_re_energy']:.6f}")
    print(f"True PCI={t['pci']} (NID1={t['nid1']}, NID2={t['nid2']})")
    print(f"Estimated PCI={e['pci']} (NID1={e['nid1']}, NID2={e['nid2']})")
    print(f"True CFO        = {t['cfo_hz']:.2f} Hz")
    print(f"Estimated IFO   = {e['ifo_hz']:.2f} Hz")
    print(f"Estimated FFO   = {e['ffo_hz']:.2f} Hz")
    print(f"Estimated CFO   = {e['fo_hz']:.2f} Hz")
    print(f"Residual CFO    = {s['residual_cfo_hz']:.2f} Hz")
    print(f"Max Doppler     = {t['doppler_hz']:.2f} Hz")
    print(f"True timing     = {t['timing_start']} samples")
    print(f"PSS timing      = {e['timing_start']} samples")
    print(f"Timing error    = {s['timing_error_samp']} samples")
    print(f"PSS metric      = {e['pss_metric']:.4f}")
    print(f"IFO grid        = {d['ifo_grid']}")


# =============================================================================
# Plotting
# =============================================================================
def plot_results(res):
    snr = res["snr_db_list"]

    plt.figure(figsize=(7, 5))
    y = np.maximum(res["pci_error_prob"], 1e-4)
    plt.semilogy(snr, y, marker='o')
    plt.grid(True, which='both')
    plt.xlabel("SNR (dB)")
    plt.ylabel("PCI error probability")
    plt.title("PCI error probability vs. SNR (log scale)")

    plt.figure(figsize=(7, 5))
    plt.errorbar(
        snr,
        res["residual_cfo_mean_hz"],
        yerr=res["residual_cfo_std_hz"],
        fmt='o-',
        capsize=4,
    )
    plt.grid(True)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Residual CFO (Hz)")
    plt.title("Residual CFO mean ± std vs. SNR")

    plt.figure(figsize=(7, 5))
    plt.errorbar(
        snr,
        res["timing_error_mean_samp"],
        yerr=res["timing_error_std_samp"],
        fmt='o-',
        capsize=4,
    )
    plt.grid(True)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Timing error (samples)")
    plt.title("Timing error mean ± std vs. SNR")

    plt.show()
if __name__ == "__main__":
    cfg = NRSimConfig(
        pci=0,
        nfft=256,
        chan_type="TDLC",
        delay_spread_ns=300.0,
        speed_kmh=120.0,
        system_cfo_ppm=5.0,
        timing_offset_samp=80,
        fo_num_candidates=3,
        oracle_rsrp_smooth_win=9,
        seed=1,
        snr_db=-14,
    )
    res_oracle = run_oracle_rsrp_sim(
        cfg,
        snr_db=-6.5,
        n_trials=10000,
        fixed_pci=0,
        verbose=True,
    )

    print("\n=== Oracle single-SSB RSRP result @ SNR = -6.5 dB ===")
    print("Trials:", res_oracle["n_trials"])
    print("Mean true RSRP (dB):", res_oracle["mean_true_rsrp_db"])
    print("Mean est  RSRP (dB):", res_oracle["mean_est_rsrp_db"])
    print("Mean error (dB):", res_oracle["mean_error_db"])
    print("Std  error (dB):", res_oracle["std_error_db"])
    print("Q90(|error|) (dB):", res_oracle["q90_abs_error_db"])
    print("P(|error| <= 3.5 dB):", res_oracle["pass_prob_within_3p5_db"])
    print("preclip_nonpositive_ratio:", res_oracle["preclip_nonpositive_ratio"])

    plot_oracle_rsrp_cdf(res_oracle)

    # multiple SSB work point 的 proxy 版本：
    # 仍然只测一个目标SSB，但每个trial做4次measurement samples，然后线性域非相干合并
    res_proxy = run_oracle_rsrp_multi_sample_proxy_sim(
        cfg,
        snr_db=-16.5,
        n_trials=10000,
        fixed_pci=0,
        num_samples=4,
        verbose=True,
    )

    print("\n=== Proxy multiple-SSB Oracle RSRP result @ SNR = -16.5 dB, Y=4 ===")
    print("Trials:", res_proxy["n_trials"])
    print("Samples per trial:", res_proxy["num_samples"])
    print("Mean true combined RSRP (dB):", res_proxy["mean_true_rsrp_db"])
    print("Mean est  combined RSRP (dB):", res_proxy["mean_est_rsrp_db"])
    print("Mean combined error (dB):", res_proxy["mean_error_db"])
    print("Std  combined error (dB):", res_proxy["std_error_db"])
    print("Q90(|combined error|) (dB):", res_proxy["q90_abs_error_db"])
    print("P(|combined error| <= 3.5 dB):", res_proxy["pass_prob_within_3p5_db"])
    print("mean_sample_preclip_nonpositive_ratio:", res_proxy["mean_sample_preclip_nonpositive_ratio"])
    print("trial_preclip_any_ratio:", res_proxy["trial_preclip_any_ratio"])

    plot_oracle_rsrp_multi_sample_proxy_cdf(res_proxy)

    # multiple SSB work point 的 proxy 版本：
    # 仍然只测一个目标SSB，但每个trial做4次measurement samples，然后线性域非相干合并
    res_proxy2 = run_oracle_rsrp_multi_sample_proxy_sim(
        cfg,
        snr_db=-16.5,
        n_trials=10000,
        fixed_pci=0,
        num_samples=3,
        verbose=True,
    )

    print("\n=== Proxy multiple-SSB Oracle RSRP result @ SNR = -16.5 dB, Y=4 ===")
    print("Trials:", res_proxy2["n_trials"])
    print("Samples per trial:", res_proxy2["num_samples"])
    print("Mean true combined RSRP (dB):", res_proxy2["mean_true_rsrp_db"])
    print("Mean est  combined RSRP (dB):", res_proxy2["mean_est_rsrp_db"])
    print("Mean combined error (dB):", res_proxy2["mean_error_db"])
    print("Std  combined error (dB):", res_proxy2["std_error_db"])
    print("Q90(|combined error|) (dB):", res_proxy2["q90_abs_error_db"])
    print("P(|combined error| <= 3.5 dB):", res_proxy2["pass_prob_within_3p5_db"])
    print("mean_sample_preclip_nonpositive_ratio:", res_proxy2["mean_sample_preclip_nonpositive_ratio"])
    print("trial_preclip_any_ratio:", res_proxy2["trial_preclip_any_ratio"])

    plot_oracle_rsrp_multi_sample_proxy_cdf(res_proxy2)

    # multiple SSB work point 的 proxy 版本：
    # 仍然只测一个目标SSB，但每个trial做4次measurement samples，然后线性域非相干合并
    res_proxy3 = run_oracle_rsrp_multi_sample_proxy_sim(
        cfg,
        snr_db=-16.5,
        n_trials=10000,
        fixed_pci=0,
        num_samples=8,
        verbose=True,
    )

    print("\n=== Proxy multiple-SSB Oracle RSRP result @ SNR = -16.5 dB, Y=8 ===")
    print("Trials:", res_proxy3["n_trials"])
    print("Samples per trial:", res_proxy3["num_samples"])
    print("Mean true combined RSRP (dB):", res_proxy3["mean_true_rsrp_db"])
    print("Mean est  combined RSRP (dB):", res_proxy3["mean_est_rsrp_db"])
    print("Mean combined error (dB):", res_proxy3["mean_error_db"])
    print("Std  combined error (dB):", res_proxy3["std_error_db"])
    print("Q90(|combined error|) (dB):", res_proxy3["q90_abs_error_db"])
    print("P(|combined error| <= 3.5 dB):", res_proxy3["pass_prob_within_3p5_db"])
    print("mean_sample_preclip_nonpositive_ratio:", res_proxy3["mean_sample_preclip_nonpositive_ratio"])
    print("trial_preclip_any_ratio:", res_proxy3["trial_preclip_any_ratio"])

    plot_oracle_rsrp_multi_sample_proxy_cdf(res_proxy3)