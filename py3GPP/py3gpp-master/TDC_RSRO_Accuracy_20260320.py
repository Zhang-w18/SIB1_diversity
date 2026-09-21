import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
from numba import njit
# 5G PSS/SSS 检测，改成了 Tx SNR, 不显示进度条

# =============================================================================
# User-provided channel model (fixed for DS unit consistency)
# =============================================================================
@njit(cache=True)
def _gen_jakes_kernel(
    n_ant_total: int,
    ds_tap: int,
    num_taps: int,
    taps: np.ndarray,          # int64
    taps_power: np.ndarray,    # float64
    wd_t: np.ndarray,          # float64
    M: int,
    seed: int,
):
    if seed >= 0:
        np.random.seed(seed)

    n_time = wd_t.size
    hf_samples = np.zeros((n_ant_total, ds_tap, n_time), dtype=np.complex128)

    scale_alpha = np.pi / (2.0 * M)

    for iant in range(n_ant_total):
        for itap in range(num_taps):
            tap_idx = taps[itap]
            tap_gain = taps_power[itap]

            alpha = np.empty(M, dtype=np.float64)
            cos_alpha = np.empty(M, dtype=np.float64)
            sin_alpha = np.empty(M, dtype=np.float64)
            phi = np.empty(M, dtype=np.float64)
            psi = np.empty(M, dtype=np.float64)

            for m in range(M):
                alpha[m] = (m + np.random.rand()) * scale_alpha
                cos_alpha[m] = np.cos(alpha[m])
                sin_alpha[m] = np.sin(alpha[m])
                phi[m] = (2.0 * np.random.rand() - 1.0) * np.pi
                psi[m] = (2.0 * np.random.rand() - 1.0) * np.pi

            for it in range(n_time):
                w = wd_t[it]
                acc_real = 0.0
                acc_imag = 0.0

                for m in range(M):
                    acc_real += np.cos(cos_alpha[m] * w + phi[m])
                    acc_imag += np.cos(sin_alpha[m] * w + psi[m])

                hf_samples[iant, tap_idx, it] += tap_gain * (acc_real + 1j * acc_imag)

    hf_samples *= np.sqrt(1.0 / M)
    return hf_samples

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

    def gen_Jakes_Accelerate(self, time_samples, seed: int = -1):
        M = 8
        wd_t = 2 * np.pi * self.doppler * time_samples

        hf_samples = _gen_jakes_kernel(
            n_ant_total=self.nTx * self.nRx,
            ds_tap=self.DS_tap,
            num_taps=self.num_taps,
            taps=self.taps.astype(np.int64),
            taps_power=self.taps_power.astype(np.float64),
            wd_t=wd_t.astype(np.float64),
            M=M,
            seed=seed,
        )

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

@dataclass(frozen=True)
class SSBPatternConfig:
    name: str
    num_symbols_per_block: int
    sss_symbol_indices: Tuple[int, ...]
    num_repeats: int
# =============================================================================
# Numerology / OFDM
# =============================================================================
@lru_cache(maxsize=None)
def normal_cp_lengths(nfft: int) -> np.ndarray:
    cp_others = int(round(144 * nfft / 2048))
    cp_first = nfft - 13 * cp_others
    return np.array([cp_first] + [cp_others] * 13, dtype=int)

@lru_cache(maxsize=None)
def cp_lengths_for_nsymbols(nfft: int, num_symbols: int) -> np.ndarray:
    cp14 = normal_cp_lengths(nfft)
    reps = int(np.ceil(num_symbols / 14))
    return np.tile(cp14, reps)[:num_symbols].astype(int)

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


def build_pattern_grid(
    pci: int,
    nfft: int,
    pattern: SSBPatternConfig,
) -> Tuple[np.ndarray, Dict]:
    nid2 = pci % 3
    nid1 = pci // 3

    total_symbols = pattern.num_symbols_per_block * pattern.num_repeats
    grid = np.zeros((total_symbols, nfft), dtype=complex)

    k_ssb = centered_bin_range(240)
    sync_local = np.arange(56, 183)
    sss_k = k_ssb[sync_local]
    sss_seq = build_sss_bank()[nid2, nid1]

    sss_symbol_positions = []
    for r in range(pattern.num_repeats):
        base = r * pattern.num_symbols_per_block
        for sym_idx in pattern.sss_symbol_indices:
            pos = base + sym_idx
            if pos >= total_symbols:
                raise ValueError(f"SSS symbol index out of range: {pos}")
            grid[pos, sss_k + nfft // 2] = sss_seq
            sss_symbol_positions.append(pos)

    info = {
        "nid1": nid1,
        "nid2": nid2,
        "k_ssb": k_ssb,
        "sss_k": sss_k,
        "sss_seq": sss_seq,
        "sss_symbol_positions": tuple(sss_symbol_positions),
        "total_symbols": total_symbols,
        "pattern_name": pattern.name,
    }
    return grid, info


def build_pattern_waveform(
    pci: int,
    nfft: int,
    pattern: SSBPatternConfig,
):
    grid, info = build_pattern_grid(pci, nfft, pattern)
    cp_seq = cp_lengths_for_nsymbols(nfft, info["total_symbols"])
    tx_syms = [ofdm_modulate_symbol(grid[l], cp_seq[l]) for l in range(info["total_symbols"])]
    tx_wave = np.concatenate(tx_syms)
    return tx_wave, grid, info, cp_seq

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
def apply_time_varying_channel(x: np.ndarray, cfg: NRSimConfig,seed: int = -1):
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
    if ch.ChanType == "AWGN":
        y = x.astype(np.complex128).copy()
        ch.time_channel = np.zeros((ch.nRx, ch.nTx, ch.DS_tap, len(y)), dtype=np.complex128)
        ch.time_channel[0, 0, 0, :] = 1.0 + 0j
        return y, ch

    nout = len(x) + ch.DS_tap - 1
    # time_samples = np.arange(nout) / (cfg.nfft * cfg.scs_hz)
    time_samples = np.arange(nout) / (cfg.nfft * cfg.scs_hz)

    # # 预热一次，首次会编译，比较慢
    # ch.gen_Jakes_Accelerate(time_samples[:8], seed=123)

    # 正式跑
    ch.gen_Jakes_Accelerate(time_samples, seed=seed)
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
def extract_symbols_from_rx_generic(
    rx: np.ndarray,
    start0: int,
    cp_seq: np.ndarray,
    nfft: int,
):
    syms = []
    ptr = start0
    for l in range(len(cp_seq)):
        slen = cp_seq[l] + nfft
        seg = rx[ptr:ptr + slen]
        if len(seg) < slen:
            seg = np.pad(seg, (0, slen - len(seg)))
        syms.append(seg)
        ptr += slen
    return syms


def extract_oracle_sss_re_pattern(
    rx: np.ndarray,
    start0: int,
    cp_seq: np.ndarray,
    nfft: int,
    info: Dict,
    fs: float,
    cfo_hz: float = 0.0,
):
    """
    提取一个 pattern 内所有 SSS symbol 上的 SSS RE
    返回 shape = [num_sss_symbols_total, 127]
    """
    n = np.arange(len(rx))
    if abs(cfo_hz) > 0:
        rx = rx * np.exp(-1j * 2 * np.pi * cfo_hz * n / fs)

    rx_syms = extract_symbols_from_rx_generic(rx, start0, cp_seq, nfft)

    sss_pos = info["sss_symbol_positions"]
    sss_re_list = []

    for pos in sss_pos:
        Y = ofdm_demodulate_symbol(rx_syms[pos], nfft, cp_seq[pos])
        sss_re = Y[info["sss_k"] + nfft // 2]
        sss_re_list.append(sss_re)

    return np.stack(sss_re_list, axis=0)
# def estimate_oracle_rsrp_from_sss_channel_multi(
#     sss_re_mat: np.ndarray,   # shape [Nsym, 127]
#     sss_seq: np.ndarray,      # shape [127]
#     noise_var: float,
#     smooth_win: int = 9,
# ):
#     """
#     对一个pattern内全部SSS symbol统一做一次RRM measurement:
#       - 每个SSS symbol单独做 LS channel estimation
#       - 每个symbol单独频域平滑
#       - 所有symbol/所有RE统一做功率平均
#     """
#     H_ls_all = []
#     H_smooth_all = []
#     denom_all = []
#     p_re_est_all = []
#
#     for i in range(sss_re_mat.shape[0]):
#         H_ls = sss_re_mat[i] / (sss_seq + 1e-12)
#         H_smooth, denom = moving_average_same_complex(H_ls, smooth_win)
#         noise_var_after_smooth = noise_var / denom
#         p_re_est = np.abs(H_smooth) ** 2 - noise_var_after_smooth
#
#         H_ls_all.append(H_ls)
#         H_smooth_all.append(H_smooth)
#         denom_all.append(denom)
#         p_re_est_all.append(p_re_est)
#
#     H_ls_all = np.stack(H_ls_all, axis=0)
#     H_smooth_all = np.stack(H_smooth_all, axis=0)
#     denom_all = np.stack(denom_all, axis=0)
#     p_re_est_all = np.stack(p_re_est_all, axis=0)
#
#     rsrp_est_lin_preclip = float(np.mean(p_re_est_all))
#     rsrp_est_lin = max(rsrp_est_lin_preclip, 1e-15)
#
#     dbg = {
#         "H_ls_all": H_ls_all,
#         "H_smooth_all": H_smooth_all,
#         "denom_all": denom_all,
#         "p_re_est_all": p_re_est_all,
#         "rsrp_est_lin_preclip": rsrp_est_lin_preclip,
#     }
#     return rsrp_est_lin, dbg
def estimate_oracle_rsrp_from_sss_channel_multi(
    sss_re_mat: np.ndarray,   # shape [Nsym, 127]
    sss_seq: np.ndarray,      # shape [127]
    noise_var: float,
    smooth_win: int = 9,
):
    """
    改进版：
    1) 先对每个symbol做 LS
    2) 跨symbol做复数域平均
    3) 对平均后的H做频域平滑
    4) 再算功率并扣噪
    """
    # [Nsym, 127]
    H_ls_all = sss_re_mat / (sss_seq[None, :] + 1e-12)

    # 先跨重复/跨symbol做复数域平均
    H_avg = np.mean(H_ls_all, axis=0)   # [127]

    # 再做频域平滑
    H_smooth, denom = moving_average_same_complex(H_avg, smooth_win)

    # 复数平均后，噪声方差降为 noise_var / Nsym
    nsym = sss_re_mat.shape[0]
    noise_var_after_avg = noise_var / nsym
    noise_var_after_smooth = noise_var_after_avg / denom

    p_re_est = np.abs(H_smooth) ** 2 - noise_var_after_smooth

    # 加权平均，边缘RE权重低一点
    weights = denom / np.sum(denom)
    rsrp_est_lin_preclip = float(np.sum(weights * p_re_est))
    rsrp_est_lin = max(rsrp_est_lin_preclip, 1e-15)

    dbg = {
        "H_ls_all": H_ls_all,
        "H_avg": H_avg,
        "H_smooth": H_smooth,
        "denom": denom,
        "noise_var_after_smooth": noise_var_after_smooth,
        "p_re_est": p_re_est,
        "rsrp_est_lin_preclip": rsrp_est_lin_preclip,
    }
    return rsrp_est_lin, dbg
# def estimate_oracle_rsrp_from_sss_channel_multi(
#     sss_re_mat: np.ndarray,   # shape [Nobs, 127]
#     sss_seq: np.ndarray,      # shape [127]
#     noise_var: float,
#     smooth_win: int = 9,
#     trim_low: int = 0,        # 可选：去掉最差的几个obs，先默认0
# ):
#     """
#     稳健版：
#     1) 每个SSS observation单独做 LS
#     2) 每个observation单独频域平滑
#     3) 每个observation单独得到每个RE的功率估计 p_re_est
#     4) 对同一个RE，跨所有observation做非相干平均
#     5) 最后 across RE 做加权平均
#
#     说明：
#     - 不做跨observation复数相干合并
#     - 更符合TDL-C下RSRP“功率平均”的目标
#     """
#
#     nobs, nsub = sss_re_mat.shape
#
#     H_ls_all = []
#     H_smooth_all = []
#     denom_all = []
#     p_re_est_all = []
#     noise_var_eff_all = []
#
#     for i in range(nobs):
#         # 每个observation单独LS
#         H_ls = sss_re_mat[i] / (sss_seq + 1e-12)
#
#         # 每个observation单独平滑
#         H_smooth, denom = moving_average_same_complex(H_ls, smooth_win)
#         noise_var_after_smooth = noise_var / denom
#
#         # 每个observation单独估计每个RE功率
#         p_re_est = np.abs(H_smooth) ** 2 - noise_var_after_smooth
#
#         H_ls_all.append(H_ls)
#         H_smooth_all.append(H_smooth)
#         denom_all.append(denom)
#         p_re_est_all.append(p_re_est)
#         noise_var_eff_all.append(noise_var_after_smooth)
#
#     H_ls_all = np.stack(H_ls_all, axis=0)              # [Nobs, Nsub]
#     H_smooth_all = np.stack(H_smooth_all, axis=0)      # [Nobs, Nsub]
#     denom_all = np.stack(denom_all, axis=0)            # [Nobs, Nsub]
#     p_re_est_all = np.stack(p_re_est_all, axis=0)      # [Nobs, Nsub]
#     noise_var_eff_all = np.stack(noise_var_eff_all, axis=0)
#
#     # 可选：对 observation 维做 trimmed mean（先默认 trim_low=0）
#     if trim_low > 0 and (2 * trim_low) < nobs:
#         p_sorted = np.sort(p_re_est_all, axis=0)
#         p_re_avg = np.mean(p_sorted[trim_low:nobs-trim_low, :], axis=0)
#     else:
#         p_re_avg = np.mean(p_re_est_all, axis=0)   # [Nsub]
#
#     # across RE 做加权平均，denom越大权重越高
#     weights = np.mean(denom_all, axis=0)
#     weights = weights / np.sum(weights)
#
#     rsrp_est_lin_preclip = float(np.sum(weights * p_re_avg))
#     rsrp_est_lin = max(rsrp_est_lin_preclip, 1e-15)
#
#     dbg = {
#         "H_ls_all": H_ls_all,
#         "H_smooth_all": H_smooth_all,
#         "denom_all": denom_all,
#         "p_re_est_all": p_re_est_all,
#         "noise_var_eff_all": noise_var_eff_all,
#         "p_re_avg": p_re_avg,
#         "rsrp_est_lin_preclip": rsrp_est_lin_preclip,
#     }
#     return rsrp_est_lin, dbg
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
    y_ch, ch = apply_time_varying_channel(tx, cfg, seed=seed if seed is not None else cfg.seed)

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
def run_single_trial_pattern_oracle_rsrp(
    cfg: NRSimConfig,
    pattern: SSBPatternConfig,
    pci: Optional[int] = None,
    seed: Optional[int] = None,
):
    """
    单次trial：
    - 生成一个pattern waveform（短时长，不是160ms）
    - TDL-C信道 + CFO + 噪声
    - oracle timing + oracle CFO
    - 对pattern内所有SSS symbol + 所有重复一起做一次measurement
    """
    if seed is not None:
        np.random.seed(seed)
    elif cfg.seed is not None:
        np.random.seed(cfg.seed)

    fs = cfg.nfft * cfg.scs_hz

    if pci is None:
        pci = int(np.random.randint(0, 1008)) if cfg.pci is None else int(cfg.pci)

    tx_pat, grid, info, cp_seq = build_pattern_waveform(pci, cfg.nfft, pattern)
    tx = np.concatenate([np.zeros(cfg.timing_offset_samp, dtype=complex), tx_pat])

    y_ch, ch = apply_time_varying_channel(tx, cfg, seed=seed if seed is not None else cfg.seed)

    ref_re_energy = get_reference_re_energy(grid)
    rx, cfo_true_hz, noise_var = apply_cfo_and_noise(
        y_ch, cfg, ref_re_energy=ref_re_energy
    )

    # noiseless true
    sss_true_re_mat = extract_oracle_sss_re_pattern(
        rx=y_ch,
        start0=cfg.timing_offset_samp,
        cp_seq=cp_seq,
        nfft=cfg.nfft,
        info=info,
        fs=fs,
        cfo_hz=0.0,
    )

    # noisy measured
    sss_est_re_mat = extract_oracle_sss_re_pattern(
        rx=rx,
        start0=cfg.timing_offset_samp,
        cp_seq=cp_seq,
        nfft=cfg.nfft,
        info=info,
        fs=fs,
        cfo_hz=cfo_true_hz,
    )

    sss_seq = info["sss_seq"]

    # true RSRP：所有SSS symbol / 所有重复一起平均
    H_true = sss_true_re_mat / (sss_seq[None, :] + 1e-12)
    rsrp_true_lin = float(np.mean(np.abs(H_true) ** 2))

    # measured RSRP
    rsrp_est_lin, rsrp_dbg = estimate_oracle_rsrp_from_sss_channel_multi(
        sss_re_mat=sss_est_re_mat,
        sss_seq=sss_seq,
        noise_var=noise_var,
        smooth_win=cfg.oracle_rsrp_smooth_win,
    )
    rsrp_true_db = float(linear_to_db(np.array([rsrp_true_lin]))[0])
    rsrp_est_db = float(linear_to_db(np.array([rsrp_est_lin]))[0])
    rsrp_err_db = rsrp_est_db - rsrp_true_db

    return {
        "cfg": cfg,
        "pattern": pattern,
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
        # "debug": {
        #     "ref_re_energy": ref_re_energy,
        #     "sss_true_re_mat": sss_true_re_mat,
        #     "sss_est_re_mat": sss_est_re_mat,
        #     "H_true": H_true,
        #     "H_ls_all": rsrp_dbg["H_ls_all"],
        #     "H_block_coh_all": rsrp_dbg["H_block_coh_all"],
        #     "H_block_smooth_all": rsrp_dbg["H_block_smooth_all"],
        #     "denom_all": rsrp_dbg["denom_all"],
        #     "p_re_est_all": rsrp_dbg["p_re_est_all"],
        #     "phase_offsets_all": rsrp_dbg["phase_offsets_all"],
        #     "noise_var_eff_all": rsrp_dbg["noise_var_eff_all"],
        #     "rsrp_block_lin_preclip_arr": rsrp_dbg["rsrp_block_lin_preclip_arr"],
        #     "y_ch": y_ch,
        #     "rx": rx,
        #     "cp_seq": cp_seq,
        #     "info": info,
        # }
    }
def run_pattern_oracle_rsrp_sim(
    cfg: NRSimConfig,
    pattern: SSBPatternConfig,
    snr_db: float,
    n_trials: int = 10000,
    fixed_pci: Optional[int] = 0,
    verbose: bool = True,
):
    cfg_local = NRSimConfig(**vars(cfg))
    cfg_local.snr_db = float(snr_db)

    rng = np.random.default_rng(cfg.seed)

    rsrp_true_db_list = []
    rsrp_est_db_list = []
    rsrp_err_db_list = []
    abs_rsrp_err_db_list = []

    rsrp_true_lin_list = []
    rsrp_est_lin_preclip_list = []

    preclip_nonpositive_count = 0

    for i in range(n_trials):
        trial_seed = int(rng.integers(0, 2**31 - 1))
        trial_pci = int(fixed_pci) if fixed_pci is not None else int(rng.integers(0, 1008))

        trial_res = run_single_trial_pattern_oracle_rsrp(
            cfg=cfg_local,
            pattern=pattern,
            pci=trial_pci,
            seed=trial_seed,
        )

        rsrp_true_db_list.append(trial_res["true"]["rsrp_db"])
        rsrp_est_db_list.append(trial_res["est"]["rsrp_db"])
        rsrp_err_db_list.append(trial_res["stats"]["rsrp_err_db"])
        abs_rsrp_err_db_list.append(trial_res["stats"]["abs_rsrp_err_db"])
        rsrp_true_lin_list.append(trial_res["true"]["rsrp_lin"])
        rsrp_est_lin_preclip_list.append(trial_res["stats"]["rsrp_est_lin_preclip"])

        if trial_res["stats"]["preclip_nonpositive_flag"]:
            preclip_nonpositive_count += 1

        if verbose and ((i + 1) % 1000 == 0 or (i + 1) == n_trials):
            print(f"[{pattern.name}] SNR={snr_db:.1f} dB, finished {i+1}/{n_trials}")

    rsrp_true_db_arr = np.array(rsrp_true_db_list, dtype=float)
    rsrp_est_db_arr = np.array(rsrp_est_db_list, dtype=float)
    rsrp_err_db_arr = np.array(rsrp_err_db_list, dtype=float)
    abs_rsrp_err_db_arr = np.array(abs_rsrp_err_db_list, dtype=float)

    rsrp_true_lin_arr = np.array(rsrp_true_lin_list, dtype=float)
    rsrp_est_lin_preclip_arr = np.array(rsrp_est_lin_preclip_list, dtype=float)

    return {
        "pattern_name": pattern.name,
        "snr_db": float(snr_db),
        "n_trials": int(n_trials),
        "oracle_true_rsrp_db_samples": rsrp_true_db_arr,
        "oracle_est_rsrp_db_samples": rsrp_est_db_arr,
        "oracle_rsrp_error_db_samples": rsrp_err_db_arr,
        "oracle_abs_rsrp_error_db_samples": abs_rsrp_err_db_arr,
        "q90_abs_error_db": float(np.percentile(abs_rsrp_err_db_arr, 90)),
        "pass_prob_within_3p5_db": float(np.mean(abs_rsrp_err_db_arr <= 3.5)),
        "preclip_nonpositive_ratio": float(preclip_nonpositive_count / n_trials),
        "mean_true_rsrp_db": float(np.mean(rsrp_true_db_arr)),
        "mean_est_rsrp_db": float(np.mean(rsrp_est_db_arr)),
        "mean_error_db": float(np.mean(rsrp_err_db_arr)),
        "std_error_db": float(np.std(rsrp_err_db_arr, ddof=0)),
        "mean_true_rsrp_lin": float(np.mean(rsrp_true_lin_arr)),
        "mean_est_rsrp_lin_preclip": float(np.mean(rsrp_est_lin_preclip_arr)),
    }
def plot_pattern_comparison_cdf(
    res_a,
    res_b,
    res_c,
    save_dir="figures",
    prefix="pattern_compare",
    max_delta_db=8.0,
):
    os.makedirs(save_dir, exist_ok=True)

    # 取绝对误差，作为 Delta RSRP
    delta_a = np.asarray(res_a["oracle_abs_rsrp_error_db_samples"], dtype=float)
    delta_b = np.asarray(res_b["oracle_abs_rsrp_error_db_samples"], dtype=float)
    delta_c = np.asarray(res_c["oracle_abs_rsrp_error_db_samples"], dtype=float)
    # 只统计 0~max_delta_db 内的样本
    delta_a_in = delta_a[delta_a <= max_delta_db]
    delta_b_in = delta_b[delta_b <= max_delta_db]
    delta_c_in = delta_c[delta_c <= max_delta_db]
    if len(delta_a_in) == 0 or len(delta_b_in) == 0:
        raise ValueError("No samples fall within the requested Delta RSRP range.")

    x1, y1 = ecdf(delta_a)
    x2, y2 = ecdf(delta_b)
    x3, y3 = ecdf(delta_c)
    plt.figure(figsize=(7, 5))
    plt.plot(x1, y1, label=res_a["pattern_name"])
    plt.plot(x2, y2, label=res_b["pattern_name"])
    plt.plot(x3, y3, label=res_c["pattern_name"])
    plt.axvline(3.5, linestyle="--", label="3.5 dB")
    plt.axhline(0.9, linestyle="--", label="90%")
    plt.grid(True)
    plt.xlim(0, max_delta_db)
    plt.ylim(0, 1.0)
    plt.xlabel("Delta RSRP (dB)")
    plt.ylabel("CDF")
    plt.title(
        f"Delta RSRP CDF Comparison @ SNR = {res_a['snr_db']:.1f} dB "
        f"(0~{max_delta_db:.1f} dB only)"
    )
    plt.legend()

    fig_path = os.path.join(
        save_dir,
        f"{prefix}_delta_rsrp_cdf_snr_{res_a['snr_db']:.1f}dB_0to{max_delta_db:.1f}dB.png"
    )
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {fig_path}")

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

def estimate_delay_support_bins_partial(
    H_ls_partial: np.ndarray,
    guard_bins: int = 1,
    wrap_bins: int = 0,
) -> np.ndarray:
    """
    针对观测到的连续 SSS 子带（长度 Nsub）做 partial-band delay-domain 支撑估计。
    注意：这是基于观测子带的等效 delay profile，不是 full-band CIR。
    """
    nsub = len(H_ls_partial)
    h_td = np.fft.ifft(H_ls_partial, norm="ortho")
    power = np.abs(h_td) ** 2

    peak = int(np.argmax(power))
    total_power = np.sum(power) + 1e-15
    peak_power = power[peak]

    # 和你原来的 full-band 版本保持相似风格
    threshold = max(0.01 * peak_power, 0.001 * total_power)

    valid = np.where(power >= threshold)[0]
    if valid.size == 0:
        valid = np.array([peak], dtype=int)

    start = max(0, int(valid.min()) - int(guard_bins))
    end = min(nsub - 1, int(valid.max()) + int(guard_bins))

    keep = np.zeros(nsub, dtype=bool)
    keep[start:end + 1] = True

    if wrap_bins > 0:
        keep[:wrap_bins] = True
        keep[-wrap_bins:] = True

    return keep



if __name__ == "__main__":
    cfg = NRSimConfig(
        pci=0,
        nfft=256,
        chan_type="TDLC",
        delay_spread_ns=300.0,
        speed_kmh=30.0,
        system_cfo_ppm=5.0,
        timing_offset_samp=80,
        fo_num_candidates=3,
        oracle_rsrp_smooth_win =9,
        seed=1,
        snr_db=-15.5,
    )

    # pattern_rep1 = SSBPatternConfig(
    #     name="rep1",
    #     num_symbols_per_block=10,
    #     sss_symbol_indices=(3, 5, 7, 9),
    #     num_repeats=1,
    # )
    pattern_rep0 = SSBPatternConfig(
        name="New SSB rep 1",
        num_symbols_per_block=10,
        sss_symbol_indices=(3, 5, 7, 9),
        num_repeats=1,
    )
    pattern_rep1 = SSBPatternConfig(
        name="NR SSB rep 8",
        num_symbols_per_block=4,
        sss_symbol_indices=(2,),
        num_repeats=8,
    )

    pattern_rep2 = SSBPatternConfig(
        name="New SSB rep 2",
        num_symbols_per_block=10,
        sss_symbol_indices=(3, 5, 7, 9),
        num_repeats=2,
    )
    _dummy_ch = ChannelInfo(
        nTx=1, nRx=1,
        Speed=cfg.speed_kmh,
        Fc=cfg.fc_hz,
        Fs=cfg.nfft * cfg.scs_hz,
        ChanType=cfg.chan_type,
        DS=cfg.delay_spread_ns,
        NFFT=cfg.nfft,
    )
    _dummy_ts = np.arange(8) / (cfg.nfft * cfg.scs_hz)
    _dummy_ch.gen_Jakes_Accelerate(_dummy_ts, seed=0)

    res_8 = run_pattern_oracle_rsrp_sim(
        cfg=cfg,
        pattern=pattern_rep1,
        snr_db=-15.5,
        n_trials=10000,
        fixed_pci=0,
        verbose=True,
    )

    res_16 = run_pattern_oracle_rsrp_sim(
        cfg=cfg,
        pattern=pattern_rep2,
        snr_db=-15.5,
        n_trials=10000,
        fixed_pci=0,
        verbose=True,
    )

    res_0 = run_pattern_oracle_rsrp_sim(
        cfg=cfg,
        pattern=pattern_rep0,
        snr_db=-15.5,
        n_trials=10000,
        fixed_pci=0,
        verbose=True,
    )
    print("\n=== Single SSB: Rep1 ===")
    print("Mean true RSRP (dB):", res_8["mean_true_rsrp_db"])
    print("Mean est  RSRP (dB):", res_8["mean_est_rsrp_db"])
    print("Mean error (dB):", res_8["mean_error_db"])
    print("Std  error (dB):", res_8["std_error_db"])
    print("Q90(|error|) (dB):", res_8["q90_abs_error_db"])
    print("P(|error| <= 3.5 dB):", res_8["pass_prob_within_3p5_db"])
    print("preclip_nonpositive_ratio:", res_8["preclip_nonpositive_ratio"])
    print("Mean true RSRP (lin):", res_8["mean_true_rsrp_lin"])
    print("Mean est  RSRP preclip (lin):", res_8["mean_est_rsrp_lin_preclip"])
    print("\n=== Double SSB: rep2 ===")
    print("Mean true RSRP (dB):", res_16["mean_true_rsrp_db"])
    print("Mean est  RSRP (dB):", res_16["mean_est_rsrp_db"])
    print("Mean error (dB):", res_16["mean_error_db"])
    print("Std  error (dB):", res_16["std_error_db"])
    print("Q90(|error|) (dB):", res_16["q90_abs_error_db"])
    print("P(|error| <= 3.5 dB):", res_16["pass_prob_within_3p5_db"])
    print("preclip_nonpositive_ratio:", res_16["preclip_nonpositive_ratio"])

    plot_pattern_comparison_cdf(
        res_8,
        res_16,
        res_0,
        save_dir="figures",
        prefix="sss_pattern_compare",
    )