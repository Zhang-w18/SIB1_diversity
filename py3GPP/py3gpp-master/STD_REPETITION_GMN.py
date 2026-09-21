# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:23:13 2025

 File name: STD_PRACH
 Author: m00829866	Version: 0.1	Date: 2025-11-17
 Description:
     1. standard PRACH Link Level Simulation

 修改记录：
 date name line xxx
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import GenTDLChannel
from GenTDLChannel import ChannelInfo
# ============================ 工具与随机源 =============================
rng = np.random.default_rng(5)
def awgn(x, snr_db,cfg):
    # x: 时域复序列
    # snr_db：你想把“每个有效子载波上的 SNR”设成多少 dB

    P_signal_td = np.mean(np.abs(x)**2) # 时域平均功率
    occ_ratio = cfg.L_RA / (cfg.N_fft_ra / cfg.N_rep) # 实际占用的子载波比例 L_RA / L_rep
    # 将“RE SNR”换算成“sample SNR”
    # sample_SNR = RE_SNR * occ_ratio
    # 因为只有 occ_ratio 的子载波有能量，能量被扩散到 N_fft 的时域采样中
    sample_snr_linear = (10 ** (snr_db / 10)) * occ_ratio
    # 对应 sample-level 的噪声功率
    n0 = P_signal_td / sample_snr_linear

    w = (rng.normal(scale=np.sqrt(n0 / 2), size=x.shape) +
         1j * rng.normal(scale=np.sqrt(n0 / 2), size=x.shape))
    return x + w
# === ADD: 规范的 139→IFFT 嵌入（不居中），与参考实现一致 ===
def prach_ifft_embed_pad(X_139_fd: np.ndarray, ifft_size: int) -> np.ndarray:
    """
    把 139 点频域序列直接放在 [0:139)，其余补零，然后 IFFT。
    与你给的参考代码一致：yuv_extend=[yuv, zeros]; td = ifft(yuv_extend)*sqrt(N)
    注意：这里不做 ifftshift 居中！
    """
    L = X_139_fd.size
    assert L <= ifft_size
    X = np.zeros(ifft_size, dtype=complex)
    X[:L] = X_139_fd
    td = np.fft.ifft(X) * math.sqrt(ifft_size)  # IFFT 增益补偿
    return td



# ============================ d_u & Type-A 参数 =============================
def compute_d_u(L_RA: int, u: int) -> int:
    """ d_u 定义：
    找到最小非负整数 q，使 (q*u) mod L_RA == 1
    若 q < L_RA/2, d_u = q；否则 d_u = L_RA - q
    """
    L = L_RA
    u_mod = u % L
    if math.gcd(u_mod, L) != 1:
        raise ValueError(f"Root u={u} is not coprime with L_RA={L}, cannot compute d_u.")
    q = 0
    while True:
        if (q * u_mod) % L == 1:
            break
        q += 1
    return q if q < L/2 else L - q

def restricted_type_a_params(L_RA: int, N_cs: int, d_u: int):
    """ 返回 (n_shift_RA, d_start, n_group_RA, nbar_shift_RA)
    或 None（若 d_u 落在未覆盖区间）
    区间:
      Case-1: N_cs <= d_u < L_RA/3
      Case-2: L_RA/3 <= d_u <= (L_RA - N_cs)/2
    """
    L = L_RA
    if N_cs <= d_u < L/3:
        n_shift_RA  = d_u // N_cs
        d_start     = 2*d_u + n_shift_RA * N_cs
        n_group_RA  = L // d_start
        nbar_shift_RA = max((L - 2*d_u - n_group_RA*d_start) / N_cs, 0)
        nbar_shift_RA = int(np.floor(nbar_shift_RA + 1e-12))
        return int(n_shift_RA), int(d_start), int(n_group_RA), int(nbar_shift_RA)

    if L/3 <= d_u <= (L - N_cs)//2:
        n_shift_RA  = (L - 2*d_u) // N_cs
        d_start     = L - 2*d_u + n_shift_RA * N_cs
        n_group_RA  = d_u // d_start
        tmp = (d_u - n_group_RA * d_start) / N_cs
        nbar_shift_RA = min(max(tmp, 0), n_shift_RA)
        nbar_shift_RA = int(np.floor(nbar_shift_RA + 1e-12))
        return int(n_shift_RA), int(d_start), int(n_group_RA), int(nbar_shift_RA)

    return None

# ============================ 38.211: Ncs 与根序列表 =============================
def _get_Ncs(LRA: int, zeroCorrelationZoneConfig: int) -> int:
    longLRA_list  = [0,13,15,18,22,26,32,38,46,59,76,93,119,167,279,419] # TODO 这里只实现了RA-SCS=1.25kHZ情况下，非限制集的N_CS取值
    shortLRA_list = [0,2,4,6,8,10,12,13,15,17,19,23,27,34,46,69]
    assert zeroCorrelationZoneConfig in range(16)
    assert LRA in (139, 839)
    return longLRA_list[zeroCorrelationZoneConfig] if LRA == 839 else shortLRA_list[zeroCorrelationZoneConfig]

def _get_sequence_number(LRA, logical_root_seq):
    rootsTableShortPreamble = [1,138,2,137,3,136,4,135,5,134,6,133,7,132,8,131,9,130,10,129,11,128,12,127,13,126,14,125,15,124,16,123,17,122,18,121,19,120,20,119,21,118,22,117,23,116,24,115,25,114,26,113,27,112,28,111,29,110,30,109,31,108,32,107,33,106,34,105,35,104,36,103,37,102,38,101,39,100,40,99,41,98,42,97,43,96,44,95,45,94,46,93,47,92,48,91,49,90,50,89,51,88,52,87,53,86,54,85,55,84,56,83,57,82,58,81,59,80,60,79,61,78,62,77,63,76,64,75,65,74,66,73,67,72,68,71,69,70]
    rootsTableLongPreamble  = [129,710,140,699,120,719,210,629,168,671,84,755,105,734,93,746,70,769,60,779,2,837,1,838,56,783,112,727,148,691,80,759,42,797,40,799,35,804,73,766,146,693,31,808,28,811,30,809,27,812,29,810,24,815,48,791,68,771,74,765,178,661,136,703,86,753,78,761,43,796,39,800,20,819,21,818,95,744,202,637,190,649,181,658,137,702,125,714,151,688,217,622,128,711,142,697,122,717,203,636,118,721,110,729,89,750,103,736,61,778,55,784,15,824,14,825,12,827,23,816,34,805,37,802,46,793,207,632,179,660,145,694,130,709,223,616,228,611,227,612,132,707,133,706,143,696,135,704,161,678,201,638,173,666,106,733,83,756,91,748,66,773,53,786,10,829,9,830,7,832,8,831,16,823,47,792,64,775,57,782,104,735,101,738,108,731,208,631,184,655,197,642,191,648,121,718,141,698,149,690,216,623,218,621,152,687,144,695,134,705,138,701,199,640,162,677,176,663,119,720,158,681,164,675,174,665,171,668,170,669,87,752,169,670,88,751,107,732,81,758,82,757,100,739,98,741,71,768,59,780,65,774,50,789,49,790,26,813,17,822,13,826,6,833,5,834,33,806,51,788,75,764,99,740,96,743,97,742,166,673,172,667,175,664,187,652,163,676,185,654,200,639,114,725,189,650,115,724,194,645,195,644,192,647,182,657,157,682,156,683,211,628,154,685,123,716,139,700,212,627,153,686,213,626,215,624,150,689,225,614,224,615,221,618,220,619,127,712,147,692,124,715,193,646,205,634,206,633,116,723,160,679,186,653,167,672,79,760,85,754,77,762,92,747,58,781,62,777,69,770,54,785,36,803,32,807,25,814,18,821,11,828,4,835,3,836,19,820,22,817,41,798,38,801,44,795,52,787,45,794,63,776,67,772,72,767,76,763,94,745,102,737,90,749,109,730,165,674,111,728,209,630,204,635,117,722,188,651,159,680,198,641,113,726,183,656,180,659,177,662,196,643,155,684,214,625,126,713,131,708,219,620,222,617,226,613,230,609,232,607,262,577,252,587,418,421,416,423,413,426,411,428,376,463,395,444,283,556,285,554,379,460,390,449,363,476,384,455,388,451,386,453,361,478,387,452,360,479,310,529,354,485,328,511,315,524,337,502,349,490,335,504,324,515,323,516,320,519,334,505,359,480,295,544,385,454,292,547,291,548,381,458,399,440,380,459,397,442,369,470,377,462,410,429,407,432,281,558,414,425,247,592,277,562,271,568,272,567,264,575,259,580,237,602,239,600,244,595,243,596,275,564,278,561,250,589,246,593,417,422,248,591,394,445,393,446,370,469,365,474,300,539,299,540,364,475,362,477,298,541,312,527,313,526,314,525,353,486,352,487,343,496,327,512,350,489,326,513,319,520,332,507,333,506,348,491,347,492,322,517,330,509,338,501,341,498,340,499,342,497,301,538,366,473,401,438,371,468,408,431,375,464,249,590,269,570,238,601,234,605,257,582,273,566,255,584,254,585,245,594,251,588,412,427,372,467,282,557,403,436,396,443,392,447,391,448,382,457,389,450,294,545,297,542,311,528,344,495,345,494,318,521,331,508,325,514,321,518,346,493,339,500,351,488,306,533,289,550,400,439,378,461,374,465,415,424,270,569,241,598,231,608,260,579,268,571,276,563,409,430,398,441,290,549,304,535,308,531,358,481,316,523,293,546,288,551,284,555,368,471,253,586,256,583,263,576,242,597,274,565,402,437,383,456,357,482,329,510,317,522,307,532,286,553,287,552,266,573,261,578,236,603,303,536,356,483,355,484,405,434,404,435,406,433,235,604,267,572,302,537,309,530,265,574,233,606,367,472,296,543,336,503,305,534,373,466,280,559,279,560,419,420,240,599,258,581,229,610]
    assert LRA in (139, 839)
    return rootsTableLongPreamble[logical_root_seq] if LRA == 839 else rootsTableShortPreamble[logical_root_seq]

# ============================ 选择 logical_root_seq 与 v =============================
def _type_a_num_preambles_for_root(LRA: int, Ncs: int, u: int) -> int:
    d_u = compute_d_u(LRA, u)
    params = restricted_type_a_params(LRA, Ncs, d_u)
    if params is None:
        return 0
    n_shift_RA, d_start, n_group_RA, nbar_shift_RA = params
    return n_shift_RA * n_group_RA + nbar_shift_RA

def _type_a_Cv_for_v(LRA: int, Ncs: int, u: int, v: int) -> int:
    d_u = compute_d_u(LRA, u)
    params = restricted_type_a_params(LRA, Ncs, d_u)
    if params is None:
        raise ValueError("Type-A params not available for this (LRA,Ncs,u).")
    n_shift_RA, d_start, n_group_RA, nbar_shift_RA = params
    w = n_shift_RA * n_group_RA + nbar_shift_RA
    v = int(v % max(w, 1))
    Cv = (v // n_shift_RA) * d_start + (v % n_shift_RA) * Ncs
    return Cv % LRA

def _select_logical_root_and_v(
    LRA: int, NCS: int, PreambleIndex: int, prach_RootSequenceIndex: int, restricted_type: str
) -> tuple[int, int]:
    """ 返回 (logical_root_seq, v) 使得该 preambleIndex 被分配到相应的逻辑根与该根内的索引 v。 """
    if NCS == 0:
        # 每个 preamble 用一个 root，按顺序从 prach_RootSequenceIndex 开始
        logical_root_seq = (prach_RootSequenceIndex + PreambleIndex) % (LRA - 1)
        v = 0
        return logical_root_seq, v

    if restricted_type == "unrestricted":
        per_root = max(LRA // NCS, 1)
        if per_root >= 64:
            return prach_RootSequenceIndex % (LRA - 1), PreambleIndex
        remain = PreambleIndex
        m = 0
        while True:
            logical_root_seq = (prach_RootSequenceIndex + m) % (LRA - 1)
            if remain < per_root:
                return logical_root_seq, remain
            remain -= per_root
            m += 1

    elif restricted_type == "typeA":
        remain = PreambleIndex
        m = 0
        while True:
            logical_root_seq = (prach_RootSequenceIndex + m) % (LRA - 1)
            u = _get_sequence_number(LRA, logical_root_seq)
            w = _type_a_num_preambles_for_root(LRA, NCS, u)
            if w <= 0:
                m += 1
                continue
            if remain < w:
                return logical_root_seq, remain
            remain -= w
            m += 1
    else:
        raise ValueError("restricted_type must be 'unrestricted' or 'typeA'.")

# ============================ PRACH 序列生成（频域） =============================
def PRACH_seq_gen(
    prach_RootSequenceIndex: int, LRA: int, zeroCorrelationZoneConfig: int,
    PreambleIndex: int, restricted_type: str = "unrestricted"
):
    """ 38.211 §6.3.3.1：生成 preamble 的频域表示 y_uv，并返回 (u, v, Cv, logical_root_seq) """
    assert LRA in (139, 839)
    NCS = _get_Ncs(LRA, zeroCorrelationZoneConfig)

    logical_root_seq, v = _select_logical_root_and_v(
        LRA=LRA, NCS=NCS, PreambleIndex=PreambleIndex,
        prach_RootSequenceIndex=prach_RootSequenceIndex, restricted_type=restricted_type
    )
    u = _get_sequence_number(LRA, logical_root_seq)

    n = np.arange(LRA)
    xu = np.exp(-1j * np.pi * u * n * (n + 1) / LRA)

    if NCS == 0:
        Cv = 0
    else:
        if restricted_type == "unrestricted":
            Cv = (v * NCS) % LRA
        else:  # typeA
            Cv = _type_a_Cv_for_v(LRA, NCS, u, v)

    xuv = np.roll(xu, Cv)          # x_u((n + Cv) mod LRA)
    yuv = np.fft.fft(xuv)          # 频域
    return yuv, (u, v, Cv, logical_root_seq)

# ============================ 配置容器 =============================
class PRACHConfig:
    def __init__(self, name: str, L_RA: int = 139, N_rep: int = 2, scs_khz: int = 15, mu: int = 0,
                 restricted_type: str = "unrestricted", prach_RootSequenceIndex: int = 0, zeroCorrelationZoneConfig: int = 8):
        self.name = name
        self.L_RA = L_RA
        self.N_rep = N_rep
        self.scs_khz = scs_khz
        self.mu = mu
        self.restricted_type = restricted_type
        self.prach_RootSequenceIndex = prach_RootSequenceIndex
        self.zeroCorrelationZoneConfig = zeroCorrelationZoneConfig

        fmt = FORMAT_DB[name]
        kappa = 1
        if L_RA < 839:
            self.N_rep = fmt["N_rep"]
            self.N_fft_ra = int(fmt["Nu"](kappa, mu))
            self.cp_len_ra = int(fmt["Ncp"](kappa, mu))
            self.delta_f_ra_hz = 15e3 * (2 ** mu)
            L_rep = self.N_fft_ra // self.N_rep
            self.Fs = self.delta_f_ra_hz * L_rep
        else:
            self.N_rep = fmt["N_rep"]
            self.N_fft_ra = int(fmt["Nu"](kappa,mu))
            self.cp_len_ra = int(fmt["Ncp"](kappa,mu))
            self.delta_f_ra_hz = 1.25e3
            L_rep = self.N_fft_ra // self.N_rep
            self.Fs = self.delta_f_ra_hz * L_rep

FORMAT_DB = {
    # name: {reps, Nu, Ncp}
    "F0": {"N_rep": 1, "Nu": lambda k, mu: 24576 * k, "Ncp": lambda k, mu: 3168 * k},
    "A1": {"N_rep": 1, "Nu": lambda k, mu: 2 * 2048 * k * 2 ** (-mu), "Ncp": lambda k, mu: 288 * k * 2 ** (-mu)},
    "A2": {"N_rep": 2, "Nu": lambda k, mu: 4 * 2048 * k * 2 ** (-mu), "Ncp": lambda k, mu: 576 * k * 2 ** (-mu)},
    "A3": {"N_rep": 3, "Nu": lambda k, mu: 6 * 2048 * k * 2 ** (-mu), "Ncp": lambda k, mu: 864 * k * 2 ** (-mu)},
    "B1": {"N_rep": 1, "Nu": lambda k, mu: 2 * 2048 * k * 2 ** (-mu), "Ncp": lambda k, mu: 216 * k * 2 ** (-mu)},
    "B2": {"N_rep": 2, "Nu": lambda k, mu: 4 * 2048 * k * 2 ** (-mu), "Ncp": lambda k, mu: 360 * k * 2 ** (-mu)},
    "B3": {"N_rep": 3, "Nu": lambda k, mu: 6 * 2048 * k * 2 ** (-mu), "Ncp": lambda k, mu: 504 * k * 2 ** (-mu)},
    "B4": {"N_rep": 12, "Nu": lambda k, mu: 12 * 2048 * k * 2 ** (-mu), "Ncp": lambda k, mu: 936 * k * 2 ** (-mu)},
    "C0": {"N_rep": 1, "Nu": lambda k, mu: 1 * 2048 * k * 2 ** (-mu), "Ncp": lambda k, mu: 1240 * k * 2 ** (-mu)},
    "C2": {"N_rep": 4, "Nu": lambda k, mu: 4 * 2048 * k * 2 ** (-mu), "Ncp": lambda k, mu: 2048 * k * 2 ** (-mu)},
}

# 示例配置（请按表替换具体数值）
CFG_C2 = PRACHConfig(
    name="C2", L_RA=139, scs_khz=15, mu=0, restricted_type="unrestricted",
    prach_RootSequenceIndex=10, zeroCorrelationZoneConfig=8
)
CFG_B4 = PRACHConfig(
    name="B4", L_RA=139, scs_khz=15, mu=0, restricted_type="unrestricted",
    prach_RootSequenceIndex=10, zeroCorrelationZoneConfig=8
)
CFG_F0 = PRACHConfig(
    name="F0", L_RA=839, scs_khz=15, mu=0, restricted_type="unrestricted",
    prach_RootSequenceIndex=10, zeroCorrelationZoneConfig=8
)
def estimate_cfo_from_cp(
    y_td: np.ndarray,
    cfg: PRACHConfig,
    tau_samp: int = 0,
) -> tuple[float, float]:
    """
    Format 0 情形：只有 1 个 OFDM 符号 + CP，使用 CP 自相关估计 CFO。

    参数：
      y_td    : 接收时域波形（含 CP），假设 preamble 起点在 tau_samp 左右
      cfg     : PRACHConfig
      tau_samp: 估计的 preamble 起点（样点），仿真里可以先假设 0 或 true_delay

    返回：
      cfo_hat : 归一化 CFO（单位：子载波间隔）
      dphi    : 相位差（rad）
    """
    L_rep = int(cfg.N_fft_ra // cfg.N_rep)  # Format 0: N_rep=1 → L_rep=N_fft_ra
    Ncp   = cfg.cp_len_ra

    # 拿出 [起点 .. 起点 + Ncp + L_rep) 这一段
    start = tau_samp
    end   = start + Ncp + L_rep
    if end > y_td.size:
        return 0.0, 0.0   # 波形不够长，估不了了

    seg = y_td[start:end]

    cp   = seg[:Ncp]          # CP 段
    tail = seg[L_rep:L_rep+Ncp]   # 符号末尾与 CP 对应的那一段

    # 相关：sum cp * conj(tail)
    R = np.vdot(cp, tail)    # = sum conj(cp[n]) * tail[n]，相位差号差无所谓，只差一个负号
    dphi = np.angle(R)       # ∈ (-π, π]

    # 归一化 CFO（每 L_rep 样点 2π 相移 = 1 子载波间隔）
    cfo_hat = -dphi / (2 * np.pi)

    return cfo_hat, dphi
# ============================ 生成 preamble 参考库 =============================
def build_preamble_bank(cfg: PRACHConfig, num_preambles: int):
    bank_fd, mapping = [], []
    for pid in range(num_preambles):
        yuv, (u, v, Cv, lroot) = PRACH_seq_gen(
            prach_RootSequenceIndex=cfg.prach_RootSequenceIndex,
            LRA=cfg.L_RA,
            zeroCorrelationZoneConfig=cfg.zeroCorrelationZoneConfig,
            PreambleIndex=pid,
            restricted_type=cfg.restricted_type
        )
        bank_fd.append(yuv)
        mapping.append((u, v, Cv, lroot))
    return bank_fd, mapping

from scipy import sparse


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
        H_eff = build_time_varying_H_sparse(h_tap_t, L)
        # 最终输出
        y = tx_wave @ H_eff


    return y, ch  # 返回波形与信道对象（你后续若想观测 doppler/ch.funs 可用）


# ============================ 发射机：按 format 拼接重复与 CP =============================
def prach_tx_one_preamble(cfg: PRACHConfig, X_fd: np.ndarray) -> np.ndarray:
    """根据 N_rep 和 CP 长度拼接一个 PRACH 短前导的时域波形（单次发射）"""
    sym = prach_ifft_embed_pad(X_fd, int(cfg.N_fft_ra/cfg.N_rep))  # 长度 N_fft_ra
    sym_cp = np.concatenate([sym[-cfg.cp_len_ra:], sym])

    if cfg.N_rep == 1:
        tx = sym_cp
    elif cfg.N_rep == 2:
        # 第二个重复通常不再加 CP（按表决定；若需要，可改为 sym_cp）
        tx = np.concatenate([sym_cp, sym])
    else:
        tx = sym_cp.copy()
        for _ in range(cfg.N_rep - 1):
            tx = np.concatenate([tx, sym])
    return tx

# ============================ 接收机：检测、CFO/TA 估计 =============================

# === REPLACED: 原单对重复块的 CFO 估计，改为跨(N_rep-1)对的相位平均 ===
def estimate_cfo_from_repetition(rx_no_cp: np.ndarray, L_rep: int) -> float:
    """
    利用相邻重复块的相位差平均来估计 CFO（归一化到“每个 N_fft_ra”）。
    rx_no_cp: 去掉CP后的连续重复块序列，长度应为 K*L_rep
    L_rep   : 每个重复块长度（= cfg.N_fft_ra）
    """
    K = rx_no_cp.size // L_rep
    K = max(K, 1)
    # 至少需要两块
    if K < 2:
        return 0.0, 0.0

    phasor_sum = 0+0j
    for k in range(K-1):
        blk0 = rx_no_cp[k*L_rep:(k+1)*L_rep]
        blk1 = rx_no_cp[(k+1)*L_rep:(k+2)*L_rep]
        phasor_sum += np.vdot(blk0, blk1)

    dphi = np.angle(phasor_sum)   # ∈ (-π, π]
    cfo_hat = dphi / (2*np.pi)    # 归一化 CFO ∈ (-0.5, 0.5]
    return cfo_hat, dphi



# === REPLACE: 按 root 检测 + 反查 v + 线性匹配取 TA ===
# === FIX: 时间域 + root/ZCZ 版本的检测 ===
def detect_preamble_and_ta_mode_time(
    cfg: PRACHConfig,
    y_td: np.ndarray,
    bank_fd: list,
    mapping: list,
    combine_mode: str = "first",
    det_threshold=None,
):
    """
    基于：
      1) 去 CP + CFO 估计/补偿（在时间域）
      2) 调用 prach_detect_by_root_zcz：按 root + ZCZ 做 preamble ID & 粗 TA
      3) 把样点级 offset 映射到 N_TA

    返回：
      best_id, delay_bin, cfo_hat, peak, N_TA, t_hat, delta_t
    （接口跟你原来的 detect_preamble_and_ta_mode 保持一致）
    """

    # --- 1) 去 CP，取若干重复块 ---
    L_rep = int(cfg.N_fft_ra // cfg.N_rep)
    take  = cfg.N_rep if cfg.N_rep > 1 else 1
    y_nocp_for_cfo = y_td[cfg.cp_len_ra : cfg.cp_len_ra + take * L_rep]

    # --- 2) CFO 估计 + 去旋 ---
    if y_nocp_for_cfo.size >= 2 * L_rep:
        cfo_hat, _ = estimate_cfo_from_repetition(y_nocp_for_cfo, L_rep=L_rep) # TODO
        # cfo_hat, _ = estimate_cfo_denoised(y_nocp_for_cfo, L_rep=L_rep)  # TODO
    else:
        cfo_hat, _ = estimate_cfo_from_cp(y_nocp_for_cfo, cfg, tau_samp=0)

        # --- 2) 在整条波形上去 CFO（仍然保留 CP 给 root 检测自己处理） ---
    n_full = np.arange(y_td.size)
    y_corr = y_td * np.exp(-1j * 2 * np.pi * cfo_hat * n_full / L_rep) #TODO L_rep改成一个ofdm符号的采样点个数

    if mapping is None:
        raise ValueError("detect_preamble_and_ta_mode_time() 需要传入 mapping（build_preamble_bank 返回的第二个值）")

    # --- 3) 按 root/ZCZ 做 preamble 检测 + 样点级 offset ---
    # 这里 indin 先默认扫 bank_fd 中所有 preamble；后面你可以改成只扫一部分
    indin = list(range(len(bank_fd)))

    pid_hat, delay_samples, best_root, v_hat, pid_again, bestCorr = prach_detect_by_root_zcz(
        cfg, y_corr, bank_fd, mapping, indin, det_threshold=det_threshold
    )

    # --- 4) TA 映射（样点 -> 秒 -> N_TA）---
    Tc = 1 / (15000 * 4096)          # NR 基本时钟
    delta_t = 16 * Tc * (2 ** cfg.mu)

    if pid_hat is None:
        # 没检测到：返回一个“空检测”的结果
        best_id   = -1
        delay_bin = 0
        t_hat     = 0.0
        N_TA      = 0
        peak      = 0.0
    else:
        best_id   = int(pid_hat)
        delay_bin = int(delay_samples)           # 样点级 delay（相对于 y_block 起点）
        t_hat     = delay_bin / cfg.Fs          # 秒
        N_TA      = int(np.round(t_hat / delta_t))
        peak      = float(bestCorr)

    return best_id, delay_bin, cfo_hat, peak, N_TA, t_hat, delta_t


# === ADD: helper — 反查 root → 该 root 下的 (pid, Cv) 列表 ===
def _root_group(mapping):
    by_root = {}
    for pid, (u, v, Cv, lroot) in enumerate(mapping):
        by_root.setdefault(lroot, []).append((pid, u, v, Cv))
    return by_root

# === ADD: helper — 给出某 root 的“参考频域 139”（优先 Cv=0），以及该 root 的 u ===
def _pick_root_ref(bank_fd, plist):
    # plist: list of (pid, u, v, Cv)
    # 选择 Cv=0 的那条作为参考；若无，就拿第一条
    plist_sorted = sorted(plist, key=lambda x: 0 if x[3]==0 else 1)
    pid_ref, u, v0, Cv0 = plist_sorted[0]
    return bank_fd[pid_ref], u, pid_ref

def prach_detect_by_root_zcz(cfg: PRACHConfig,
                             y_td: np.ndarray,
                             bank_fd: list,
                             mapping: list,
                             indin: list,
                             det_threshold=None
                             ):
    """
    两阶段检测：
      1) 暴力 per-preamble 相关，找出最有可能的 root（best_root）。
      2) 在 best_root 内，用 NCS/ZCZ 结构按 Cv 区分不同 v（preamble index），
         并给出粗 TA（offset_samples）。

    返回:
      indout         : 检测到的 preamble index（0..63），或 None
      offset_samples : 相对第一个有效符号起点的粗 TA 样点
      best_root      : 逻辑 root index
      v_hat          : 该 root 下的 v
      pid_hat        : 同 indout（冗余）
      bestCorr_glob  : 最终使用的相关峰值
    """
    Fs      = cfg.Fs
    LRA     = cfg.L_RA
    delta_f = cfg.delta_f_ra_hz

    # 单块长度 & 起点（假定 y_td 含 CP）
    L_rep        = int(cfg.N_fft_ra // cfg.N_rep)
    start        = cfg.cp_len_ra
    prachDuration = cfg.N_rep - 1 if cfg.name == "C2" else cfg.N_rep

    # === 预处理 root 分组 ===
    roots = _root_group(mapping)  # {lroot: [(pid,u,v,Cv), ...]}

    if indin is None or len(indin) == 0:
        allowed_pids = list(range(len(mapping)))
    else:
        allowed_pids = list(indin)

    # =========================================================
    # 阶段 1：per-preamble 暴力相关，找出“最强 root”
    # =========================================================
    # 记录每个 root 内“最佳 preamble”的相关结果
    root_best = {}  # lroot -> dict(bestCorr, maxpos, pid)

    for pid in allowed_pids:
        u, v, Cv, lroot = mapping[pid]

        # 如果这个 root 不在 roots 里，跳过（理论上不会）
        if lroot not in roots:
            continue

        X_fd = bank_fd[pid]

        # 生成与 TX 完全一致的单块参考
        sym_ref = prach_ifft_embed_pad(X_fd, L_rep)
        H_ref   = np.fft.fft(sym_ref)

        # 多重复块累加相关
        cp_local = np.zeros(L_rep, dtype=float)
        for k in range(int(prachDuration)):
            s = start + k * L_rep
            rx = y_td[s:s+L_rep]
            if rx.size < L_rep:
                break
            Rk = np.fft.ifft(np.fft.fft(rx) * np.conj(H_ref))
            cp_local += np.abs(Rk) ** 2

        bestCorr_pid = float(np.max(cp_local))
        maxpos_pid   = int(np.argmax(cp_local))

        if lroot not in root_best or bestCorr_pid > root_best[lroot]["bestCorr"]:
            root_best[lroot] = {
                "bestCorr": bestCorr_pid,
                "maxpos"  : maxpos_pid,
                "pid"     : pid
            }

    if not root_best:
        # 一个 root 都没统计到，直接返回“未检测到”
        return None, 0, None, None, None, 0.0

    # 从所有 root 中选“最强”的那个
    best_root     = None
    bestCorr_glob = -1.0
    for lroot, info in root_best.items():
        if info["bestCorr"] > bestCorr_glob:
            bestCorr_glob = info["bestCorr"]
            best_root     = lroot

    # 如果根本没选出 root（极端情况），也返回“未检测到”
    if best_root is None:
        return None, 0, None, None, None, 0.0

    # 为调试打印一下 root 选择结果
    info_root = root_best[best_root]
    # print(f"[STAGE1] best_root={best_root}, "
    #       f"bestCorr={info_root['bestCorr']:.3e}, maxpos={info_root['maxpos']}, pid_ref={info_root['pid']}")

    # =========================================================
    # 阶段 2：只在 best_root 内，用 Type-A NCS/ZCZ 区分 v
    # =========================================================
    plist = roots[best_root]  # [(pid,u,v,Cv), ...]

    # 1) 选该 root 的“参考 preamble”（优先 Cv=0），与 debug 一致
    Xocc_139_ref, u_ref, pid_ref = _pick_root_ref(bank_fd, plist)
    sym_ref_root = prach_ifft_embed_pad(Xocc_139_ref, L_rep)
    H_ref_root   = np.fft.fft(sym_ref_root)

    # 2) 用 root 参考重新算一次 cp_root（和你 debug 时完全一致）
    cp_root = np.zeros(L_rep, dtype=float)
    delay_list = []
    for k in range(int(prachDuration)):
        s = start + k * L_rep
        rx = y_td[s:s+L_rep]
        if rx.size < L_rep:
            break
        Rk = np.fft.ifft(np.fft.fft(rx) * np.conj(H_ref_root))
        cp_k = np.abs(Rk) ** 2  # 第 k 个重复块的功率
        cp_root += cp_k  # 非相干累加

        # 按符号的 delay：只对 cp_k 取 argmax
        delay_k = int(np.argmax(cp_k))
        delay_list.append(delay_k)
    peak_root = float(np.max(cp_root))

    if len(delay_list) > 0:
        delays = np.array(delay_list, dtype=float)

        # 映射到单位圆上的角：theta = 2π * delay / L_rep
        angles = 2 * np.pi * delays / L_rep
        z = np.exp(1j * angles)
        z_mean = np.mean(z)
        mean_angle = np.angle(z_mean)  # (-π, π]

        if mean_angle < 0:
            mean_angle += 2 * np.pi  # 映射到 [0, 2π)

        delay_avg = L_rep * mean_angle / (2 * np.pi)
        maxpos = int(round(delay_avg)) % L_rep
    else:
        # 极端情况：没有任何符号（一般不会发生），兜底用 cp 的 argmax
        maxpos = int(np.argmax(cp_root))

    # 3) 该 root 下所有 v 的理论 cyclicShift 位置（你已经验证过）
    NCS = _get_Ncs(LRA, cfg.zeroCorrelationZoneConfig)
    zcz = (NCS / LRA) * (Fs / delta_f) if NCS > 0 else 0.0

    cv_list = [(pid, v, Cv) for (pid, uu, v, Cv) in plist]
    cyclicShift = np.array(
        [ (Cv % LRA) / LRA * (Fs / delta_f) for (_, v, Cv) in cv_list ],
        dtype=float
    )
    cyclicShift = np.mod(cyclicShift, L_rep)  # 限定到 [0, L_rep)

    # 4) 圆环距离：在 best_root 内选离 maxpos 最近的那个 v
    diff     = np.abs(maxpos - cyclicShift)
    circdist = np.minimum(diff, L_rep - diff)

    idx_v    = int(np.argmin(circdist))
    best_dist = float(circdist[idx_v])

    # 为了稳一点，可以保留“距离不超过 zcz/2”的约束
    if zcz > 0 and best_dist > zcz / 2:
        # 说明在 best_root 内所有 v 与峰位置都不算“合理接近”，认定为检测失败
        # print(f"[STAGE2] best_root={best_root}, but best_dist={best_dist:.2f} > zcz/2={zcz/2:.2f}, treat as no detection.")
        return None, 0, best_root, None, None, bestCorr_glob

    pid_cand, v_cand, Cv_cand = cv_list[idx_v]

    # 5) 粗 TA：offs = maxpos - cyclicShift[v_hat]（映射到 [0, L_rep)）
    offs = maxpos - cyclicShift[idx_v]
    offs = float(np.mod(offs, L_rep))

    indout         = pid_cand
    offset_samples = int(round(offs))
    v_hat          = v_cand
    pid_hat        = pid_cand

    if det_threshold is not None and peak_root < det_threshold:
        # print(f"[THR] best_root={best_root}, peak={peak_root:.3e} < thr={det_threshold:.3e}, treat as no detection.")
        return None, 0, best_root, None, None, peak_root

    # print(f"[STAGE2] best_root={best_root}, v_hat={v_hat}, pid_hat={pid_hat}, "
    #       f"maxpos={maxpos}, offs={offset_samples}, best_dist={best_dist:.2f}, zcz={zcz:.2f}, peak={peak_root:.3e}")

    # 最后一个返回量现在是 peak_root，作为“检测统计量”
    return indout, offset_samples, best_root, v_hat, pid_hat, peak_root

def run_one_trial(
    cfg: PRACHConfig,
    bank_fd: list,
    mapping: list,
    true_pid: int,
    snr_db: float,
    true_delay: int,
    true_cfo: float = 0.0,
    use_cfo: bool = True,
    combine_mode: str = "noncoherent",
    det_threshold=None,
):
    """
    跑一次试验：
      1) 生成一个 preamble 波形（单次发射）
      2) 加 TA、CFO、AWGN
      3) 调用 detect_preamble_and_ta_mode_time
      返回：
        detected_pid, N_TA_hat, success_bool
    """
    # --- 1) 生成发射波形 ---
    X = bank_fd[true_pid]
    tx = prach_tx_one_preamble(cfg, X)   # 含 CP、重复结构
    tx = np.roll(tx, true_delay)

    # --- 2) 注入 CFO（如果需要） ---
    y = tx.copy()


    # --- 3) 加 AWGN ---
    # y, h = apply_tdl_c(tx, cfg.Fs, seed=1, ds=300e-9, fc_hz=7e9, speed_kmh=3.0, nfft=2048, normalize=True)
    y, h = apply_tdl_c_time_variable(tx, fs=cfg.Fs,
                                     speed_kmh=3,
                                     fc_hz=3.5e9,
                                     tdl_type='TDLC',
                                     tdl_ds_ns=300,
                                     nfft_hint=2048)

    # === FIX: CFO 归一化应以单块 IFFT 长度 L_rep 为分母 ===
    if use_cfo and abs(true_cfo) > 0:
        L_rep = int(cfg.N_fft_ra // cfg.N_rep)
        n = np.arange(len(y))
        y *= np.exp(1j * 2 * np.pi * true_cfo * n / L_rep) # TODO L_rep

    y = awgn(y, snr_db,cfg)
    # TODO

    # --- 4) 检测 ---
    best_id, delay_bin, cfo_hat, peak, N_TA_hat, t_hat, delta_t = detect_preamble_and_ta_mode_time(
        cfg,
        y,
        bank_fd,
        mapping,
        combine_mode=combine_mode,
        det_threshold=det_threshold,
    )
    # --- 5) 判成功：ID 正确 & TA 误差在 1/2 CP 以内 ---

    # 真值 TA（时间） = “roll 的样点” / Fs
    t_true = true_delay / cfg.Fs  # 秒

    # 参考 15k/2048 下的 1/2 CP 时间
    Fs_ref = 15e3 * 2048  # 30.72 MHz
    N_half_cp_ref = 144  # 你定义的“1/2 CP = 144 个采样”
    T_half_cp = N_half_cp_ref / Fs_ref  # 秒

    ta_ok = abs(t_hat - t_true) <= T_half_cp

    success =(best_id == true_pid) and ta_ok

    # 返回 preamble ID 和是否检测成功（Required SNR 口径只关心这个）
    return best_id, success
def complex_awgn_noise(shape, sigma2):
    return (rng.normal(scale=np.sqrt(sigma2/2), size=shape) +
            1j*rng.normal(scale=np.sqrt(sigma2/2), size=shape))
def calibrate_threshold_for_fa(
    cfg: PRACHConfig,
    bank_fd: list,
    mapping: list,
    target_fa: float = 1e-3,   # 0.1%
    num_trials: int = 20000,
    combine_mode: str = "noncoherent",
    sigma2=0,
):
    """
    用纯噪声输入标定门限 det_threshold，使得：
        P_FA = P(peak > det_threshold | no signal) ≈ target_fa

    做法：
      1) 生成与一个 preamble 一样长的噪声块
      2) 用 det_threshold=None 调用检测器，拿到 peak（但不关心 best_id）
      3) 收集全部 peak 的经验分布，取 (1 - target_fa) 分位数作为门限
    """
    # 生成一个参考 preamble，拿到长度
    X0 = bank_fd[0]
    tx0 = prach_tx_one_preamble(cfg, X0)
    L = tx0.size

    peaks = []
    for _ in range(num_trials):
        # noise = (rng.normal(size=L) + 1j * rng.normal(size=L)) / np.sqrt(2)
        noise = complex_awgn_noise(L, sigma2)
        # det_threshold=None：只想拿 peak，不要门限裁决
        best_id, delay_bin, cfo_hat, peak, N_TA_hat, t_hat, delta_t = \
            detect_preamble_and_ta_mode_time(
                cfg,
                noise,
                bank_fd,
                mapping,
                combine_mode=combine_mode,
                det_threshold=None,
            )
        peaks.append(peak)

    peaks = np.array(peaks)
    peaks_sorted = np.sort(peaks)

    # 经验分位数：P(peak <= thr) ≈ 1 - target_fa
    idx = int(np.ceil((1 - target_fa) * len(peaks_sorted))) - 1
    idx = max(0, min(idx, len(peaks_sorted) - 1))
    det_threshold = peaks_sorted[idx]

    # 实际的经验 FA（>thr 的比例）
    p_fa_empirical = np.mean(peaks > det_threshold)

    print(f"[CAL] target P_FA={target_fa:.3e}, "
          f"got det_threshold={det_threshold:.3e}, empirical P_FA≈{p_fa_empirical:.3e}")

    return det_threshold, p_fa_empirical



import os,csv
from datetime import datetime
def sigma2_from_re_snr(cfg, snr_db, P_ref_td):
    """snr_db 定义为：有效子载波(=L_RA)上的 RE-SNR"""
    L_rep = int(cfg.N_fft_ra // cfg.N_rep)
    occ_ratio = cfg.L_RA / L_rep
    sample_snr_linear = (10 ** (snr_db / 10.0)) * occ_ratio
    return P_ref_td / sample_snr_linear
def cfg_signature(cfg):
    # 你也可以按需再加字段（如 RootIndex / ZCZ / restricted_type 等）
    return f"name={cfg.name}|LRA={cfg.L_RA}|Nfft={cfg.N_fft_ra}|Nrep={cfg.N_rep}|ZCZ={cfg.zeroCorrelationZoneConfig}|rtype={cfg.restricted_type}"
def load_thr_csv(csv_path):
    """返回 dict: (cfg_key, snr_key) -> row(dict)"""
    table = {}
    if not os.path.exists(csv_path):
        return table

    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                cfg_key = row["cfg_key"]
                snr_key = float(row["snr_key"])
                table[(cfg_key, snr_key)] = row
            except Exception:
                continue
    return table


def get_or_calibrate_threshold(
        cfg, bank_fd, mapping,
        snr_db: float,
        P_ref_td: float,
        csv_path: str,
        target_fa: float = 1e-3,
        num_trials: int = 20000,
        combine_mode: str = "noncoherent",
        rng=None,
):
    cfg_key = cfg_signature(cfg)
    snr_key = round(float(snr_db), 1)

    table = load_thr_csv(csv_path)

    # 命中：直接用
    if (cfg_key, snr_key) in table:
        row = table[(cfg_key, snr_key)]
        thr = float(row["det_threshold"])
        return thr

    # 未命中：现场标定
    sigma2 = float(sigma2_from_re_snr(cfg, snr_db, P_ref_td))
    thr, pfa = calibrate_threshold_for_fa(
        cfg,
        bank_fd,
        mapping,
        sigma2=sigma2,
        target_fa=target_fa,
        num_trials=4000,
        combine_mode="noncoherent",
    )
    append_thr_csv(csv_path, {
        "cfg_key": cfg_key,
        "snr_key": snr_key,
        "snr_db": float(snr_db),
        "sigma2": sigma2,
        "det_threshold": thr,
        "p_fa_emp": pfa,
        "target_fa": target_fa,
        "num_trials": num_trials,
    })

    return thr
def append_thr_csv(csv_path, row_dict):
    """不存在则创建并写表头；存在则 append 行"""
    file_exists = os.path.exists(csv_path)
    fieldnames = ["cfg_key", "snr_key", "snr_db", "sigma2", "det_threshold", "p_fa_emp", "target_fa", "num_trials"]

    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(row_dict)


def simulate_preamble_error_curve(
    cfg: PRACHConfig,
    snr_db_list,
    num_trials: int = 1000,
    true_pid: int = 11,
    true_delay: int = 128,
    true_cfo: float = 0.05,
    use_cfo: bool = True,
    csv_prefix: str = "preamble_err_curve", thr_csv="fa_thr_table.csv", fa_trials=5000, target_fa=1e-3
):
    """
    统计“检测错误概率”：
        P_err = 1 - P_det

    并画 log-scale 曲线 + 保存 CSV。
    """
    NUM_PREAMBLES = 64
    bank_fd, mapping = build_preamble_bank(cfg, NUM_PREAMBLES)

    # === 先用纯噪声标定门限，使 P_FA≈0.1% ===
    target_fa = 1*1e-3

    p_fa_emp = 1 * 1e-3 # for test

    p_md_list = []
    tx0 = prach_tx_one_preamble(cfg, bank_fd[0])
    P_ref_td = float(np.mean(np.abs(tx0) ** 2))

    for snr in snr_db_list:
        det_threshold = get_or_calibrate_threshold(
            cfg, bank_fd, mapping,
            snr_db=snr,
            P_ref_td=P_ref_td,
            csv_path=thr_csv,
            target_fa=target_fa,
            num_trials=fa_trials,
            combine_mode="noncoherent",
            rng=rng
        )
        ok_cnt = 0
        for _ in range(num_trials):
            print(_)
            _, success = run_one_trial(
                cfg=cfg,
                bank_fd=bank_fd,
                mapping=mapping,
                true_pid=true_pid,
                snr_db=snr,
                true_delay=true_delay,
                true_cfo=true_cfo,
                use_cfo=use_cfo,
                combine_mode="noncoherent",
                det_threshold=det_threshold,
            )
            if success:
                ok_cnt += 1

        p_det = ok_cnt / num_trials
        p_md = 1 - p_det
        p_md_list.append(p_md)

        print(f"SNR={snr:5.1f} dB, P_MD={p_md:.6f}")

    # ===== 保存 CSV =====
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"{csv_prefix}_{cfg.name}_pid{true_pid}_{ts}.csv"

    with open(csv_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["SNR_dB", "P_MD"])
        for snr, p in zip(snr_db_list, p_md_list):
            writer.writerow([snr, p])

    print(f"➡ CSV saved: {csv_name}")

    # ===== 在“P_FA≈target_fa”下，找 P_MD≈1% 的 Required SNR =====
    target_md = 0.01  # 1%
    req_snr = None
    for snr, p in zip(snr_db_list, p_md_list):
        if p <= target_md:
            req_snr = snr
            break

    if req_snr is not None:
        print(f"Required SNR ≈ {req_snr:.1f} dB "
              f"(P_MD≤{target_md:.2%}, P_FA≈{p_fa_emp:.3e}, thr={det_threshold:.3e})")
    else:
        print(f"⚠ 在扫描的 SNR 范围内，未达到 P_MD≤{target_md:.2%}，"
              f"当前 P_FA≈{p_fa_emp:.3e}, thr={det_threshold:.3e}")

    # ===== 画图：纵轴对数（P_MD）=====
    plt.figure()
    plt.semilogy(snr_db_list, p_md_list, marker="o")
    plt.xlabel("SNR [dB]")
    plt.ylabel("Missed Detection Probability P_MD (log scale)")
    plt.title(f"PRACH {cfg.name} P_MD vs SNR (pid={true_pid}, delay={true_delay})")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()

    return snr_db_list, p_md_list


#
if __name__ == "__main__":
    # cfg = CFG_C2
    # cfg = CFG_B4
    cfg = CFG_F0
    snr_list = np.arange(-6,-4, 2)  # -12, -10, ..., 12 dB
    ppm_bs = 0.05  # BS 频偏 [ppm]
    ppm_ue = 0.10  # UE 频偏 [ppm]
    delta_f_cfo = (ppm_ue + ppm_bs) * 1e-6 * 3.5e9
    true_cfo = delta_f_cfo / cfg.delta_f_ra_hz  # ε

    simulate_preamble_error_curve(
        cfg,
        snr_db_list=snr_list,
        num_trials=100,
        true_pid=12,
        true_delay=50,   # 已在 ZCZ 范围内
        true_cfo=true_cfo,
        use_cfo=True,
        csv_prefix="test_preamble_err_test"
    )